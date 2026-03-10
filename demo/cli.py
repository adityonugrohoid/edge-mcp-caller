#!/usr/bin/env python3
"""Interactive CLI demo — end-to-end specialist model → MCP servers (filesystem + git).

User query → Ollama (edge-mcp-caller) → JSON parse → MCP tools/call → result.

Usage:
    python demo/cli.py                        # interactive mode, current directory
    python demo/cli.py /path/to/explore       # interactive, specify target directory
    python demo/cli.py --repo /path/to/repo   # enable git tools for a repo
    python demo/cli.py -n 10                  # batch mode: run 10 eval examples
    python demo/cli.py -n 5 --verbose         # batch mode with per-query detail
"""

import argparse
import asyncio
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn
from rich.table import Table
from rich.text import Text

# Import our mcp/client.py without shadowing the installed `mcp` SDK package.
# The local mcp/ directory name conflicts with the PyPI `mcp` package, so we
# load our module directly by file path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "mcp_client_bridge", PROJECT_ROOT / "mcp" / "client.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MCPClientBridge = _mod.MCPClientBridge
parse_model_output = _mod.parse_model_output
GIT_TOOLS = _mod.GIT_TOOLS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "edge-mcp-caller:latest"
EVAL_FILE = PROJECT_ROOT / "data" / "eval.jsonl"

console = Console()


# ---------------------------------------------------------------------------
# Git repo auto-detection
# ---------------------------------------------------------------------------


def detect_git_repo(dirs: list[str]) -> str | None:
    """Try to find a git repo root from the given directories."""
    for d in dirs:
        try:
            result = subprocess.run(
                ["git", "-C", d, "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    return None


# ---------------------------------------------------------------------------
# Eval data loader
# ---------------------------------------------------------------------------


def load_eval_queries(n: int) -> list[dict]:
    """Load n eval examples from data/eval.jsonl.

    Returns:
        List of {"query": str, "expected_tool": str, "expected_args": dict}
    """
    examples = []
    with open(EVAL_FILE) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            msgs = entry["messages"]
            query = msgs[0]["content"]
            expected = json.loads(msgs[1]["content"])
            examples.append({
                "query": query,
                "expected_tool": expected["tool"],
                "expected_args": expected["args"],
            })
            if len(examples) >= n:
                break
    return examples


# ---------------------------------------------------------------------------
# Ollama inference
# ---------------------------------------------------------------------------


async def call_model(client: httpx.AsyncClient, query: str) -> dict:
    """Call the specialist model and return full response data.

    Returns:
        {
            "raw": str, "tool": str|None, "args": dict,
            "prompt_tokens": int, "eval_tokens": int,
            "latency_ms": float, "error": str|None,
            "payload": dict,  # the exact request sent to Ollama
        }
    """
    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "options": {"temperature": 0},
        "messages": [{"role": "user", "content": query}],
    }

    start = time.perf_counter()
    resp = await client.post(OLLAMA_URL, json=payload, timeout=30.0)
    elapsed_ms = (time.perf_counter() - start) * 1000

    data = resp.json()

    if "error" in data:
        return {
            "raw": "",
            "tool": None,
            "args": {},
            "prompt_tokens": 0,
            "eval_tokens": 0,
            "latency_ms": elapsed_ms,
            "error": data["error"],
            "payload": payload,
        }

    content = data.get("message", {}).get("content", "").strip()
    parsed = parse_model_output(content)

    # Extract Ollama-reported metrics
    total_ns = data.get("total_duration", 0)
    load_ns = data.get("load_duration", 0)
    model_latency_ms = (total_ns - load_ns) / 1_000_000

    return {
        "raw": content,
        "tool": parsed["tool"],
        "args": parsed["args"],
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "eval_tokens": data.get("eval_count", 0),
        "latency_ms": model_latency_ms,
        "error": parsed["error"],
        "payload": payload,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def show_banner(allowed_dirs: list[str], repo_path: str | None) -> None:
    """Show startup banner."""
    dirs_text = ", ".join(allowed_dirs)
    banner = Text()
    banner.append("Edge MCP Caller", style="bold magenta")
    banner.append(" — specialist 270M model + MCP servers\n\n")
    banner.append("Model: ", style="dim")
    banner.append(f"{MODEL_NAME}\n")

    # Servers
    servers = []
    if allowed_dirs:
        servers.append(f"filesystem ({dirs_text})")
    if repo_path:
        servers.append(f"git ({repo_path})")
    banner.append("Servers: ", style="dim")
    banner.append(", ".join(servers) + "\n\n")

    banner.append("Commands:\n", style="dim")
    banner.append("  tools   ", style="cyan")
    banner.append("— list available MCP tools\n")
    banner.append("  help    ", style="cyan")
    banner.append("— show this help\n")
    banner.append("  quit    ", style="cyan")
    banner.append("— exit\n")
    banner.append("\nAsk about your filesystem or git repository.")
    console.print(Panel(banner, border_style="blue"))


def show_model_step(result: dict) -> None:
    """Display the model inference step with full detail."""
    console.print()

    # 1. Prompt sent to model
    prompt_json = json.dumps(result["payload"], indent=2)
    console.print(Panel(
        prompt_json,
        title="[bold]1. Ollama Request[/bold]",
        subtitle=f"POST {OLLAMA_URL}",
        border_style="blue",
    ))

    # 2. Model response
    if result["error"]:
        console.print(Panel(
            f"[red]Error:[/red] {result['error']}\n[dim]Raw:[/dim] {escape(result['raw'])}",
            title="[bold]2. Model Response[/bold]",
            border_style="red",
        ))
        return

    response_lines = [
        f'[cyan]Raw output:[/cyan]  {escape(result["raw"])}',
        f'[cyan]Parsed tool:[/cyan] {result["tool"]}',
        f'[cyan]Parsed args:[/cyan] {json.dumps(result["args"])}',
    ]
    console.print(Panel(
        "\n".join(response_lines),
        title="[bold]2. Model Response[/bold]",
        border_style="blue",
    ))

    # 3. Model metrics
    metrics = Table(title="Model Metrics", border_style="dim", show_header=False)
    metrics.add_column("Metric", style="dim")
    metrics.add_column("Value", style="bold")
    metrics.add_row("Prompt tokens", str(result["prompt_tokens"]))
    metrics.add_row("Output tokens", str(result["eval_tokens"]))
    metrics.add_row("Inference latency", f'{result["latency_ms"]:.0f}ms')
    metrics.add_row("Schema tokens", "0 (tools baked into weights)")
    console.print(metrics)


def show_mcp_step(
    tool: str, args: dict, server: str,
    result_text: str | None, error: str | None, latency_ms: float,
) -> None:
    """Display the MCP tool execution step with full detail."""
    server_label = f"@modelcontextprotocol/server-{server}"

    # 3. MCP request
    mcp_request = json.dumps(
        {"method": "tools/call", "params": {"name": tool, "arguments": args}},
        indent=2,
    )
    console.print(Panel(
        mcp_request,
        title="[bold]3. MCP Request[/bold]",
        subtitle=f"stdio → {server_label}",
        border_style="yellow",
    ))

    # 4. MCP response
    if error:
        console.print(Panel(
            f"[red]{error}[/red]",
            title="[bold]4. MCP Response[/bold]",
            border_style="red",
        ))
    else:
        text = result_text or "(empty response)"
        lines = text.split("\n")
        if len(lines) > 50:
            display = "\n".join(lines[:50])
            display += f"\n[dim]... ({len(lines) - 50} more lines, {len(text)} chars total)[/dim]"
        else:
            display = text
        console.print(Panel(
            display,
            title="[bold]4. MCP Response[/bold]",
            border_style="green",
        ))

    # 5. MCP metrics
    metrics = Table(title="MCP Metrics", border_style="dim", show_header=False)
    metrics.add_column("Metric", style="dim")
    metrics.add_column("Value", style="bold")
    metrics.add_row("Server", server_label)
    metrics.add_row("Tool called", tool)
    metrics.add_row("Execution latency", f"{latency_ms:.0f}ms")
    result_len = len(result_text) if result_text else 0
    metrics.add_row("Response size", f"{result_len} chars")
    console.print(metrics)


def show_summary(model_result: dict, mcp_latency_ms: float, server: str) -> None:
    """Display end-to-end summary."""
    total_ms = model_result["latency_ms"] + mcp_latency_ms
    table = Table(title="End-to-End Summary", border_style="magenta")
    table.add_column("Stage", style="cyan")
    table.add_column("Detail")
    table.add_column("Time", justify="right", style="bold")
    table.add_row(
        "Model inference",
        f'{model_result["prompt_tokens"]} prompt → {model_result["eval_tokens"]} output tokens',
        f'{model_result["latency_ms"]:.0f}ms',
    )
    table.add_row(
        f"MCP execution ({server})",
        f'{model_result["tool"]}({json.dumps(model_result["args"])})',
        f"{mcp_latency_ms:.0f}ms",
    )
    table.add_row("Total", "", f"{total_ms:.0f}ms", style="bold magenta")
    console.print(table)


def show_tools(tools: list[dict]) -> None:
    """Display available MCP tools as a table."""
    table = Table(title="Available MCP Tools", border_style="blue")
    table.add_column("Tool", style="cyan")
    table.add_column("Server", style="dim")
    table.add_column("Description")
    for t in tools:
        table.add_row(t["name"], t.get("server", ""), t["description"])
    console.print(table)


# ---------------------------------------------------------------------------
# Pipeline execution (shared between interactive and batch)
# ---------------------------------------------------------------------------


async def run_pipeline(
    http: httpx.AsyncClient,
    bridge: MCPClientBridge,
    query: str,
    verbose: bool = True,
) -> dict:
    """Run full pipeline for a single query. Returns metrics dict.

    Args:
        http: HTTP client for Ollama.
        bridge: Connected MCP bridge.
        query: User query string.
        verbose: If True, show detailed per-step panels.

    Returns:
        {
            "query": str, "tool": str|None, "args": dict,
            "prompt_tokens": int, "eval_tokens": int,
            "model_latency_ms": float, "mcp_latency_ms": float,
            "mcp_result": str|None, "mcp_error": str|None,
            "model_error": str|None, "server": str|None,
        }
    """
    if verbose:
        console.print(f"\n[bold]=== Pipeline Start ===[/bold]")

    # Model inference
    model_result = await call_model(http, query)

    if verbose:
        show_model_step(model_result)

    if model_result["error"]:
        return {
            "query": query,
            "raw_model_output": model_result["raw"],
            "tool": model_result["tool"],
            "args": model_result["args"],
            "prompt_tokens": model_result["prompt_tokens"],
            "eval_tokens": model_result["eval_tokens"],
            "model_latency_ms": model_result["latency_ms"],
            "mcp_latency_ms": 0.0,
            "mcp_result": None,
            "mcp_error": None,
            "model_error": model_result["error"],
            "server": None,
        }

    # Determine server
    server = bridge.server_for_tool(model_result["tool"])

    # MCP execution
    mcp_latency_ms = 0.0
    mcp_result_text = None
    mcp_error = None
    try:
        mcp_start = time.perf_counter()
        mcp_result_text = await bridge.call_tool(
            model_result["tool"], model_result["args"]
        )
        mcp_latency_ms = (time.perf_counter() - mcp_start) * 1000
    except Exception as e:
        mcp_latency_ms = (time.perf_counter() - mcp_start) * 1000
        mcp_error = str(e)

    if verbose:
        show_mcp_step(
            model_result["tool"], model_result["args"], server,
            mcp_result_text, mcp_error, mcp_latency_ms,
        )
        show_summary(model_result, mcp_latency_ms, server)
        console.print("[bold]=== Pipeline End ===[/bold]")

    return {
        "query": query,
        "raw_model_output": model_result["raw"],
        "tool": model_result["tool"],
        "args": model_result["args"],
        "prompt_tokens": model_result["prompt_tokens"],
        "eval_tokens": model_result["eval_tokens"],
        "model_latency_ms": model_result["latency_ms"],
        "mcp_latency_ms": mcp_latency_ms,
        "mcp_result": mcp_result_text,
        "mcp_error": mcp_error,
        "model_error": None,
        "server": server,
    }


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------


async def run_batch(
    allowed_dirs: list[str], repo_path: str | None, n: int, verbose: bool,
) -> None:
    """Run n eval examples through the full pipeline."""
    servers = []
    if allowed_dirs:
        servers.append("filesystem")
    if repo_path:
        servers.append("git")

    console.print(Panel(
        f"[bold magenta]Batch Mode[/bold magenta] — {n} eval queries through full pipeline\n"
        f"Model: {MODEL_NAME}\n"
        f"Eval file: {EVAL_FILE}\n"
        f"Servers: {', '.join(servers)}\n"
        f"Allowed dirs: {', '.join(allowed_dirs)}\n"
        f"Repo path: {repo_path or '(none)'}\n"
        f"Verbose: {verbose}",
        border_style="blue",
    ))

    # Load eval data
    examples = load_eval_queries(n)
    actual_n = len(examples)
    if actual_n < n:
        console.print(f"[yellow]Warning:[/yellow] eval set has {actual_n} examples (requested {n})")

    # Connect MCP
    console.print("[dim]Starting MCP servers...[/dim]")
    bridge = MCPClientBridge(allowed_dirs=allowed_dirs, repo_path=repo_path)
    try:
        await bridge.connect()
    except Exception as e:
        console.print(f"[red]Failed to start MCP servers:[/red] {e}")
        return

    tools = await bridge.list_tools()
    console.print(f"[green]Connected.[/green] {len(tools)} tools available.\n")

    # Run queries
    results: list[dict] = []
    all_start = time.perf_counter()

    async with httpx.AsyncClient() as http:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[dim]{task.fields[status]}[/dim]"),
            console=console,
            disable=verbose,  # hide progress bar when verbose (panels take over)
        ) as progress:
            task = progress.add_task("Running pipeline...", total=actual_n, status="")

            for i, ex in enumerate(examples):
                progress.update(task, status=f"query {i+1}: {ex['query'][:40]}...")

                if verbose:
                    console.print(f"\n[bold cyan]--- Query {i+1}/{actual_n} ---[/bold cyan]")
                    console.print(f"[dim]Query:[/dim] {ex['query']}")
                    console.print(f"[dim]Expected:[/dim] {ex['expected_tool']}({json.dumps(ex['expected_args'])})")

                r = await run_pipeline(http, bridge, ex["query"], verbose=verbose)

                # Add eval comparison
                r["expected_tool"] = ex["expected_tool"]
                r["expected_args"] = ex["expected_args"]
                r["tool_correct"] = r["tool"] == ex["expected_tool"]
                r["args_correct"] = r["args"] == ex["expected_args"]
                r["combined_correct"] = r["tool_correct"] and r["args_correct"]
                r["mcp_success"] = r["mcp_error"] is None and r["model_error"] is None

                results.append(r)
                progress.advance(task)

    total_time = (time.perf_counter() - all_start) * 1000
    await bridge.close()

    # Aggregate metrics
    tool_correct = sum(1 for r in results if r["tool_correct"])
    args_correct = sum(1 for r in results if r["args_correct"])
    combined_correct = sum(1 for r in results if r["combined_correct"])
    mcp_success = sum(1 for r in results if r["mcp_success"])
    model_errors = sum(1 for r in results if r["model_error"])
    mcp_errors = sum(1 for r in results if r["mcp_error"])
    avg_prompt = sum(r["prompt_tokens"] for r in results) / actual_n if actual_n else 0
    avg_eval = sum(r["eval_tokens"] for r in results) / actual_n if actual_n else 0
    avg_model_lat = sum(r["model_latency_ms"] for r in results) / actual_n if actual_n else 0
    avg_mcp_lat = sum(r["mcp_latency_ms"] for r in results if r["mcp_latency_ms"] > 0) / max(1, actual_n - model_errors)
    avg_total_lat = avg_model_lat + avg_mcp_lat

    # Per-tool breakdown
    tool_stats: dict[str, dict] = {}
    for r in results:
        t = r["expected_tool"]
        if t not in tool_stats:
            tool_stats[t] = {"count": 0, "tool_ok": 0, "args_ok": 0, "combined_ok": 0, "mcp_ok": 0}
        tool_stats[t]["count"] += 1
        tool_stats[t]["tool_ok"] += int(r["tool_correct"])
        tool_stats[t]["args_ok"] += int(r["args_correct"])
        tool_stats[t]["combined_ok"] += int(r["combined_correct"])
        tool_stats[t]["mcp_ok"] += int(r["mcp_success"])

    # Display results
    console.print(f"\n{'='*60}")

    # Accuracy table
    acc_table = Table(title=f"Batch Results — {actual_n} queries", border_style="magenta")
    acc_table.add_column("Metric", style="cyan")
    acc_table.add_column("Count", justify="right")
    acc_table.add_column("Rate", justify="right", style="bold")
    acc_table.add_row("Tool accuracy", f"{tool_correct}/{actual_n}", f"{tool_correct/actual_n*100:.1f}%")
    acc_table.add_row("Args accuracy", f"{args_correct}/{actual_n}", f"{args_correct/actual_n*100:.1f}%")
    acc_table.add_row("Combined accuracy", f"{combined_correct}/{actual_n}", f"{combined_correct/actual_n*100:.1f}%")
    acc_table.add_row("MCP execution success", f"{mcp_success}/{actual_n}", f"{mcp_success/actual_n*100:.1f}%")
    acc_table.add_row("Model errors", str(model_errors), "")
    acc_table.add_row("MCP errors", str(mcp_errors), "")
    console.print(acc_table)

    # Per-tool table
    tool_table = Table(title="Per-Tool Breakdown", border_style="blue")
    tool_table.add_column("Tool", style="cyan")
    tool_table.add_column("Count", justify="right")
    tool_table.add_column("Tool Acc", justify="right")
    tool_table.add_column("Args Acc", justify="right")
    tool_table.add_column("Combined", justify="right", style="bold")
    tool_table.add_column("MCP OK", justify="right")
    for t in sorted(tool_stats):
        s = tool_stats[t]
        c = s["count"]
        tool_table.add_row(
            t, str(c),
            f"{s['tool_ok']/c*100:.1f}%",
            f"{s['args_ok']/c*100:.1f}%",
            f"{s['combined_ok']/c*100:.1f}%",
            f"{s['mcp_ok']/c*100:.1f}%",
        )
    console.print(tool_table)

    # Latency table
    lat_table = Table(title="Latency & Token Metrics", border_style="dim")
    lat_table.add_column("Metric", style="dim")
    lat_table.add_column("Value", style="bold")
    lat_table.add_row("Avg prompt tokens", f"{avg_prompt:.0f}")
    lat_table.add_row("Avg output tokens", f"{avg_eval:.0f}")
    lat_table.add_row("Schema tokens", "0 (tools baked into weights)")
    lat_table.add_row("Avg model latency", f"{avg_model_lat:.0f}ms")
    lat_table.add_row("Avg MCP latency", f"{avg_mcp_lat:.0f}ms")
    lat_table.add_row("Avg total latency", f"{avg_total_lat:.0f}ms")
    lat_table.add_row("Total wall time", f"{total_time/1000:.1f}s")
    lat_table.add_row("Throughput", f"{actual_n/(total_time/1000):.1f} queries/sec")
    console.print(lat_table)

    # Show failures if any
    failures = [r for r in results if not r["combined_correct"] or r["mcp_error"]]
    if failures:
        console.print(f"\n[yellow]Failures ({len(failures)}):[/yellow]")
        fail_table = Table(border_style="red", show_lines=True)
        fail_table.add_column("#", style="dim", width=4)
        fail_table.add_column("Query", max_width=40)
        fail_table.add_column("Expected")
        fail_table.add_column("Got")
        fail_table.add_column("Issue", style="red")
        for r in failures[:20]:  # cap at 20 shown
            idx = results.index(r) + 1
            expected = f"{r['expected_tool']}({json.dumps(r['expected_args'])})"
            got = f"{r['tool']}({json.dumps(r['args'])})" if r["tool"] else "(none)"
            issue = []
            if r["model_error"]:
                issue.append(f"model: {r['model_error']}")
            if not r["tool_correct"]:
                issue.append("wrong tool")
            elif not r["args_correct"]:
                issue.append("wrong args")
            if r["mcp_error"]:
                issue.append(f"mcp: {r['mcp_error'][:30]}")
            fail_table.add_row(str(idx), r["query"][:40], expected[:50], got[:50], "; ".join(issue))
        if len(failures) > 20:
            console.print(f"  [dim]... and {len(failures) - 20} more[/dim]")
        console.print(fail_table)

    # Save raw results
    RESULTS_DIR = PROJECT_ROOT / "results"
    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / "cli_batch.json"

    save_data = {
        "metadata": {
            "model": MODEL_NAME,
            "num_queries": actual_n,
            "allowed_dirs": allowed_dirs,
            "repo_path": repo_path,
            "total_time_s": round(total_time / 1000, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "summary": {
            "tool_accuracy": round(tool_correct / actual_n * 100, 1),
            "args_accuracy": round(args_correct / actual_n * 100, 1),
            "combined_accuracy": round(combined_correct / actual_n * 100, 1),
            "mcp_success_rate": round(mcp_success / actual_n * 100, 1),
            "model_errors": model_errors,
            "mcp_errors": mcp_errors,
            "avg_prompt_tokens": round(avg_prompt, 1),
            "avg_eval_tokens": round(avg_eval, 1),
            "avg_model_latency_ms": round(avg_model_lat, 1),
            "avg_mcp_latency_ms": round(avg_mcp_lat, 1),
            "avg_total_latency_ms": round(avg_total_lat, 1),
        },
        "per_tool": {
            t: {
                "count": s["count"],
                "tool_accuracy": round(s["tool_ok"] / s["count"] * 100, 1),
                "args_accuracy": round(s["args_ok"] / s["count"] * 100, 1),
                "combined_accuracy": round(s["combined_ok"] / s["count"] * 100, 1),
                "mcp_success_rate": round(s["mcp_ok"] / s["count"] * 100, 1),
            }
            for t, s in sorted(tool_stats.items())
        },
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    console.print(f"\n[green]Raw results saved:[/green] {output_file}")

    console.print(f"[dim]MCP servers stopped.[/dim]")


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------


async def run_interactive(allowed_dirs: list[str], repo_path: str | None) -> None:
    """Main interactive loop."""
    show_banner(allowed_dirs, repo_path)

    # Connect to MCP servers
    servers = []
    if allowed_dirs:
        servers.append("filesystem")
    if repo_path:
        servers.append("git")
    console.print(f"[dim]Starting MCP servers ({', '.join(servers)})...[/dim]")

    bridge = MCPClientBridge(allowed_dirs=allowed_dirs, repo_path=repo_path)
    try:
        await bridge.connect()
    except Exception as e:
        console.print(f"[red]Failed to start MCP servers:[/red] {e}")
        console.print("[dim]Make sure Node.js 18+ is installed (npx must be available)[/dim]")
        return

    # List tools on startup
    tools = await bridge.list_tools()
    console.print(f"[green]Connected.[/green] {len(tools)} tools available.\n")

    # Verify Ollama is running
    async with httpx.AsyncClient() as http:
        try:
            resp = await http.get("http://localhost:11434/api/tags", timeout=5.0)
            models = [m["name"] for m in resp.json().get("models", [])]
            if MODEL_NAME not in models:
                console.print(f"[yellow]Warning:[/yellow] {MODEL_NAME} not found in Ollama.")
                console.print(f"[dim]Available: {', '.join(models)}[/dim]")
                console.print(f"[dim]Run: ollama pull {MODEL_NAME}[/dim]\n")
        except Exception:
            console.print("[yellow]Warning:[/yellow] Cannot reach Ollama at localhost:11434")
            console.print("[dim]Make sure Ollama is running: ollama serve[/dim]\n")

    # Interactive loop
    async with httpx.AsyncClient() as http:
        while True:
            try:
                query = console.input("\n[bold blue]>[/bold blue] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Bye.[/dim]")
                break

            if not query:
                continue

            cmd = query.lower()
            if cmd in ("quit", "exit", "q"):
                console.print("[dim]Bye.[/dim]")
                break
            elif cmd == "tools":
                show_tools(tools)
                continue
            elif cmd == "help":
                show_banner(allowed_dirs, repo_path)
                continue

            await run_pipeline(http, bridge, query, verbose=True)

    await bridge.close()
    console.print("[dim]MCP servers stopped.[/dim]")


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Edge MCP Caller — specialist 270M model + MCP servers (filesystem + git)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python demo/cli.py                          # interactive, auto-detect git\n"
            "  python demo/cli.py /path/to/dir             # interactive with custom dir\n"
            "  python demo/cli.py --repo /path/to/repo     # explicit git repo path\n"
            "  python demo/cli.py --no-git                 # filesystem only, skip git\n"
            "  python demo/cli.py -n 10                    # batch: 10 eval queries\n"
            "  python demo/cli.py -n 5 --verbose           # batch with per-query detail\n"
        ),
    )
    parser.add_argument(
        "dirs", nargs="*", default=[],
        help="Allowed directories for MCP filesystem server (default: current directory)",
    )
    parser.add_argument(
        "--repo", default=None,
        help="Git repository path for MCP git server (default: auto-detect from dirs)",
    )
    parser.add_argument(
        "--no-git", action="store_true",
        help="Disable git server (filesystem only)",
    )
    parser.add_argument(
        "-n", "--num-runs", type=int, default=0,
        help="Number of eval examples to run in batch mode (0 = interactive)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show per-query detail in batch mode (default: progress bar only)",
    )

    args = parser.parse_args()

    # Resolve allowed directories
    allowed_dirs = [str(Path(d).resolve()) for d in args.dirs] if args.dirs else [str(Path.cwd())]
    for d in allowed_dirs:
        if not Path(d).is_dir():
            console.print(f"[red]Error:[/red] '{d}' is not a directory")
            sys.exit(1)

    # Resolve git repo path
    repo_path: str | None = None
    if args.no_git:
        repo_path = None
    elif args.repo:
        repo_path = str(Path(args.repo).resolve())
        if not Path(repo_path).is_dir():
            console.print(f"[red]Error:[/red] repo path '{repo_path}' is not a directory")
            sys.exit(1)
    else:
        # Auto-detect git repo from allowed dirs
        repo_path = detect_git_repo(allowed_dirs)
        if repo_path:
            console.print(f"[dim]Auto-detected git repo: {repo_path}[/dim]")

    if args.num_runs > 0:
        asyncio.run(run_batch(allowed_dirs, repo_path, args.num_runs, args.verbose))
    else:
        asyncio.run(run_interactive(allowed_dirs, repo_path))


if __name__ == "__main__":
    main()
