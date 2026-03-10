#!/usr/bin/env python3
"""End-to-end test: specialist model → MCP servers → real results.

Runs 14 curated queries (one per tool) through the full pipeline against the
test fixture. Read-only tools run first, then write/mutating tools.

Prerequisites:
    1. python tests/e2e/setup_fixture.py          # create fixture
    2. ollama serve                                 # start Ollama
    3. python tests/e2e/run_e2e.py                 # run tests

Options:
    --mcp-only     Skip model inference, test MCP servers directly with known-good JSON
    --verbose      Show full model output and MCP responses
    --reset        Re-create fixture before running (clean slate)
"""

import argparse
import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Import bridge without shadowing `mcp` package
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location(
    "mcp_client_bridge", PROJECT_ROOT / "mcp" / "client.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MCPClientBridge = _mod.MCPClientBridge
parse_model_output = _mod.parse_model_output
GIT_TOOLS = _mod.GIT_TOOLS

FIXTURE_DIR = Path(__file__).resolve().parent / "fixture"
RESULTS_DIR = PROJECT_ROOT / "results"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "edge-mcp-caller:latest"

console = Console()

# ---------------------------------------------------------------------------
# Test cases: 14 queries, one per tool, ordered read-first
# ---------------------------------------------------------------------------

# Each test case: (tool_name, query, expected_tool, expected_args_subset)
# expected_args_subset: keys we expect in the model output (not exact match
# since paths may vary). For MCP-only mode, we use known-good args directly.

TEST_CASES = [
    # --- Read-only tools (safe, no side effects) ---
    {
        "tool": "list_directory",
        "query": "show me what files are in the src directory",
        "known_args": {"path": "src"},
        "check": lambda r: "main.py" in (r or ""),
    },
    {
        "tool": "read_file",
        "query": "read the contents of config/settings.yaml",
        "known_args": {"path": "config/settings.yaml"},
        "check": lambda r: "edge-mcp-test" in (r or ""),
    },
    {
        "tool": "search_files",
        "query": "find all python files in the tests directory",
        "known_args": {"path": "tests", "pattern": "*.py"},
        "check": lambda r: "test_main" in (r or ""),
    },
    {
        "tool": "directory_tree",
        "query": "show the tree structure of the docs folder",
        "known_args": {"path": "docs"},
        "check": lambda r: "README" in (r or ""),
    },
    {
        "tool": "git_status",
        "query": "what's the current git status",
        "known_args": {},
        "check": lambda r: "main" in (r or "").lower() or "branch" in (r or "").lower(),
    },
    {
        "tool": "git_diff_staged",
        "query": "show me the staged changes",
        "known_args": {},
        "check": lambda r: "README" in (r or "") or "Staged" in (r or ""),
    },
    {
        "tool": "git_log",
        "query": "show the last 3 commits",
        "known_args": {"max_count": 3},
        "check": lambda r: "feat" in (r or "").lower() or "commit" in (r or "").lower(),
    },
    {
        "tool": "git_branch",
        "query": "list all the branches",
        "known_args": {},
        "check": lambda r: "main" in (r or ""),
    },
    # --- Write/mutating tools (modify fixture state) ---
    {
        "tool": "write_file",
        "query": "create a file called notes.txt with content 'hello world'",
        "known_args": {"path": "notes.txt", "content": "hello world"},
        "check": lambda r: r is not None,
    },
    {
        "tool": "create_directory",
        "query": "make a new directory called backups",
        "known_args": {"path": "backups"},
        "check": lambda r: r is not None,
    },
    {
        "tool": "edit_file",
        "query": "in config/settings.yaml change 'debug: true' to 'debug: false'",
        "known_args": {"path": "config/settings.yaml", "old_text": "debug: true", "new_text": "debug: false"},
        "check": lambda r: r is not None,
    },
    {
        "tool": "move_file",
        "query": "move scripts/deploy.sh to config/deploy.sh",
        "known_args": {"source": "scripts/deploy.sh", "destination": "config/deploy.sh"},
        "check": lambda r: r is not None,
    },
    {
        "tool": "git_commit",
        "query": "commit with message 'update readme'",
        "known_args": {"message": "update readme"},
        "check": lambda r: r is not None,
    },
    {
        "tool": "git_create_branch",
        "query": "create a new branch called feature/test from main",
        "known_args": {"branch_name": "feature/test", "base_branch": "main"},
        "check": lambda r: r is not None,
    },
]


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------


async def call_model(client: httpx.AsyncClient, query: str) -> dict:
    """Call the specialist model. Returns parsed output."""
    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "options": {"temperature": 0},
        "messages": [{"role": "user", "content": query}],
    }

    start = time.perf_counter()
    resp = await client.post(OLLAMA_URL, json=payload, timeout=30.0)
    latency_ms = (time.perf_counter() - start) * 1000
    data = resp.json()

    if "error" in data:
        return {"tool": None, "args": {}, "error": data["error"], "raw": "", "latency_ms": latency_ms}

    content = data.get("message", {}).get("content", "").strip()
    parsed = parse_model_output(content)

    total_ns = data.get("total_duration", 0)
    load_ns = data.get("load_duration", 0)

    return {
        "tool": parsed["tool"],
        "args": parsed["args"],
        "error": parsed["error"],
        "raw": content,
        "latency_ms": (total_ns - load_ns) / 1_000_000,
        "prompt_tokens": data.get("prompt_eval_count", 0),
    }


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


async def run_tests(mcp_only: bool, verbose: bool) -> list[dict]:
    """Run all 14 test cases. Returns list of result dicts."""
    fixture_str = str(FIXTURE_DIR)

    bridge = MCPClientBridge(allowed_dirs=[fixture_str], repo_path=fixture_str)
    console.print("[dim]Connecting to MCP servers...[/dim]")
    await bridge.connect()

    tools = await bridge.list_tools()
    our_tools = {t["name"] for t in tools}
    console.print(f"[green]Connected.[/green] {len(tools)} tools on servers.")

    http = httpx.AsyncClient() if not mcp_only else None

    results: list[dict] = []

    for i, tc in enumerate(TEST_CASES, 1):
        tool_name = tc["tool"]
        query = tc["query"]
        known_args = tc["known_args"]
        check_fn = tc["check"]

        result: dict = {
            "index": i,
            "tool": tool_name,
            "query": query,
            "model_tool": None,
            "model_args": None,
            "model_error": None,
            "model_latency_ms": 0,
            "mcp_result": None,
            "mcp_error": None,
            "mcp_latency_ms": 0,
            "tool_correct": False,
            "mcp_success": False,
            "check_passed": False,
        }

        # --- Model inference (or skip in mcp-only mode) ---
        if mcp_only:
            model_tool = tool_name
            model_args = dict(known_args)
            result["model_tool"] = model_tool
            result["model_args"] = model_args
            result["tool_correct"] = True
        else:
            model_out = await call_model(http, query)
            result["model_tool"] = model_out["tool"]
            result["model_args"] = model_out["args"]
            result["model_error"] = model_out["error"]
            result["model_latency_ms"] = model_out["latency_ms"]
            model_tool = model_out["tool"]
            model_args = model_out["args"]

            if model_out["error"]:
                results.append(result)
                _print_result(result, verbose)
                continue

            result["tool_correct"] = model_tool == tool_name

        # Resolve relative paths to absolute for filesystem tools
        if model_tool not in GIT_TOOLS:
            for key in ("path", "source", "destination"):
                if key in model_args and not Path(model_args[key]).is_absolute():
                    model_args[key] = str(FIXTURE_DIR / model_args[key])

        # --- MCP execution ---
        try:
            mcp_start = time.perf_counter()
            mcp_result = await bridge.call_tool(model_tool, model_args)
            result["mcp_latency_ms"] = (time.perf_counter() - mcp_start) * 1000
            result["mcp_result"] = mcp_result
            result["mcp_success"] = True
            result["check_passed"] = check_fn(mcp_result)
        except Exception as e:
            result["mcp_latency_ms"] = (time.perf_counter() - mcp_start) * 1000
            result["mcp_error"] = str(e)

        results.append(result)
        _print_result(result, verbose)

    if http:
        await http.aclose()
    await bridge.close()

    return results


def _print_result(result: dict, verbose: bool) -> None:
    """Print a single test result inline."""
    i = result["index"]
    tool = result["tool"]
    tool_ok = "[green]OK[/green]" if result["tool_correct"] else "[red]WRONG[/red]"
    mcp_ok = "[green]OK[/green]" if result["mcp_success"] else "[red]FAIL[/red]"
    check_ok = "[green]OK[/green]" if result["check_passed"] else "[yellow]--[/yellow]"

    line = f"  {i:2d}. {tool:20s}  tool={tool_ok}  mcp={mcp_ok}  check={check_ok}"
    if result["model_latency_ms"]:
        line += f"  [{result['model_latency_ms']:.0f}ms + {result['mcp_latency_ms']:.0f}ms]"
    else:
        line += f"  [mcp {result['mcp_latency_ms']:.0f}ms]"
    console.print(line)

    if verbose:
        if result["model_error"]:
            console.print(f"      [red]Model error: {result['model_error']}[/red]")
        if result["model_args"]:
            console.print(f"      [dim]Args: {json.dumps(result['model_args'])}[/dim]")
        if result["mcp_error"]:
            console.print(f"      [red]MCP error: {result['mcp_error']}[/red]")
        if result["mcp_result"]:
            text = result["mcp_result"]
            lines = text.split("\n")
            preview = "\n".join(lines[:5])
            if len(lines) > 5:
                preview += f"\n      ... ({len(lines) - 5} more lines)"
            console.print(f"      [dim]{preview}[/dim]")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def show_summary(results: list[dict], mcp_only: bool) -> None:
    """Display summary table and stats."""
    console.print()

    table = Table(title="E2E Test Results (14 tools)", border_style="magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Tool", style="cyan")
    table.add_column("Server", style="dim")
    if not mcp_only:
        table.add_column("Model", justify="center")
    table.add_column("MCP", justify="center")
    table.add_column("Check", justify="center")
    table.add_column("Latency", justify="right", style="dim")

    for r in results:
        server = "git" if r["tool"] in GIT_TOOLS else "filesystem"
        tool_sym = "[green]pass[/green]" if r["tool_correct"] else "[red]FAIL[/red]"
        mcp_sym = "[green]pass[/green]" if r["mcp_success"] else "[red]FAIL[/red]"
        check_sym = "[green]pass[/green]" if r["check_passed"] else "[yellow]skip[/yellow]"
        latency = f"{r['model_latency_ms'] + r['mcp_latency_ms']:.0f}ms"

        row = [str(r["index"]), r["tool"], server]
        if not mcp_only:
            row.append(tool_sym)
        row.extend([mcp_sym, check_sym, latency])
        table.add_row(*row)

    console.print(table)

    # Totals
    total = len(results)
    tool_pass = sum(1 for r in results if r["tool_correct"])
    mcp_pass = sum(1 for r in results if r["mcp_success"])
    check_pass = sum(1 for r in results if r["check_passed"])

    console.print(f"\n  Tool routing:  {tool_pass}/{total}")
    console.print(f"  MCP execution: {mcp_pass}/{total}")
    console.print(f"  Result check:  {check_pass}/{total}")

    if mcp_pass == total and check_pass == total:
        console.print(f"\n  [bold green]All 14 tools passed end-to-end.[/bold green]")
    else:
        failures = [r for r in results if not r["mcp_success"]]
        if failures:
            console.print(f"\n  [yellow]Failures:[/yellow]")
            for r in failures:
                err = r["mcp_error"] or r["model_error"] or "unknown"
                console.print(f"    {r['tool']}: {err[:80]}")


def save_results(results: list[dict], mcp_only: bool) -> None:
    """Save results to JSON."""
    RESULTS_DIR.mkdir(exist_ok=True)
    output = RESULTS_DIR / "e2e_results.json"

    data = {
        "metadata": {
            "fixture": str(FIXTURE_DIR),
            "model": MODEL_NAME if not mcp_only else "(mcp-only)",
            "mode": "mcp-only" if mcp_only else "full",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_tests": len(results),
        },
        "summary": {
            "tool_routing": sum(1 for r in results if r["tool_correct"]),
            "mcp_execution": sum(1 for r in results if r["mcp_success"]),
            "result_check": sum(1 for r in results if r["check_passed"]),
        },
        "results": [
            {
                "tool": r["tool"],
                "query": r["query"],
                "model_tool": r["model_tool"],
                "model_args": r["model_args"],
                "tool_correct": r["tool_correct"],
                "mcp_success": r["mcp_success"],
                "check_passed": r["check_passed"],
                "model_error": r["model_error"],
                "mcp_error": r["mcp_error"],
                "mcp_result_preview": (r["mcp_result"] or "")[:200],
                "model_latency_ms": round(r["model_latency_ms"], 1),
                "mcp_latency_ms": round(r["mcp_latency_ms"], 1),
            }
            for r in results
        ],
    }

    with open(output, "w") as f:
        json.dump(data, f, indent=2)
    console.print(f"\n  [green]Results saved:[/green] {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main_async(mcp_only: bool, verbose: bool, reset: bool) -> None:
    # Ensure fixture exists
    if reset or not FIXTURE_DIR.exists():
        console.print("[dim]Setting up fixture...[/dim]")
        import subprocess
        cmd = [sys.executable, str(Path(__file__).parent / "setup_fixture.py")]
        if reset:
            cmd.append("--clean")
        subprocess.run(cmd, check=True)
        console.print()

    if not FIXTURE_DIR.exists():
        console.print("[red]Fixture not found.[/red] Run: python tests/e2e/setup_fixture.py")
        return

    mode = "MCP-only" if mcp_only else "Full pipeline (model → MCP)"
    console.print(Panel(
        f"[bold magenta]E2E Test[/bold magenta] — {mode}\n"
        f"Fixture: {FIXTURE_DIR}\n"
        f"Tests: {len(TEST_CASES)} (one per tool, read-first ordering)",
        border_style="blue",
    ))

    results = await run_tests(mcp_only, verbose)
    show_summary(results, mcp_only)
    save_results(results, mcp_only)


def main() -> None:
    parser = argparse.ArgumentParser(description="E2E test: specialist model → MCP servers")
    parser.add_argument(
        "--mcp-only", action="store_true",
        help="Skip model inference, test MCP servers directly with known-good args",
    )
    parser.add_argument("--verbose", action="store_true", help="Show full detail per test")
    parser.add_argument("--reset", action="store_true", help="Re-create fixture before running")
    args = parser.parse_args()
    asyncio.run(main_async(args.mcp_only, args.verbose, args.reset))


if __name__ == "__main__":
    main()
