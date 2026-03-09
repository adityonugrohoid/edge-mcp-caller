#!/usr/bin/env python3
"""Head-to-head benchmark: specialist vs raw Gemma vs FunctionGemma vs GPT-OSS-120B.

Runs all models against the eval set (360 examples) and compares:
- Tool accuracy (correct tool name)
- Argument accuracy (correct args, exact match)
- Combined accuracy (both tool + args correct)
- Prompt token count (schema-prompting vs specialist)
- Inference latency (excluding model load time)

Each model is tested in its intended mode:
- edge-mcp-caller: bare user query, no schema (specialist, tools in weights)
- gemma3:270m: few-shot system prompt + JSON format (best-effort schema prompting)
- functiongemma:270m: Ollama tools API with full schemas (designed interface)
- openai/gpt-oss-120b: NVIDIA NIM API with OpenAI tools format (ceiling reference)
"""

import asyncio
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = PROJECT_ROOT / "data" / "eval.jsonl"
TOOLS_FILE = PROJECT_ROOT / "tools" / "filesystem.json"
RESULTS_DIR = PROJECT_ROOT / "results"

OLLAMA_URL = "http://localhost:11434/api/chat"
NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NIM_RPM = 40  # Rate limit: 40 requests per minute

load_dotenv(PROJECT_ROOT / ".env")

# Models to benchmark (order matters for display)
MODELS = {
    "edge-mcp-caller": {
        "ollama_name": "edge-mcp-caller:latest",
        "description": "Specialist (tools in weights)",
        "mode": "specialist",
    },
    "gemma3-270m-raw": {
        "ollama_name": "gemma3:270m",
        "description": "Raw Gemma 3 270M (few-shot prompt)",
        "mode": "schema_prompt",
    },
    "functiongemma-270m": {
        "ollama_name": "functiongemma:270m",
        "description": "FunctionGemma (Ollama tools API)",
        "mode": "tools_api",
    },
    "gpt-oss-120b": {
        "ollama_name": "openai/gpt-oss-120b",
        "description": "GPT-OSS 120B (NIM API, ceiling)",
        "mode": "nim_tools_api",
    },
}

# All valid tools for per-tool breakdown
ALL_TOOLS = ["list_directory", "read_file", "search_files", "write_file", "create_directory"]

# System prompt for raw Gemma 3 (few-shot, best-effort)
RAW_GEMMA_SYSTEM_PROMPT = (
    "You are a tool router. Given a user query about filesystem operations, "
    "output a JSON tool call. Output ONLY the JSON object, nothing else.\n\n"
    "Available tools:\n"
    "- list_directory(path): list files and directories at a path\n"
    "- read_file(path): read contents of a file\n"
    "- search_files(path, pattern): search for files matching a glob pattern\n"
    "- write_file(path, content): create or overwrite a file with content\n"
    "- create_directory(path): create a new directory\n\n"
    "Examples:\n"
    'User: "show files in docs/" → {"tool":"list_directory","args":{"path":"docs/"}}\n'
    'User: "read config.yaml" → {"tool":"read_file","args":{"path":"config.yaml"}}\n'
    'User: "find *.py in lib/" → {"tool":"search_files","args":{"path":"lib/","pattern":"*.py"}}\n'
    'User: "what\'s in the tests folder" → {"tool":"list_directory","args":{"path":"tests/"}}\n'
    'User: "show me README.md" → {"tool":"read_file","args":{"path":"README.md"}}\n'
    'User: "find all .js files in src/" → {"tool":"search_files","args":{"path":"src/","pattern":"*.js"}}\n'
    'User: "save hello world to output.txt" → {"tool":"write_file","args":{"path":"output.txt","content":"hello world"}}\n'
    'User: "make a new folder called utils" → {"tool":"create_directory","args":{"path":"utils/"}}'
)

console = Console()


# ---------------------------------------------------------------------------
# Tool schema for FunctionGemma (Ollama tools API format)
# ---------------------------------------------------------------------------


def load_tools_for_ollama() -> list[dict]:
    """Convert filesystem.json to Ollama tools API format."""
    tools_data = json.loads(TOOLS_FILE.read_text())
    ollama_tools = []
    for tool in tools_data["tools"]:
        ollama_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["inputSchema"],
            },
        })
    return ollama_tools


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------


async def call_specialist(
    client: httpx.AsyncClient, query: str
) -> dict:
    """Call edge-mcp-caller: bare query, no schema."""
    payload = {
        "model": MODELS["edge-mcp-caller"]["ollama_name"],
        "stream": False,
        "options": {"temperature": 0},
        "messages": [{"role": "user", "content": query}],
    }
    resp = await client.post(OLLAMA_URL, json=payload)
    data = resp.json()
    return parse_response(data, mode="specialist")


async def call_raw_gemma(
    client: httpx.AsyncClient, query: str
) -> dict:
    """Call raw Gemma 3 270M with few-shot system prompt + JSON format."""
    payload = {
        "model": MODELS["gemma3-270m-raw"]["ollama_name"],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": RAW_GEMMA_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    }
    resp = await client.post(OLLAMA_URL, json=payload)
    data = resp.json()
    return parse_response(data, mode="schema_prompt")


async def call_functiongemma(
    client: httpx.AsyncClient, query: str, tools: list[dict]
) -> dict:
    """Call FunctionGemma via Ollama tools API."""
    payload = {
        "model": MODELS["functiongemma-270m"]["ollama_name"],
        "stream": False,
        "options": {"temperature": 0},
        "messages": [{"role": "user", "content": query}],
        "tools": tools,
    }
    resp = await client.post(OLLAMA_URL, json=payload)
    data = resp.json()
    return parse_response(data, mode="tools_api")


async def call_gpt_oss(
    client: httpx.AsyncClient, query: str, tools: list[dict]
) -> dict:
    """Call GPT-OSS-120B via NVIDIA NIM API with OpenAI tools format."""
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        return {
            "tool": None, "args": {}, "raw_content": "",
            "prompt_tokens": 0, "eval_tokens": 0, "latency_ms": 0,
            "error": "NVIDIA_API_KEY not set",
        }

    payload = {
        "model": "openai/gpt-oss-120b",
        "temperature": 0,
        "messages": [{"role": "user", "content": query}],
        "tools": tools,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    t0 = time.monotonic()
    resp = await client.post(NIM_URL, json=payload, headers=headers)
    latency_ms = (time.monotonic() - t0) * 1000

    data = resp.json()

    result = {
        "tool": None,
        "args": {},
        "raw_content": "",
        "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
        "eval_tokens": data.get("usage", {}).get("completion_tokens", 0),
        "latency_ms": latency_ms,
        "error": None,
    }

    if "error" in data:
        result["error"] = data["error"].get("message", str(data["error"]))
        return result

    choices = data.get("choices", [])
    if not choices:
        result["error"] = "no_choices"
        return result

    msg = choices[0].get("message", {})
    result["raw_content"] = msg.get("content") or ""

    tool_calls = msg.get("tool_calls", [])
    if tool_calls:
        tc = tool_calls[0]
        fn = tc.get("function", {})
        result["tool"] = fn.get("name")
        # NIM returns arguments as JSON string
        args_raw = fn.get("arguments", "{}")
        if isinstance(args_raw, str):
            try:
                result["args"] = json.loads(args_raw)
            except json.JSONDecodeError:
                result["args"] = {}
                result["error"] = "invalid_args_json"
        else:
            result["args"] = args_raw

    return result


def parse_response(data: dict, mode: str) -> dict:
    """Parse Ollama response into standardized result dict."""
    result = {
        "tool": None,
        "args": {},
        "raw_content": "",
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "eval_tokens": data.get("eval_count", 0),
        "latency_ms": 0,
        "error": None,
    }

    # Calculate latency (exclude load time for fairness)
    total_ns = data.get("total_duration", 0)
    load_ns = data.get("load_duration", 0)
    result["latency_ms"] = (total_ns - load_ns) / 1_000_000

    if "error" in data:
        result["error"] = data["error"]
        return result

    msg = data.get("message", {})

    if mode == "tools_api":
        # FunctionGemma returns structured tool_calls
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            tc = tool_calls[0]
            fn = tc.get("function", {})
            result["tool"] = fn.get("name")
            result["args"] = fn.get("arguments", {})
        result["raw_content"] = msg.get("content", "")
    else:
        # Specialist and raw Gemma return JSON in content
        content = msg.get("content", "").strip()
        result["raw_content"] = content
        try:
            parsed = json.loads(content)
            result["tool"] = parsed.get("tool")
            result["args"] = parsed.get("args", {})
        except (json.JSONDecodeError, TypeError):
            result["error"] = "invalid_json"

    return result


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------


def normalize_args(args: dict) -> dict:
    """Normalize args for comparison (strip trailing slashes, lowercase)."""
    normalized = {}
    for k, v in args.items():
        if isinstance(v, str):
            # Don't normalize path values — exact match matters
            normalized[k] = v
        else:
            normalized[k] = v
    return normalized


def score_result(predicted: dict, expected: dict) -> dict:
    """Score a single prediction against expected output."""
    expected_tool = expected.get("tool")
    expected_args = normalize_args(expected.get("args", {}))

    pred_tool = predicted.get("tool")
    pred_args = normalize_args(predicted.get("args", {}))

    tool_correct = pred_tool == expected_tool
    args_correct = pred_args == expected_args
    combined_correct = tool_correct and args_correct

    return {
        "tool_correct": tool_correct,
        "args_correct": args_correct,
        "combined_correct": combined_correct,
        "expected_tool": expected_tool,
        "predicted_tool": pred_tool,
        "expected_args": expected_args,
        "predicted_args": pred_args,
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


async def run_benchmark() -> tuple[dict, list[dict]]:
    """Run full benchmark across all models. Returns (results, examples)."""
    console.print("[bold cyan]Edge MCP Caller — Benchmark (Step 4)[/bold cyan]\n")

    # Load eval data
    examples = []
    with open(EVAL_FILE) as f:
        for line in f:
            ex = json.loads(line)
            query = ex["messages"][0]["content"]
            expected = json.loads(ex["messages"][1]["content"])
            examples.append({
                "query": query,
                "expected": expected,
                "category": ex.get("category", "clean"),
            })

    console.print(f"Loaded {len(examples)} eval examples\n")

    # Load tools for FunctionGemma / GPT-OSS
    ollama_tools = load_tools_for_ollama()

    # Check if GPT-OSS is available
    nim_available = bool(os.environ.get("NVIDIA_API_KEY"))
    if not nim_available:
        console.print("[yellow]NVIDIA_API_KEY not set — skipping GPT-OSS-120B[/yellow]\n")

    # Warm up local models (first call loads model into VRAM)
    console.print("[bold]Warming up models...[/bold]")
    async with httpx.AsyncClient(timeout=120.0) as client:
        warmup_query = "list files in src/"
        await call_specialist(client, warmup_query)
        await call_raw_gemma(client, warmup_query)
        await call_functiongemma(client, warmup_query, ollama_tools)
    console.print("   All models loaded.\n")

    # Run benchmark
    all_results = {}

    async with httpx.AsyncClient(timeout=120.0) as client:
        for model_key, model_info in MODELS.items():
            # Skip GPT-OSS if no API key
            if model_key == "gpt-oss-120b" and not nim_available:
                continue

            console.print(f"[bold]Benchmarking: {model_info['description']}[/bold] ({model_info['ollama_name']})")
            if model_key == "gpt-oss-120b":
                console.print(f"   [dim](rate limited: {NIM_RPM} RPM → ~{len(examples) * 60 // NIM_RPM // 60}m {len(examples) * 60 // NIM_RPM % 60}s estimated)[/dim]")

            results = []
            errors = 0
            rpm_window_start = time.monotonic()
            rpm_count = 0

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"  {model_key}", total=len(examples))

                for ex in examples:
                    query = ex["query"]

                    # Rate limiting for NIM API
                    if model_key == "gpt-oss-120b":
                        rpm_count += 1
                        if rpm_count >= NIM_RPM:
                            elapsed = time.monotonic() - rpm_window_start
                            if elapsed < 60:
                                wait = 60 - elapsed + 1
                                await asyncio.sleep(wait)
                            rpm_window_start = time.monotonic()
                            rpm_count = 0

                    try:
                        if model_key == "edge-mcp-caller":
                            pred = await call_specialist(client, query)
                        elif model_key == "gemma3-270m-raw":
                            pred = await call_raw_gemma(client, query)
                        elif model_key == "functiongemma-270m":
                            pred = await call_functiongemma(client, query, ollama_tools)
                        elif model_key == "gpt-oss-120b":
                            pred = await call_gpt_oss(client, query, ollama_tools)
                        else:
                            continue

                        score = score_result(pred, ex["expected"])
                        results.append({
                            "query": query,
                            "prediction": pred,
                            "score": score,
                        })

                        if pred.get("error"):
                            errors += 1

                    except Exception as e:
                        errors += 1
                        results.append({
                            "query": query,
                            "prediction": {"error": str(e)},
                            "score": {
                                "tool_correct": False,
                                "args_correct": False,
                                "combined_correct": False,
                                "expected_tool": ex["expected"].get("tool"),
                                "predicted_tool": None,
                            },
                        })

                    progress.advance(task)

            all_results[model_key] = {
                "model": model_info,
                "results": results,
                "errors": errors,
            }
            console.print()

    return all_results, examples


def compute_metrics(all_results: dict, examples_with_category: list[dict] | None = None) -> dict:
    """Compute aggregate metrics from benchmark results."""
    if examples_with_category is None:
        examples_with_category = []
    metrics = {}

    for model_key, data in all_results.items():
        results = data["results"]
        n = len(results)

        # Overall accuracy
        tool_acc = sum(1 for r in results if r["score"]["tool_correct"]) / n
        args_acc = sum(1 for r in results if r["score"]["args_correct"]) / n
        combined_acc = sum(1 for r in results if r["score"]["combined_correct"]) / n

        # Per-tool breakdown
        per_tool = {}
        for tool_name in ALL_TOOLS:
            tool_results = [r for r in results if r["score"]["expected_tool"] == tool_name]
            if tool_results:
                per_tool[tool_name] = {
                    "count": len(tool_results),
                    "tool_acc": sum(1 for r in tool_results if r["score"]["tool_correct"]) / len(tool_results),
                    "args_acc": sum(1 for r in tool_results if r["score"]["args_correct"]) / len(tool_results),
                    "combined_acc": sum(1 for r in tool_results if r["score"]["combined_correct"]) / len(tool_results),
                }

        # Per-category breakdown (uses "category" tag from eval set)
        per_category = {}
        for category in ["clean", "messy", "adversarial"]:
            cat_indices = [i for i, ex_data in enumerate(examples_with_category)
                          if ex_data.get("category") == category]
            if cat_indices:
                cat_results = [results[i] for i in cat_indices if i < len(results)]
                if cat_results:
                    per_category[category] = {
                        "count": len(cat_results),
                        "tool_acc": sum(1 for r in cat_results if r["score"]["tool_correct"]) / len(cat_results),
                        "args_acc": sum(1 for r in cat_results if r["score"]["args_correct"]) / len(cat_results),
                        "combined_acc": sum(1 for r in cat_results if r["score"]["combined_correct"]) / len(cat_results),
                    }

        # Token and latency stats (exclude errors)
        valid = [r for r in results if not r["prediction"].get("error")]
        prompt_tokens = [r["prediction"]["prompt_tokens"] for r in valid if r["prediction"].get("prompt_tokens")]
        latencies = [r["prediction"]["latency_ms"] for r in valid if r["prediction"].get("latency_ms")]

        metrics[model_key] = {
            "description": data["model"]["description"],
            "ollama_name": data["model"]["ollama_name"],
            "total": n,
            "errors": data["errors"],
            "tool_accuracy": tool_acc,
            "args_accuracy": args_acc,
            "combined_accuracy": combined_acc,
            "per_tool": per_tool,
            "per_category": per_category,
            "avg_prompt_tokens": sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else 0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "median_latency_ms": sorted(latencies)[len(latencies) // 2] if latencies else 0,
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
        }

    return metrics


def print_results(metrics: dict) -> None:
    """Print formatted benchmark results."""
    console.print("\n[bold cyan]═══ Benchmark Results ═══[/bold cyan]\n")

    # Overall comparison table
    table = Table(title="Overall Accuracy", show_lines=True)
    table.add_column("Model", style="cyan", min_width=30)
    table.add_column("Tool Acc", justify="right")
    table.add_column("Args Acc", justify="right")
    table.add_column("Combined", justify="right", style="bold")
    table.add_column("Avg Prompt Tokens", justify="right")
    table.add_column("Avg Latency (ms)", justify="right")
    table.add_column("Errors", justify="right")

    model_order = [k for k in MODELS if k in metrics]
    for model_key in model_order:
        m = metrics[model_key]
        combined_str = f"{m['combined_accuracy']:.1%}"
        table.add_row(
            f"{m['description']}\n({m['ollama_name']})",
            f"{m['tool_accuracy']:.1%}",
            f"{m['args_accuracy']:.1%}",
            combined_str,
            f"{m['avg_prompt_tokens']:.0f}",
            f"{m['avg_latency_ms']:.0f}",
            str(m["errors"]),
        )

    console.print(table)

    # Per-tool breakdown
    for model_key in model_order:
        m = metrics[model_key]
        tool_table = Table(title=f"Per-Tool Breakdown: {m['description']}")
        tool_table.add_column("Tool", style="cyan")
        tool_table.add_column("Count", justify="right")
        tool_table.add_column("Tool Acc", justify="right")
        tool_table.add_column("Args Acc", justify="right")
        tool_table.add_column("Combined", justify="right", style="bold")

        for tool_name in ALL_TOOLS:
            pt = m["per_tool"].get(tool_name, {})
            if pt:
                tool_table.add_row(
                    tool_name,
                    str(pt["count"]),
                    f"{pt['tool_acc']:.1%}",
                    f"{pt['args_acc']:.1%}",
                    f"{pt['combined_acc']:.1%}",
                )

        console.print(tool_table)

        # Per-category breakdown (if available)
        if m.get("per_category"):
            cat_table = Table(title=f"Per-Category Breakdown: {m['description']}")
            cat_table.add_column("Category", style="cyan")
            cat_table.add_column("Count", justify="right")
            cat_table.add_column("Tool Acc", justify="right")
            cat_table.add_column("Args Acc", justify="right")
            cat_table.add_column("Combined", justify="right", style="bold")

            for cat_name in ["clean", "messy", "adversarial"]:
                pc = m["per_category"].get(cat_name, {})
                if pc:
                    cat_table.add_row(
                        cat_name,
                        str(pc["count"]),
                        f"{pc['tool_acc']:.1%}",
                        f"{pc['args_acc']:.1%}",
                        f"{pc['combined_acc']:.1%}",
                    )

            console.print(cat_table)

    # Latency comparison
    lat_table = Table(title="Latency Comparison (ms, excluding model load)")
    lat_table.add_column("Model", style="cyan", min_width=30)
    lat_table.add_column("Average", justify="right")
    lat_table.add_column("Median", justify="right")
    lat_table.add_column("P95", justify="right")

    for model_key in model_order:
        m = metrics[model_key]
        lat_table.add_row(
            m["description"],
            f"{m['avg_latency_ms']:.0f}",
            f"{m['median_latency_ms']:.0f}",
            f"{m['p95_latency_ms']:.0f}",
        )

    console.print(lat_table)

    # Prompt efficiency comparison
    console.print("\n[bold]Prompt Efficiency:[/bold]")
    specialist_tokens = metrics["edge-mcp-caller"]["avg_prompt_tokens"]
    for model_key in model_order:
        if model_key == "edge-mcp-caller":
            continue
        other_tokens = metrics[model_key]["avg_prompt_tokens"]
        if specialist_tokens > 0 and other_tokens > 0:
            ratio = other_tokens / specialist_tokens
            console.print(
                f"  {metrics[model_key]['description']}: "
                f"{other_tokens:.0f} tokens/request "
                f"({ratio:.0f}x more than specialist's {specialist_tokens:.0f})"
            )


def generate_report_html(metrics: dict) -> str:
    """Generate HTML report for benchmark results."""
    model_order = [k for k in MODELS if k in metrics]

    rows = []
    for model_key in model_order:
        m = metrics[model_key]
        rows.append(f"""
        <tr>
            <td><strong>{m['description']}</strong><br><code>{m['ollama_name']}</code></td>
            <td>{m['tool_accuracy']:.1%}</td>
            <td>{m['args_accuracy']:.1%}</td>
            <td><strong>{m['combined_accuracy']:.1%}</strong></td>
            <td>{m['avg_prompt_tokens']:.0f}</td>
            <td>{m['avg_latency_ms']:.0f}</td>
        </tr>""")

    per_tool_rows = []
    for model_key in model_order:
        m = metrics[model_key]
        for tool_name in ALL_TOOLS:
            pt = m["per_tool"].get(tool_name, {})
            if pt:
                per_tool_rows.append(f"""
        <tr>
            <td>{m['description']}</td>
            <td><code>{tool_name}</code></td>
            <td>{pt['count']}</td>
            <td>{pt['tool_acc']:.1%}</td>
            <td>{pt['args_acc']:.1%}</td>
            <td><strong>{pt['combined_acc']:.1%}</strong></td>
        </tr>""")

    # Per-category rows
    per_cat_rows = []
    for model_key in model_order:
        m = metrics[model_key]
        for cat_name in ["clean", "messy", "adversarial"]:
            pc = m.get("per_category", {}).get(cat_name, {})
            if pc:
                per_cat_rows.append(f"""
        <tr>
            <td>{m['description']}</td>
            <td>{cat_name}</td>
            <td>{pc['count']}</td>
            <td>{pc['tool_acc']:.1%}</td>
            <td>{pc['args_acc']:.1%}</td>
            <td><strong>{pc['combined_acc']:.1%}</strong></td>
        </tr>""")

    # Build summary cards dynamically
    summary_cards = []
    for model_key in model_order:
        m = metrics[model_key]
        summary_cards.append(f"""
        <div class="summary-card">
            <div>{m['description']}</div>
            <div class="metric">{m['combined_accuracy']:.1%}</div>
            <div>combined accuracy</div>
        </div>""")

    # Build latency rows dynamically
    latency_rows = []
    for model_key in model_order:
        m = metrics[model_key]
        latency_rows.append(
            f"<tr><td>{m['description']}</td>"
            f"<td>{m['avg_latency_ms']:.0f}</td>"
            f"<td>{m['median_latency_ms']:.0f}</td>"
            f"<td>{m['p95_latency_ms']:.0f}</td></tr>"
        )

    # Build prompt efficiency lines
    specialist_tokens = metrics["edge-mcp-caller"]["avg_prompt_tokens"]
    efficiency_lines = [
        f"<p>Specialist model uses <strong>{specialist_tokens:.0f}</strong> tokens per request (no schema in prompt).</p>"
    ]
    for model_key in model_order:
        if model_key == "edge-mcp-caller":
            continue
        m = metrics[model_key]
        if m["avg_prompt_tokens"] > 0:
            ratio = m["avg_prompt_tokens"] / max(specialist_tokens, 1)
            efficiency_lines.append(
                f"<p>{m['description']} uses <strong>{m['avg_prompt_tokens']:.0f}</strong> "
                f"tokens per request ({ratio:.0f}x more).</p>"
            )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Edge MCP Caller — Benchmark Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; color: #1a1a1a; }}
        h1 {{ color: #2563eb; }}
        h2 {{ color: #374151; margin-top: 2em; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
        th, td {{ border: 1px solid #d1d5db; padding: 10px 14px; text-align: left; }}
        th {{ background: #f3f4f6; font-weight: 600; }}
        tr:nth-child(even) {{ background: #f9fafb; }}
        code {{ background: #e5e7eb; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
        .highlight {{ background: #dcfce7; font-weight: bold; }}
        .metric {{ font-size: 2em; font-weight: bold; color: #2563eb; }}
        .summary {{ display: flex; gap: 2em; margin: 1em 0; flex-wrap: wrap; }}
        .summary-card {{ flex: 1; min-width: 180px; padding: 1em; background: #f0f9ff; border-radius: 8px; border: 1px solid #bae6fd; }}
    </style>
</head>
<body>
    <h1>Edge MCP Caller — Benchmark Report</h1>
    <p>Head-to-head comparison of {len(model_order)} models on {metrics['edge-mcp-caller']['total']} eval examples (MCP filesystem tool calling). All runs at temperature=0.</p>

    <div class="summary">
        {"".join(summary_cards)}
    </div>

    <h2>Overall Results</h2>
    <table>
        <thead>
            <tr><th>Model</th><th>Tool Acc</th><th>Args Acc</th><th>Combined</th><th>Avg Prompt Tokens</th><th>Avg Latency (ms)</th></tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>

    <h2>Per-Tool Breakdown</h2>
    <table>
        <thead>
            <tr><th>Model</th><th>Tool</th><th>Count</th><th>Tool Acc</th><th>Args Acc</th><th>Combined</th></tr>
        </thead>
        <tbody>
            {"".join(per_tool_rows)}
        </tbody>
    </table>

    <h2>Per-Category Breakdown</h2>
    <table>
        <thead>
            <tr><th>Model</th><th>Category</th><th>Count</th><th>Tool Acc</th><th>Args Acc</th><th>Combined</th></tr>
        </thead>
        <tbody>
            {"".join(per_cat_rows)}
        </tbody>
    </table>

    <h2>Latency Comparison</h2>
    <table>
        <thead>
            <tr><th>Model</th><th>Average (ms)</th><th>Median (ms)</th><th>P95 (ms)</th></tr>
        </thead>
        <tbody>
            {"".join(latency_rows)}
        </tbody>
    </table>

    <h2>Prompt Efficiency</h2>
    {"".join(efficiency_lines)}

    <hr>
    <p><em>Generated by <code>eval/benchmark.py</code> — temperature=0, deterministic</em></p>
</body>
</html>"""
    return html


def save_results(all_results: dict, metrics: dict) -> None:
    """Save benchmark results and HTML report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save raw results (without full prediction details to keep file manageable)
    summary = {
        "eval_examples": metrics["edge-mcp-caller"]["total"],
        "models": {},
    }
    for model_key, m in metrics.items():
        summary["models"][model_key] = m

    # Save per-example results separately for analysis
    detailed = {}
    for model_key, data in all_results.items():
        detailed[model_key] = []
        for r in data["results"]:
            detailed[model_key].append({
                "query": r["query"],
                "expected_tool": r["score"].get("expected_tool"),
                "expected_args": r["score"].get("expected_args"),
                "predicted_tool": r["score"].get("predicted_tool"),
                "predicted_args": r["score"].get("predicted_args"),
                "raw_output": r["prediction"].get("raw_content", ""),
                "tool_correct": r["score"]["tool_correct"],
                "args_correct": r["score"]["args_correct"],
                "combined_correct": r["score"]["combined_correct"],
                "prompt_tokens": r["prediction"].get("prompt_tokens", 0),
                "eval_tokens": r["prediction"].get("eval_tokens", 0),
                "latency_ms": r["prediction"].get("latency_ms", 0),
                "error": r["prediction"].get("error"),
            })

    # Write files
    results_path = RESULTS_DIR / "benchmark.json"
    results_path.write_text(json.dumps(summary, indent=2))
    console.print(f"\n[bold]Results saved:[/bold] {results_path}")

    detailed_path = RESULTS_DIR / "benchmark_detailed.json"
    detailed_path.write_text(json.dumps(detailed, indent=2))
    console.print(f"[bold]Detailed results:[/bold] {detailed_path}")

    report_path = RESULTS_DIR / "report.html"
    html = generate_report_html(metrics)
    report_path.write_text(html)
    console.print(f"[bold]HTML report:[/bold] {report_path}")


async def main() -> None:
    all_results, eval_examples = await run_benchmark()
    metrics = compute_metrics(all_results, eval_examples)
    print_results(metrics)
    save_results(all_results, metrics)
    console.print("\n[bold green]Benchmark complete![/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
