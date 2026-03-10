#!/usr/bin/env python3
"""Head-to-head benchmark: specialist vs raw Gemma vs FunctionGemma vs GPT-OSS-120B.

Supports scaling evaluation across tool subsets (3/5/8/14) to find where
270M accuracy degrades as tool count increases.

Each model is tested in its intended mode:
- edge-mcp-caller: bare user query, no schema (specialist, tools in weights)
- gemma3:270m: few-shot system prompt + JSON format (best-effort schema prompting)
- functiongemma:270m: Ollama tools API with full schemas (designed interface)
- openai/gpt-oss-120b: NVIDIA NIM API with OpenAI tools format (ceiling reference)

Usage:
    python eval/benchmark.py                          # 14-tool, 30/tool
    python eval/benchmark.py --subset 3               # 3-tool subset
    python eval/benchmark.py --subset 8 --per-tool 50 # 8-tool, 50/tool
    python eval/benchmark.py --subset all              # run all subsets (scaling curve)
"""

import argparse
import asyncio
import json
import os
import random
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
FS_TOOLS_FILE = PROJECT_ROOT / "tools" / "filesystem.json"
GIT_TOOLS_FILE = PROJECT_ROOT / "tools" / "git.json"
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

# ---------------------------------------------------------------------------
# Tool subsets for scaling benchmark
# ---------------------------------------------------------------------------

SUBSET_3 = ["list_directory", "read_file", "search_files"]
SUBSET_5 = SUBSET_3 + ["write_file", "create_directory"]
SUBSET_8 = SUBSET_5 + ["edit_file", "move_file", "directory_tree"]
SUBSET_14 = SUBSET_8 + [
    "git_status", "git_diff_staged", "git_commit",
    "git_log", "git_branch", "git_create_branch",
]

SUBSETS = {
    3: SUBSET_3,
    5: SUBSET_5,
    8: SUBSET_8,
    14: SUBSET_14,
}

# Categories
CATEGORIES = ["clean", "messy", "disambiguation"]

# Few-shot examples for raw Gemma, keyed by tool
RAW_GEMMA_FEW_SHOT = {
    "list_directory": [
        ('show files in docs/', '{"tool":"list_directory","args":{"path":"docs/"}}'),
        ('what\'s in the tests folder', '{"tool":"list_directory","args":{"path":"tests/"}}'),
    ],
    "read_file": [
        ('read config.yaml', '{"tool":"read_file","args":{"path":"config.yaml"}}'),
        ('show me README.md', '{"tool":"read_file","args":{"path":"README.md"}}'),
    ],
    "search_files": [
        ('find *.py in lib/', '{"tool":"search_files","args":{"path":"lib/","pattern":"*.py"}}'),
        ('find all .js files in src/', '{"tool":"search_files","args":{"path":"src/","pattern":"*.js"}}'),
    ],
    "write_file": [
        ('save hello world to output.txt', '{"tool":"write_file","args":{"path":"output.txt","content":"hello world"}}'),
        ('write TODO: fix later to notes.txt', '{"tool":"write_file","args":{"path":"notes.txt","content":"TODO: fix later"}}'),
    ],
    "create_directory": [
        ('make a new folder called utils', '{"tool":"create_directory","args":{"path":"utils/"}}'),
        ('create directory tests/unit/', '{"tool":"create_directory","args":{"path":"tests/unit/"}}'),
    ],
    "edit_file": [
        ('in config.json replace debug=false with debug=true', '{"tool":"edit_file","args":{"path":"config.json","old_text":"debug=false","new_text":"debug=true"}}'),
        ('change port=3000 to port=8080 in server.conf', '{"tool":"edit_file","args":{"path":"server.conf","old_text":"port=3000","new_text":"port=8080"}}'),
    ],
    "move_file": [
        ('move old.txt to archive/old.txt', '{"tool":"move_file","args":{"source":"old.txt","destination":"archive/old.txt"}}'),
        ('rename utils.py to helpers.py', '{"tool":"move_file","args":{"source":"utils.py","destination":"helpers.py"}}'),
    ],
    "directory_tree": [
        ('show the full tree of src/', '{"tool":"directory_tree","args":{"path":"src/"}}'),
        ('tree structure of project/', '{"tool":"directory_tree","args":{"path":"project/"}}'),
    ],
    "git_status": [
        ('check git status', '{"tool":"git_status","args":{}}'),
        ('any uncommitted changes?', '{"tool":"git_status","args":{}}'),
    ],
    "git_diff_staged": [
        ('show staged changes', '{"tool":"git_diff_staged","args":{}}'),
        ('what\'s staged for commit', '{"tool":"git_diff_staged","args":{}}'),
    ],
    "git_commit": [
        ('commit with message fix auth bug', '{"tool":"git_commit","args":{"message":"fix auth bug"}}'),
        ('git commit refactor: clean up utils', '{"tool":"git_commit","args":{"message":"refactor: clean up utils"}}'),
    ],
    "git_log": [
        ('show commit history', '{"tool":"git_log","args":{}}'),
        ('show last 5 commits', '{"tool":"git_log","args":{"max_count":5}}'),
    ],
    "git_branch": [
        ('list branches', '{"tool":"git_branch","args":{}}'),
        ('show all branches', '{"tool":"git_branch","args":{}}'),
    ],
    "git_create_branch": [
        ('create branch feature/auth', '{"tool":"git_create_branch","args":{"branch_name":"feature/auth"}}'),
        ('create branch fix/login from develop', '{"tool":"git_create_branch","args":{"branch_name":"fix/login","base_branch":"develop"}}'),
    ],
}

console = Console()


# ---------------------------------------------------------------------------
# Tool schema loading
# ---------------------------------------------------------------------------


def load_all_tool_schemas() -> dict[str, dict]:
    """Load all tool schemas from filesystem.json and git.json, keyed by name."""
    schemas = {}
    for tools_file in [FS_TOOLS_FILE, GIT_TOOLS_FILE]:
        data = json.loads(tools_file.read_text())
        for tool in data["tools"]:
            schemas[tool["name"]] = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["inputSchema"],
                },
            }
    return schemas


def get_tools_for_subset(all_schemas: dict[str, dict], subset_tools: list[str]) -> list[dict]:
    """Filter tool schemas to only include tools in the subset."""
    return [all_schemas[name] for name in subset_tools if name in all_schemas]


def build_raw_gemma_prompt(subset_tools: list[str]) -> str:
    """Build few-shot system prompt for raw Gemma, scoped to subset tools."""
    tool_descriptions = {
        "list_directory": "list_directory(path): list files and directories at a path",
        "read_file": "read_file(path): read contents of a file",
        "search_files": "search_files(path, pattern): search for files matching a glob pattern",
        "write_file": "write_file(path, content): create or overwrite a file with content",
        "create_directory": "create_directory(path): create a new directory",
        "edit_file": "edit_file(path, old_text, new_text): replace specific text in a file",
        "move_file": "move_file(source, destination): move or rename a file or directory",
        "directory_tree": "directory_tree(path): get recursive tree structure of a directory",
        "git_status": "git_status(): show working tree status",
        "git_diff_staged": "git_diff_staged(): show staged changes",
        "git_commit": "git_commit(message): commit staged changes with a message",
        "git_log": "git_log(max_count?): show commit history, optionally limited",
        "git_branch": "git_branch(): list all branches",
        "git_create_branch": "git_create_branch(branch_name, base_branch?): create a new branch",
    }

    tools_text = "\n".join(f"- {tool_descriptions[t]}" for t in subset_tools)

    examples_text = ""
    for tool in subset_tools:
        for query, response in RAW_GEMMA_FEW_SHOT.get(tool, []):
            examples_text += f'User: "{query}" → {response}\n'

    return (
        "You are a tool router. Given a user query, "
        "output a JSON tool call. Output ONLY the JSON object, nothing else.\n\n"
        f"Available tools:\n{tools_text}\n\n"
        f"Examples:\n{examples_text}"
    )


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
    client: httpx.AsyncClient, query: str, system_prompt: str
) -> dict:
    """Call raw Gemma 3 270M with few-shot system prompt + JSON format."""
    payload = {
        "model": MODELS["gemma3-270m-raw"]["ollama_name"],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": system_prompt},
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


def score_result(predicted: dict, expected: dict) -> dict:
    """Score a single prediction against expected output."""
    expected_tool = expected.get("tool")
    expected_args = expected.get("args", {})

    pred_tool = predicted.get("tool")
    pred_args = predicted.get("args", {})

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
# Data loading and sampling
# ---------------------------------------------------------------------------


def load_eval_data(subset_tools: list[str], per_tool: int, seed: int = 42) -> list[dict]:
    """Load eval examples filtered by subset tools, sampled to per_tool each."""
    all_examples = []
    with open(EVAL_FILE) as f:
        for line in f:
            ex = json.loads(line)
            query = ex["messages"][0]["content"]
            expected = json.loads(ex["messages"][1]["content"])
            tool = expected.get("tool")
            if tool in subset_tools:
                all_examples.append({
                    "query": query,
                    "expected": expected,
                    "category": ex.get("category", "clean"),
                })

    # Sample per_tool examples per tool (stratified)
    rng = random.Random(seed)
    by_tool: dict[str, list[dict]] = {}
    for ex in all_examples:
        tool = ex["expected"]["tool"]
        by_tool.setdefault(tool, []).append(ex)

    sampled = []
    for tool in subset_tools:
        pool = by_tool.get(tool, [])
        if per_tool > 0 and len(pool) > per_tool:
            sampled.extend(rng.sample(pool, per_tool))
        else:
            sampled.extend(pool)

    rng.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# Main benchmark for a single subset
# ---------------------------------------------------------------------------


async def run_subset_benchmark(
    subset_size: int,
    per_tool: int,
) -> tuple[dict, list[dict]]:
    """Run benchmark for one tool subset. Returns (all_results, examples)."""
    subset_tools = SUBSETS[subset_size]

    console.print(f"\n[bold cyan]═══ {subset_size}-Tool Benchmark ═══[/bold cyan]")
    console.print(f"Tools: {', '.join(subset_tools)}\n")

    # Load eval data
    examples = load_eval_data(subset_tools, per_tool)
    console.print(f"Loaded {len(examples)} eval examples ({per_tool}/tool, {subset_size} tools)\n")

    if not examples:
        console.print("[red]No eval examples found for this subset![/red]")
        return {}, examples

    # Load tool schemas for this subset
    all_schemas = load_all_tool_schemas()
    subset_schemas = get_tools_for_subset(all_schemas, subset_tools)

    # Build raw Gemma prompt for this subset
    raw_gemma_prompt = build_raw_gemma_prompt(subset_tools)

    # Check if GPT-OSS is available
    nim_available = bool(os.environ.get("NVIDIA_API_KEY"))
    if not nim_available:
        console.print("[yellow]NVIDIA_API_KEY not set — skipping GPT-OSS-120B[/yellow]\n")

    # Warm up local models
    console.print("[bold]Warming up models...[/bold]")
    async with httpx.AsyncClient(timeout=120.0) as client:
        warmup_query = "list files in src/"
        await call_specialist(client, warmup_query)
        await call_raw_gemma(client, warmup_query, raw_gemma_prompt)
        await call_functiongemma(client, warmup_query, subset_schemas)
    console.print("   All models loaded.\n")

    # Run benchmark
    all_results = {}

    async with httpx.AsyncClient(timeout=120.0) as client:
        for model_key, model_info in MODELS.items():
            if model_key == "gpt-oss-120b" and not nim_available:
                continue

            console.print(f"[bold]Benchmarking: {model_info['description']}[/bold] ({model_info['ollama_name']})")
            if model_key == "gpt-oss-120b":
                est_min = len(examples) * 60 // NIM_RPM // 60
                est_sec = len(examples) * 60 // NIM_RPM % 60
                console.print(f"   [dim](rate limited: {NIM_RPM} RPM → ~{est_min}m {est_sec}s estimated)[/dim]")

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
                            pred = await call_raw_gemma(client, query, raw_gemma_prompt)
                        elif model_key == "functiongemma-270m":
                            pred = await call_functiongemma(client, query, subset_schemas)
                        elif model_key == "gpt-oss-120b":
                            pred = await call_gpt_oss(client, query, subset_schemas)
                        else:
                            continue

                        score = score_result(pred, ex["expected"])
                        results.append({
                            "query": query,
                            "category": ex["category"],
                            "prediction": pred,
                            "score": score,
                        })

                        if pred.get("error"):
                            errors += 1

                    except Exception as e:
                        errors += 1
                        results.append({
                            "query": query,
                            "category": ex["category"],
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


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def compute_metrics(
    all_results: dict,
    subset_tools: list[str],
) -> dict:
    """Compute aggregate metrics from benchmark results."""
    metrics = {}

    for model_key, data in all_results.items():
        results = data["results"]
        n = len(results)
        if n == 0:
            continue

        # Overall accuracy
        tool_acc = sum(1 for r in results if r["score"]["tool_correct"]) / n
        args_acc = sum(1 for r in results if r["score"]["args_correct"]) / n
        combined_acc = sum(1 for r in results if r["score"]["combined_correct"]) / n

        # Per-tool breakdown
        per_tool = {}
        for tool_name in subset_tools:
            tool_results = [r for r in results if r["score"]["expected_tool"] == tool_name]
            if tool_results:
                per_tool[tool_name] = {
                    "count": len(tool_results),
                    "tool_acc": sum(1 for r in tool_results if r["score"]["tool_correct"]) / len(tool_results),
                    "args_acc": sum(1 for r in tool_results if r["score"]["args_correct"]) / len(tool_results),
                    "combined_acc": sum(1 for r in tool_results if r["score"]["combined_correct"]) / len(tool_results),
                }

        # Per-category breakdown
        per_category = {}
        for category in CATEGORIES:
            cat_results = [r for r in results if r.get("category") == category]
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


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_results(metrics: dict, subset_size: int, subset_tools: list[str]) -> None:
    """Print formatted benchmark results for a single subset."""
    console.print(f"\n[bold cyan]═══ {subset_size}-Tool Results ═══[/bold cyan]\n")

    # Overall comparison table
    table = Table(title=f"Overall Accuracy ({subset_size} tools)", show_lines=True)
    table.add_column("Model", style="cyan", min_width=30)
    table.add_column("Tool Acc", justify="right")
    table.add_column("Args Acc", justify="right")
    table.add_column("Combined", justify="right", style="bold")
    table.add_column("Avg Tokens", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("Errors", justify="right")

    model_order = [k for k in MODELS if k in metrics]
    for model_key in model_order:
        m = metrics[model_key]
        table.add_row(
            f"{m['description']}\n({m['ollama_name']})",
            f"{m['tool_accuracy']:.1%}",
            f"{m['args_accuracy']:.1%}",
            f"{m['combined_accuracy']:.1%}",
            f"{m['avg_prompt_tokens']:.0f}",
            f"{m['avg_latency_ms']:.0f}ms",
            str(m["errors"]),
        )

    console.print(table)

    # Per-tool breakdown (specialist only, to keep output manageable)
    specialist_key = "edge-mcp-caller"
    if specialist_key in metrics:
        m = metrics[specialist_key]
        tool_table = Table(title=f"Per-Tool: {m['description']} ({subset_size} tools)")
        tool_table.add_column("Tool", style="cyan")
        tool_table.add_column("Count", justify="right")
        tool_table.add_column("Tool Acc", justify="right")
        tool_table.add_column("Args Acc", justify="right")
        tool_table.add_column("Combined", justify="right", style="bold")

        for tool_name in subset_tools:
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

    # Per-category breakdown (specialist only)
    if specialist_key in metrics and metrics[specialist_key].get("per_category"):
        m = metrics[specialist_key]
        cat_table = Table(title=f"Per-Category: {m['description']} ({subset_size} tools)")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Count", justify="right")
        cat_table.add_column("Tool Acc", justify="right")
        cat_table.add_column("Args Acc", justify="right")
        cat_table.add_column("Combined", justify="right", style="bold")

        for cat_name in CATEGORIES:
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


def print_scaling_summary(all_subset_metrics: dict[int, dict]) -> None:
    """Print scaling curve summary across all subsets."""
    console.print(f"\n[bold cyan]═══ Scaling Curve ═══[/bold cyan]\n")

    table = Table(title="Tool Routing Accuracy by Subset Size", show_lines=True)
    table.add_column("Model", style="cyan", min_width=30)
    for size in sorted(all_subset_metrics.keys()):
        table.add_column(f"{size}-tool", justify="right")

    model_order = [k for k in MODELS if any(k in m for m in all_subset_metrics.values())]
    for model_key in model_order:
        row = [MODELS[model_key]["description"]]
        for size in sorted(all_subset_metrics.keys()):
            m = all_subset_metrics[size].get(model_key)
            row.append(f"{m['tool_accuracy']:.1%}" if m else "—")
        table.add_row(*row)

    console.print(table)

    # Combined accuracy table
    table2 = Table(title="Combined Accuracy by Subset Size", show_lines=True)
    table2.add_column("Model", style="cyan", min_width=30)
    for size in sorted(all_subset_metrics.keys()):
        table2.add_column(f"{size}-tool", justify="right")

    for model_key in model_order:
        row = [MODELS[model_key]["description"]]
        for size in sorted(all_subset_metrics.keys()):
            m = all_subset_metrics[size].get(model_key)
            row.append(f"{m['combined_accuracy']:.1%}" if m else "—")
        table2.add_row(*row)

    console.print(table2)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------


def save_subset_results(
    all_results: dict,
    metrics: dict,
    subset_size: int,
) -> None:
    """Save benchmark results for a single subset."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Summary
    summary = {
        "subset_size": subset_size,
        "subset_tools": SUBSETS[subset_size],
        "models": metrics,
    }
    summary_path = RESULTS_DIR / f"benchmark_v03_{subset_size}tool.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    console.print(f"[bold]Results saved:[/bold] {summary_path}")

    # Detailed per-example results
    detailed = {}
    for model_key, data in all_results.items():
        detailed[model_key] = []
        for r in data["results"]:
            detailed[model_key].append({
                "query": r["query"],
                "category": r.get("category", "clean"),
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

    detailed_path = RESULTS_DIR / f"benchmark_v03_{subset_size}tool_detailed.json"
    detailed_path.write_text(json.dumps(detailed, indent=2))
    console.print(f"[bold]Detailed:[/bold] {detailed_path}")


def save_scaling_results(all_subset_metrics: dict[int, dict]) -> None:
    """Save scaling curve summary."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scaling = {}
    for size, metrics in sorted(all_subset_metrics.items()):
        scaling[str(size)] = {}
        for model_key, m in metrics.items():
            scaling[str(size)][model_key] = {
                "tool_accuracy": m["tool_accuracy"],
                "args_accuracy": m["args_accuracy"],
                "combined_accuracy": m["combined_accuracy"],
                "avg_prompt_tokens": m["avg_prompt_tokens"],
                "avg_latency_ms": m["avg_latency_ms"],
                "total": m["total"],
                "errors": m["errors"],
            }

    path = RESULTS_DIR / "benchmark_v03_scaling.json"
    path.write_text(json.dumps(scaling, indent=2))
    console.print(f"\n[bold]Scaling summary:[/bold] {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(description="Edge MCP Caller Benchmark")
    parser.add_argument(
        "--subset", type=str, default="14",
        help="Tool subset size: 3, 5, 8, 14, or 'all' for scaling curve (default: 14)",
    )
    parser.add_argument(
        "--per-tool", type=int, default=30,
        help="Number of eval examples per tool (default: 30, 0 = all)",
    )
    args = parser.parse_args()

    console.print("[bold cyan]Edge MCP Caller — Benchmark (v0.3)[/bold cyan]")
    console.print(f"Per-tool sample: {args.per_tool if args.per_tool > 0 else 'all'}\n")

    if args.subset == "all":
        # Run all subsets for scaling curve
        all_subset_metrics: dict[int, dict] = {}
        for size in [3, 5, 8, 14]:
            results, examples = await run_subset_benchmark(size, args.per_tool)
            if results:
                metrics = compute_metrics(results, SUBSETS[size])
                print_results(metrics, size, SUBSETS[size])
                save_subset_results(results, metrics, size)
                all_subset_metrics[size] = metrics

        if all_subset_metrics:
            print_scaling_summary(all_subset_metrics)
            save_scaling_results(all_subset_metrics)
    else:
        # Single subset
        subset_size = int(args.subset)
        if subset_size not in SUBSETS:
            console.print(f"[red]Invalid subset: {subset_size}. Must be 3, 5, 8, or 14.[/red]")
            return

        results, examples = await run_subset_benchmark(subset_size, args.per_tool)
        if results:
            metrics = compute_metrics(results, SUBSETS[subset_size])
            print_results(metrics, subset_size, SUBSETS[subset_size])
            save_subset_results(results, metrics, subset_size)

    console.print("\n[bold green]Benchmark complete![/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
