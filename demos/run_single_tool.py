#!/usr/bin/env python3
"""Run a single tool through the full verbose pipeline for GIF recording.

Connects to real MCP servers, calls the real model via Ollama, and displays
the complete journey from natural language query to MCP server result using
rich panels. Designed to be recorded with asciinema for per-tool GIF demos.

Usage:
    python demos/run_single_tool.py --tool list_directory
    python demos/run_single_tool.py --tool git_status
    python demos/run_single_tool.py --list                  # show available tools
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
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Import bridge without shadowing `mcp` package
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "mcp_client_bridge", PROJECT_ROOT / "mcp" / "client.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MCPClientBridge = _mod.MCPClientBridge
parse_model_output = _mod.parse_model_output
GIT_TOOLS = _mod.GIT_TOOLS

FIXTURE_DIR = PROJECT_ROOT / "tests" / "e2e" / "fixture"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "edge-mcp-caller:latest"

console = Console()

# ---------------------------------------------------------------------------
# Test cases — one per tool (same as tests/e2e/run_e2e.py)
# ---------------------------------------------------------------------------

TEST_CASES = {
    "list_directory": {
        "query": "show me what files are in the src directory",
        "known_args": {"path": "src"},
    },
    "read_file": {
        "query": "read the contents of config/settings.yaml",
        "known_args": {"path": "config/settings.yaml"},
    },
    "search_files": {
        "query": "find all python files in the tests directory",
        "known_args": {"path": "tests", "pattern": "*.py"},
    },
    "directory_tree": {
        "query": "show the tree structure of the docs folder",
        "known_args": {"path": "docs"},
    },
    "git_status": {
        "query": "what's the current git status",
        "known_args": {},
    },
    "git_diff_staged": {
        "query": "show me the staged changes",
        "known_args": {},
    },
    "git_log": {
        "query": "show the last 3 commits",
        "known_args": {"max_count": 3},
    },
    "git_branch": {
        "query": "list all the branches",
        "known_args": {},
    },
    "write_file": {
        "query": "create a file called notes.txt with content 'hello world'",
        "known_args": {"path": "notes.txt", "content": "hello world"},
    },
    "create_directory": {
        "query": "make a new directory called backups",
        "known_args": {"path": "backups"},
    },
    "edit_file": {
        "query": "in config/settings.yaml change 'debug: true' to 'debug: false'",
        "known_args": {"path": "config/settings.yaml", "old_text": "debug: true", "new_text": "debug: false"},
    },
    "move_file": {
        "query": "move scripts/deploy.sh to config/deploy.sh",
        "known_args": {"source": "scripts/deploy.sh", "destination": "config/deploy.sh"},
    },
    "git_commit": {
        "query": "commit with message 'update readme'",
        "known_args": {"message": "update readme"},
    },
    "git_create_branch": {
        "query": "create a new branch called feature/test from main",
        "known_args": {"branch_name": "feature/test", "base_branch": "main"},
    },
}

# Ordered tool list for numbering
TOOL_ORDER = [
    "list_directory", "read_file", "search_files", "directory_tree",
    "git_status", "git_diff_staged", "git_log", "git_branch",
    "write_file", "create_directory", "edit_file", "move_file",
    "git_commit", "git_create_branch",
]


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------


async def call_model(client: httpx.AsyncClient, query: str) -> dict:
    """Call the specialist model. Returns full response data."""
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
# Display
# ---------------------------------------------------------------------------


def show_header(tool_name: str, server: str, query: str, tool_num: int) -> None:
    """Display the tool header panel."""
    header = Text()
    header.append(f"Edge MCP Caller", style="bold magenta")
    header.append(f" — {tool_name}\n", style="bold")
    header.append(f"Server: ", style="dim")
    header.append(f"{server}\n")
    header.append(f"Query:  ", style="dim")
    header.append(f'"{query}"')
    console.print(Panel(
        header,
        title=f"[bold cyan]Tool {tool_num}/14[/bold cyan]",
        border_style="magenta",
    ))


def show_ollama_request(payload: dict) -> None:
    """Display the Ollama request panel."""
    console.print(Panel(
        json.dumps(payload, indent=2),
        title="[bold]1. Ollama Request[/bold]",
        subtitle=f"POST {OLLAMA_URL}",
        border_style="blue",
    ))


def show_model_response(result: dict) -> None:
    """Display model response panel and metrics table."""
    if result["error"]:
        console.print(Panel(
            f"[red]Error:[/red] {result['error']}\n[dim]Raw:[/dim] {escape(result['raw'])}",
            title="[bold]2. Model Response[/bold]",
            border_style="red",
        ))
        return

    lines = [
        f'[cyan]Raw:[/cyan]    {escape(result["raw"])}',
        f'[cyan]Tool:[/cyan]   {result["tool"]}',
        f'[cyan]Args:[/cyan]   {json.dumps(result["args"])}',
    ]
    console.print(Panel(
        "\n".join(lines),
        title="[bold]2. Model Response[/bold]",
        border_style="blue",
    ))

    metrics = Table(title="Model Metrics", border_style="dim", show_header=False)
    metrics.add_column("Metric", style="dim")
    metrics.add_column("Value", style="bold")
    metrics.add_row("Prompt tokens", str(result["prompt_tokens"]))
    metrics.add_row("Output tokens", str(result["eval_tokens"]))
    metrics.add_row("Latency", f'{result["latency_ms"]:.0f}ms')
    metrics.add_row("Schema tokens", "0 (tools baked into weights)")
    console.print(metrics)


def show_mcp_request(tool: str, args: dict, server: str) -> None:
    """Display the MCP request panel."""
    if server == "filesystem":
        server_label = "@modelcontextprotocol/server-filesystem"
    else:
        server_label = "mcp-server-git"

    mcp_payload = json.dumps(
        {"method": "tools/call", "params": {"name": tool, "arguments": args}},
        indent=2,
    )
    console.print(Panel(
        mcp_payload,
        title="[bold]3. MCP Request[/bold]",
        subtitle=f"stdio -> {server_label}",
        border_style="yellow",
    ))


def show_mcp_response(result_text: str | None, error: str | None, latency_ms: float, tool: str, server: str) -> None:
    """Display MCP response panel and metrics table."""
    if server == "filesystem":
        server_label = "@modelcontextprotocol/server-filesystem"
    else:
        server_label = "mcp-server-git"

    if error:
        console.print(Panel(
            f"[red]{error}[/red]",
            title="[bold]4. MCP Response[/bold]",
            border_style="red",
        ))
    else:
        text = result_text or "(empty response)"
        lines = text.split("\n")
        if len(lines) > 30:
            display = "\n".join(lines[:30])
            display += f"\n[dim]... ({len(lines) - 30} more lines, {len(text)} chars total)[/dim]"
        else:
            display = text
        console.print(Panel(
            display,
            title="[bold]4. MCP Response[/bold]",
            border_style="green",
        ))

    metrics = Table(title="MCP Metrics", border_style="dim", show_header=False)
    metrics.add_column("Metric", style="dim")
    metrics.add_column("Value", style="bold")
    metrics.add_row("Server", server_label)
    metrics.add_row("Tool", tool)
    metrics.add_row("Latency", f"{latency_ms:.0f}ms")
    result_len = len(result_text) if result_text else 0
    metrics.add_row("Response size", f"{result_len} chars")
    console.print(metrics)


def show_e2e_summary(model_result: dict, mcp_latency_ms: float, server: str) -> None:
    """Display end-to-end summary table."""
    total_ms = model_result["latency_ms"] + mcp_latency_ms
    table = Table(title="End-to-End Summary", border_style="magenta")
    table.add_column("Stage", style="cyan")
    table.add_column("Detail")
    table.add_column("Time", justify="right", style="bold")
    table.add_row(
        "Model inference",
        f'{model_result["prompt_tokens"]} prompt -> {model_result["eval_tokens"]} output tokens',
        f'{model_result["latency_ms"]:.0f}ms',
    )
    table.add_row(
        f"MCP execution",
        f'{model_result["tool"]}({json.dumps(model_result["args"])})',
        f"{mcp_latency_ms:.0f}ms",
    )
    table.add_row("Total", "", f"{total_ms:.0f}ms", style="bold magenta")
    console.print(table)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


async def run_single_tool(tool_name: str) -> None:
    """Run the full verbose pipeline for a single tool."""
    if tool_name not in TEST_CASES:
        console.print(f"[red]Unknown tool:[/red] {tool_name}")
        console.print(f"[dim]Available: {', '.join(TOOL_ORDER)}[/dim]")
        return

    tc = TEST_CASES[tool_name]
    query = tc["query"]
    tool_num = TOOL_ORDER.index(tool_name) + 1
    server = "git" if tool_name in GIT_TOOLS else "filesystem"
    fixture_str = str(FIXTURE_DIR)

    # Ensure fixture exists
    if not FIXTURE_DIR.exists():
        console.print("[red]Fixture not found.[/red] Run: python tests/e2e/setup_fixture.py")
        return

    # Header
    show_header(tool_name, server, query, tool_num)

    # Connect MCP
    console.print("[dim]Connecting to MCP servers...[/dim]")
    bridge = MCPClientBridge(allowed_dirs=[fixture_str], repo_path=fixture_str)
    await bridge.connect()
    console.print("[green]Connected.[/green]\n")

    # Model inference
    async with httpx.AsyncClient() as http:
        model_result = await call_model(http, query)

    # 1. Ollama Request
    show_ollama_request(model_result["payload"])

    # 2. Model Response + Metrics
    show_model_response(model_result)

    if model_result["error"]:
        await bridge.close()
        return

    # Resolve relative paths for filesystem tools
    model_args = dict(model_result["args"])
    if model_result["tool"] not in GIT_TOOLS:
        for key in ("path", "source", "destination"):
            if key in model_args and not Path(model_args[key]).is_absolute():
                model_args[key] = str(FIXTURE_DIR / model_args[key])

    # 3. MCP Request
    show_mcp_request(model_result["tool"], model_args, server)

    # Execute MCP call
    mcp_latency_ms = 0.0
    mcp_result_text = None
    mcp_error = None
    try:
        mcp_start = time.perf_counter()
        mcp_result_text = await bridge.call_tool(model_result["tool"], model_args)
        mcp_latency_ms = (time.perf_counter() - mcp_start) * 1000
    except Exception as e:
        mcp_latency_ms = (time.perf_counter() - mcp_start) * 1000
        mcp_error = str(e)

    # 4. MCP Response + Metrics
    show_mcp_response(mcp_result_text, mcp_error, mcp_latency_ms, model_result["tool"], server)

    # End-to-End Summary
    show_e2e_summary(model_result, mcp_latency_ms, server)

    await bridge.close()

    # Hold for GIF recording
    time.sleep(2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single tool through the full verbose pipeline (for GIF recording)",
    )
    parser.add_argument(
        "--tool", type=str, default=None,
        help="Tool name to demo (e.g. list_directory, git_status)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available tools and exit",
    )
    args = parser.parse_args()

    if args.list:
        for i, tool in enumerate(TOOL_ORDER, 1):
            server = "git" if tool in GIT_TOOLS else "filesystem"
            query = TEST_CASES[tool]["query"]
            console.print(f"  {i:2d}. [cyan]{tool:20s}[/cyan]  [dim]{server:10s}[/dim]  {query}")
        return

    if not args.tool:
        parser.print_help()
        return

    asyncio.run(run_single_tool(args.tool))


if __name__ == "__main__":
    main()
