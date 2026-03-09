#!/usr/bin/env python3
"""Interactive CLI demo — end-to-end specialist model → MCP filesystem server.

User query → Ollama (edge-mcp-caller) → JSON parse → MCP tools/call → result.

Usage:
    python demo/cli.py                        # default: current directory
    python demo/cli.py /path/to/explore       # specify target directory
    python demo/cli.py ~/projects ~/documents # multiple allowed directories
"""

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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "edge-mcp-caller:latest"

console = Console()


# ---------------------------------------------------------------------------
# Ollama inference
# ---------------------------------------------------------------------------


async def call_model(client: httpx.AsyncClient, query: str) -> dict:
    """Call the specialist model and return parsed response.

    Returns:
        {"raw": str, "tool": str|None, "args": dict, "latency_ms": float, "error": str|None}
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
            "latency_ms": elapsed_ms,
            "error": data["error"],
        }

    content = data.get("message", {}).get("content", "").strip()
    parsed = parse_model_output(content)

    return {
        "raw": content,
        "tool": parsed["tool"],
        "args": parsed["args"],
        "latency_ms": elapsed_ms,
        "error": parsed["error"],
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def show_banner(allowed_dirs: list[str]) -> None:
    """Show startup banner."""
    dirs_text = ", ".join(allowed_dirs)
    banner = Text()
    banner.append("Edge MCP Caller", style="bold magenta")
    banner.append(" — specialist 270M model + MCP filesystem server\n\n")
    banner.append("Model: ", style="dim")
    banner.append(f"{MODEL_NAME}\n")
    banner.append("Allowed dirs: ", style="dim")
    banner.append(f"{dirs_text}\n\n")
    banner.append("Commands:\n", style="dim")
    banner.append("  tools   ", style="cyan")
    banner.append("— list available MCP tools\n")
    banner.append("  help    ", style="cyan")
    banner.append("— show this help\n")
    banner.append("  quit    ", style="cyan")
    banner.append("— exit\n")
    banner.append("\nOr just ask a question about your filesystem.")
    console.print(Panel(banner, border_style="blue"))


def show_model_output(result: dict) -> None:
    """Display the model's parsed tool call."""
    if result["error"]:
        console.print(f"  [red]Model error:[/red] {result['error']}")
        console.print(f"  [dim]Raw output:[/dim] {escape(result['raw'])}")
        return

    args_str = json.dumps(result["args"], indent=2)
    console.print(f"  [cyan]Tool:[/cyan]    {result['tool']}")
    console.print(f"  [cyan]Args:[/cyan]    {args_str}")
    console.print(f"  [dim]Latency: {result['latency_ms']:.0f}ms[/dim]")


def show_mcp_result(mcp_result: dict) -> None:
    """Display the MCP server's response."""
    if mcp_result["error"]:
        console.print(f"  [red]MCP error:[/red] {mcp_result['error']}")
        return

    text = mcp_result["result"] or "(empty response)"
    # Truncate very long results for display
    lines = text.split("\n")
    if len(lines) > 50:
        display = "\n".join(lines[:50])
        console.print(Panel(display, title="Result", border_style="green"))
        console.print(f"  [dim]... ({len(lines) - 50} more lines, {len(text)} chars total)[/dim]")
    else:
        console.print(Panel(text, title="Result", border_style="green"))


def show_tools(tools: list[dict]) -> None:
    """Display available MCP tools as a table."""
    table = Table(title="Available MCP Tools", border_style="blue")
    table.add_column("Tool", style="cyan")
    table.add_column("Description")
    for t in tools:
        table.add_row(t["name"], t["description"])
    console.print(table)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def run(allowed_dirs: list[str]) -> None:
    """Main interactive loop."""
    show_banner(allowed_dirs)

    # Connect to MCP server
    console.print("[dim]Starting MCP filesystem server...[/dim]")
    bridge = MCPClientBridge(allowed_dirs)
    try:
        await bridge.connect()
    except Exception as e:
        console.print(f"[red]Failed to start MCP server:[/red] {e}")
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
                show_banner(allowed_dirs)
                continue

            # Step 1: Model inference
            console.print("\n[dim]Model inference...[/dim]")
            model_result = await call_model(http, query)
            show_model_output(model_result)

            if model_result["error"]:
                continue

            # Step 2: MCP tool execution
            console.print("\n[dim]Executing MCP tool...[/dim]")
            try:
                mcp_result = await bridge.call_tool(
                    model_result["tool"], model_result["args"]
                )
                show_mcp_result({
                    "tool": model_result["tool"],
                    "args": model_result["args"],
                    "result": mcp_result,
                    "error": None,
                })
            except Exception as e:
                console.print(f"  [red]MCP error:[/red] {e}")

    await bridge.close()
    console.print("[dim]MCP server stopped.[/dim]")


def main() -> None:
    """Entry point."""
    # Parse allowed directories from CLI args
    if len(sys.argv) > 1:
        allowed_dirs = [str(Path(d).resolve()) for d in sys.argv[1:]]
    else:
        allowed_dirs = [str(Path.cwd())]

    # Validate directories exist
    for d in allowed_dirs:
        if not Path(d).is_dir():
            console.print(f"[red]Error:[/red] '{d}' is not a directory")
            sys.exit(1)

    asyncio.run(run(allowed_dirs))


if __name__ == "__main__":
    main()
