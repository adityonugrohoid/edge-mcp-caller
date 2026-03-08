#!/usr/bin/env python3
"""Generate synthetic training data for the specialist MCP tool caller.

Uses NVIDIA NIM API (deepseek-v3.1) to create diverse (user_query, tool_call)
pairs for 3 MCP filesystem tools: list_directory, read_file, search_files.

Output: data/train.jsonl + data/eval.jsonl in HuggingFace chat format.
"""

import asyncio
import json
import os
import random
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = PROJECT_ROOT / "data" / "train.jsonl"
EVAL_FILE = PROJECT_ROOT / "data" / "eval.jsonl"

EVAL_RATIO = 0.1
EXAMPLES_PER_TOOL = 1200  # ~3600 total
BATCH_SIZE = 20  # ~12s per batch with llama-3.1-70b
MODEL = "meta/llama-3.1-70b-instruct"
BASE_URL = "https://integrate.api.nvidia.com/v1"
MAX_CONCURRENT = 5  # 5 concurrent × ~12s each ≈ 25 RPM, under 40 RPM limit
REQUEST_TIMEOUT = 60  # seconds per API call
RETRY_MAX = 5
RETRY_BASE_DELAY = 3  # seconds, exponential backoff
DELAY_BETWEEN_REQUESTS = 1  # seconds between requests

console = Console()

# ---------------------------------------------------------------------------
# Tool definitions (reference — NOT passed to the specialist model)
# ---------------------------------------------------------------------------

TOOLS_SPEC: dict[str, dict] = {
    "list_directory": {
        "description": "Get a listing of files and directories in a specified path",
        "args": {"path": "string (path to directory)"},
        "required": ["path"],
    },
    "read_file": {
        "description": "Read the complete contents of a file",
        "args": {"path": "string (path to file)"},
        "required": ["path"],
    },
    "search_files": {
        "description": "Recursively search for files/directories matching a pattern",
        "args": {
            "path": "string (starting path for search)",
            "pattern": "string (glob pattern to match)",
        },
        "required": ["path", "pattern"],
    },
}

# ---------------------------------------------------------------------------
# Diversity dimensions — combinatorial coverage
# ---------------------------------------------------------------------------

QUERY_STYLES = [
    "casual and conversational (e.g., 'hey, what's in src?')",
    "direct command (e.g., 'list the contents of src/')",
    "question form (e.g., 'what files are in the test directory?')",
    "implicit intent — user describes a goal, not the action (e.g., 'I need to understand the project layout')",
    "technical/precise (e.g., 'enumerate entries in /home/user/project/lib')",
    "beginner/confused (e.g., 'how do I see what's inside a folder?')",
    "context-rich — user provides background/reason (e.g., 'I'm debugging a test, show me test files')",
    "terse/minimal (e.g., 'src/', 'show config.yaml', 'find *.py')",
]

PATH_CONTEXTS = [
    "web project (React/Next.js/Vue — src/, components/, pages/, public/, package.json)",
    "Python project (src/, tests/, scripts/, setup.py, pyproject.toml, requirements.txt)",
    "Rust project (src/, Cargo.toml, target/, benches/, examples/)",
    "Go project (cmd/, pkg/, internal/, go.mod, go.sum)",
    "generic project root (README.md, LICENSE, .gitignore, docs/, Makefile)",
    "deep nested paths (src/components/auth/providers/oauth/, lib/internal/utils/)",
    "absolute Linux paths (/home/user/projects/, /etc/nginx/, /var/log/)",
    "dot-prefixed paths (.github/workflows/, .vscode/, .env, .dockerignore)",
    "monorepo (packages/, apps/, libs/, turbo.json, pnpm-workspace.yaml)",
    "data/ML project (data/, models/, notebooks/, experiments/, configs/)",
]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_generation_prompt(
    tool_name: str, style: str, path_ctx: str, n: int
) -> str:
    tool = TOOLS_SPEC[tool_name]
    return f"""Generate exactly {n} diverse training examples for a specialist AI model that routes natural language user queries to MCP filesystem tool calls.

TARGET TOOL: {tool_name}
TOOL DESCRIPTION: {tool["description"]}
TOOL ARGS: {json.dumps(tool["args"])}
REQUIRED ARGS: {json.dumps(tool["required"])}

QUERY STYLE FOR THIS BATCH: {style}
PATH CONTEXT FOR THIS BATCH: {path_ctx}

RULES:
1. Each example is a user query (natural language) paired with the correct JSON tool call.
2. User queries must NOT mention tool names, JSON, or API details — they're natural language.
3. Every query in this batch must be unique — vary wording, paths, and specifics.
4. Paths should be realistic for the given path context.
5. For search_files: patterns must be valid globs (*.py, *.ts, test_*, Dockerfile, etc.).
6. The JSON tool call format is exactly: {{"tool": "{tool_name}", "args": {{...}}}}
7. User queries should be 1-2 sentences max.
8. Include a few queries with minor typos, abbreviations, or informal grammar for realism.
9. Do NOT include trailing slashes on file paths (only on directory paths where appropriate).
10. Args values should be strings, not objects.

OUTPUT FORMAT: Return ONLY a JSON array of objects with "query" and "tool_call" fields.
Example:
[
  {{"query": "what's in the src folder?", "tool_call": {{"tool": "list_directory", "args": {{"path": "src/"}}}}}},
  {{"query": "show me the test directory contents", "tool_call": {{"tool": "list_directory", "args": {{"path": "tests/"}}}}}}
]

Return ONLY the JSON array. No markdown fences, no explanation."""


# ---------------------------------------------------------------------------
# NVIDIA NIM API client (OpenAI-compatible)
# ---------------------------------------------------------------------------


async def nim_chat(
    http_client: httpx.AsyncClient,
    api_key: str,
    prompt: str,
) -> str:
    """Send a chat completion request to NVIDIA NIM API with retry."""
    for attempt in range(RETRY_MAX):
        resp = await http_client.post(
            f"{BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 429:
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            console.print(
                f"[yellow]Rate limited, retrying in {delay}s "
                f"(attempt {attempt + 1}/{RETRY_MAX})...[/yellow]"
            )
            await asyncio.sleep(delay)
            continue
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    raise httpx.HTTPStatusError(
        "Rate limited after max retries",
        request=resp.request,
        response=resp,
    )


# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------


def parse_examples(text: str, tool_name: str) -> list[dict]:
    """Parse and validate generated examples from LLM response text."""
    text = text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if "```" in text:
            text = text[: text.rfind("```")]
        text = text.strip()

    examples = json.loads(text)
    if not isinstance(examples, list):
        return []

    valid: list[dict] = []
    for ex in examples:
        if "query" not in ex or "tool_call" not in ex:
            continue
        tc = ex["tool_call"]
        if tc.get("tool") != tool_name:
            continue
        args = tc.get("args", {})
        if not all(k in args for k in TOOLS_SPEC[tool_name]["required"]):
            continue
        # Ensure all arg values are strings
        if not all(isinstance(v, str) for v in args.values()):
            continue

        valid.append(
            {
                "messages": [
                    {"role": "user", "content": ex["query"]},
                    {
                        "role": "assistant",
                        "content": json.dumps(tc, separators=(",", ":")),
                    },
                ]
            }
        )
    return valid


async def generate_batch(
    http_client: httpx.AsyncClient,
    api_key: str,
    tool_name: str,
    style: str,
    path_ctx: str,
    n: int,
) -> list[dict]:
    """Generate a single batch of examples via NVIDIA NIM API."""
    prompt = build_generation_prompt(tool_name, style, path_ctx, n)

    try:
        text = await nim_chat(http_client, api_key, prompt)
        return parse_examples(text, tool_name)

    except json.JSONDecodeError as e:
        console.print(f"[yellow]JSON parse error ({tool_name}): {e}[/yellow]")
        return []
    except httpx.HTTPStatusError as e:
        console.print(
            f"[red]HTTP {e.response.status_code} ({tool_name}): {e}[/red]"
        )
        return []
    except Exception as e:
        console.print(f"[red]Batch error ({tool_name}): {e}[/red]")
        return []


async def generate_all(
    http_client: httpx.AsyncClient,
    api_key: str,
) -> list[dict]:
    """Generate all training examples across tools and diversity dimensions."""
    all_examples: list[dict] = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Build task list: pick (style, path_context) combos for each tool
    tasks_spec: list[tuple[str, str, str, int]] = []
    for tool_name in TOOLS_SPEC:
        batches_needed = EXAMPLES_PER_TOOL // BATCH_SIZE
        combos = [
            (style, path_ctx)
            for style in QUERY_STYLES
            for path_ctx in PATH_CONTEXTS
        ]
        random.shuffle(combos)
        # Use enough combos to cover batches_needed, cycling if needed
        selected = (combos * ((batches_needed // len(combos)) + 1))[:batches_needed]
        for style, path_ctx in selected:
            tasks_spec.append((tool_name, style, path_ctx, BATCH_SIZE))

    total_batches = len(tasks_spec)
    console.print(
        f"[bold]Generating {total_batches} batches "
        f"({BATCH_SIZE} examples each) across {len(TOOLS_SPEC)} tools...[/bold]\n"
    )

    async def run_one(
        spec: tuple[str, str, str, int],
        progress: Progress,
        task_id: int,
    ) -> list[dict]:
        async with semaphore:
            result = await generate_batch(http_client, api_key, *spec)
            progress.advance(task_id)
            await asyncio.sleep(DELAY_BETWEEN_REQUESTS)
            return result

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Generating", total=total_batches)
        coros = [run_one(spec, progress, task_id) for spec in tasks_spec]
        results = await asyncio.gather(*coros)

    for batch in results:
        all_examples.extend(batch)

    return all_examples


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def deduplicate(examples: list[dict]) -> list[dict]:
    """Remove examples with duplicate user queries (case-insensitive)."""
    seen: set[str] = set()
    unique: list[dict] = []
    for ex in examples:
        key = ex["messages"][0]["content"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique


def save_dataset(examples: list[dict]) -> tuple[int, int]:
    """Shuffle, split into train/eval, and save as JSONL."""
    random.shuffle(examples)
    split_idx = int(len(examples) * (1 - EVAL_RATIO))
    train = examples[:split_idx]
    eval_ = examples[split_idx:]

    TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(TRAIN_FILE, "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")

    with open(EVAL_FILE, "w") as f:
        for ex in eval_:
            f.write(json.dumps(ex) + "\n")

    return len(train), len(eval_)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(n_train: int, n_eval: int) -> None:
    """Print a summary table of the generated dataset."""
    console.print(f"\n[bold green]Saved:[/bold green]")
    console.print(f"  Train: {TRAIN_FILE} ({n_train} examples)")
    console.print(f"  Eval:  {EVAL_FILE} ({n_eval} examples)")

    # Per-tool breakdown
    table = Table(title="Dataset Summary")
    table.add_column("Tool", style="cyan")
    table.add_column("Train", justify="right")
    table.add_column("Eval", justify="right")

    with open(TRAIN_FILE) as f:
        train_data = [json.loads(line) for line in f]
    with open(EVAL_FILE) as f:
        eval_data = [json.loads(line) for line in f]

    for tool_name in TOOLS_SPEC:
        t_count = sum(
            1
            for ex in train_data
            if json.loads(ex["messages"][1]["content"]).get("tool") == tool_name
        )
        e_count = sum(
            1
            for ex in eval_data
            if json.loads(ex["messages"][1]["content"]).get("tool") == tool_name
        )
        table.add_row(tool_name, str(t_count), str(e_count))

    table.add_row(
        "[bold]Total[/bold]", f"[bold]{n_train}[/bold]", f"[bold]{n_eval}[/bold]"
    )
    console.print(table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    load_dotenv()

    api_key = os.getenv("NVIDIA_API_KEY", "")
    if not api_key:
        console.print(
            "[red]NVIDIA_API_KEY not set. Add it to .env (see .env.example).[/red]"
        )
        sys.exit(1)

    target = EXAMPLES_PER_TOOL * len(TOOLS_SPEC)
    console.print("[bold cyan]Edge MCP Caller — Training Data Generator[/bold cyan]")
    console.print(f"Model: {MODEL} via NVIDIA NIM")
    console.print(f"Target: ~{target} examples across {len(TOOLS_SPEC)} tools\n")

    async with httpx.AsyncClient() as http_client:
        examples = await generate_all(http_client, api_key)

    console.print(f"\n[green]Generated {len(examples)} raw examples[/green]")

    # Deduplicate
    examples = deduplicate(examples)
    console.print(f"[green]After dedup: {len(examples)} unique examples[/green]")

    if not examples:
        console.print(
            "[red]No examples generated. Check NVIDIA_API_KEY and try again.[/red]"
        )
        sys.exit(1)

    # Save
    n_train, n_eval = save_dataset(examples)
    print_summary(n_train, n_eval)


if __name__ == "__main__":
    asyncio.run(main())
