#!/usr/bin/env python3
"""Clean bad examples from dataset and backfill to reach 1200/tool target.

Removes:
  - list_directory pointing at file-like paths (no trailing slash, has extension)
  - search_files with pattern="*" (should be list_directory)
  - search_files with empty path="" (ambiguous)

Then generates replacement examples via NVIDIA NIM API to hit 1200 per tool.
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

# Reuse config and helpers from generate_dataset
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.generate_dataset import (
    TOOLS_SPEC,
    QUERY_STYLES,
    PATH_CONTEXTS,
    EVAL_RATIO,
    build_generation_prompt,
    parse_examples,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = PROJECT_ROOT / "data" / "train.jsonl"
EVAL_FILE = PROJECT_ROOT / "data" / "eval.jsonl"

TARGET_PER_TOOL = 1200
MODEL = "meta/llama-3.1-70b-instruct"
BASE_URL = "https://integrate.api.nvidia.com/v1"
BATCH_SIZE = 20
MAX_CONCURRENT = 5
REQUEST_TIMEOUT = 60
RETRY_MAX = 5
RETRY_BASE_DELAY = 3
DELAY_BETWEEN_REQUESTS = 1

console = Console()


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


KNOWN_EXTENSIONLESS_FILES = {
    "Makefile", "Dockerfile", "Vagrantfile", "Gemfile", "Rakefile",
    "Procfile", "LICENSE", "README", "CHANGELOG", "ROADMAP", "CODEOWNERS",
    "Cargo.lock", "Brewfile", "Justfile", "Taskfile",
}

SEARCH_INTENT_KEYWORDS = [
    "find", "search", "locate", "where", "look for", "any", "all the",
]


def is_bad_example(ex: dict) -> bool:
    """Return True if the example has a quality issue."""
    tc = json.loads(ex["messages"][1]["content"])
    tool = tc["tool"]
    args = tc["args"]
    path = args.get("path", "")
    query = ex["messages"][0]["content"].lower()

    # --- Round 1 rules (unchanged) ---

    # list_directory on file-like paths (has extension, no trailing slash)
    if tool == "list_directory":
        last = path.rstrip("/").split("/")[-1] if path else ""
        if "." in last and not path.endswith("/"):
            return True

    # search_files with wildcard-only pattern
    if tool == "search_files" and args.get("pattern") == "*":
        return True

    # search_files with empty path
    if tool == "search_files" and path == "":
        return True

    # --- Round 2 rules (new) ---

    # read_file where query mentions "directory"/"folder" (should be list_directory)
    if tool == "read_file":
        if any(kw in query for kw in ["director", "folder", "dir "]):
            return True
        # read_file with path ending in / (definitely a directory)
        if path.endswith("/"):
            return True

    # search_files with exact filename pattern + no search intent in query
    if tool == "search_files":
        pattern = args.get("pattern", "")
        if pattern and "*" not in pattern and "?" not in pattern:
            if not any(kw in query for kw in SEARCH_INTENT_KEYWORDS):
                return True

    return False


def clean_dataset() -> tuple[list[dict], dict[str, int]]:
    """Load, clean, and return good examples + deficit per tool."""
    with open(TRAIN_FILE) as f:
        train = [json.loads(l) for l in f]
    with open(EVAL_FILE) as f:
        eval_ = [json.loads(l) for l in f]
    all_ex = train + eval_

    clean = [ex for ex in all_ex if not is_bad_example(ex)]
    removed = len(all_ex) - len(clean)

    # Calculate deficit per tool
    deficit: dict[str, int] = {}
    for tool_name in TOOLS_SPEC:
        count = sum(
            1
            for ex in clean
            if json.loads(ex["messages"][1]["content"])["tool"] == tool_name
        )
        deficit[tool_name] = max(0, TARGET_PER_TOOL - count)

    console.print(f"[green]Cleaned: {removed} bad examples removed[/green]")
    console.print(f"[green]Remaining: {len(clean)} clean examples[/green]")

    return clean, deficit


# ---------------------------------------------------------------------------
# Backfill generation (with dedup against existing)
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


async def generate_backfill(
    http_client: httpx.AsyncClient,
    api_key: str,
    deficit: dict[str, int],
    existing_queries: set[str],
) -> list[dict]:
    """Generate backfill examples for tools that are under target."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    all_new: list[dict] = []

    # Build tasks for each tool that needs backfill
    tasks_spec: list[tuple[str, str, str, int]] = []
    for tool_name, needed in deficit.items():
        if needed <= 0:
            continue
        batches = (needed // BATCH_SIZE) + (1 if needed % BATCH_SIZE else 0)
        # Add extra batches to account for dedup/validation losses
        batches = int(batches * 1.3)
        combos = [
            (style, path_ctx)
            for style in QUERY_STYLES
            for path_ctx in PATH_CONTEXTS
        ]
        random.shuffle(combos)
        selected = (combos * ((batches // len(combos)) + 1))[:batches]
        for style, path_ctx in selected:
            tasks_spec.append((tool_name, style, path_ctx, BATCH_SIZE))

    if not tasks_spec:
        console.print("[green]No backfill needed![/green]")
        return []

    total = len(tasks_spec)
    console.print(
        f"\n[bold]Backfilling {total} batches "
        f"({BATCH_SIZE} examples each)...[/bold]\n"
    )

    async def run_one(
        spec: tuple[str, str, str, int],
        progress: Progress,
        task_id: int,
    ) -> list[dict]:
        tool_name, style, path_ctx, n = spec
        prompt = build_generation_prompt(tool_name, style, path_ctx, n)
        async with semaphore:
            try:
                text = await nim_chat(http_client, api_key, prompt)
                result = parse_examples(text, tool_name)
                # Filter out bad examples immediately
                result = [ex for ex in result if not is_bad_example(ex)]
                # Filter out duplicates against existing
                result = [
                    ex
                    for ex in result
                    if ex["messages"][0]["content"].lower().strip()
                    not in existing_queries
                ]
            except Exception as e:
                console.print(f"[red]Backfill error ({tool_name}): {e}[/red]")
                result = []
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
        task_id = progress.add_task("Backfilling", total=total)
        coros = [run_one(spec, progress, task_id) for spec in tasks_spec]
        results = await asyncio.gather(*coros)

    for batch in results:
        all_new.extend(batch)

    return all_new


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_dataset(examples: list[dict]) -> tuple[int, int]:
    """Shuffle, split into train/eval, and save as JSONL."""
    random.shuffle(examples)
    split_idx = int(len(examples) * (1 - EVAL_RATIO))
    train = examples[:split_idx]
    eval_ = examples[split_idx:]

    with open(TRAIN_FILE, "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")

    with open(EVAL_FILE, "w") as f:
        for ex in eval_:
            f.write(json.dumps(ex) + "\n")

    return len(train), len(eval_)


def print_summary(n_train: int, n_eval: int) -> None:
    """Print a summary table of the final dataset."""
    table = Table(title="Final Dataset")
    table.add_column("Tool", style="cyan")
    table.add_column("Train", justify="right")
    table.add_column("Eval", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("vs Target", justify="right")

    with open(TRAIN_FILE) as f:
        train_data = [json.loads(line) for line in f]
    with open(EVAL_FILE) as f:
        eval_data = [json.loads(line) for line in f]

    for tool_name in TOOLS_SPEC:
        t = sum(
            1
            for ex in train_data
            if json.loads(ex["messages"][1]["content"]).get("tool") == tool_name
        )
        e = sum(
            1
            for ex in eval_data
            if json.loads(ex["messages"][1]["content"]).get("tool") == tool_name
        )
        total = t + e
        pct = f"{total / TARGET_PER_TOOL * 100:.0f}%"
        table.add_row(tool_name, str(t), str(e), str(total), pct)

    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{n_train}[/bold]",
        f"[bold]{n_eval}[/bold]",
        f"[bold]{n_train + n_eval}[/bold]",
        f"[bold]{(n_train + n_eval) / (TARGET_PER_TOOL * 3) * 100:.0f}%[/bold]",
    )
    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    load_dotenv()

    api_key = os.getenv("NVIDIA_API_KEY", "")
    if not api_key:
        console.print("[red]NVIDIA_API_KEY not set.[/red]")
        sys.exit(1)

    console.print("[bold cyan]Edge MCP Caller — Clean & Backfill[/bold cyan]\n")

    # Step 1: Clean
    clean_examples, deficit = clean_dataset()

    table = Table(title="Backfill Plan")
    table.add_column("Tool", style="cyan")
    table.add_column("Clean", justify="right")
    table.add_column("Need", justify="right", style="yellow")
    for tool_name in TOOLS_SPEC:
        count = sum(
            1
            for ex in clean_examples
            if json.loads(ex["messages"][1]["content"])["tool"] == tool_name
        )
        table.add_row(tool_name, str(count), str(deficit[tool_name]))
    console.print(table)

    # Build set of existing queries for dedup
    existing_queries = {
        ex["messages"][0]["content"].lower().strip() for ex in clean_examples
    }

    # Step 2: Backfill
    async with httpx.AsyncClient() as http_client:
        new_examples = await generate_backfill(
            http_client, api_key, deficit, existing_queries
        )

    console.print(f"\n[green]Generated {len(new_examples)} new examples[/green]")

    # Merge, capping each tool at TARGET_PER_TOOL
    merged = list(clean_examples)
    for tool_name in TOOLS_SPEC:
        existing_count = sum(
            1
            for ex in clean_examples
            if json.loads(ex["messages"][1]["content"])["tool"] == tool_name
        )
        new_for_tool = [
            ex
            for ex in new_examples
            if json.loads(ex["messages"][1]["content"])["tool"] == tool_name
        ]
        take = min(len(new_for_tool), TARGET_PER_TOOL - existing_count)
        merged.extend(new_for_tool[:take])
        console.print(
            f"  {tool_name}: {existing_count} existing + {take} new = "
            f"{existing_count + take}"
        )

    # Step 3: Save
    n_train, n_eval = save_dataset(merged)
    console.print(f"\n[bold green]Saved {n_train + n_eval} total examples[/bold green]")
    print_summary(n_train, n_eval)


if __name__ == "__main__":
    asyncio.run(main())
