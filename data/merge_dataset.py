#!/usr/bin/env python3
"""Merge all generated JSONL data into train/eval splits.

Reads ALL data/generated/*.jsonl files, validates each example,
deduplicates by query, and creates stratified train/eval splits.

Usage:
    python data/merge_dataset.py              # merge + split
    python data/merge_dataset.py --dry-run    # validate only, don't write
"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = PROJECT_ROOT / "data" / "generated"
TRAIN_FILE = PROJECT_ROOT / "data" / "train.jsonl"
EVAL_FILE = PROJECT_ROOT / "data" / "eval.jsonl"

EVAL_RATIO = 0.10  # 90/10 split

VALID_TOOLS = {"list_directory", "read_file", "search_files", "write_file", "create_directory"}

TOOL_REQUIRED_ARGS: dict[str, list[str]] = {
    "list_directory": ["path"],
    "read_file": ["path"],
    "search_files": ["path", "pattern"],
    "write_file": ["path", "content"],
    "create_directory": ["path"],
}

# File extensions that indicate a file, not a directory
FILE_EXTENSIONS = re.compile(
    r"\.(py|js|ts|tsx|jsx|json|yaml|yml|toml|cfg|ini|conf|md|txt|html|css|"
    r"scss|less|xml|csv|sql|sh|bash|zsh|rb|go|rs|java|kt|c|cpp|h|hpp|"
    r"swift|m|ex|exs|erl|hs|lua|r|pl|pm|php|vue|svelte|lock|log|env|"
    r"dockerfile|makefile|gitignore|dockerignore|editorconfig|prettierrc|"
    r"eslintrc|babelrc|tf|hcl|proto|graphql|gql)$",
    re.IGNORECASE,
)

console = Console()


# ---------------------------------------------------------------------------
# Category detection
# ---------------------------------------------------------------------------


def detect_category(filename: str) -> str:
    """Detect example category from the source filename.

    Naming convention: {tool}_{category}_{timestamp}.jsonl
    Fallback for test files (test_*.jsonl) → clean.
    """
    name = Path(filename).stem.lower()

    if "_clean_" in name:
        return "clean"
    if "_messy_" in name:
        return "messy"
    if "_adversarial_" in name:
        return "adversarial"

    # Test files from trial runs → clean
    if name.startswith("test_"):
        return "clean"

    # Default: clean
    return "clean"


# ---------------------------------------------------------------------------
# Validation (11 checks)
# ---------------------------------------------------------------------------


def validate_example(ex: dict, source_file: str = "") -> tuple[bool, str]:
    """Validate a single training example. Returns (valid, reason)."""
    # 1. Top-level structure
    if not isinstance(ex, dict) or "messages" not in ex:
        return False, "missing 'messages' key"

    msgs = ex["messages"]

    # 2. Correct messages format (user + assistant)
    if not isinstance(msgs, list) or len(msgs) != 2:
        return False, f"expected 2 messages, got {len(msgs) if isinstance(msgs, list) else type(msgs)}"

    if msgs[0].get("role") != "user" or msgs[1].get("role") != "assistant":
        return False, "expected [user, assistant] roles"

    # 3. Non-empty user query
    query = msgs[0].get("content", "")
    if not query or not query.strip():
        return False, "empty user query"

    # 4. Valid JSON in assistant content
    assistant_content = msgs[1].get("content", "")
    try:
        tool_call = json.loads(assistant_content)
    except (json.JSONDecodeError, TypeError):
        return False, "assistant content is not valid JSON"

    # 5. Tool in VALID_TOOLS
    tool = tool_call.get("tool")
    if tool not in VALID_TOOLS:
        return False, f"unknown tool '{tool}'"

    # 6. Required args present
    args = tool_call.get("args", {})
    required = TOOL_REQUIRED_ARGS.get(tool, [])
    for arg in required:
        if arg not in args:
            return False, f"missing required arg '{arg}' for {tool}"

    # 7. All arg values are strings
    for k, v in args.items():
        if not isinstance(v, str):
            return False, f"arg '{k}' is {type(v).__name__}, expected string"

    # 8. No trailing slash on file paths (list_directory and create_directory are ok)
    path_val = args.get("path", "")
    if tool in ("read_file", "write_file") and path_val.endswith("/"):
        return False, f"trailing slash on {tool} path: {path_val}"

    # 9. search_files pattern should not be just "*"
    if tool == "search_files":
        pattern = args.get("pattern", "")
        if pattern == "*":
            return False, "search_files pattern is bare '*' (should use list_directory)"

    # 10. create_directory path should end with /
    if tool == "create_directory" and not path_val.endswith("/"):
        return False, f"create_directory path missing trailing slash: {path_val}"

    # 11. write_file content should be non-empty
    if tool == "write_file":
        content = args.get("content", "")
        if not content.strip():
            return False, "write_file content is empty"

    return True, ""


# ---------------------------------------------------------------------------
# Loading and deduplication
# ---------------------------------------------------------------------------


def load_all_generated() -> list[tuple[dict, str]]:
    """Load all examples from data/generated/*.jsonl.

    Returns list of (example_dict, source_filename) tuples.
    """
    if not GENERATED_DIR.exists():
        console.print(f"[red]Generated directory not found: {GENERATED_DIR}[/red]")
        return []

    files = sorted(GENERATED_DIR.glob("*.jsonl"))
    if not files:
        console.print(f"[red]No .jsonl files found in {GENERATED_DIR}[/red]")
        return []

    all_examples: list[tuple[dict, str]] = []
    parse_errors = 0

    for filepath in files:
        with open(filepath) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                    all_examples.append((ex, filepath.name))
                except json.JSONDecodeError:
                    parse_errors += 1
                    if parse_errors <= 5:
                        console.print(
                            f"[yellow]JSON parse error: {filepath.name}:{line_num}[/yellow]"
                        )

    console.print(f"Loaded {len(all_examples)} raw examples from {len(files)} files")
    if parse_errors:
        console.print(f"[yellow]{parse_errors} JSON parse errors skipped[/yellow]")

    return all_examples


def deduplicate(examples: list[tuple[dict, str]]) -> list[tuple[dict, str]]:
    """Remove examples with duplicate user queries (case-insensitive)."""
    seen: set[str] = set()
    unique: list[tuple[dict, str]] = []

    for ex, source in examples:
        query = ex.get("messages", [{}])[0].get("content", "").lower().strip()
        if query and query not in seen:
            seen.add(query)
            unique.append((ex, source))

    removed = len(examples) - len(unique)
    if removed:
        console.print(f"Deduplicated: removed {removed} duplicates")

    return unique


# ---------------------------------------------------------------------------
# Split and save
# ---------------------------------------------------------------------------


def split_and_save(
    examples: list[tuple[dict, str]], dry_run: bool = False
) -> tuple[int, int]:
    """Stratified 90/10 split by (tool, category). Save to train/eval JSONL.

    Eval examples include a "category" tag for downstream robustness reporting.
    Train examples do NOT include category tags.
    """
    # Group by (tool, category)
    groups: dict[tuple[str, str], list[tuple[dict, str]]] = defaultdict(list)

    for ex, source in examples:
        tool_call = json.loads(ex["messages"][1]["content"])
        tool = tool_call["tool"]
        category = detect_category(source)
        groups[(tool, category)].append((ex, source))

    train_examples: list[dict] = []
    eval_examples: list[dict] = []

    for (tool, category), group in groups.items():
        random.shuffle(group)
        split_idx = max(1, int(len(group) * (1 - EVAL_RATIO)))
        train_part = group[:split_idx]
        eval_part = group[split_idx:]

        for ex, _ in train_part:
            train_examples.append(ex)

        for ex, _ in eval_part:
            eval_ex = dict(ex)
            eval_ex["category"] = category
            eval_examples.append(eval_ex)

    random.shuffle(train_examples)
    random.shuffle(eval_examples)

    if not dry_run:
        TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)

        with open(TRAIN_FILE, "w") as f:
            for ex in train_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        with open(EVAL_FILE, "w") as f:
            for ex in eval_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        console.print(f"\n[bold green]Saved:[/bold green]")
        console.print(f"  Train: {TRAIN_FILE} ({len(train_examples)} examples)")
        console.print(f"  Eval:  {EVAL_FILE} ({len(eval_examples)} examples)")

    return len(train_examples), len(eval_examples)


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------


def print_summary(
    valid_examples: list[tuple[dict, str]],
    invalid_count: int,
    n_train: int,
    n_eval: int,
) -> None:
    """Print per-tool x per-category distribution matrix."""
    # Build counts
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ex, source in valid_examples:
        tool_call = json.loads(ex["messages"][1]["content"])
        tool = tool_call["tool"]
        category = detect_category(source)
        counts[tool][category] += 1

    # Distribution table
    table = Table(title="Dataset Distribution (Tool x Category)")
    table.add_column("Tool", style="cyan")
    table.add_column("Clean", justify="right")
    table.add_column("Messy", justify="right")
    table.add_column("Adversarial", justify="right")
    table.add_column("Total", justify="right", style="bold")

    all_tools = sorted(VALID_TOOLS)
    categories = ["clean", "messy", "adversarial"]
    grand_total = 0
    cat_totals = defaultdict(int)

    for tool in all_tools:
        row_total = 0
        row_vals = []
        for cat in categories:
            c = counts[tool][cat]
            row_vals.append(str(c))
            row_total += c
            cat_totals[cat] += c
        grand_total += row_total
        table.add_row(tool, *row_vals, str(row_total))

    table.add_row(
        "[bold]Total[/bold]",
        *[f"[bold]{cat_totals[c]}[/bold]" for c in categories],
        f"[bold]{grand_total}[/bold]",
    )

    console.print()
    console.print(table)

    # Category ratios
    if grand_total > 0:
        console.print(f"\n[bold]Category ratios:[/bold]")
        for cat in categories:
            pct = cat_totals[cat] / grand_total * 100
            console.print(f"  {cat}: {cat_totals[cat]} ({pct:.1f}%)")

    # Split info
    console.print(f"\n[bold]Split:[/bold] {n_train} train / {n_eval} eval")
    if invalid_count:
        console.print(f"[yellow]Rejected: {invalid_count} invalid examples[/yellow]")

    # Sample 5 random examples for spot-check
    if valid_examples:
        console.print(f"\n[bold]Random sample (5 examples):[/bold]")
        sample = random.sample(valid_examples, min(5, len(valid_examples)))
        for ex, source in sample:
            query = ex["messages"][0]["content"][:80]
            tool_call = json.loads(ex["messages"][1]["content"])
            tool = tool_call["tool"]
            args_str = json.dumps(tool_call["args"], ensure_ascii=False)
            if len(args_str) > 60:
                args_str = args_str[:57] + "..."
            console.print(f"  [{detect_category(source)}] {tool}: {query}")
            console.print(f"    args: {args_str}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    console.print("[bold cyan]Edge MCP Caller — Dataset Merge Pipeline[/bold cyan]\n")

    # Load
    raw = load_all_generated()
    if not raw:
        sys.exit(1)

    # Validate
    valid: list[tuple[dict, str]] = []
    invalid_count = 0
    invalid_reasons: dict[str, int] = defaultdict(int)

    for ex, source in raw:
        ok, reason = validate_example(ex, source)
        if ok:
            valid.append((ex, source))
        else:
            invalid_count += 1
            invalid_reasons[reason] += 1

    console.print(f"Validated: {len(valid)} valid, {invalid_count} invalid")

    if invalid_reasons:
        console.print(f"\n[yellow]Rejection reasons:[/yellow]")
        for reason, count in sorted(invalid_reasons.items(), key=lambda x: -x[1]):
            console.print(f"  {count:4d}  {reason}")

    # Deduplicate
    valid = deduplicate(valid)

    if not valid:
        console.print("[red]No valid examples after validation + dedup.[/red]")
        sys.exit(1)

    # Split and save
    if dry_run:
        console.print("\n[yellow]DRY RUN — not writing files[/yellow]")

    n_train, n_eval = split_and_save(valid, dry_run=dry_run)

    # Summary
    print_summary(valid, invalid_count, n_train, n_eval)

    # write_file round-trip JSON parse test
    write_file_count = 0
    write_file_parse_ok = 0
    for ex, _ in valid:
        tool_call = json.loads(ex["messages"][1]["content"])
        if tool_call["tool"] == "write_file":
            write_file_count += 1
            try:
                # Verify the full line round-trips through JSON
                line = json.dumps(ex, ensure_ascii=False)
                reparsed = json.loads(line)
                inner = json.loads(reparsed["messages"][1]["content"])
                assert inner["tool"] == "write_file"
                assert isinstance(inner["args"]["content"], str)
                assert len(inner["args"]["content"].strip()) > 0
                write_file_parse_ok += 1
            except (json.JSONDecodeError, AssertionError, KeyError):
                pass

    if write_file_count > 0:
        console.print(
            f"\n[bold]write_file round-trip test:[/bold] "
            f"{write_file_parse_ok}/{write_file_count} passed"
        )


if __name__ == "__main__":
    main()
