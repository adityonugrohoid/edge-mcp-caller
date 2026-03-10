#!/usr/bin/env python3
"""Generate a visually appealing demo recording for GitHub/LinkedIn.

This script replays the e2e test output with controlled timing, making
it look natural and readable in a GIF. Run it inside asciinema rec.

Usage:
    asciinema rec demos/e2e-demo.cast --cols 90 --rows 42 --overwrite \
        -c "python demos/record_demo.py"
"""

import sys
import time


def type_out(text: str, delay: float = 0.04) -> None:
    """Simulate typing a command."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)


def print_slow(text: str, delay: float = 0.02) -> None:
    """Print text line by line with a small delay."""
    for line in text.split("\n"):
        print(line)
        sys.stdout.flush()
        time.sleep(delay)


def pause(seconds: float = 0.5) -> None:
    time.sleep(seconds)


# ANSI color helpers
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"
BGREEN = "\033[1;32m"
BMAGENTA = "\033[1;35m"
BCYAN = "\033[1;36m"


def main():
    # --- Command prompt ---
    pause(0.5)
    sys.stdout.write(f"{BGREEN}${RESET} ")
    sys.stdout.flush()
    type_out("python tests/e2e/run_e2e.py --reset", delay=0.03)
    pause(0.3)
    print()
    pause(0.8)

    # --- Fixture setup ---
    print_slow(f"""{DIM}Setting up fixture...{RESET}
{DIM}Creating fixture at tests/e2e/fixture/{RESET}
  Created directory structure (12 dirs)
  Created 30 files (.py, .js, .tsx, .go, .yaml, .sql, .sh, .md)
  Initialized git repo: 4 commits, 4 branches
  Staged: README.md | Unstaged: src/main.py
""", delay=0.04)
    pause(0.6)

    # --- Banner ---
    print(f"{'─' * 78}")
    print(f" {BMAGENTA}E2E Test{RESET} — Full pipeline (model → MCP)")
    print(f" Fixture: tests/e2e/fixture/")
    print(f" Servers: {CYAN}filesystem{RESET} (npx) + {CYAN}git{RESET} (uvx)")
    print(f" Tests:   14 (one per tool, read-first ordering)")
    print(f"{'─' * 78}")
    pause(0.5)

    print(f"\n{DIM}Connecting to MCP servers...{RESET}")
    pause(1.0)
    print(f"{GREEN}Connected.{RESET} 26 tools on servers.\n")
    pause(0.4)

    # --- Test results (simulated with realistic timing) ---
    tests = [
        ("list_directory",    "filesystem", 224,   4, True, "Listed src/ → app.js, components/, main.py, utils/"),
        ("read_file",         "filesystem", 127,   3, True, "Read config/settings.yaml → got YAML config"),
        ("search_files",      "filesystem", 153,   5, True, "Found *.py in tests/ → 3 files"),
        ("directory_tree",    "filesystem", 113,   3, True, "Tree of docs/ → API.md, CHANGELOG.md, README.md"),
        ("git_status",        "git",         95,   8, True, "On main, staged README.md, unstaged main.py"),
        ("git_diff_staged",   "git",        107,   8, True, "Showed staged README.md diff"),
        ("git_log",           "git",        120,  13, True, "3 commits: test → fix → docs"),
        ("git_branch",        "git",         85,   8, True, "4 branches: develop, feature/redesign, main, staging"),
        ("write_file",        "filesystem", 154,   3, True, "Created notes.txt → 'hello world'"),
        ("create_directory",  "filesystem", 113,   3, True, "Created backups/ directory"),
        ("edit_file",         "filesystem", 222,   5, True, "config/settings.yaml: debug: true → false"),
        ("move_file",         "filesystem", 169,   3, True, "scripts/deploy.sh → config/deploy.sh"),
        ("git_commit",        "git",        112,  14, True, "Committed staged changes: 'update readme'"),
        ("git_create_branch", "git",        183,   7, True, "Created feature/test from main"),
    ]

    for i, (tool, server, model_ms, mcp_ms, passed, detail) in enumerate(tests, 1):
        total_ms = model_ms + mcp_ms
        server_color = CYAN if server == "filesystem" else YELLOW

        # Simulate model inference time
        sys.stdout.write(f"  {DIM}{i:2d}.{RESET} {CYAN}{tool:20s}{RESET}")
        sys.stdout.flush()
        time.sleep(0.15)  # brief pause to simulate work

        # Show result
        status = f"{BGREEN}pass{RESET}" if passed else f"{RED}FAIL{RESET}"
        line = f"  {status}  {server_color}{server:10s}{RESET}  {DIM}{total_ms:4d}ms{RESET}"
        print(line)
        sys.stdout.flush()
        time.sleep(0.08)

    pause(0.8)

    # --- Summary table ---
    print()
    print(f"{'═' * 78}")
    print(f"  {BMAGENTA}E2E Test Results — 14 tools × 2 MCP servers{RESET}")
    print(f"{'═' * 78}")
    print()

    # Table header
    print(f"  {'#':>3}  {'Tool':<20} {'Server':<12} {'Model':^7} {'MCP':^7} {'Check':^7} {'Latency':>8}")
    print(f"  {'─'*3}  {'─'*20} {'─'*12} {'─'*7} {'─'*7} {'─'*7} {'─'*8}")

    for i, (tool, server, model_ms, mcp_ms, passed, _) in enumerate(tests, 1):
        total_ms = model_ms + mcp_ms
        p = f"{GREEN}pass{RESET}"
        server_color = CYAN if server == "filesystem" else YELLOW
        print(f"  {i:3d}  {CYAN}{tool:<20}{RESET} {server_color}{server:<12}{RESET} {p:>16} {p:>16} {p:>16} {DIM}{total_ms:>5d}ms{RESET}")
        sys.stdout.flush()
        time.sleep(0.06)

    print(f"  {'─'*3}  {'─'*20} {'─'*12} {'─'*7} {'─'*7} {'─'*7} {'─'*8}")
    pause(0.5)

    # --- Totals ---
    print()
    print(f"  Tool routing:  {BGREEN}14{RESET}/{BCYAN}14{RESET}")
    print(f"  MCP execution: {BGREEN}14{RESET}/{BCYAN}14{RESET}")
    print(f"  Result check:  {BGREEN}14{RESET}/{BCYAN}14{RESET}")
    pause(0.3)
    print()
    print(f"  {BGREEN}All 14 tools passed end-to-end.{RESET}")
    print()

    # --- Key stats ---
    print(f"  {DIM}Specialist model:{RESET} Gemma 3 270M (Q8_0, 291 MB)")
    print(f"  {DIM}Avg latency:{RESET}      ~150ms (model ~130ms + MCP ~7ms)")
    print(f"  {DIM}Prompt tokens:{RESET}    ~20 per request (no schemas in prompt)")
    print(f"  {DIM}Servers:{RESET}          filesystem (8 tools) + git (6 tools)")
    print()
    pause(3.0)


if __name__ == "__main__":
    main()
