#!/bin/bash
# Generate per-tool GIF demos for all 14 tools.
#
# Records real model + MCP execution via asciinema, converts to GIF via agg.
# Groups tools so fixture resets only happen before mutating tools.
#
# Prerequisites:
#   - Ollama running with edge-mcp-caller:latest
#   - Node.js 18+ (filesystem MCP server)
#   - uvx (git MCP server)
#   - asciinema (installed)
#   - agg (at /tmp/agg or on PATH)
#
# Usage:
#   bash demos/generate_all_gifs.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/per-tool"
SETUP_FIXTURE="$PROJECT_ROOT/tests/e2e/setup_fixture.py"
RUN_TOOL="$SCRIPT_DIR/run_single_tool.py"
VENV="$PROJECT_ROOT/.venv/bin/activate"

# Activate venv for fixture setup commands
if [ -f "$VENV" ]; then
    source "$VENV"
fi

# Find agg binary
if command -v agg &>/dev/null; then
    AGG="agg"
elif [ -x /tmp/agg ]; then
    AGG="/tmp/agg"
else
    echo "ERROR: agg not found. Install it or download to /tmp/agg"
    echo "  curl -L https://github.com/asciinema/agg/releases/latest/download/agg-x86_64-unknown-linux-gnu -o /tmp/agg && chmod +x /tmp/agg"
    exit 1
fi

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Tool ordering — read-only first, then mutating
READ_ONLY_TOOLS=(
    "01:list_directory"
    "02:read_file"
    "03:search_files"
    "04:directory_tree"
    "05:git_status"
    "06:git_diff_staged"
    "07:git_log"
    "08:git_branch"
)

MUTATING_TOOLS=(
    "09:write_file"
    "10:create_directory"
    "11:edit_file"
    "12:move_file"
    "13:git_commit"
    "14:git_create_branch"
)

record_tool() {
    local num="$1"
    local tool="$2"
    local cast_file="$OUTPUT_DIR/${num}_${tool}.cast"
    local gif_file="$OUTPUT_DIR/${num}_${tool}.gif"

    echo "  Recording $num $tool..."
    asciinema rec "$cast_file" \
        --cols 100 --rows 35 --overwrite \
        -c "bash -c '. $VENV && python $RUN_TOOL --tool $tool'" \
        2>/dev/null

    echo "  Converting to GIF..."
    "$AGG" "$cast_file" "$gif_file" \
        --font-size 14 \
        --last-frame-duration 4 \
        --theme monokai \
        2>/dev/null

    local size
    size=$(du -h "$gif_file" | cut -f1)
    echo "  -> $gif_file ($size)"
}

echo "=== Per-Tool GIF Generation ==="
echo "Output: $OUTPUT_DIR"
echo ""

# --- Read-only tools: single fixture setup ---
echo "[1/2] Setting up fixture for read-only tools..."
python "$SETUP_FIXTURE" --clean
echo ""

echo "Recording read-only tools (8)..."
for entry in "${READ_ONLY_TOOLS[@]}"; do
    num="${entry%%:*}"
    tool="${entry##*:}"
    record_tool "$num" "$tool"
done
echo ""

# --- Mutating tools: reset fixture before each ---
echo "[2/2] Recording mutating tools (6) — fixture reset before each..."
for entry in "${MUTATING_TOOLS[@]}"; do
    num="${entry%%:*}"
    tool="${entry##*:}"
    echo "  Resetting fixture..."
    python "$SETUP_FIXTURE" --clean 2>/dev/null
    record_tool "$num" "$tool"
done
echo ""

# --- Summary ---
echo "=== Done ==="
echo ""
echo "Generated GIFs:"
ls -lh "$OUTPUT_DIR"/*.gif 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'
echo ""

total_gifs=$(ls "$OUTPUT_DIR"/*.gif 2>/dev/null | wc -l)
echo "$total_gifs/14 GIFs generated."
