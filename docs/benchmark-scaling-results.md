# Scaling Benchmark Results

**Date**: 2026-03-10
**Model**: edge-mcp-caller:latest (Gemma 3 270M, Q8_0, 291 MB)
**Training data**: 14,033 examples across 14 tools
**Eval data**: 1,568 examples, 112/tool, stratified (76 clean + 18 messy + 18 disambiguation per tool)
**Sampling**: 30 examples per tool per subset, seed=42

## Purpose

Measure how specialist accuracy degrades as tool count increases from 3 to 14, and compare against three baselines at each scale point. The specialist is trained on all 14 tools simultaneously — these subsets test whether a wider training vocabulary hurts narrow-scope performance.

## Models Under Test

| Model | Size | Interface | Tokens/Request |
|-------|------|-----------|----------------|
| **Specialist** (edge-mcp-caller) | 270M | Bare query, no schema | 18–21 |
| **Raw Gemma 3** (gemma3:270m) | 270M | Few-shot system prompt + JSON format | 265–1,092 |
| **FunctionGemma** (functiongemma:270m) | 270M | Ollama tools API with full schemas | 262–967 |
| **GPT-OSS 120B** (openai/gpt-oss-120b) | 120B | NIM API with OpenAI tools format | 244–647 |

Each model is tested in its designed interface. The specialist receives only the user query (no tool definitions); baselines receive the appropriate tool schemas for the subset.

---

## Scaling Curve: Overall Accuracy

### Tool Routing Accuracy

| Model | 3-tool | 5-tool | 8-tool | 14-tool | Trend |
|-------|--------|--------|--------|---------|-------|
| **Specialist** | **100.0%** | **100.0%** | **99.6%** | **99.8%** | Flat |
| GPT-OSS 120B | 90.0% | 75.3% | 71.2% | 72.6% | Degrading |
| Raw Gemma 3 | 61.1% | 52.7% | 48.3% | 33.3% | Collapsing |
| FunctionGemma | 30.0% | 27.3% | 30.0% | 21.4% | Collapsing |

### Combined Accuracy (Tool + Args Exact Match)

| Model | 3-tool | 5-tool | 8-tool | 14-tool | Trend |
|-------|--------|--------|--------|---------|-------|
| **Specialist** | **98.9%** | **99.3%** | **99.2%** | **99.5%** | Flat |
| GPT-OSS 120B | 33.3% | 25.3% | 27.5% | 41.7% | Noisy |
| FunctionGemma | 20.0% | 17.3% | 11.2% | 13.6% | Degrading |
| Raw Gemma 3 | 13.3% | 12.7% | 10.4% | 11.4% | Degrading |

### Args Accuracy

| Model | 3-tool | 5-tool | 8-tool | 14-tool |
|-------|--------|--------|--------|---------|
| **Specialist** | **98.9%** | **99.3%** | **99.6%** | **99.8%** |
| GPT-OSS 120B | 33.3% | 26.0% | 27.9% | 44.3% |
| FunctionGemma | 22.2% | 18.7% | 13.8% | 27.9% |
| Raw Gemma 3 | 17.8% | 16.0% | 12.9% | 15.7% |

### Avg Prompt Tokens

| Model | 3-tool | 5-tool | 8-tool | 14-tool |
|-------|--------|--------|--------|---------|
| **Specialist** | **18** | **20** | **21** | **20** |
| GPT-OSS 120B | 244 | 330 | 479 | 647 |
| FunctionGemma | 262 | 413 | 681 | 967 |
| Raw Gemma 3 | 265 | 419 | 708 | 1,092 |

The specialist prompt token count stays constant (~20) regardless of tool count because schemas are baked into weights. Baselines grow linearly with tool count — schemas must be passed in the prompt.

---

## Per-Tool Breakdown: Specialist

### 3-Tool Subset (read operations)

| Tool | n | Tool Routing | Args Match | Combined |
|------|---|-------------|------------|----------|
| list_directory | 30 | 100.0% | 96.7% | 96.7% |
| read_file | 30 | 100.0% | 100.0% | 100.0% |
| search_files | 30 | 100.0% | 100.0% | 100.0% |
| **Overall** | **90** | **100.0%** | **98.9%** | **98.9%** |

### 5-Tool Subset (read + write operations)

| Tool | n | Tool Routing | Args Match | Combined |
|------|---|-------------|------------|----------|
| list_directory | 30 | 100.0% | 96.7% | 96.7% |
| read_file | 30 | 100.0% | 100.0% | 100.0% |
| search_files | 30 | 100.0% | 100.0% | 100.0% |
| write_file | 30 | 100.0% | 100.0% | 100.0% |
| create_directory | 30 | 100.0% | 100.0% | 100.0% |
| **Overall** | **150** | **100.0%** | **99.3%** | **99.3%** |

### 8-Tool Subset (all filesystem)

| Tool | n | Tool Routing | Args Match | Combined |
|------|---|-------------|------------|----------|
| list_directory | 30 | 100.0% | 96.7% | 96.7% |
| read_file | 30 | 100.0% | 100.0% | 100.0% |
| search_files | 30 | 100.0% | 100.0% | 100.0% |
| write_file | 30 | 100.0% | 100.0% | 100.0% |
| create_directory | 30 | 100.0% | 100.0% | 100.0% |
| edit_file | 30 | 100.0% | 100.0% | 100.0% |
| move_file | 30 | 100.0% | 100.0% | 100.0% |
| directory_tree | 30 | 96.7% | 100.0% | 96.7% |
| **Overall** | **240** | **99.6%** | **99.6%** | **99.2%** |

### 14-Tool Subset (filesystem + git)

| Tool | n | Tool Routing | Args Match | Combined |
|------|---|-------------|------------|----------|
| list_directory | 30 | 100.0% | 96.7% | 96.7% |
| read_file | 30 | 100.0% | 100.0% | 100.0% |
| search_files | 30 | 100.0% | 100.0% | 100.0% |
| write_file | 30 | 100.0% | 100.0% | 100.0% |
| create_directory | 30 | 100.0% | 100.0% | 100.0% |
| edit_file | 30 | 100.0% | 100.0% | 100.0% |
| move_file | 30 | 100.0% | 100.0% | 100.0% |
| directory_tree | 30 | 96.7% | 100.0% | 96.7% |
| git_status | 30 | 100.0% | 100.0% | 100.0% |
| git_diff_staged | 30 | 100.0% | 100.0% | 100.0% |
| git_commit | 30 | 100.0% | 100.0% | 100.0% |
| git_log | 30 | 100.0% | 100.0% | 100.0% |
| git_branch | 30 | 100.0% | 100.0% | 100.0% |
| git_create_branch | 30 | 100.0% | 100.0% | 100.0% |
| **Overall** | **420** | **99.8%** | **99.8%** | **99.5%** |

---

## Per-Tool Breakdown: GPT-OSS 120B

### 3-Tool Subset

| Tool | n | Tool Routing | Args Match | Combined |
|------|---|-------------|------------|----------|
| list_directory | 30 | 100.0% | 16.7% | 16.7% |
| read_file | 30 | 76.7% | 76.7% | 76.7% |
| search_files | 30 | 93.3% | 6.7% | 6.7% |
| **Overall** | **90** | **90.0%** | **33.3%** | **33.3%** |

### 5-Tool Subset

| Tool | n | Tool Routing | Args Match | Combined |
|------|---|-------------|------------|----------|
| list_directory | 30 | 100.0% | 23.3% | 23.3% |
| read_file | 30 | 80.0% | 80.0% | 80.0% |
| search_files | 30 | 96.7% | 3.3% | 3.3% |
| write_file | 30 | 16.7% | 16.7% | 16.7% |
| create_directory | 30 | 83.3% | 6.7% | 3.3% |
| **Overall** | **150** | **75.3%** | **26.0%** | **25.3%** |

### 8-Tool Subset

| Tool | n | Tool Routing | Args Match | Combined |
|------|---|-------------|------------|----------|
| list_directory | 30 | 100.0% | 13.3% | 13.3% |
| read_file | 30 | 86.7% | 86.7% | 86.7% |
| search_files | 30 | 93.3% | 3.3% | 3.3% |
| write_file | 30 | 33.3% | 33.3% | 33.3% |
| create_directory | 30 | 73.3% | 3.3% | 0.0% |
| edit_file | 30 | 50.0% | 50.0% | 50.0% |
| move_file | 30 | 33.3% | 26.7% | 26.7% |
| directory_tree | 30 | 100.0% | 6.7% | 6.7% |
| **Overall** | **240** | **71.2%** | **27.9%** | **27.5%** |

### 14-Tool Subset

| Tool | n | Tool Routing | Args Match | Combined |
|------|---|-------------|------------|----------|
| list_directory | 30 | 100.0% | 13.3% | 13.3% |
| read_file | 30 | 86.7% | 86.7% | 86.7% |
| search_files | 30 | 90.0% | 0.0% | 0.0% |
| write_file | 30 | 33.3% | 33.3% | 33.3% |
| create_directory | 30 | 76.7% | 0.0% | 0.0% |
| edit_file | 30 | 56.7% | 56.7% | 56.7% |
| move_file | 30 | 26.7% | 20.0% | 20.0% |
| directory_tree | 30 | 96.7% | 3.3% | 3.3% |
| git_status | 30 | 100.0% | 100.0% | 100.0% |
| git_diff_staged | 30 | 70.0% | 86.7% | 56.7% |
| git_commit | 30 | 16.7% | 16.7% | 16.7% |
| git_log | 30 | 93.3% | 50.0% | 50.0% |
| git_branch | 30 | 93.3% | 76.7% | 70.0% |
| git_create_branch | 30 | 76.7% | 76.7% | 76.7% |
| **Overall** | **420** | **72.6%** | **44.3%** | **41.7%** |

---

## Per-Tool Breakdown: Raw Gemma 3 270M

### 3-Tool Subset

| Tool | n | Tool Routing | Args Match | Combined |
|------|---|-------------|------------|----------|
| list_directory | 30 | 46.7% | 6.7% | 6.7% |
| read_file | 30 | 46.7% | 43.3% | 30.0% |
| search_files | 30 | 90.0% | 3.3% | 3.3% |
| **Overall** | **90** | **61.1%** | **17.8%** | **13.3%** |

### 5-Tool Subset

| Tool | n | Tool Routing | Args Match | Combined |
|------|---|-------------|------------|----------|
| list_directory | 30 | 93.3% | 6.7% | 6.7% |
| read_file | 30 | 43.3% | 46.7% | 33.3% |
| search_files | 30 | 23.3% | 10.0% | 6.7% |
| write_file | 30 | 43.3% | 13.3% | 13.3% |
| create_directory | 30 | 60.0% | 3.3% | 3.3% |
| **Overall** | **150** | **52.7%** | **16.0%** | **12.7%** |

### 8-Tool Subset

| Tool | n | Tool Routing | Args Match | Combined |
|------|---|-------------|------------|----------|
| list_directory | 30 | 86.7% | 6.7% | 6.7% |
| read_file | 30 | 43.3% | 16.7% | 10.0% |
| search_files | 30 | 26.7% | 13.3% | 10.0% |
| write_file | 30 | 30.0% | 10.0% | 10.0% |
| create_directory | 30 | 40.0% | 3.3% | 3.3% |
| edit_file | 30 | 10.0% | 0.0% | 0.0% |
| move_file | 30 | 66.7% | 50.0% | 43.3% |
| directory_tree | 30 | 83.3% | 3.3% | 0.0% |
| **Overall** | **240** | **48.3%** | **12.9%** | **10.4%** |

### 14-Tool Subset

| Tool | n | Tool Routing | Args Match | Combined |
|------|---|-------------|------------|----------|
| list_directory | 30 | 93.3% | 0.0% | 0.0% |
| read_file | 30 | 43.3% | 6.7% | 6.7% |
| search_files | 30 | 3.3% | 3.3% | 0.0% |
| write_file | 30 | 36.7% | 16.7% | 16.7% |
| create_directory | 30 | 26.7% | 0.0% | 0.0% |
| edit_file | 30 | 0.0% | 0.0% | 0.0% |
| move_file | 30 | 63.3% | 30.0% | 30.0% |
| directory_tree | 30 | 30.0% | 0.0% | 0.0% |
| git_status | 30 | 16.7% | 30.0% | 16.7% |
| git_diff_staged | 30 | 40.0% | 46.7% | 40.0% |
| git_commit | 30 | 13.3% | 3.3% | 3.3% |
| git_log | 30 | 10.0% | 30.0% | 6.7% |
| git_branch | 30 | 43.3% | 53.3% | 40.0% |
| git_create_branch | 30 | 46.7% | 0.0% | 0.0% |
| **Overall** | **420** | **33.3%** | **15.7%** | **11.4%** |

---

## Per-Tool Breakdown: FunctionGemma 270M

### 3-Tool Subset

| Tool | n | Tool Routing | Args Match | Combined |
|------|---|-------------|------------|----------|
| list_directory | 30 | 40.0% | 40.0% | 33.3% |
| read_file | 30 | 23.3% | 16.7% | 16.7% |
| search_files | 30 | 26.7% | 10.0% | 10.0% |
| **Overall** | **90** | **30.0%** | **22.2%** | **20.0%** |

### 5-Tool Subset

| Tool | n | Tool Routing | Args Match | Combined |
|------|---|-------------|------------|----------|
| list_directory | 30 | 40.0% | 36.7% | 30.0% |
| read_file | 30 | 20.0% | 16.7% | 16.7% |
| search_files | 30 | 23.3% | 6.7% | 6.7% |
| write_file | 30 | 33.3% | 20.0% | 20.0% |
| create_directory | 30 | 20.0% | 13.3% | 13.3% |
| **Overall** | **150** | **27.3%** | **18.7%** | **17.3%** |

### 8-Tool Subset

| Tool | n | Tool Routing | Args Match | Combined |
|------|---|-------------|------------|----------|
| list_directory | 30 | 26.7% | 26.7% | 10.0% |
| read_file | 30 | 23.3% | 16.7% | 16.7% |
| search_files | 30 | 60.0% | 23.3% | 23.3% |
| write_file | 30 | 16.7% | 6.7% | 6.7% |
| create_directory | 30 | 26.7% | 10.0% | 6.7% |
| edit_file | 30 | 30.0% | 10.0% | 10.0% |
| move_file | 30 | 3.3% | 0.0% | 0.0% |
| directory_tree | 30 | 53.3% | 16.7% | 16.7% |
| **Overall** | **240** | **30.0%** | **13.8%** | **11.2%** |

### 14-Tool Subset

| Tool | n | Tool Routing | Args Match | Combined |
|------|---|-------------|------------|----------|
| list_directory | 30 | 13.3% | 20.0% | 3.3% |
| read_file | 30 | 23.3% | 13.3% | 13.3% |
| search_files | 30 | 53.3% | 13.3% | 13.3% |
| write_file | 30 | 3.3% | 0.0% | 0.0% |
| create_directory | 30 | 6.7% | 6.7% | 3.3% |
| edit_file | 30 | 20.0% | 6.7% | 6.7% |
| move_file | 30 | 0.0% | 0.0% | 0.0% |
| directory_tree | 30 | 23.3% | 16.7% | 13.3% |
| git_status | 30 | 6.7% | 73.3% | 6.7% |
| git_diff_staged | 30 | 40.0% | 96.7% | 40.0% |
| git_commit | 30 | 20.0% | 0.0% | 0.0% |
| git_log | 30 | 6.7% | 43.3% | 6.7% |
| git_branch | 30 | 83.3% | 100.0% | 83.3% |
| git_create_branch | 30 | 0.0% | 0.0% | 0.0% |
| **Overall** | **420** | **21.4%** | **27.9%** | **13.6%** |

---

## Per-Category Breakdown: Specialist

| Category | 3-tool | 5-tool | 8-tool | 14-tool |
|----------|--------|--------|--------|---------|
| Clean (tool routing) | 100.0% | 100.0% | 100.0% | 100.0% |
| Clean (combined) | 98.4% | 99.0% | 99.4% | 99.7% |
| Messy (tool routing) | 100.0% | 100.0% | 100.0% | 100.0% |
| Messy (combined) | 100.0% | 100.0% | 100.0% | 100.0% |
| Disambiguation (tool routing) | 100.0% | 100.0% | 97.6% | 98.6% |
| Disambiguation (combined) | 100.0% | 100.0% | 97.6% | 98.6% |

Messy queries (typos, slang, abbreviations) are at 100% combined across all subsets. Disambiguation queries (semantically overlapping tools) show the only routing errors at 8+ tools.

---

## Latency and Efficiency

| Model | 3-tool | 5-tool | 8-tool | 14-tool |
|-------|--------|--------|--------|---------|
| **Specialist** avg | 149ms | 160ms | 164ms | 153ms |
| **Specialist** median | 147ms | 151ms | 151ms | 142ms |
| **Specialist** p95 | 181ms | 258ms | 272ms | 266ms |
| Raw Gemma avg | 488ms | 564ms | 670ms | 905ms |
| FunctionGemma avg | 180ms | 203ms | 249ms | 235ms |
| GPT-OSS avg | 1,014ms | 1,246ms | 1,416ms | 1,111ms |
| GPT-OSS median | 815ms | 745ms | 842ms | 842ms |
| GPT-OSS p95 | 2,515ms | 2,484ms | 3,060ms | 2,767ms |

Specialist latency is constant (~150ms median) regardless of tool count. Raw Gemma latency nearly doubles from 3 to 14 tools due to longer few-shot prompts.

---

## Failure Analysis: Specialist

Only 2 unique failures across all 420 queries at 14-tool scale:

### Failure 1: Tokenization Stutter (list_directory)
- **Query**: "what's in retrieval/indexes/"
- **Expected**: `{"tool":"list_directory","args":{"path":"retrieval/indexes/"}}`
- **Got**: `{"tool":"list_directory","args":{"path":"ret retrieval/indexes/"}}`
- **Category**: clean
- **Appears in**: all 4 subsets (same seeded sample)
- **Root cause**: Tokenization artifact — the model emits a partial token `ret` before the full path. The word "retrieval" likely spans a subword boundary that occasionally misfires. Tool routing is correct; only the args have a stutter prefix.

### Failure 2: Disambiguation — directory_tree vs list_directory
- **Query**: "explore the entire contents of data/"
- **Expected**: `{"tool":"directory_tree","args":{"path":"data/"}}`
- **Got**: `{"tool":"list_directory","args":{"path":"data/"}}`
- **Category**: disambiguation
- **Appears in**: 8-tool and 14-tool subsets (directory_tree not in 3/5-tool)
- **Root cause**: "explore the entire contents" is semantically ambiguous — both list_directory and directory_tree are plausible. The model favors the more common tool. This is a genuine disambiguation edge case, not a capability failure.

Both failures are edge cases with correct intent. Zero JSON parse errors, zero MCP failures, zero refusals across all runs.

---

## Baseline Failure Patterns

### GPT-OSS 120B
- **Trailing slash on paths**: Adds trailing `/` to file paths (e.g., `config.yaml/`) — format mismatch, not intelligence failure
- **search_files args**: Merges path and pattern into a single glob (e.g., `src/*.py` instead of `path:"src/", pattern:"*.py"`)
- **write_file routing**: Often routes to `create_file` or returns text content instead of tool call
- **git_commit routing**: Frequently returns advisory text ("you should commit") instead of a tool call
- **Strength**: git_status (100%), read_file (86.7%), git_create_branch (76.7%) — simple tools with obvious intent

### Raw Gemma 3 270M
- **Tool routing collapse**: 33.3% at 14 tools — random guessing territory
- **edit_file**: 0% routing — never correctly identifies edit intent
- **JSON errors**: 3 parse failures (garbled output)
- **Prompt length**: 1,092 avg tokens at 14 tools — the few-shot prompt overwhelms the 270M context
- **Strength**: move_file (63.3%) and list_directory (93.3%) — few-shot examples closely match queries

### FunctionGemma 270M
- **Designed for tools API but can't use it**: Despite receiving proper tool schemas, routes correctly only 21.4% at 14 tools
- **git_branch anomaly**: 83.3% tool routing — likely matches a trained pattern from its original FunctionGemma training
- **git_diff_staged args anomaly**: 96.7% args accuracy despite 40% tool routing — when it doesn't pick the right tool, it often still produces empty args `{}` that happen to match
- **move_file**: 0% across all metrics at 14 tools
- **No refusals**: FunctionGemma doesn't refuse these queries (unlike earlier 3-tool benchmarks where 85/360 were refusals) — it just routes incorrectly

---

## Key Findings

1. **No scaling wall found.** The 270M specialist maintains 99.5% combined accuracy from 3 to 14 tools. The breaking point — if it exists — is beyond 14 tools.

2. **Data quality matters more than tool count.** An earlier dataset with invented paths (non-extractable args) achieved only 48.6% combined despite 95.3% tool routing. Switching to extraction-only rules produced 99.5% combined — same model architecture, better data.

3. **Token efficiency is the killer feature.** The specialist uses 18-21 tokens/request regardless of tool count. At 14 tools, baselines need 647-1,092 tokens — a 32x to 55x overhead just for tool schemas.

4. **GPT-OSS 120B (444x larger) achieves 41.7% combined** — the specialist outperforms it by 58 percentage points. GPT-OSS has decent tool routing (72.6%) but fails on args formatting. Its strength is git tools with simple/no args.

5. **Baseline models degrade with tool count; specialist doesn't.** Raw Gemma drops from 61.1% to 33.3% tool routing (3→14). The specialist stays flat at ~100%. This validates the core thesis: baking tools into weights eliminates the prompt-length tax.

6. **Git tools are trivially easy for the specialist** — all 6 at 100% combined. Most git tools have zero or one arg, and the semantic space (commit, branch, diff, status, log) is well-separated.

7. **Messy queries are not the problem.** 100% combined across all subsets for messy category. The model handles typos, slang, and abbreviations perfectly. The only failures are in disambiguation (semantically overlapping tools).

---

## Result Files

| File | Description |
|------|-------------|
| `results/benchmark_v03_3tool.json` | 3-tool summary metrics |
| `results/benchmark_v03_3tool_detailed.json` | 3-tool per-example results |
| `results/benchmark_v03_5tool.json` | 5-tool summary metrics |
| `results/benchmark_v03_5tool_detailed.json` | 5-tool per-example results |
| `results/benchmark_v03_8tool.json` | 8-tool summary metrics |
| `results/benchmark_v03_8tool_detailed.json` | 8-tool per-example results |
| `results/benchmark_v03_14tool.json` | 14-tool summary metrics |
| `results/benchmark_v03_14tool_detailed.json` | 14-tool per-example results |
