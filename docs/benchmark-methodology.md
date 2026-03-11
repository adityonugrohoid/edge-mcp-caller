# Benchmark Methodology

How we tested 4 models across scaling subsets (3/5/8/14 tools), and why the results are trustworthy.

---

## Table of Contents

1. [Eval Dataset](#1-eval-dataset)
2. [Models Tested](#2-models-tested)
3. [How Each Model Receives Tool Definitions](#3-how-each-model-receives-tool-definitions)
4. [Scoring Method](#4-scoring-method)
5. [Scaling Benchmark Design](#5-scaling-benchmark-design)
6. [Known Limitations & Caveats](#6-known-limitations--caveats) (incl. Synthetic Eval Data, Why Not BFCL)
7. [Fairness Analysis](#7-fairness-analysis)

---

## 1. Eval Dataset

- **Source**: `data/eval.jsonl` — 1,568 held-out examples (10% of 15,601 total)
- **Distribution**: 112 per tool × 14 tools, stratified (76 clean + 18 messy + 18 disambiguation per tool)
- **Generation**: Claude Code agents (Sonnet 4.6), extraction-only rules per `docs/generation-standard.md`
- **Format**: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "{\"tool\":\"...\",\"args\":{...}}"}], "category": "clean|messy|disambiguation"}`
- **Not cherry-picked**: Same eval set used for all models, no filtering

### Example

```
User: "in config.json replace 'DEBUG=false' with 'DEBUG=true'"
Expected: {"tool": "edit_file", "args": {"path": "config.json", "old_text": "DEBUG=false", "new_text": "DEBUG=true"}}
```

---

## 2. Models Tested

| Model | Size | Source | Interface | Purpose |
|---|---|---|---|---|
| **edge-mcp-caller** | 270M (Q8_0, 291 MB) | LoRA fine-tune of `unsloth/gemma-3-270m-it` | Bare user query, no schema | Specialist under test |
| **gemma3:270m** | 270M (291 MB) | Ollama registry | Few-shot system prompt + `format: json` | Unfine-tuned baseline |
| **functiongemma:270m** | 270M (300 MB) | Ollama registry (Google) | Ollama native tools API | Google's tool-calling specialist |
| **openai/gpt-oss-120b** | 120B | NVIDIA NIM API | OpenAI tools API | Ceiling reference (440x larger) |

All runs use `temperature: 0` for deterministic, reproducible results.

**Why GPT-OSS 120B?** It was selected as the largest freely accessible model with native OpenAI-compatible tool calling (via NVIDIA NIM, no API key cost). The comparison is not "specialist vs best function caller" — it is "specialist vs what a developer would actually deploy at scale." GPT-4o would score higher but costs per-request and requires cloud; the point is that even a 120B open model with proper tool calling API cannot match a 270M specialist on fixed tools.

---

## 3. How Each Model Receives Tool Definitions

Each model is tested using **its intended/designed interface** — we do not force a single format on all models.

### 3.1 Specialist (edge-mcp-caller) — No Schema

The specialist receives **only the user query**. No tool names, descriptions, or schemas are passed at inference time. Tool knowledge is baked into model weights during fine-tuning.

```json
{
  "model": "edge-mcp-caller:latest",
  "stream": false,
  "options": {"temperature": 0},
  "messages": [
    {"role": "user", "content": "in config.json replace 'DEBUG=false' with 'DEBUG=true'"}
  ]
}
```

**Prompt tokens**: ~18-21 (constant regardless of tool count)

### 3.2 Raw Gemma 3 270M — Few-Shot System Prompt

Raw Gemma 3 270M does not support Ollama's tools API. We use a few-shot system prompt with 2 examples per tool (scoped to the current subset) plus `format: "json"`.

```json
{
  "model": "gemma3:270m",
  "stream": false,
  "format": "json",
  "options": {"temperature": 0},
  "messages": [
    {"role": "system", "content": "You are a tool router... [2 examples per tool]"},
    {"role": "user", "content": "..."}
  ]
}
```

**Prompt tokens**: 265 (3-tool) → 1,092 (14-tool) — scales with tool count

### 3.3 FunctionGemma 270M — Ollama Native Tools API

FunctionGemma receives tool definitions using the standard Ollama tools API — the same interface any developer would use. Tool schemas are loaded from `tools/filesystem.json` and `tools/git.json`, filtered to the current subset.

```json
{
  "model": "functiongemma:270m",
  "stream": false,
  "options": {"temperature": 0},
  "messages": [{"role": "user", "content": "..."}],
  "tools": [... tool schemas for current subset ...]
}
```

**Prompt tokens**: 262 (3-tool) → 967 (14-tool) — scales with tool count

### 3.4 GPT-OSS 120B — OpenAI Tools API via NVIDIA NIM

GPT-OSS receives **identical tool definitions** as FunctionGemma in the OpenAI tools format.

```json
POST https://integrate.api.nvidia.com/v1/chat/completions
{
  "model": "openai/gpt-oss-120b",
  "temperature": 0,
  "messages": [{"role": "user", "content": "..."}],
  "tools": [... same schemas as FunctionGemma ...]
}
```

**Prompt tokens**: 244 (3-tool) → 647 (14-tool)
**Rate limiting**: 40 requests/minute (NIM free tier)

---

## 4. Scoring Method

### Strict Exact Match (primary metric)

Each prediction is scored on three metrics:

- **Tool accuracy**: `predicted_tool == expected_tool`
- **Argument accuracy**: `predicted_args == expected_args` (exact dictionary match)
- **Combined accuracy**: both tool AND args correct

No normalization is applied. `packages` ≠ `packages/`, and `path="", pattern="src/**/*.js"` ≠ `path="src/", pattern="*.js"`. A fuzzy matcher would improve baseline scores but would also require a normalization layer in production — adding complexity that the specialist eliminates entirely.

### Category Breakdown

Every eval example is tagged with a category:
- **Clean** (70%): Well-formed queries with explicit paths/args
- **Messy** (15%): Typos, slang, abbreviations, voice transcription — args still extractable
- **Disambiguation** (15%): Queries where multiple tools could apply — tests routing intelligence

---

## 5. Scaling Benchmark Design

The specialist is trained on **all 14 tools simultaneously**. We benchmark at 4 subset sizes to measure whether accuracy degrades as tool count increases.

| Subset | Tools | What it tests |
|--------|-------|---------------|
| 3-tool | list_directory, read_file, search_files | Baseline read operations |
| 5-tool | + write_file, create_directory | Read/write disambiguation |
| 8-tool | + edit_file, move_file, directory_tree | Similar-tool disambiguation |
| 14-tool | + 6 git tools | Cross-server routing |

**Sampling**: 30 examples per tool per subset, seed=42, stratified by category. Each subset uses the same seeded sample, so 3-tool examples are a strict subset of 5-tool examples, etc.

**All 4 models are benchmarked at each subset size.** FunctionGemma and GPT-OSS receive only the tool schemas for the current subset — they don't see tools outside the subset.

---

## 6. Known Limitations & Caveats

### 6.1 Eval Format = Deployment Contract

The eval data defines the **deployment contract** — the exact JSON format that `mcp/client.py` expects to parse into MCP `tools/call` requests. This contract specifies:
- Directories include trailing slashes (`src/`, not `src`)
- Glob patterns are simple (`*.js`, not `**/*.js`)
- `path` and `pattern` are cleanly separated
- `edit_file` uses `{path, old_text, new_text}` (bridge converts to MCP format)
- Git tools omit `repo_path` (bridge injects from config)

The specialist was trained on this contract; the other models were not.

### 6.2 GPT-OSS Failure Patterns

- **Trailing slash**: Adds `/` to file paths or drops it from directory paths — format mismatch
- **Glob merging**: Combines path and pattern into recursive glob (`src/**/*.js` instead of separate fields)
- **Text responses**: Some "how do I..." queries return advisory text instead of tool calls
- **git_commit routing**: Often returns text advice instead of a tool call

### 6.3 FunctionGemma Failure Patterns

- Routes correctly only 21.4% at 14 tools despite receiving proper tool schemas
- Some tools (move_file, write_file) at 0-3.3% routing — near random
- git_branch anomaly: 83.3% routing — likely matches a trained pattern

### 6.4 Synthetic Eval Data

Both training and eval data were generated by Claude Code agents (Sonnet 4.6) using the same extraction-only rules, then split 90/10. This means the eval set shares the same distribution as the training set — standard practice for ML train/test splits, but worth noting.

The 15% "messy" category (typos, slang, abbreviations, voice transcription) stress-tests robustness beyond clean synthetic queries. However, real-world user input may diverge further from these patterns. The eval measures deployment accuracy for the defined contract, not open-domain robustness.

### 6.5 Raw Gemma Failure Patterns

- Tool routing collapses to 33.3% at 14 tools (random territory)
- Few-shot prompt grows to 1,092 tokens at 14 tools — overwhelms the 270M context
- edit_file: 0% routing — never correctly identifies edit intent
- Occasional JSON parse failures (3 errors across runs)

### 6.6 Why Not BFCL?

BFCL (Berkeley Function Calling Leaderboard) tests generalist function calling — models receive arbitrary, previously unseen tool schemas. The specialist cannot run on BFCL by design: it knows only 14 tools baked into its weights.

This is not a limitation to work around — it IS the thesis. BFCL measures schema generalization. We measure deployment accuracy for a fixed tool set. These are different problems with different optimal solutions.

---

## 7. Fairness Analysis

### 7.1 Is each model tested in its intended mode?

| Model | Interface Used | Fair? |
|---|---|---|
| Specialist | Bare query (designed interface) | **Yes** |
| Raw Gemma 3 | Few-shot system prompt | **Best available** (no tools API) |
| FunctionGemma | Ollama native tools API | **Yes** |
| GPT-OSS 120B | OpenAI tools API via NIM | **Yes** |

### 7.2 Do all tool-capable models receive identical tool definitions?

**Yes.** FunctionGemma and GPT-OSS receive the exact same schemas from `tools/filesystem.json` and `tools/git.json`, filtered to the current subset.

### 7.3 Is the specialist unfairly advantaged?

The specialist has two structural advantages:

1. **Format familiarity**: Trained on data matching the eval contract (trailing slashes, glob decomposition)
2. **Zero prompt overhead**: No tokens spent on tool definitions

Both are **by design** — they ARE the specialist thesis. The benchmark measures whether baking tool knowledge into weights is more effective than passing schemas at inference time.

In production, the specialist and its deployment contract are co-designed — the model IS the contract. A generalist never adapts to your specific output format; it guesses every time. The benchmark measures exactly this difference.

### 7.4 Reproducibility

- All results at `temperature: 0` (deterministic)
- Full per-example results in `results/benchmark_v03_*_detailed.json`
- Tool schemas in `tools/filesystem.json` and `tools/git.json`
- Eval data in `data/eval.jsonl`
- Benchmark script: `eval/benchmark.py`
- Random seed: 42

---

*This document is part of the [edge-mcp-caller](https://github.com/adityonugrohoid/edge-mcp-caller) project.*
