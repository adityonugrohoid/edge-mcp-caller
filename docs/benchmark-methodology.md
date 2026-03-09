# Benchmark Methodology

How we tested 4 models on 360 MCP filesystem tool-calling examples, and why the results are trustworthy.

---

## Table of Contents

1. [Eval Dataset](#1-eval-dataset)
2. [Models Tested](#2-models-tested)
3. [How Each Model Receives Tool Definitions](#3-how-each-model-receives-tool-definitions)
4. [Scoring Method](#4-scoring-method)
5. [Known Limitations & Caveats](#5-known-limitations--caveats)
6. [Fairness Analysis](#6-fairness-analysis)
7. [Raw Results Summary](#7-raw-results-summary)

---

## 1. Eval Dataset

- **Source**: `data/eval.jsonl` — 360 held-out examples (10% of 3600 total)
- **Distribution**: 104 list_directory, 125 read_file, 131 search_files
- **Generation**: Synthetic via NVIDIA NIM API (`meta/llama-3.1-70b-instruct`), validated by 9-check quality pipeline
- **Format**: Each example is `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "{\"tool\":\"...\",\"args\":{...}}"}]}`
- **Not cherry-picked**: Same eval set used for all models, no filtering

### Example

```
User: "show me all .js files in src/components/"
Expected: {"tool": "search_files", "args": {"path": "src/components/", "pattern": "*.js"}}
```

---

## 2. Models Tested

| Model | Size | Source | Interface | Purpose |
|---|---|---|---|---|
| **edge-mcp-caller** | 270M (Q8_0, 291 MB) | Our LoRA fine-tune of `unsloth/gemma-3-270m-it` | Bare user query, no schema | Specialist under test |
| **gemma3:270m** | 270M (291 MB) | Ollama registry | Few-shot system prompt + `format: json` | Unfine-tuned baseline |
| **functiongemma:270m** | 270M (300 MB) | Ollama registry (Google) | Ollama native tools API | Google's tool-calling specialist |
| **openai/gpt-oss-120b** | 120B | NVIDIA NIM API | OpenAI tools API | Ceiling reference (440x larger) |

All runs use `temperature: 0` for deterministic, reproducible results.

---

## 3. How Each Model Receives Tool Definitions

This is the most critical section. Each model is tested using **its intended/designed interface** — we do not force a single format on all models.

### 3.1 Specialist (edge-mcp-caller) — No Schema

The specialist receives **only the user query**. No tool names, descriptions, or schemas are passed at inference time. Tool knowledge is baked into the model weights during fine-tuning.

```json
{
  "model": "edge-mcp-caller:latest",
  "stream": false,
  "options": {"temperature": 0},
  "messages": [
    {"role": "user", "content": "show me all .js files in src/components/"}
  ]
}
```

**Prompt tokens**: 13–35 (avg 20)

This is the specialist thesis: zero schema overhead, all capacity focused on intent routing.

### 3.2 Raw Gemma 3 270M — Few-Shot System Prompt

Raw Gemma 3 270M **does not support Ollama's tools API**. Sending a `tools` array returns:

```json
{"error": "registry.ollama.ai/library/gemma3:270m does not support tools"}
```

Since the model has no native tool-calling capability, we use the best available alternative: a few-shot system prompt with 6 examples (2 per tool) that teaches the output format, plus Ollama's `format: "json"` to constrain output to valid JSON.

```json
{
  "model": "gemma3:270m",
  "stream": false,
  "format": "json",
  "options": {"temperature": 0},
  "messages": [
    {"role": "system", "content": "You are a tool router... [6 few-shot examples]"},
    {"role": "user", "content": "show me all .js files in src/components/"}
  ]
}
```

**System prompt** (859 chars):
```
You are a tool router. Given a user query about filesystem operations,
output a JSON tool call. Output ONLY the JSON object, nothing else.

Available tools:
- list_directory(path): list files and directories at a path
- read_file(path): read contents of a file
- search_files(path, pattern): search for files matching a glob pattern

Examples:
User: "show files in docs/" → {"tool":"list_directory","args":{"path":"docs/"}}
User: "read config.yaml" → {"tool":"read_file","args":{"path":"config.yaml"}}
User: "find *.py in lib/" → {"tool":"search_files","args":{"path":"lib/","pattern":"*.py"}}
User: "what's in the tests folder" → {"tool":"list_directory","args":{"path":"tests/"}}
User: "show me README.md" → {"tool":"read_file","args":{"path":"README.md"}}
User: "find all .js files in src/" → {"tool":"search_files","args":{"path":"src/","pattern":"*.js"}}
```

**Prompt tokens**: 262–284 (avg 269)

**Justification**: This is the fairest treatment for a model without native tool support. The few-shot prompt teaches the exact output format (`{"tool": ..., "args": {...}}`), provides balanced examples (2 per tool), and uses `format: "json"` to prevent free-text responses. Without this, the model would respond conversationally (e.g., "The `src` directory contains the source code of your program.").

**Why not more few-shot examples?** More examples would consume more prompt tokens and arguably give the raw model an unfair advantage — it would be closer to in-context learning than zero-shot tool calling. 6 examples (2 per tool) is a balanced middle ground.

### 3.3 FunctionGemma 270M — Ollama Native Tools API

FunctionGemma is Google's purpose-built function-calling model. Ollama has a dedicated `RENDERER functiongemma` and `PARSER functiongemma` that handle the tool-calling format natively.

We send tool definitions using **the standard Ollama tools API** — the same interface any developer would use:

```json
{
  "model": "functiongemma:270m",
  "stream": false,
  "options": {"temperature": 0},
  "messages": [
    {"role": "user", "content": "show me all .js files in src/components/"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "list_directory",
        "description": "Get a detailed listing of all files and directories in a specified path. Results distinguish between files and directories with [FILE] and [DIR] prefixes.",
        "parameters": {
          "type": "object",
          "properties": {
            "path": {"type": "string", "description": "Path to the directory to list"}
          },
          "required": ["path"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read the complete contents of a file from the file system. Handles various text encodings.",
        "parameters": {
          "type": "object",
          "properties": {
            "path": {"type": "string", "description": "Path to the file to read"}
          },
          "required": ["path"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "search_files",
        "description": "Recursively search for files and directories matching a pattern.",
        "parameters": {
          "type": "object",
          "properties": {
            "path": {"type": "string", "description": "Starting path for the search"},
            "pattern": {"type": "string", "description": "Glob pattern to match against file/directory names"}
          },
          "required": ["path", "pattern"]
        }
      }
    }
  ]
}
```

**Prompt tokens**: 257–279 (avg 264)

FunctionGemma returns structured `tool_calls` in its response (not raw text), which we parse directly:

```json
{
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [{"function": {"name": "list_directory", "arguments": {"path": "src/components/"}}}]
  }
}
```

### 3.4 GPT-OSS 120B — OpenAI Tools API via NVIDIA NIM

GPT-OSS 120B is accessed via the NVIDIA NIM API using the **standard OpenAI chat completions format** with tools. It receives the **identical tool definitions** as FunctionGemma, in the same JSON schema format.

```json
POST https://integrate.api.nvidia.com/v1/chat/completions
Authorization: Bearer $NVIDIA_API_KEY

{
  "model": "openai/gpt-oss-120b",
  "temperature": 0,
  "messages": [
    {"role": "user", "content": "show me all .js files in src/components/"}
  ],
  "tools": [same 3 tool definitions as FunctionGemma]
}
```

**Prompt tokens**: 240–258 (avg 246)

GPT-OSS returns `tool_calls` in the standard OpenAI format (arguments as JSON string):

```json
{
  "choices": [{
    "message": {
      "tool_calls": [{"function": {"name": "search_files", "arguments": "{\"path\":\"\",\"pattern\":\"src/components/**/*.js\"}"}}]
    }
  }]
}
```

**Rate limiting**: 40 requests/minute (NIM free tier). Total benchmark time: ~9 minutes for 360 examples.

---

## 4. Scoring Method

### 4.1 Strict Exact Match (Current)

Each prediction is scored on three metrics:

- **Tool accuracy**: `predicted_tool == expected_tool`
- **Argument accuracy**: `predicted_args == expected_args` (exact dictionary match)
- **Combined accuracy**: both tool AND args correct

No normalization is applied. `packages` ≠ `packages/`, and `path="", pattern="src/**/*.js"` ≠ `path="src/", pattern="*.js"`.

### 4.2 Impact of Trailing Slash Normalization

Trailing slashes on directory paths are functionally equivalent at the filesystem level (`packages` and `packages/` resolve to the same directory). If we normalize by stripping trailing slashes from `path` values before comparison:

| Model | Strict Combined | Normalized Combined | Difference |
|---|---|---|---|
| Specialist | 90.8% | 93.3% | +9 examples |
| Raw Gemma 3 | 13.3% | 14.2% | +3 examples |
| FunctionGemma | 18.1% | 23.6% | +20 examples |
| GPT-OSS 120B | 23.3% | 49.4% | +94 examples |

GPT-OSS benefits most because it consistently drops trailing slashes. FunctionGemma also gains significantly. Our specialist also benefits (+9), proving the slash penalty affects all models.

**Decision**: We report strict exact match as the primary metric. The normalized scores are reported alongside for transparency. Readers can choose which metric better reflects their use case.

---

## 5. Known Limitations & Caveats

### 5.1 Eval Format = Deployment Contract

The eval data defines the **deployment contract** — the exact JSON format that the downstream MCP client wrapper (`mcp/client.py`) expects to receive and parse into valid MCP `tools/call` requests. This contract specifies:
- Paths include trailing slashes for directories (`src/`, not `src`)
- Glob patterns are simple (`*.js`, `test*`), not recursive (`**/*.js`)
- `path` and `pattern` are cleanly separated (path holds the directory, pattern holds the glob)

All models are measured against this same contract. Our specialist was trained on it; the other models were not. When GPT-OSS outputs `path=""` with `pattern="src/**/*.js"`, that's not just a formatting preference — it's a different MCP call that the client wrapper would need to handle specially, or it would fail at the MCP server.

This is the core thesis: a specialist learns the exact conventions of its deployment context and produces protocol-compliant output every time. Generalists apply their own formatting preferences, which may or may not be compatible with the downstream system.

### 5.2 GPT-OSS Glob Path Merging

In 76 of 360 examples, GPT-OSS merges the search path into the glob pattern:

```
Expected: path="src/components/", pattern="*.js"
GPT-OSS:  path="",              pattern="src/components/**/*.js"
```

This is a valid alternative interpretation — many search implementations accept recursive globs. But it produces a different MCP `tools/call` request than intended, so it's scored as incorrect under strict matching.

### 5.3 GPT-OSS Text Responses

In 7 of 360 examples, GPT-OSS returns a text response instead of a `tool_call`. These queries use "how do I" phrasing that sounds like asking for help rather than requesting a tool call:

- "How do I find all the go files in the cmd directory?"
- "how do I find all .rs files in src?"
- "Can you show me the /etc/passwd file?"

The model interprets these as questions about methodology rather than tool-call requests. This is a reasonable interpretation, though our eval expects a tool call. Our specialist handles all of these correctly because it was trained to always emit a tool call.

### 5.4 FunctionGemma Refusals

In 85 of 360 examples (23.6%), FunctionGemma refuses to act:

- "I apologize, but I cannot assist with retrieving or listing file or directory contents."
- "Could you please specify the exact directory path you would like to list?"

This is a genuine limitation of the model at 270M scale — it doesn't have enough capacity to confidently route all query phrasings. These are scored as failures (no tool selected = wrong tool).

### 5.5 Ambiguous Eval Examples

A small number of eval examples have debatable ground truth:

- "what's in /home/user/projects/src" — expected `read_file`, but `list_directory` is arguably correct (no file extension, looks like a directory)
- "I'm debugging a test, show me test files in tests/" — expected `list_directory`, but `search_files` for test files is reasonable
- "what's the deal with tests?" — expected `search_files`, but this is vague enough that `list_directory` is defensible

All models (including ours) get some of these wrong, and all are scored consistently against the same expected answers.

---

## 6. Fairness Analysis

### 6.1 Is each model tested in its intended mode?

| Model | Interface Used | Is This Fair? |
|---|---|---|
| Specialist | Bare query (designed interface) | **Yes** — this is its purpose |
| Raw Gemma 3 | Few-shot system prompt | **Best available** — model lacks tools API |
| FunctionGemma | Ollama native tools API | **Yes** — its designed interface |
| GPT-OSS 120B | OpenAI tools API via NIM | **Yes** — its designed interface |

### 6.2 Do all tool-capable models receive identical tool definitions?

**Yes.** FunctionGemma and GPT-OSS receive the **exact same tool definitions** from `tools/filesystem.json`, converted to the OpenAI tools format. The JSON schema, descriptions, parameter names, and required fields are identical.

### 6.3 Is the specialist unfairly advantaged?

The specialist has two structural advantages:

1. **Format familiarity**: It was trained on data from the same generator, so it knows the exact output conventions (trailing slashes, glob decomposition). Other models apply their own conventions.

2. **Zero prompt overhead**: No tokens are spent on tool definitions, so all model capacity is focused on intent routing.

Both advantages are **by design** — they are the specialist thesis. The benchmark measures whether baking tool knowledge into weights is more effective than passing schemas at inference time. The answer is yes, even accounting for format bias.

### 6.4 What about Raw Gemma's few-shot prompt?

Raw Gemma cannot use the tools API (`"does not support tools"` error). The alternatives were:

1. **Skip it entirely** — lose the "untrained baseline" data point
2. **Use a zero-shot prompt** (no examples) — even worse performance, unfairly harsh
3. **Use a few-shot prompt** (our approach) — teaches output format with balanced examples

We chose option 3 with 6 examples (2 per tool). This gives Raw Gemma a fair chance to understand the task while keeping the prompt reasonable. It's the same approach a developer would use to get tool-calling behavior from a non-tool-calling model.

The few-shot prompt also uses `format: "json"` to constrain output to valid JSON. Without this, Raw Gemma produces conversational text ("The src directory contains the source code...") which would score 0% — unfair to the model.

### 6.5 Are there false failures caused by prompt formatting?

| Failure Type | Cause | Models Affected | Fair? |
|---|---|---|---|
| Trailing slash (`packages` vs `packages/`) | Convention mismatch | All models (GPT-OSS most) | **No** — should normalize |
| Glob merging (`path=""` + recursive pattern) | Alternative decomposition | GPT-OSS | **Debatable** — valid but different MCP call |
| Tool refusals | Model uncertainty | FunctionGemma (85 cases) | **Yes** — genuine failure |
| Text responses instead of tool calls | Query interpretation | GPT-OSS (7 cases) | **Yes** — should call tools |
| Invalid JSON / hallucination | Model capacity limit | Raw Gemma (3 cases) | **Yes** — genuine failure |
| Pattern format (`test*` vs `test_*` vs `*test*`) | Ambiguous intent | All models | **Yes** — but eval truth is debatable |

---

## 7. Raw Results Summary

### Strict Exact Match (primary metric)

| Model | Tool Acc | Args Acc | Combined | Avg Prompt Tokens | Avg Latency |
|---|---|---|---|---|---|
| **Specialist (270M)** | **99.2%** | **90.8%** | **90.8%** | **20** | **142ms** |
| GPT-OSS 120B | 76.4% | 23.6% | 23.3% | 246 | 817ms |
| FunctionGemma 270M | 38.1% | 20.3% | 18.1% | 264 | 146ms |
| Raw Gemma 3 270M | 42.2% | 25.3% | 13.3% | 269 | 452ms |

### With Trailing Slash Normalization

| Model | Strict Combined | Normalized Combined | Change |
|---|---|---|---|
| **Specialist (270M)** | **90.8%** | **93.3%** | +2.5% |
| GPT-OSS 120B | 23.3% | 49.4% | +26.1% |
| FunctionGemma 270M | 18.1% | 23.6% | +5.5% |
| Raw Gemma 3 270M | 13.3% | 14.2% | +0.9% |

### Reproducibility

- All results at `temperature: 0` (deterministic)
- Full raw I/O saved in `results/benchmark_detailed.json`
- Tool definitions from `tools/filesystem.json`
- Eval data from `data/eval.jsonl` (unchanged since generation)
- Benchmark script: `eval/benchmark.py`

---

*This document is part of the [edge-mcp-caller](https://github.com/adityonugrohoid/edge-mcp-caller) project.*
