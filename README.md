<div align="center">

# Edge MCP Caller

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)](#)

**270M specialist model that bakes MCP tool knowledge into weights — 99.5% accuracy across 14 tools, beating a 120B model with 32x fewer prompt tokens.**

[Getting Started](#getting-started) | [Architecture](#architecture) | [Benchmark](#benchmark) | [Scaling Results](#scaling-results)

</div>

---

## The Thesis

Google's FunctionGemma takes Gemma 3 270M and makes it a **generalist** function caller — pass any tool schema in the prompt, and it routes to the right tool.

We take the **same base model** and make it a **specialist** — tool definitions baked directly into model weights. No schemas in the prompt. Query in (~20 tokens), JSON tool call out.

> A 270M model has no business being a generalist. Make it a specialist, deploy it on the edge, and let it do one job perfectly.

## Results

420 eval examples across 14 tools, deterministic (temp=0), each model tested in its native interface.

| Model | Size | Tool Routing | Combined | Prompt Tokens | Latency |
|-------|------|-------------|----------|---------------|---------|
| **Ours (specialist)** | **270M** | **99.8%** | **99.5%** | **20** | **153ms** |
| GPT-OSS 120B (NIM API) | 120B | 72.6% | 41.7% | 647 | 1,111ms |
| FunctionGemma (Ollama tools API) | 270M | 21.4% | 13.6% | 967 | 235ms |
| Raw Gemma 3 (few-shot prompt) | 270M | 33.3% | 11.4% | 1,092 | 905ms |

The specialist uses 20 tokens per request regardless of tool count. Baselines scale linearly — 647 to 1,092 tokens at 14 tools — because they must pass tool schemas in every prompt.

## Features

- **14 tools, 2 MCP servers** — filesystem (8 tools) + git (6 tools)
- **99.5% combined accuracy** — 12 of 14 tools at 100%, no degradation from 3 to 14 tools
- **32x fewer tokens** — 20 tokens vs 647+ per request (schemas baked into weights)
- **Edge-ready** — 291 MB Q8_0, runs on phones, laptops, Raspberry Pi
- **153ms avg latency** — fully local, no cloud dependency, zero API cost
- **MCP-native** — calls real MCP servers via standard protocol

## Tool Set (14 tools)

### Filesystem MCP Server (8 tools)

| Tool | Args |
|------|------|
| `list_directory` | `path` |
| `read_file` | `path` |
| `search_files` | `path`, `pattern` |
| `write_file` | `path`, `content` |
| `create_directory` | `path` |
| `edit_file` | `path`, `old_text`, `new_text` |
| `move_file` | `source`, `destination` |
| `directory_tree` | `path` |

### Git MCP Server (6 tools)

| Tool | Args |
|------|------|
| `git_status` | _(none)_ |
| `git_diff_staged` | _(none)_ |
| `git_commit` | `message` |
| `git_log` | `max_count` _(optional)_ |
| `git_branch` | _(none)_ |
| `git_create_branch` | `branch_name`, `base_branch` _(optional)_ |

## Architecture

```
Generalist (FunctionGemma / GPT-OSS):
  Input:  [~600 tokens of schemas] + [query]  = ~650 tokens
  Model:  params split between parsing schemas + routing intent

Specialist (ours):
  Input:  [query only]  = ~20 tokens
  Model:  params fully focused on routing intent for known tools
  Output: {"tool": "list_directory", "args": {"path": "src/"}}
```

### Inference Pipeline

```
User query (~20 tokens)
    ↓
Gemma 3 270M (specialist, tools in weights)
    ↓ {"tool": "git_commit", "args": {"message": "fix auth bug"}}
MCP Client Bridge (JSON → MCP tools/call)
    ↓
MCP Server (filesystem or git, standard, unchanged)
    ↓
Result to user
```

## Scaling Results

The specialist is trained on all 14 tools simultaneously. We benchmark subsets to test if accuracy degrades as tool count increases.

### Specialist Scaling Curve

| Subset | Tools | Tool Routing | Combined | Tokens |
|--------|-------|-------------|----------|--------|
| 3-tool | list_directory, read_file, search_files | 100.0% | 98.9% | 18 |
| 5-tool | + write_file, create_directory | 100.0% | 99.3% | 20 |
| 8-tool | + edit_file, move_file, directory_tree | 99.6% | 99.2% | 21 |
| 14-tool | + 6 git tools | 99.8% | 99.5% | 20 |

**No degradation found.** The 270M specialist handles 14 tools as well as 3.

### Baseline Scaling (Combined Accuracy)

| Model | 3-tool | 5-tool | 8-tool | 14-tool |
|-------|--------|--------|--------|---------|
| **Specialist** | **98.9%** | **99.3%** | **99.2%** | **99.5%** |
| GPT-OSS 120B | 33.3% | 25.3% | 27.5% | 41.7% |
| FunctionGemma | 20.0% | 17.3% | 11.2% | 13.6% |
| Raw Gemma 3 | 13.3% | 12.7% | 10.4% | 11.4% |

Full per-tool and per-category breakdowns in [`docs/benchmark-scaling-results.md`](docs/benchmark-scaling-results.md).

## Getting Started

### Prerequisites

- Python 3.12+
- NVIDIA GPU with 8GB+ VRAM (for training) or free Google Colab
- Ollama (for inference)
- Node.js 18+ (for filesystem MCP server)
- uv / uvx (for git MCP server)

### Installation

```bash
git clone https://github.com/adityonugrohoid/edge-mcp-caller.git
cd edge-mcp-caller
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Usage

```bash
# Step 1: Generate training data (Claude Code agents → data/generated/)
#         Then merge: python data/merge_dataset.py

# Step 2: Fine-tune (LoRA on Gemma 3 270M)
python train/finetune.py

# Step 3: Merge adapter + convert to GGUF for Ollama
python train/merge_and_convert.py

# Step 4: Benchmark (scaling curve across 4 models)
python eval/benchmark.py                    # 14-tool, 30/tool
python eval/benchmark.py --subset 3         # 3-tool subset
python eval/benchmark.py --subset all       # full scaling curve

# Step 5: Interactive demo (query → model → MCP servers → result)
python demo/cli.py                          # interactive, auto-detect git
python demo/cli.py /path/to/dir             # specify allowed directory
python demo/cli.py --repo /path/to/repo     # explicit git repo path
python demo/cli.py --no-git                 # filesystem only
python demo/cli.py -n 10                    # batch mode
python demo/cli.py -n 5 --verbose           # batch with detail
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Base Model | Gemma 3 270M (`unsloth/gemma-3-270m-it`) |
| Fine-tuning | Unsloth (LoRA r=128, BF16) |
| Training Data | Claude Code agents (Sonnet 4.6), extraction-only rules |
| Inference | Ollama (Q8_0 GGUF, 291 MB) |
| MCP Servers | @modelcontextprotocol/server-filesystem (npx), mcp-server-git (uvx) |
| Language | Python 3.12+ |
| Hardware | RTX 4060 Laptop (8GB VRAM) — single consumer GPU |

## Project Structure

```
edge-mcp-caller/
├── data/
│   ├── merge_dataset.py            # Merge generated/ → train.jsonl + eval.jsonl
│   ├── generated/                  # Raw batches (timestamped, accumulative)
│   ├── train.jsonl                 # 14,033 training examples
│   └── eval.jsonl                  # 1,568 eval examples
├── tools/
│   ├── filesystem.json             # 8 filesystem tool schemas
│   └── git.json                    # 6 git tool schemas
├── train/
│   ├── finetune.py                 # LoRA fine-tune via Unsloth
│   └── merge_and_convert.py        # Merge adapter + GGUF conversion
├── eval/
│   └── benchmark.py                # Scaling benchmark (3/5/8/14 tools × 4 models)
├── mcp/
│   └── client.py                   # JSON → MCP tools/call bridge (14 tools, 2 servers)
├── demo/
│   └── cli.py                      # Interactive CLI demo
├── docs/
│   ├── benchmark-scaling-results.md # Full scaling curve results
│   ├── generation-standard.md      # Data generation rules for all 14 tools
│   ├── training-lessons.md         # Training/conversion troubleshooting
│   ├── benchmark-methodology.md    # Benchmark fairness analysis
│   └── real-world-use-cases.md     # Edge deployment scenarios
├── models/                         # Adapters + GGUF (gitignored)
└── results/                        # Benchmark outputs
```

## Key Design Decisions

- **Specialist = router + extractor, not generator.** The model picks the right tool and extracts arguments from the query. It does not invent paths or generate content.
- **Extraction-only training data.** Every argument in the expected output must be extractable from the user's query. See [`docs/generation-standard.md`](docs/generation-standard.md).
- **Simplified model output for complex tools.** `edit_file` outputs `{path, old_text, new_text}` — the MCP bridge converts to the server's `{path, edits: [{oldText, newText}]}` format. Git tools omit `repo_path` — bridge injects from config.
- **No `mcp/__init__.py`.** The local `mcp/` directory would shadow the PyPI `mcp` package. `demo/cli.py` uses importlib to load it.

## License

This project is licensed under the [MIT License](LICENSE).

## Author

**Adityo Nugroho** ([@adityonugrohoid](https://github.com/adityonugrohoid))

## Acknowledgments

- [Google Gemma 3](https://ai.google.dev/gemma) — base model
- [FunctionGemma](https://ai.google.dev/gemma/docs/functiongemma) — generalist benchmark target
- [Model Context Protocol](https://modelcontextprotocol.io/) — tool calling standard
- [Unsloth](https://unsloth.ai/) — efficient fine-tuning
- [Amazon SLM Tool Calling Paper](https://arxiv.org/abs/2512.15943) — proving tiny models can compete
- [Microsoft SLM Fine-tuning Guide](https://github.com/microsoft/slm-finetuning-for-function-calling) — baking tools into weights
