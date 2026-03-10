# Edge MCP Caller

## Overview
Specialist 270M model fine-tuned as an edge MCP tool caller. Takes raw Gemma 3 270M and fine-tunes it with tool knowledge baked directly into weights — no schema prompts needed. Achieves 99.5% combined accuracy across 14 tools vs GPT-OSS-120B's 41.7% and FunctionGemma's 13.6%, with 32x fewer prompt tokens.

## Core Thesis
A 270M model has no business being a generalist. Make it a specialist: bake tool knowledge into weights, deploy on the edge, one job done perfectly.

### Specialist vs Baselines (14-tool benchmark, 420 eval)
- **Ours (specialist)**: 20 tokens/request (query only), **99.5% combined** — 12/14 tools at 100%
- **GPT-OSS 120B (ceiling)**: 647 tokens/request via NIM tools API, 41.7% combined
- **FunctionGemma (generalist)**: 967 tokens/request via Ollama tools API, 13.6% combined
- **Raw Gemma 3 270M (baseline)**: 1092 tokens/request (few-shot), 11.4% combined

## Tech Stack
- **Language**: Python 3.12+
- **Base model**: Gemma 3 270M (`unsloth/gemma-3-270m-it`)
- **Fine-tuning**: Unsloth (LoRA r=128, BF16), single consumer GPU
- **Training data**: Claude Code agents (Sonnet 4.6) orchestrated by Opus 4.6, extraction-only rules
- **Inference**: Ollama / llama.cpp (Q8_0, 291 MB)
- **MCP**: Python MCP client SDK + @modelcontextprotocol/server-filesystem (npx) + mcp-server-git (uvx)
- **CLI output**: rich (tables, progress)
- **Config**: python-dotenv (.env)

## Architecture
```
Pipeline:
  1. Claude agents → data/generated/<timestamp>_<tool>.jsonl (accumulate)
     data/merge_dataset.py → data/train.jsonl + data/eval.jsonl
  2. train/finetune.py → models/<adapter>
  3. train/merge_and_convert.py → models/<merged-gguf>
  4. eval/benchmark.py → results/benchmark_*.json
  5. demo/cli.py → interactive MCP tool calling demo
```

### Data Flow (inference)
```
User query (~20 tokens)
    ↓
Gemma 3 270M (specialist, tools in weights)
    ↓ {"tool": "git_commit", "args": {"message": "fix auth bug"}}
MCP Client Bridge (JSON → MCP tools/call, arg translation, server routing)
    ↓
MCP Server (filesystem via npx, or git via uvx)
    ↓
Result to user
```

## Key Files
- `data/merge_dataset.py` — Merge all `data/generated/*.jsonl` → validate → dedup → stratified train/eval split
- `train/finetune.py` — LoRA fine-tuning via Unsloth
- `train/merge_and_convert.py` — Merge LoRA adapter + convert to GGUF for Ollama
- `eval/benchmark.py` — Scaling benchmark: 4 models × 4 subsets (3/5/8/14 tools)
- `mcp/client.py` — Bridge: model JSON → MCP tools/call (14 tools, 2 servers, arg translation, server routing)
- `demo/cli.py` — Interactive CLI demo (dual-server, auto-detect git, batch mode)
- `tools/filesystem.json` — Filesystem MCP server tool definitions (8 tools, reference only, NOT passed to model)
- `tools/git.json` — Git MCP server tool definitions (6 tools, reference only)
- `docs/generation-standard.md` — Data generation standard for all 14 tools
- `docs/benchmark-scaling-results.md` — Full scaling curve results with per-tool breakdowns
- `docs/benchmark-methodology.md` — Benchmark methodology and fairness analysis
- `docs/training-lessons.md` — Training/conversion troubleshooting guide
- `docs/real-world-use-cases.md` — Edge deployment scenarios

## Data Generation Directory Structure
```
data/
├── generated/                              # Raw batches (never overwritten)
│   ├── 20260309_143022_list_directory.jsonl # Timestamped per run
│   ├── 20260309_143022_write_file.jsonl
│   └── ...
├── train.jsonl          # 14,033 examples (built by merge step)
├── eval.jsonl           # 1,568 examples (built by merge step)
└── archive/             # Historical datasets (not used)
```

## Commands
```bash
# Step 1: Generate training data (Claude agents → data/generated/)
python data/merge_dataset.py            # merge all data/generated/ → train.jsonl + eval.jsonl

# Step 2: Fine-tune
python train/finetune.py

# Step 3: Merge adapter + convert to GGUF
python train/merge_and_convert.py

# Step 4: Benchmark
python eval/benchmark.py                    # 14-tool, 30/tool
python eval/benchmark.py --subset 3         # 3-tool subset
python eval/benchmark.py --subset all       # full scaling curve (3/5/8/14)

# Step 5: Demo (connects filesystem + git MCP servers)
python demo/cli.py                          # interactive, auto-detect git
python demo/cli.py /path/to/dir             # interactive with custom dir
python demo/cli.py --repo /path/to/repo     # explicit git repo path
python demo/cli.py --no-git                 # filesystem only, skip git
python demo/cli.py -n 10                    # batch: 10 eval examples
python demo/cli.py -n 5 --verbose           # batch with per-query detail
```

## Complete Tool Set (14 tools, 2 MCP servers)

### Filesystem MCP Server (8 tools)
| # | Tool | Args (model output) |
|---|------|---------------------|
| 1 | `list_directory` | `path` |
| 2 | `read_file` | `path` |
| 3 | `search_files` | `path`, `pattern` |
| 4 | `write_file` | `path`, `content` |
| 5 | `create_directory` | `path` |
| 6 | `edit_file` | `path`, `old_text`, `new_text` |
| 7 | `move_file` | `source`, `destination` |
| 8 | `directory_tree` | `path` |

### Git MCP Server (6 tools)
| # | Tool | Args (model output) |
|---|------|---------------------|
| 9 | `git_status` | _(none — bridge injects repo_path)_ |
| 10 | `git_diff_staged` | _(none)_ |
| 11 | `git_commit` | `message` |
| 12 | `git_log` | `max_count` _(optional, integer)_ |
| 13 | `git_branch` | _(none)_ |
| 14 | `git_create_branch` | `branch_name`, `base_branch` _(optional)_ |

**Bridge translations** (`mcp/client.py`):
- `edit_file`: model outputs `{path, old_text, new_text}` → bridge converts to MCP `{path, edits: [{oldText, newText}]}`
- Git tools: model omits `repo_path` → bridge injects from config
- `git_branch`: MCP server requires `branch_type` → bridge defaults to `"local"`
- Filesystem server: `npx -y @modelcontextprotocol/server-filesystem`
- Git server: `uvx mcp-server-git --repository <path>`
- See `docs/generation-standard.md` for full data rules.

## Training Patterns

### LoRA on Consumer Hardware
- RTX 4060 Laptop (8GB VRAM) handles sub-1B models easily
- LoRA r=128, BF16, 3 epochs — ~2h on 14K examples
- paged_adamw_8bit optimizer for memory efficiency
- gradient_checkpointing with `use_reentrant=False`

### Critical: trl v0.29 API Changes
```python
# Use SFTConfig (not TrainingArguments), processing_class (not tokenizer), max_length (not max_seq_length)
from trl import SFTConfig, SFTTrainer
sft_config = SFTConfig(..., max_length=2048)
trainer = SFTTrainer(model=model, processing_class=tokenizer, args=sft_config)
```
When in doubt, introspect: `inspect.signature(SFTConfig.__init__)`

### GGUF Conversion: Always Include Modelfile Metadata
```bash
# ALWAYS copy template + stop tokens from base model
ollama show gemma3:270m --modelfile
# Add TEMPLATE + PARAMETER blocks to your Modelfile
# A bare `FROM file.gguf` produces garbage output
```

### CRITICAL: Unsloth Strips `added_tokens_decoder`
Unsloth's `save_pretrained_merged()` drops `added_tokens_decoder` from `tokenizer_config.json`. The GGUF converter uses this section to mark `<start_of_turn>` (105) and `<end_of_turn>` (106) as special tokens (USER_DEFINED type). Without it, they get type UNKNOWN, and Ollama tokenizes them as character sequences instead of single tokens — the model can't understand its prompts.

**Fix**: Always restore the base model's `tokenizer_config.json` to the merged directory:
```python
hf_hub_download(BASE_MODEL, "tokenizer_config.json", local_dir=str(MERGED_DIR))
```

### Symptom → Diagnosis
| Symptom | Cause | Fix |
|---------|-------|-----|
| Garbage output | Missing chat template | Copy TEMPLATE block from base model |
| Infinite generation | Missing stop tokens | Copy PARAMETER stop blocks |
| Empty response + ~7 eval tokens | Missing `added_tokens_decoder` in tokenizer_config.json | Restore base model's tokenizer_config.json before GGUF conversion |
| `loras are not yet implemented` | Ollama runtime LoRA unsupported | Use full merge approach |

## Token Policy
Never limit tokens on any API call — no `max_tokens`, `num_ctx`, `num_predict`.

## Benchmark Results

### Scaling Curve (specialist, trained on all 14 tools)
| Subset | Tool Routing | Combined | Avg Tokens |
|--------|-------------|----------|------------|
| 3-tool | 100.0% | 98.9% | 18 |
| 5-tool | 100.0% | 99.3% | 20 |
| 8-tool | 99.6% | 99.2% | 21 |
| 14-tool | **99.8%** | **99.5%** | 20 |

No degradation from 3 to 14 tools. Full results in `docs/benchmark-scaling-results.md`.

### 14-Tool Per-Model Comparison (420 eval, 30/tool)
| Model | Tool Routing | Combined | Tokens | Latency |
|-------|-------------|----------|--------|---------|
| Specialist (270M) | 99.8% | 99.5% | 20 | 153ms |
| GPT-OSS 120B | 72.6% | 41.7% | 647 | 1,111ms |
| Raw Gemma 3 270M | 33.3% | 11.4% | 1,092 | 905ms |
| FunctionGemma 270M | 21.4% | 13.6% | 967 | 235ms |

### Dataset Stats
- 15,601 total examples (14,033 train / 1,568 eval) across 14 tools
- Generated by Claude Code agents (Sonnet 4.6), extraction-only rules
- Per-tool: ~1,000 each, evenly distributed
- Category split: 70% clean / 15% messy / 15% disambiguation
- Eval: 112/tool, stratified (76 clean + 18 messy + 18 disambiguation)
- 0 invalid examples after validation pipeline

## Key Patterns
- Standard chat format for training (no special tokens, no schema in prompt)
- JSON output format: `{"tool": "name", "args": {...}}`
- MCP client wrapper translates JSON → MCP tools/call protocol
- `mcp/` directory has NO `__init__.py` — avoids shadowing the PyPI `mcp` package; `demo/cli.py` uses importlib to load
- Data generation is accumulative — each run writes timestamped files to `data/generated/`, never overwrites
- Merge step reads ALL `data/generated/*.jsonl`, deduplicates, validates, splits into train/eval
- Specialist = router + extractor, NOT generator — every arg must be extractable from the query
- Training data includes messy real-world queries: typos, slang, abbreviations, grammar errors, voice transcription
- Data quality pipeline: generate → validate → dedup → stratified split → verify
- Head-to-head eval against FunctionGemma, GPT-OSS-120B, and raw Gemma on identical eval set
- Each model tested in its native interface (tools API, few-shot, or bare query)
- Eval format = deployment contract (what `mcp/client.py` expects to parse)
