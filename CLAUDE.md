# Edge MCP Caller

## Overview
Specialist 270M model fine-tuned as an edge MCP tool caller. Takes raw Gemma 3 270M (same base Google used for FunctionGemma) and fine-tunes it with tool knowledge baked directly into weights — no schema prompts needed. Achieves 90.8% combined accuracy vs FunctionGemma's 18.1% and GPT-OSS-120B's 23.3%, with 13x fewer prompt tokens.

## Core Thesis
A 270M model has no business being a generalist. Make it a specialist: bake tool knowledge into weights, deploy on the edge, one job done perfectly.

### Specialist vs Generalist
- **GPT-OSS 120B (ceiling)**: 246 tokens/request via NIM tools API, 23.3% combined (76.4% tool routing but 23.6% args — format mismatch)
- **FunctionGemma (generalist)**: 264 tokens/request via Ollama tools API, 18.1% combined, 85/360 refusals
- **Raw Gemma 3 270M (baseline)**: 269 tokens/request (few-shot, no tools API support), 13.3% combined
- **Ours (specialist)**: 20 tokens/request (query only), **90.8% combined** — beats 120B ceiling, 13x fewer tokens

## Tech Stack
- **Language**: Python 3.12+
- **Base model**: Gemma 3 270M (`google/gemma-3-270m-pt`) — NOT FunctionGemma
- **Fine-tuning**: Unsloth (LoRA/QLoRA), single consumer GPU or free Colab
- **Training data (v0.1)**: Synthetic generation via NVIDIA NIM API (llama-3.1-70b-instruct)
- **Training data (v0.2+)**: Claude Code agents (Sonnet 4.6) orchestrated by Opus 4.6 — parallel generation, no API rate limits
- **Inference**: Ollama / llama.cpp
- **MCP**: Python MCP client SDK + @modelcontextprotocol/server-filesystem
- **CLI output**: rich (tables, progress)
- **Config**: python-dotenv (.env)

## Architecture
```
Pipeline:
  1a. Claude agents (v0.2+)      → data/generated/<timestamp>_<tool>.jsonl (accumulate)
  1b. data/generate_dataset.py   → data/generated/ (NIM API, v0.1 legacy)
  1c. Merge step                 → data/train.jsonl + data/eval.jsonl (from all generated/)
  2.  train/finetune.py          → models/<adapter>
  3.  train/merge_and_convert.py → models/<merged-gguf>
  4.  eval/benchmark.py          → results/benchmark.json + results/report.html
  5.  demo/cli.py                → interactive MCP tool calling demo
```

### Data Flow (inference)
```
User query (~10 tokens)
    ↓
Gemma 3 270M (specialist, tools in weights)
    ↓ {"tool": "list_directory", "args": {"path": "src/"}}
MCP Client Wrapper (JSON → MCP tools/call)
    ↓
MCP Filesystem Server (standard, unchanged)
    ↓
Result to user
```

## Key Files
- `data/generate_dataset.py` — Synthetic training data generation via NVIDIA NIM API (v0.1 legacy)
- `data/merge_dataset.py` — Merge all `data/generated/*.jsonl` → validate → dedup → stratified train/eval split
- `data/clean_and_backfill.py` — Quality validation, bad example removal, and backfill (v0.1 legacy)
- `data/archive/` — Archived baseline datasets for comparison
- `train/finetune.py` — LoRA/QLoRA fine-tuning via Unsloth
- `train/merge_and_convert.py` — Merge LoRA adapter + convert to GGUF for Ollama
- `eval/benchmark.py` — Head-to-head benchmark: specialist vs raw Gemma vs FunctionGemma vs GPT-OSS-120B, per-category reporting
- `docs/benchmark-methodology.md` — Benchmark methodology, fairness analysis, known caveats
- `mcp/client.py` — Bridge: model JSON output → MCP tools/call protocol (5 tools: list_directory, read_file, search_files, write_file, create_directory)
- `demo/cli.py` — Interactive CLI demo
- `tools/filesystem.json` — MCP filesystem tool definitions (reference only, NOT passed to model)
- `docs/training-lessons.md` — Battle-tested training/conversion troubleshooting guide
- `docs/real-world-use-cases.md` — Real-world deployment scenarios for edge specialist models

## Data Generation Directory Structure
```
data/
├── generated/                              # Raw batches (never overwritten)
│   ├── 20260309_143022_list_directory.jsonl # Timestamped per run
│   ├── 20260309_143022_write_file.jsonl
│   ├── 20260309_180000_write_file.jsonl    # Another run, more data
│   └── ...
├── train.jsonl          # Built by merge step from all generated/
├── eval.jsonl           # Built by merge step from all generated/
└── archive/             # v0.1 baseline datasets
```

## Commands
```bash
# Step 1: Generate training data (v0.2+ uses Claude agents, writes to data/generated/)
# Legacy: python data/generate_dataset.py (NIM API, v0.1)
python data/merge_dataset.py            # merge all data/generated/ → train.jsonl + eval.jsonl

# Step 2: Fine-tune
python train/finetune.py

# Step 3: Merge adapter + convert to GGUF
python train/merge_and_convert.py

# Step 4: Benchmark
python eval/benchmark.py

# Step 5: Demo (end-to-end: query → model → MCP server → result)
python demo/cli.py                  # interactive mode, current directory
python demo/cli.py /path/to/dir    # interactive, specify allowed directory
python demo/cli.py -n 10            # batch mode: 10 eval examples
python demo/cli.py -n 360           # batch mode: full eval set
python demo/cli.py -n 5 --verbose   # batch with per-query detail
```

## MVP Tools (v0.1 — Filesystem Read-Only)
| Tool | Args | Example |
|------|------|---------|
| `list_directory` | `path: string` | "what's in the project root?" |
| `read_file` | `path: string` | "show me the README" |
| `search_files` | `path: string, pattern: string` | "find all Python files" |

## v0.2 Tools (Filesystem Read + Write)
| Tool | Args | Example |
|------|------|---------|
| `list_directory` | `path: string` | "what's in the project root?" |
| `read_file` | `path: string` | "show me the README" |
| `search_files` | `path: string, pattern: string` | "find all Python files" |
| `write_file` | `path: string, content: string` | "save 'hello world' to output.txt" |
| `create_directory` | `path: string` | "make a new folder called utils" |

## Training Patterns

### QLoRA on Consumer Hardware (from spatial-llm)
- RTX 4060 Laptop (8GB VRAM) handles sub-1B models easily
- 4-bit NF4 quantization + LoRA r=16 → ~3GB total VRAM
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

### v0.2 (919 eval, 5 tools, 3 categories)
- **95.3% tool routing** (876/919) — the model correctly identifies intent
- **48.6% strict exact-match** — bottlenecked by path ambiguity, not model capability
- Path analysis: 329/429 args failures are path mismatches (model generates plausible alternative paths like `solar_panels/monitoring/` vs `solar/monitoring/`)
- Per-tool tool routing: create_directory 100%, read_file 97.8%, search_files 95.1%, list_directory 93.4%, write_file 90.3%
- Per-category tool routing: messy 97.6%, adversarial 96.2%, clean 94.4% (clean has longer, more ambiguous paths)
- 22 avg prompt tokens, 341ms avg latency
- Results: `results/benchmark_v02_specialist.json`

### v0.1 (360 eval, 3 tools)
`python demo/cli.py -n 360` — full eval set through live MCP pipeline:
- **99.2% tool accuracy, 90.8% combined, 100% MCP success** (0 parse errors, 0 MCP failures)
- 20 avg prompt tokens, 151ms model latency, 111ms MCP latency, 263ms total e2e
- 33 failures are all args formatting (trailing slashes, pattern syntax) — no tool routing or JSON errors
- Results saved to `results/cli_batch.json` with full raw I/O per query

## v0.2 Dataset Stats
- 9174 total examples (8255 train / 919 eval) across 5 tools
- Generated by Claude Code agents (Sonnet 4.6), 14 runs, 63 JSONL files
- Per-tool balance: 1827-1847 each (±0.5%)
- Category split: 65.8% clean / 22.8% messy / 11.3% adversarial
- 0 invalid examples, 16 deduped, 1847/1847 write_file round-trip pass
- Eval examples tagged by category for per-category robustness reporting

## Key Patterns
- Standard chat format for training (no special tokens, no schema in prompt)
- JSON output format: `{"tool": "name", "args": {...}}`
- MCP client wrapper translates JSON → MCP tools/call protocol
- `mcp/` directory has NO `__init__.py` — avoids shadowing the PyPI `mcp` package; `demo/cli.py` uses importlib to load
- **v0.1 data gen**: NVIDIA NIM API (llama-3.1-70b-instruct), overwrites train/eval each run
- **v0.2+ data gen**: Claude Code agents (Sonnet 4.6), accumulate-then-merge pattern
- Data generation is accumulative — each run writes timestamped files to `data/generated/`, never overwrites
- Merge step reads ALL `data/generated/*.jsonl`, deduplicates, validates, splits into train/eval
- Multiple runs across sessions build a richer, more diverse dataset over time
- Training data includes messy real-world queries: typos, slang, abbreviations, grammar errors, voice transcription artifacts, filler/hedging, self-corrections, redundant phrasing
- Data split target: 60% clean, 25% messy/noisy, 15% adversarial disambiguation (read vs write)
- Eval set tagged by category for robustness breakdown (clean acc vs messy acc vs disambiguation acc)
- Data quality pipeline: generate → validate (11 checks) → dedup → stratified split → verify
- Head-to-head eval against FunctionGemma, GPT-OSS-120B, and raw Gemma on identical eval set
- Benchmark methodology documented in `docs/benchmark-methodology.md`
- Each model tested in its native interface (tools API, few-shot, or bare query)
- Eval format = deployment contract (what `mcp/client.py` expects to parse)
