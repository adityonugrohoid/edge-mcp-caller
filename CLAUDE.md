# Edge MCP Caller

## Overview
Specialist 270M model fine-tuned as an edge MCP tool caller. Takes raw Gemma 3 270M (same base Google used for FunctionGemma) and fine-tunes it with tool knowledge baked directly into weights — no schema prompts needed. Targets beating FunctionGemma's 85% accuracy on MCP filesystem operations with 40x shorter prompts.

## Core Thesis
A 270M model has no business being a generalist. Make it a specialist: bake tool knowledge into weights, deploy on the edge, one job done perfectly.

### Specialist vs Generalist
- **FunctionGemma (generalist)**: passes ~400 tokens of tool schemas in every prompt, 270M params split between schema parsing + intent routing, 85% accuracy fine-tuned
- **Ours (specialist)**: ~10 token prompts (query only), 270M params fully focused on intent routing for known tools, target >85% accuracy

## Tech Stack
- **Language**: Python 3.12+
- **Base model**: Gemma 3 270M (`google/gemma-3-270m-pt`) — NOT FunctionGemma
- **Fine-tuning**: Unsloth (LoRA/QLoRA), single consumer GPU or free Colab
- **Training data**: Synthetic generation via NVIDIA NIM API (llama-3.1-70b-instruct)
- **Inference**: Ollama / llama.cpp
- **MCP**: Python MCP client SDK + @modelcontextprotocol/server-filesystem
- **CLI output**: rich (tables, progress)
- **Config**: python-dotenv (.env)

## Architecture
```
Pipeline:
  1. data/generate_dataset.py   → data/train.jsonl + data/eval.jsonl
  2. train/finetune.py          → models/<adapter>
  3. train/merge_and_convert.py  → models/<merged-gguf>
  4. eval/benchmark.py           → results/benchmark.json + results/report.html
  5. demo/cli.py                 → interactive MCP tool calling demo
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
- `data/generate_dataset.py` — Synthetic training data generation via NVIDIA NIM API
- `data/clean_and_backfill.py` — Quality validation, bad example removal, and backfill
- `data/archive/` — Archived baseline datasets for comparison
- `train/finetune.py` — LoRA/QLoRA fine-tuning via Unsloth
- `train/merge_and_convert.py` — Merge LoRA adapter + convert to GGUF for Ollama
- `eval/benchmark.py` — Head-to-head benchmark: raw vs fine-tuned vs FunctionGemma vs GPT-4
- `mcp/client.py` — Bridge: model JSON output → MCP tools/call protocol
- `demo/cli.py` — Interactive CLI demo
- `tools/filesystem.json` — MCP filesystem tool definitions (reference only, NOT passed to model)
- `docs/training-lessons.md` — Battle-tested training/conversion troubleshooting guide

## Commands
```bash
# Step 1: Generate training data
python data/generate_dataset.py
python data/clean_and_backfill.py  # validate + fix + backfill to 1200/tool

# Step 2: Fine-tune
python train/finetune.py

# Step 3: Merge adapter + convert to GGUF
python train/merge_and_convert.py

# Step 4: Benchmark
python eval/benchmark.py

# Step 5: Demo
python demo/cli.py
```

## MVP Tools (v0.1 — Filesystem Read-Only)
| Tool | Args | Example |
|------|------|---------|
| `list_directory` | `path: string` | "what's in the project root?" |
| `read_file` | `path: string` | "show me the README" |
| `search_files` | `path: string, pattern: string` | "find all Python files" |

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

### Symptom → Diagnosis
| Symptom | Cause | Fix |
|---------|-------|-----|
| Garbage output | Missing chat template | Copy TEMPLATE block from base model |
| Infinite generation | Missing stop tokens | Copy PARAMETER stop blocks |
| `loras are not yet implemented` | Ollama runtime LoRA unsupported | Use full merge approach |

## Token Policy
Never limit tokens on any API call — no `max_tokens`, `num_ctx`, `num_predict`.

## Key Patterns
- Standard chat format for training (no special tokens, no schema in prompt)
- JSON output format: `{"tool": "name", "args": {...}}`
- MCP client wrapper translates JSON → MCP tools/call protocol
- Synthetic data generated by NVIDIA NIM API (llama-3.1-70b-instruct) with diversity constraints
- Data quality pipeline: generate → validate (9 checks) → clean → backfill → verify
- Head-to-head eval against FunctionGemma on identical eval set
