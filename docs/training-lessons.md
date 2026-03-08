# Training & Deployment Lessons Learned

Battle-tested solutions from fine-tuning sub-1B models on consumer hardware. Consolidated from the spatial-llm and edge-mcp-caller projects. Every issue below was hit in practice.

---

## Table of Contents

1. [HuggingFace Ecosystem API Breakage](#1-huggingface-ecosystem-api-breakage)
2. [Unsloth-Specific Gotchas](#2-unsloth-specific-gotchas)
3. [QLoRA vs Full Precision on Consumer Hardware](#3-qlora-vs-full-precision-on-consumer-hardware)
4. [GGUF Conversion — The Full Failure Catalog](#4-gguf-conversion--the-full-failure-catalog)
5. [Ollama Modelfile — Non-Negotiable Metadata](#5-ollama-modelfile--non-negotiable-metadata)
6. [Gemma 3 Specific Issues](#6-gemma-3-specific-issues)
7. [Small Reasoning Model Loops](#7-small-reasoning-model-loops)
8. [Model Selection Insights](#8-model-selection-insights)
9. [Synthetic Data Generation](#9-synthetic-data-generation)
10. [Symptom → Diagnosis Quick Reference](#10-symptom--diagnosis-quick-reference)

---

## 1. HuggingFace Ecosystem API Breakage

The `trl`, `peft`, and `transformers` libraries break API between minor versions. Never trust tutorial code — always introspect the installed version.

### trl v0.24+ / v0.29+ Breaking Changes

**Fix 1: `tokenizer` → `processing_class`**
```python
# OLD (tutorials, docs, context7 examples)
SFTTrainer(model=model, tokenizer=tokenizer, ...)

# NEW (trl v0.29+)
SFTTrainer(model=model, processing_class=tokenizer, ...)
```

**Fix 2: `TrainingArguments` → `SFTConfig`**
```python
# OLD
from transformers import TrainingArguments
args = TrainingArguments(...)
SFTTrainer(args=args, max_seq_length=2048)

# NEW (trl v0.29+)
from trl import SFTConfig
args = SFTConfig(..., max_length=2048)
SFTTrainer(args=args)
```

**Fix 3: `max_seq_length` → `max_length`**
```python
# OLD
SFTConfig(max_seq_length=2048)

# NEW (trl v0.29+)
SFTConfig(max_length=2048)
```

**Fix 4: `trainer.tokenizer` removed**
```python
# OLD
trainer.tokenizer(text, return_tensors="pt")

# NEW — use the tokenizer variable directly
tokenizer(text, return_tensors="pt")
```

**Fix 5: `warmup_ratio` deprecated**
```python
# Deprecated in v5.2+
SFTConfig(warmup_ratio=0.03)  # warning: use warmup_steps instead

# NEW
SFTConfig(warmup_steps=20)
```

### How to Diagnose

Always introspect the installed API before trusting docs:
```python
import inspect
sig = inspect.signature(SFTConfig.__init__)
print([p for p in sig.parameters if 'max' in p or 'length' in p])
# → ['max_steps', 'max_grad_norm', 'length_column_name', 'max_length']
```

### Takeaway

When using HuggingFace ecosystem libraries, **always inspect the actual installed signatures**. `python -c "help(Class.__init__)"` is faster than debugging.

---

## 2. Unsloth-Specific Gotchas

### `FastModel` not `FastLanguageModel`

Unsloth's newer API (2026.3+) uses `FastModel` for loading — it handles both text and vision models:
```python
# OLD
from unsloth import FastLanguageModel

# NEW (Unsloth 2026.3+)
from unsloth import FastModel
model, tokenizer = FastModel.from_pretrained(...)
model = FastModel.get_peft_model(model, ...)
```

### Chat Template Name

For Gemma 3, use `"gemma-3"` (not `"gemma3"`):
```python
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
```

### Response Masking

Train only on model outputs, not user prompts:
```python
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part="<start_of_turn>user\n",
    response_part="<start_of_turn>model\n",
)
```

### BOS Token Stripping

Unsloth handles BOS insertion automatically. Strip if template adds it:
```python
text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
text = text.removeprefix("<bos>")
```

### Don't Quantize Tiny Models

For 270M models, full BF16 precision fits in ~1.5GB VRAM. 4-bit quantization hurts quality at this scale and is unnecessary:
```python
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-270m-it",
    load_in_4bit=False,   # Full precision for tiny models
    load_in_8bit=False,
    full_finetuning=False, # LoRA, not full fine-tune
)
```

---

## 3. QLoRA vs Full Precision on Consumer Hardware

### RTX 4060 Laptop (8GB VRAM) — What Fits

| Model Size | Method | VRAM Used | Fits? |
|---|---|---|---|
| 270M (BF16) | LoRA full precision | ~1.5GB | Easily |
| 270M (BF16) | Full fine-tune | ~2.5GB | Yes |
| 600M-1B (4-bit) | QLoRA | ~3GB | Yes |
| 1B-3B (4-bit) | QLoRA | ~4-6GB | Tight |
| 7B (4-bit) | QLoRA | ~8GB | Barely |
| 13B+ | Any | >8GB | No — need cloud |

### QLoRA VRAM Budget (sub-1B reference)

From spatial-llm (Qwen3-0.6B, 751M params):
- 4-bit model: ~400MB
- LoRA adapters: ~10MB (4.6M trainable / 756M total = 0.6%)
- Optimizer (paged_adamw_8bit): ~500MB
- Gradients + activations (gradient checkpointing): ~1-2GB
- **Total: ~3GB** — no need for Colab T4

### Training Speed Reference

| Project | Model | Examples | Epochs | Steps | Time |
|---|---|---|---|---|---|
| spatial-llm | Qwen3-0.6B (QLoRA) | 39 | 3 | 15 | 57 seconds |
| edge-mcp-caller | Gemma 3 270M (LoRA BF16) | 3240 | 3 | 609 | 55 minutes |

### Takeaway

For sub-1B models, 8GB VRAM is plenty. Don't default to Colab. The 270M model is a sweet spot — full precision fits with room to spare.

---

## 4. GGUF Conversion — The Full Failure Catalog

### The Pipeline

```
LoRA adapter (safetensors)
    ↓ merge (Unsloth or PEFT)
Full merged model (HF safetensors)
    ↓ convert_hf_to_gguf.py (llama.cpp)
GGUF file (BF16)
    ↓ llama-quantize (llama.cpp)
GGUF file (Q8_0 / Q4_K_M)
    ↓ ollama create -f Modelfile
Ollama model registry
```

### Unsloth's Built-in GGUF Export — Why It Fails

`model.save_pretrained_gguf()` is convenient but unreliable on headless/restricted systems:

| Issue | Error | Root Cause |
|---|---|---|
| System package deps | `EOFError: EOF when reading a line` | Needs `libcurl`, `cmake`, `libssl` via sudo |
| Interactive prompt | `EOFError` | Prompts for `sudo apt-get` confirmation |
| Quantizer not found | `No working quantizer found` | Looks in repo root, build puts it in `build/bin/` |

**Verdict:** Use the manual 3-step pipeline instead.

### Manual Pipeline (Recommended)

```
save_pretrained_merged()  →  convert_hf_to_gguf.py  →  llama-quantize
     (Unsloth)                  (llama.cpp Python)        (llama.cpp C++)
```

**Step 1: Merge LoRA into base model**
```python
model.save_pretrained_merged("models/merged", tokenizer, save_method="merged_16bit")
```

**Step 2: Convert HF → BF16 GGUF**
```bash
python ~/.unsloth/llama.cpp/convert_hf_to_gguf.py models/merged/ \
  --outfile models/gguf/model-bf16.gguf --outtype bf16
```

**Step 3: Quantize**
```bash
~/.unsloth/llama.cpp/build/bin/llama-quantize \
  models/gguf/model-bf16.gguf models/gguf/model-q8_0.gguf Q8_0
```

### Building llama.cpp Without Sudo

CURL/SSL are only for HTTP downloads, not conversion. Build without them:
```bash
pip install cmake  # pip cmake, no sudo needed
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/.unsloth/llama.cpp
cd ~/.unsloth/llama.cpp
cmake -B build -DLLAMA_CURL=OFF -DGGML_CUDA=OFF
cmake --build build -j$(nproc) --target llama-quantize
```

**Conversion quality is identical** regardless of CURL/SSL support.

### Red Herring: llama.cpp Version Mismatch

Tested by checking out Ollama's exact pinned llama.cpp commit — produced **identical file** (same SHA256 hash). GGUF format IS compatible across versions.

```bash
# How to find Ollama's pinned commit
ollama --version  # → 0.17.7
curl -s "https://raw.githubusercontent.com/ollama/ollama/v0.17.7/Makefile.sync" | head -3
```

### Ollama LoRA Adapter Limitation

Ollama supports `ADAPTER` in Modelfiles, but runtime LoRA is **not implemented for all architectures**. Only reliable path: full merge via `merge_and_unload()` or `save_pretrained_merged()`.

### Size Progression Reference

| Stage | Format | Size (270M model) |
|---|---|---|
| Base model (HF, BF16) | safetensors | 540 MB |
| LoRA adapter only | safetensors | 116 MB |
| Merged model (HF, BF16) | safetensors | 540 MB |
| BF16 GGUF | .gguf | 536 MB |
| Q8_0 GGUF | .gguf | 272 MB |
| Q4_K_M GGUF | .gguf | ~150 MB |

---

## 5. Ollama Modelfile — Non-Negotiable Metadata

A bare `FROM file.gguf` produces garbage output and infinite generation. **Always** include:

### Must-Have: TEMPLATE + Stop Tokens

```bash
# 1. Get template from base model
ollama show gemma3:270m --modelfile

# 2. Build Modelfile with metadata
cat > Modelfile << 'EOF'
FROM /path/to/your-model-q8_0.gguf
TEMPLATE """{{- range $i, $_ := .Messages }}...full template..."""
PARAMETER stop <end_of_turn>
PARAMETER top_p 0.95
PARAMETER top_k 64
EOF

# 3. Register
ollama create my-model -f Modelfile
```

### Why Bare Modelfiles Fail

When Ollama pulls a registry model, it includes **multiple layers**: weights, chat template, parameters, license. A custom GGUF only has weights. Ollama falls back to `{{ .Prompt }}` — raw text with:
- **No chat formatting** → model receives garbage instead of structured messages
- **No stop tokens** → model never knows when to stop → infinite generation

### Model-Specific Stop Tokens

| Model Family | Stop Tokens |
|---|---|
| Gemma 3 | `<end_of_turn>` |
| Qwen3 / ChatML | `<\|im_start\|>`, `<\|im_end\|>` |
| Llama 3 | `<\|eot_id\|>` |

---

## 6. Gemma 3 Specific Issues

### Vocab Size Mismatch (262144 vs 262145)

**Symptom:** `AssertionError: max(tokenizer.vocab.values()) < vocab_size`

**Root cause:** Gemma 3 tokenizer has 262145 tokens (IDs 0–262144), but model config says `vocab_size: 262144` and embedding has 262144 rows.

**Fix:** Pad embedding + update config:
```python
import torch, json
import safetensors.torch as st

config = json.loads(Path("config.json").read_text())
tensors = st.load_file("model.safetensors")
embed = tensors["model.embed_tokens.weight"]

# Pad one zero row
pad = torch.zeros(1, embed.shape[1], dtype=embed.dtype)
tensors["model.embed_tokens.weight"] = torch.cat([embed, pad], dim=0)
st.save_file(tensors, "model.safetensors")

config["vocab_size"] = 262145
Path("config.json").write_text(json.dumps(config, indent=2))
```

### Missing tokenizer.model After Merge

**Symptom:** `NotImplementedError: BPE pre-tokenizer was not recognized`

**Root cause:** Unsloth's `save_pretrained_merged()` saves `tokenizer.json` + `tokenizer_config.json` but drops the sentencepiece `tokenizer.model` file. llama.cpp's converter needs it.

**Fix:** Download original tokenizer files from HuggingFace:
```python
from huggingface_hub import hf_hub_download
for f in ["tokenizer.model", "special_tokens_map.json", "added_tokens.json"]:
    hf_hub_download("unsloth/gemma-3-270m-it", f, local_dir="models/merged")
```

### Ungated Model Access

`google/gemma-3-270m-it` is gated (requires HF_TOKEN + license acceptance). Use `unsloth/gemma-3-270m-it` instead — identical weights, freely accessible.

---

## 7. Small Reasoning Model Loops

*From spatial-llm project. Not relevant to Gemma 3 270M (non-reasoning) but critical knowledge for model selection.*

### The Problem

Small distilled reasoning models can converge to stable attractors at low temperatures, producing infinite thinking loops:

| Model | Loop Rate | Notes |
|---|---|---|
| Qwen3-0.6B | 20-50% | Abandoned |
| Qwen3-1.7B | ~20% | Abandoned |
| DeepSeek-R1:1.5B | ~33% at temp 0.6 | Abandoned |
| LFM2.5-Thinking:1.2B | 0% (36 runs) | Selected |

### Root Cause

arxiv:2512.12895 — small distilled reasoning models converge to stable attractors at low temperature. The thinking process enters a self-reinforcing loop.

### Temperature Recommendations

| Model Type | Recommended Temp |
|---|---|
| Non-reasoning (Gemma 3, Llama) | 0.0 – 0.1 |
| Reasoning (DeepSeek-R1) | 0.6 (helps but doesn't eliminate loops) |
| Thinking (LFM2.5-Thinking) | 0.6 (no loops observed) |

---

## 8. Model Selection Insights

*From spatial-llm benchmarking across 20+ models.*

### Non-Monotonic Scaling

Bigger isn't always better. 24B models can tie with 120B+ on specific tasks. Model architecture and training data matter more than raw parameter count.

### Fine-Tuning Can Degrade Performance

Fine-tuning (instruction tuning) degrades spatial reasoning in some model families:
- Llama 3.1 70B variants: dropped from 0.909 to <0.353 after fine-tuning
- Not all capabilities transfer — fine-tuning for chat can hurt non-chat tasks

### Sub-4B Baseline Reality

All sub-4B baseline models (no fine-tuning) fail at spatial reasoning completely. The 270M specialist model approach works because we're training for one specific task.

---

## 9. Synthetic Data Generation

### API Provider Selection

| Provider | Best Model | Speed | Limit | Cost |
|---|---|---|---|---|
| NVIDIA NIM | llama-3.1-70b-instruct | 12s/batch of 20 | 40 RPM | Free unlimited |
| Ollama Cloud | gemini-3-flash-preview | Varies | Weekly cap | Free tier |
| Anthropic | Claude | — | — | Requires API key (Max200 plan ≠ API) |

### Batch Generation Lessons

- **Batch size 20** is sweet spot for llama-70b (12s/batch, reliable JSON)
- **5 concurrent** requests with 1s delay stays under 40 RPM
- **Always strip markdown fences** from LLM JSON output (```json blocks)
- **DeepSeek v3.1** is too slow (~45s for 10 examples, thinking model overhead)
- **Ollama Cloud** weekly limit burns fast (~15% for 599 examples)

### Quality Validation (9 Checks)

Run all checks exhaustively on all examples (not sampled):

| Check | Detects | Prevalence |
|---|---|---|
| list_directory on file-like paths | Extension + no trailing slash | 2.7% |
| search_files pattern="*" | Should be list_directory | 2.4% |
| search_files empty path | Ambiguous intent | — |
| read_file with "directory/folder" in query | Wrong tool routing | 0.9% |
| read_file path ending in "/" | Definitely a directory | — |
| search_files exact filename, no search intent | Ambiguous routing | 1.1% |
| Duplicate user queries | Case-insensitive dedup | — |
| Missing required args | Incomplete tool calls | — |
| Non-string arg values | Type validation | — |

### Data Pipeline is Idempotent

`clean_and_backfill.py` can be run repeatedly until target count (1200/tool) is reached. Each run cleans bad examples and generates replacements.

---

## 10. Symptom → Diagnosis Quick Reference

### Training Errors

| Symptom | Cause | Fix |
|---|---|---|
| `TypeError: tokenizer` | trl v0.29+ API change | Use `processing_class=tokenizer` |
| `TypeError: max_seq_length` | trl v0.29+ API change | Use `max_length` in `SFTConfig` |
| `AttributeError: trainer.tokenizer` | Removed in newer trl | Use `tokenizer` variable directly |
| `AssertionError` on generate | Unsloth expects torch.Tensor | Use `tokenizer(text, return_tensors="pt")` not `apply_chat_template(return_tensors="pt")` |
| CUDA OOM | Model too large for VRAM | Use QLoRA (4-bit) or reduce batch size |

### GGUF Conversion Errors

| Symptom | Cause | Fix |
|---|---|---|
| `EOFError: EOF when reading a line` | Unsloth prompts for sudo | Use manual pipeline |
| `No working quantizer found` | Unsloth can't find llama-quantize | Build llama.cpp manually |
| `AssertionError: vocab_size` | Tokenizer/embedding mismatch | Pad embedding + update config |
| `NotImplementedError: BPE pre-tokenizer` | Missing tokenizer.model | Download from HuggingFace |
| `loras are not yet implemented` | Ollama runtime LoRA unsupported | Use full merge approach |

### Inference Errors

| Symptom | Cause | Fix |
|---|---|---|
| Garbage output (random, mixed languages) | Missing chat template in Modelfile | Copy TEMPLATE from base model |
| Infinite generation, never stops | Missing stop tokens | Add `PARAMETER stop` to Modelfile |
| `unexpected EOF` | Model crashes after garbage | Add template + stop tokens |
| Model hangs, no output | Same root cause as garbage | Enable streaming to debug |
| Thinking loops (infinite `<think>`) | Small reasoning model attractor | Use non-reasoning model or LFM2.5 |
