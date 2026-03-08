# Training Lessons Learned

Battle-tested solutions from fine-tuning sub-1B models on consumer hardware (spatial-llm project). These apply directly to Gemma 3 270M fine-tuning.

## `trl` SFTTrainer API Breaking Changes (v0.29)

The `trl` library (v0.29) shipped breaking API changes from what most tutorials and docs show. Three things broke in sequence:

### Fix 1: `tokenizer` → `processing_class`
```python
# OLD (tutorials, docs, context7 examples)
SFTTrainer(model=model, tokenizer=tokenizer, ...)

# NEW (trl v0.29+)
SFTTrainer(model=model, processing_class=tokenizer, ...)
```

### Fix 2: `TrainingArguments` → `SFTConfig`
```python
# OLD
from transformers import TrainingArguments
args = TrainingArguments(...)
SFTTrainer(args=args, max_seq_length=2048)

# NEW (trl v0.29+) — SFTConfig wraps TrainingArguments + SFT-specific params
from trl import SFTConfig
args = SFTConfig(..., max_length=2048)  # max_length lives in config now
SFTTrainer(args=args)  # no max_seq_length here
```

### Fix 3: `max_seq_length` → `max_length`
```python
# OLD
SFTConfig(max_seq_length=2048)

# NEW (trl v0.29+)
SFTConfig(max_length=2048)
```

### How to Diagnose
Instead of guessing, introspect the actual installed API:
```python
import inspect
sig = inspect.signature(SFTConfig.__init__)
print([p for p in sig.parameters if 'max' in p or 'length' in p])
# → ['max_steps', 'max_grad_norm', 'length_column_name', 'max_length']
```

### Root Cause
Context7 docs and most online examples are written for `trl` v0.4-0.7. The latest versions reorganized the API significantly. HuggingFace ecosystem breaks frequently between minor versions.

### Takeaway
When using HuggingFace ecosystem libraries (`trl`, `peft`, `transformers`), **always inspect the actual installed signatures** before trusting tutorial code.

## QLoRA on Consumer Hardware

### Setup
- RTX 4060 Laptop, 8GB VRAM, WSL2
- Sub-1B model (Qwen3-0.6B, 751M params)
- 4-bit NF4 quantization + LoRA r=16

### VRAM Budget
- 4-bit model: ~400MB
- LoRA adapters: ~10MB (4.6M trainable / 756M total = 0.6%)
- Optimizer (paged_adamw_8bit): ~500MB
- Gradients + activations (gradient checkpointing): ~1-2GB
- **Total: ~3GB** — no need for Colab T4

### Training Stats (reference)
- 39 examples, 3 epochs, 15 steps
- **57 seconds** total training time
- Loss: 1.60 → 0.83
- Token accuracy: 71% → 81%

### Takeaway
For sub-1B models (including Gemma 3 270M) with QLoRA, 8GB VRAM is plenty. Don't default to Colab — check the math first.

## GGUF Conversion for Ollama: The Missing Chat Template

Converting a fine-tuned HuggingFace model to GGUF for Ollama has multiple failure modes. The real root cause is almost always missing Modelfile metadata.

### The Pipeline
```
LoRA adapter (safetensors)
    ↓ merge_and_unload()
Full merged model (safetensors)
    ↓ convert_hf_to_gguf.py
GGUF file (Q8_0)
    ↓ ollama create -f Modelfile
Ollama model registry
```

### Failure Modes (in order of what we tried)

| # | Approach | Result | Error |
|---|----------|--------|-------|
| 1 | F16 merged GGUF, bare Modelfile (`FROM file.gguf`) | Garbage output → crash | `unexpected EOF` |
| 2 | LoRA adapter GGUF + `FROM base` + `ADAPTER` | Immediate failure | `loras are not yet implemented` |
| 3 | Q8_0 merged GGUF, bare Modelfile | Garbage output, hangs | No stop tokens → infinite loop |
| 4 | Q8_0 merged GGUF + full chat template in Modelfile | **Works** | — |

### Red Herring: llama.cpp Version Mismatch

We suspected GGUF format incompatibility between bleeding-edge llama.cpp and Ollama's pinned version. Verified by checking out Ollama's exact pinned commit and re-running conversion — **produced identical file** (same SHA256 hash). GGUF format IS compatible across versions.

```bash
# How to find Ollama's pinned llama.cpp commit
ollama --version  # → 0.17.7
curl -s "https://raw.githubusercontent.com/ollama/ollama/v0.17.7/Makefile.sync" | head -3
# FETCH_HEAD=ec98e2002
```

### The Actual Root Cause: Missing Modelfile Metadata

When you use `FROM file.gguf` without metadata, Ollama only gets the weights. Falls back to `{{ .Prompt }}` template with:
- **No chat formatting** — model receives raw text instead of properly formatted sequences
- **No stop tokens** — model never knows when to stop → infinite garbage loops

### The Fix: Copy Template from Base Model
```bash
# 1. Extract the original model's template and parameters
ollama show <base-model> --modelfile

# 2. Add TEMPLATE and PARAMETER blocks to your Modelfile
# FROM /path/to/your-model.q8.gguf
#
# TEMPLATE """
# ... (full chat template from original model) ...
# """
#
# PARAMETER stop "<|im_start|>"
# PARAMETER stop "<|im_end|>"
# PARAMETER temperature 0.6

# 3. Recreate
ollama create your-model -f Modelfile
```

### Ollama LoRA Adapter Limitation
Ollama supports `ADAPTER` in Modelfiles, but **runtime LoRA is not implemented for all architectures**. Only reliable path: full merge via `merge_and_unload()`.

### Symptom → Diagnosis Guide

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Garbage output (random, mixed languages) | Missing chat template | Copy TEMPLATE block |
| Infinite generation, never stops | Missing stop tokens | Copy PARAMETER stop blocks |
| `unexpected EOF` | Model crashes after garbage | Add template + tokens |
| `loras are not yet implemented` | Runtime LoRA unsupported | Use full merge approach |
| Model hangs, no output | Same as garbage | Enable streaming to debug |

## Pad Token Configuration

Many tokenizers don't set `pad_token` by default. Always explicitly set it:
```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

## Model Loading (QLoRA Template)

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)
```

## LoRA Target Modules

Varies by architecture. Check model config for attention/MLP layer names:
```python
# Example for Qwen/LFM — adjust for Gemma 3
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # verify for Gemma 3
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

## Training Config Template (trl v0.29+)

```python
from trl import SFTConfig, SFTTrainer

sft_config = SFTConfig(
    output_dir="models/adapter",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_length=2048,
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    save_strategy="epoch",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    args=sft_config,
)
```

## Thinking Loop Problem (small reasoning models)

Small distilled reasoning models (Qwen3-0.6B, DeepSeek-R1:1.5b) can converge to stable attractors at low temperatures, producing infinite thinking loops. Not relevant for Gemma 3 270M (not a reasoning model), but worth knowing if exploring alternatives.
