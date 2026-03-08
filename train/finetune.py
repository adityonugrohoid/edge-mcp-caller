#!/usr/bin/env python3
"""Fine-tune Gemma 3 270M as a specialist MCP tool caller using Unsloth LoRA.

Loads training data from data/train.jsonl and data/eval.jsonl,
fine-tunes with LoRA on the instruction-tuned base, saves adapter to models/adapter/.

Hardware: RTX 4060 Laptop (8GB VRAM) — 270M model fits at full BF16 precision.
"""

import json
from pathlib import Path

from datasets import Dataset
from rich.console import Console
from rich.table import Table
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = PROJECT_ROOT / "data" / "train.jsonl"
EVAL_FILE = PROJECT_ROOT / "data" / "eval.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "models" / "adapter"

# Model — ungated Unsloth mirror (no HF_TOKEN needed)
MODEL_NAME = "unsloth/gemma-3-270m-it"
MAX_SEQ_LENGTH = 512  # Our examples are ~50-80 tokens

# LoRA — high rank for tiny model, maximize capacity
LORA_R = 128
LORA_ALPHA = 128
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training
BATCH_SIZE = 4
GRAD_ACCUM = 4  # Effective batch size = 16
EPOCHS = 3
LR = 2e-4
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
SEED = 3407

console = Console()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(path: Path) -> Dataset:
    """Load JSONL training data into a HuggingFace Dataset.

    Input format:  {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
    Output format: {"conversations": [{"role": "user", ...}, {"role": "assistant", ...}]}
    """
    examples = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            examples.append({
                "conversations": [
                    {"role": "user", "content": ex["messages"][0]["content"]},
                    {"role": "assistant", "content": ex["messages"][1]["content"]},
                ]
            })
    return Dataset.from_list(examples)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    console.print("[bold cyan]Edge MCP Caller — Fine-Tuning (Step 2)[/bold cyan]\n")

    # 1. Load model (full precision — 270M is small enough)
    console.print(f"[bold]1. Loading model:[/bold] {MODEL_NAME}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
    )

    # 2. Add LoRA adapters
    console.print(f"[bold]2. Adding LoRA:[/bold] r={LORA_R}, alpha={LORA_ALPHA}")
    model = FastModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGETS,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    # 3. Setup Gemma 3 chat template
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    # 4. Load and format training data
    console.print("[bold]3. Loading data...[/bold]")
    train_ds = load_data(TRAIN_FILE)
    eval_ds = load_data(EVAL_FILE)

    def formatting_func(examples: dict) -> dict:
        """Apply Gemma 3 chat template to conversations."""
        texts = []
        for convo in examples["conversations"]:
            text = tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            # Unsloth handles BOS insertion — strip if template added it
            texts.append(text.removeprefix("<bos>"))
        return {"text": texts}

    train_ds = train_ds.map(formatting_func, batched=True)
    eval_ds = eval_ds.map(formatting_func, batched=True)

    console.print(f"   Train: {len(train_ds)} examples")
    console.print(f"   Eval:  {len(eval_ds)} examples")
    console.print(f"   Sample:\n   [dim]{train_ds[0]['text'][:300]}[/dim]\n")

    # 5. Setup trainer
    console.print("[bold]4. Configuring trainer...[/bold]")
    steps_per_epoch = len(train_ds) // (BATCH_SIZE * GRAD_ACCUM)
    console.print(
        f"   Effective batch size: {BATCH_SIZE * GRAD_ACCUM} "
        f"({steps_per_epoch} steps/epoch × {EPOCHS} epochs = "
        f"~{steps_per_epoch * EPOCHS} total steps)"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=EPOCHS,
            learning_rate=LR,
            warmup_ratio=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            seed=SEED,
            output_dir=str(OUTPUT_DIR),
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",
            bf16=True,
        ),
    )

    # Train only on model responses, not user prompts
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    # Show masked token stats
    tokenized = tokenizer(train_ds[0]["text"], return_tensors="pt")
    console.print(f"   Tokenized sample length: {tokenized['input_ids'].shape[1]} tokens")

    # 6. Train
    console.print("\n[bold]5. Training...[/bold]\n")
    stats = trainer.train()

    # 7. Save adapter
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # 8. Summary
    console.print(f"\n[bold green]Training complete![/bold green]")
    table = Table(title="Training Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Total steps", str(stats.global_step))
    table.add_row("Train loss", f"{stats.training_loss:.4f}")
    table.add_row("Adapter saved", str(OUTPUT_DIR))
    console.print(table)


if __name__ == "__main__":
    main()
