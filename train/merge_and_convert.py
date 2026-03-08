#!/usr/bin/env python3
"""Merge LoRA adapter into base model and convert to GGUF for Ollama.

Loads the fine-tuned adapter from models/adapter/, merges it with the base
Gemma 3 270M-IT model, saves merged HF model, converts to GGUF via
llama.cpp's convert script, quantizes, and registers with Ollama.

Uses llama.cpp tools directly instead of Unsloth's built-in GGUF export
to avoid system package dependencies (libcurl, libssl).
"""

import json
import subprocess
import sys
from pathlib import Path

import torch
import safetensors.torch as st
from huggingface_hub import hf_hub_download
from rich.console import Console

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = PROJECT_ROOT / "models" / "adapter"
MERGED_DIR = PROJECT_ROOT / "models" / "merged"
GGUF_DIR = PROJECT_ROOT / "models" / "gguf"
MODELFILE_PATH = PROJECT_ROOT / "models" / "Modelfile"

LLAMA_CPP_DIR = Path.home() / ".unsloth" / "llama.cpp"
CONVERTER_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
QUANTIZER_BIN = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"

BASE_MODEL = "unsloth/gemma-3-270m-it"
OLLAMA_MODEL_NAME = "edge-mcp-caller"
QUANTIZATION = "Q8_0"  # High quality for tiny model

# Gemma 3 Ollama template — copied from `ollama show gemma3:270m --modelfile`
OLLAMA_TEMPLATE = """{{- $systemPromptAdded := false }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 }}
{{- if eq .Role "user" }}<start_of_turn>user
{{- if (and (not $systemPromptAdded) $.System) }}
{{- $systemPromptAdded = true }}
{{ $.System }}
{{ end }}
{{ .Content }}<end_of_turn>
{{ if $last }}<start_of_turn>model
{{ end }}
{{- else if eq .Role "assistant" }}<start_of_turn>model
{{ .Content }}{{ if not $last }}<end_of_turn>
{{ end }}
{{- end }}
{{- end }}"""

console = Console()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    console.print("[bold cyan]Edge MCP Caller — Merge & Convert (Step 3)[/bold cyan]\n")

    # Verify llama.cpp tools exist
    if not CONVERTER_SCRIPT.exists():
        console.print(f"[red]Missing: {CONVERTER_SCRIPT}[/red]")
        console.print("[red]Run: git clone https://github.com/ggml-org/llama.cpp ~/.unsloth/llama.cpp[/red]")
        sys.exit(1)
    if not QUANTIZER_BIN.exists():
        console.print(f"[red]Missing: {QUANTIZER_BIN}[/red]")
        console.print("[red]Build llama.cpp first (see README)[/red]")
        sys.exit(1)

    # 1. Load base model + LoRA adapter and save merged
    console.print("[bold]1. Loading base model + adapter...[/bold]")

    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=str(ADAPTER_DIR),
        max_seq_length=512,
        load_in_4bit=False,
    )

    console.print("[bold]2. Saving merged model (HuggingFace format)...[/bold]")
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(
        str(MERGED_DIR),
        tokenizer,
        save_method="merged_16bit",
    )
    console.print(f"   Merged model saved: {MERGED_DIR}\n")

    # 3. Fix tokenizer and vocab (Unsloth merge drops tokenizer.model)
    console.print("[bold]3. Restoring original tokenizer files...[/bold]")
    for fname in ["tokenizer.model", "special_tokens_map.json", "added_tokens.json"]:
        hf_hub_download(BASE_MODEL, fname, local_dir=str(MERGED_DIR))
    console.print("   Downloaded tokenizer.model + special_tokens_map.json\n")

    # Fix vocab size mismatch: tokenizer has 262145 tokens, embedding has 262144
    console.print("[bold]4. Fixing vocab size mismatch...[/bold]")
    config_path = MERGED_DIR / "config.json"
    config = json.loads(config_path.read_text())
    tensors = st.load_file(str(MERGED_DIR / "model.safetensors"))
    embed = tensors["model.embed_tokens.weight"]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MERGED_DIR))
    tok_size = max(tok.get_vocab().values()) + 1

    if embed.shape[0] < tok_size:
        pad_rows = tok_size - embed.shape[0]
        console.print(f"   Padding embedding: {embed.shape[0]} → {tok_size} (+{pad_rows} rows)")
        pad = torch.zeros(pad_rows, embed.shape[1], dtype=embed.dtype)
        tensors["model.embed_tokens.weight"] = torch.cat([embed, pad], dim=0)
        st.save_file(tensors, str(MERGED_DIR / "model.safetensors"))
        config["vocab_size"] = tok_size
        config_path.write_text(json.dumps(config, indent=2))
    else:
        console.print("   Vocab sizes match, no fix needed.")
    console.print()

    # 5. Convert HF → GGUF (BF16)
    console.print("[bold]5. Converting HF → GGUF (BF16)...[/bold]")
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    bf16_gguf = GGUF_DIR / f"{OLLAMA_MODEL_NAME}-bf16.gguf"

    result = subprocess.run(
        [
            sys.executable,
            str(CONVERTER_SCRIPT),
            str(MERGED_DIR),
            "--outfile", str(bf16_gguf),
            "--outtype", "bf16",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        console.print(f"[red]Conversion failed:\n{result.stderr}[/red]")
        sys.exit(1)
    size_mb = bf16_gguf.stat().st_size / (1024 * 1024)
    console.print(f"   BF16 GGUF: {bf16_gguf} ({size_mb:.0f} MB)\n")

    # 6. Quantize BF16 → Q8_0
    console.print(f"[bold]6. Quantizing BF16 → {QUANTIZATION}...[/bold]")
    quantized_gguf = GGUF_DIR / f"{OLLAMA_MODEL_NAME}-{QUANTIZATION.lower()}.gguf"

    result = subprocess.run(
        [
            str(QUANTIZER_BIN),
            str(bf16_gguf),
            str(quantized_gguf),
            QUANTIZATION,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        console.print(f"[red]Quantization failed:\n{result.stderr}[/red]")
        sys.exit(1)
    size_mb = quantized_gguf.stat().st_size / (1024 * 1024)
    console.print(f"   Quantized GGUF: {quantized_gguf} ({size_mb:.0f} MB)\n")

    # Clean up BF16 intermediate
    bf16_gguf.unlink()
    console.print("   Cleaned up BF16 intermediate.\n")

    # 7. Generate Modelfile
    console.print("[bold]7. Generating Modelfile...[/bold]")

    modelfile_content = f"""FROM {quantized_gguf}
TEMPLATE \"\"\"{OLLAMA_TEMPLATE}\"\"\"
PARAMETER stop <end_of_turn>
PARAMETER top_p 0.95
PARAMETER top_k 64
"""

    MODELFILE_PATH.write_text(modelfile_content)
    console.print(f"   Modelfile saved: {MODELFILE_PATH}\n")

    # 8. Register with Ollama
    console.print(f"[bold]8. Registering with Ollama as '{OLLAMA_MODEL_NAME}'...[/bold]")
    result = subprocess.run(
        ["ollama", "create", OLLAMA_MODEL_NAME, "-f", str(MODELFILE_PATH)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        console.print(f"[red]Ollama create failed: {result.stderr}[/red]")
        sys.exit(1)

    console.print(f"   {result.stdout.strip()}\n")

    # 9. Verify
    console.print("[bold]9. Verifying...[/bold]")
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if OLLAMA_MODEL_NAME in line or "NAME" in line:
            console.print(f"   {line}")

    console.print(f"\n[bold green]Done! Run with: ollama run {OLLAMA_MODEL_NAME}[/bold green]")


if __name__ == "__main__":
    main()
