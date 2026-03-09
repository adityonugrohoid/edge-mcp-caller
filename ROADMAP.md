# Roadmap

## v0.1 — MVP: Specialist Beats Generalist (current)

Prove that a specialist 270M model (tools baked in weights) beats FunctionGemma (generalist, schemas in prompt) on MCP filesystem operations.

- [x] Acquire Gemma 3 270M base model from HuggingFace
- [x] Define 3 MCP filesystem tool schemas (list_directory, read_file, search_files)
- [x] Build synthetic data generator (NVIDIA NIM API → 3600 training examples)
- [x] Fine-tune with Unsloth LoRA (standard chat format, JSON output)
- [x] Merge adapter + convert to GGUF for Ollama
- [x] Build eval harness (tool selection accuracy + argument exact match)
- [x] Run FunctionGemma on same eval set
- [x] Head-to-head benchmark report (accuracy, prompt tokens, latency)
- [ ] MCP client wrapper (JSON → MCP tools/call)
- [ ] Interactive CLI demo

**Win condition**: match or beat FunctionGemma 85% accuracy with 40x fewer prompt tokens.

**Result**: 90.8% combined accuracy (temp=0, deterministic) vs GPT-OSS-120B's 23.3%, FunctionGemma's 18.1%, raw Gemma's 13.3% — with 13x fewer prompt tokens (20 vs 264). See [`docs/benchmark-methodology.md`](docs/benchmark-methodology.md).

## v0.2 — Write Operations

Expand tool set to include write operations.

- [ ] Add write_file, create_directory tools (5 total)
- [ ] Generate additional training data for write intent
- [ ] Retrain and benchmark
- [ ] Prove: model handles read vs write intent disambiguation

## v0.3 — Multi-Argument Tools

Add tools with structured, multi-field arguments.

- [ ] Add edit_file(path, old_text, new_text) tool
- [ ] Generate training data with multi-field args
- [ ] Retrain and benchmark
- [ ] Prove: model generates correct multi-field tool calls

## v0.4 — Multi-MCP Server

Expand to a second MCP server (git or memory).

- [ ] Add second MCP server's tools to training set
- [ ] Retrain specialist for combined tool set
- [ ] Benchmark cross-server routing accuracy
- [ ] Prove: specialist can scale to multiple servers

## v0.5 — Agentic Chains

Multi-step tool calling: list → read → decide → act.

- [ ] Train on multi-turn tool calling sequences
- [ ] Implement agentic loop in demo
- [ ] Prove: specialist handles agentic workflows

## v1.0 — Packaged Edge Product

Drop-in local MCP tool caller, packaged for distribution.

- [ ] Ollama model registry publication
- [ ] Docker image for edge deployment
- [ ] Documentation for custom tool training
- [ ] Benchmark on BFCL leaderboard

## Version History
- 2026-03-09: Step 4 complete — 4-model benchmark (temp=0): 90.8% specialist vs 23.3% GPT-OSS-120B vs 18.1% FunctionGemma vs 13.3% raw Gemma
- 2026-03-09: Step 3 complete — merged LoRA + GGUF Q8_0 (272 MB), registered in Ollama
- 2026-03-09: Step 2 complete — fine-tuned Gemma 3 270M-IT with LoRA r=128 (3 epochs, 609 steps, 55 min)
- 2026-03-09: Step 1 complete — 3600 clean training examples generated (1200/tool)
- 2026-03-09: v0.1 scaffolded, concept finalized
