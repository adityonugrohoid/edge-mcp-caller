# Roadmap

## v0.1 — MVP: Specialist Beats Generalist (current)

Prove that a specialist 270M model (tools baked in weights) beats FunctionGemma (generalist, schemas in prompt) on MCP filesystem operations.

- [ ] Acquire Gemma 3 270M base model from HuggingFace
- [ ] Define 3 MCP filesystem tool schemas (list_directory, read_file, search_files)
- [ ] Build synthetic data generator (Claude API → 3-5K training examples)
- [ ] Fine-tune with Unsloth LoRA (standard chat format, JSON output)
- [ ] Merge adapter + convert to GGUF for Ollama
- [ ] Build eval harness (tool selection accuracy + argument exact match)
- [ ] Run FunctionGemma (base + fine-tuned) on same eval set
- [ ] Head-to-head benchmark report (accuracy, prompt tokens, latency)
- [ ] MCP client wrapper (JSON → MCP tools/call)
- [ ] Interactive CLI demo
- [ ] Benchmark results page (HTML)

**Win condition**: match or beat FunctionGemma 85% accuracy with 40x fewer prompt tokens.

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
- 2026-03-09: v0.1 scaffolded, concept finalized
