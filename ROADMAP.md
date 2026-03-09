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
- [x] MCP client wrapper (JSON → MCP tools/call)
- [x] Interactive CLI demo

**Win condition**: match or beat FunctionGemma 85% accuracy with 40x fewer prompt tokens.

**Result**: 90.8% combined accuracy (temp=0, deterministic) vs GPT-OSS-120B's 23.3%, FunctionGemma's 18.1%, raw Gemma's 13.3% — with 13x fewer prompt tokens (20 vs 264). Full 360-query MCP pipeline run: 100% MCP execution success, 263ms avg e2e latency, 0 JSON parse errors. See [`docs/benchmark-methodology.md`](docs/benchmark-methodology.md).

## v0.2 — Write Operations

Expand tool set to include write operations. Prove read vs write intent disambiguation.

- [x] Define write_file(path, content) and create_directory(path) tool schemas
- [x] Switch data generation to Claude Code agents (Sonnet 4.6, orchestrated by Opus 4.6)
- [x] Implement accumulate-then-merge data pipeline (timestamped batches in `data/generated/`, no overwrites)
- [x] Generate training data across 3 categories:
  - [x] Clean well-formed queries (65.8% — 6038 examples)
  - [x] Messy real-world queries (22.8% — 2096 examples): typos, slang, abbreviations, grammar errors, voice transcription, filler/hedging, self-corrections
  - [x] Adversarial disambiguation (11.3% — 1040 examples): read vs write confusion pairs
- [x] Build `data/merge_dataset.py` — merge all `data/generated/` → train/eval split with dedup + validation (11 checks)
- [x] Tag eval examples by category for robustness breakdown reporting
- [x] Update `eval/benchmark.py` for 5 tools + per-category accuracy reporting
- [x] Retrain specialist on 5-tool dataset (8255 train examples)
- [x] Benchmark: overall accuracy + per-category breakdown (clean / messy / disambiguation)
- [x] Prove: model handles read vs write intent disambiguation AND noisy real-world queries

**Result**: 95.3% tool routing accuracy across 919 eval examples (5 tools, 3 categories). 48.6% strict exact-match — bottlenecked by path ambiguity (329/429 args failures are path mismatches where model generates plausible alternative paths, not trailing-slash issues). create_directory 100% tool routing. Per-category tool accuracy: clean 94.4%, messy 97.6%, adversarial 96.2% — model handles noisy queries and read/write disambiguation well. 22 avg prompt tokens, 341ms avg latency.

## v0.3 — Full Filesystem + Unified Generation + Scaling Experiment

Complete filesystem server (8 tools). Add Git MCP server schemas (6 tools). Design unified generation standard for all 14 tools. Generate data once, train once, benchmark at each tool count to find 270M breaking point.

### Infrastructure
- [ ] Archive v0.2 generated data (polluted — invented paths, generated content)
- [ ] Add edit_file(path, old_text, new_text), move_file(source, destination), directory_tree(path) to filesystem schemas
- [ ] Define 6 git tool schemas in `tools/git.json` (git_status, git_diff_staged, git_commit, git_log, git_branch, git_create_branch)
- [ ] Create unified generation standard (`docs/generation-standard.md`) for all 14 tools
- [ ] Update merge_dataset.py validation for 14 tools
- [ ] Update MCP client bridge for edit_file translation (simplified → MCP edits[] format)

### Data + Training
- [ ] Generate fresh training data for ALL 14 tools (~1000/tool, even distribution)
- [ ] Categories: 70% clean, 15% messy, 15% disambiguation — all with extractable args only
- [ ] Retrain specialist on full 14-tool dataset

### Scaling Benchmark
- [ ] Eval at 3-tool subset (v0.1 tools: list_directory, read_file, search_files)
- [ ] Eval at 5-tool subset (v0.2 tools: + write_file, create_directory)
- [ ] Eval at 8-tool subset (full filesystem: + edit_file, move_file, directory_tree)
- [ ] Eval at 14-tool subset (filesystem + git)
- [ ] Report: scaling curve — where does 270M accuracy degrade?
- [ ] Prove: multi-field extraction + similar-tool disambiguation + cross-server routing

## v0.4 — Git MCP Server Integration

Wire up the Git MCP server end-to-end. Model already trained on git tools from v0.3 data.

- [ ] Implement Git MCP client bridge (repo_path injection from config)
- [ ] End-to-end demo: git queries → specialist → Git MCP server
- [ ] Live pipeline benchmark (like v0.1 CLI batch run)
- [ ] Prove: specialist routes across 2 MCP servers with 14 tools in production

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
- 2026-03-10: v0.3 infrastructure — archived v0.2 data, defined all 14 tools (8 filesystem + 6 git), created unified generation standard, scaling experiment design
- 2026-03-09: v0.2 benchmark complete — 95.3% tool routing (919 eval, 5 tools), 48.6% strict match (path ambiguity bottleneck); per-category tool acc: clean 94.4%, messy 97.6%, adversarial 96.2%
- 2026-03-09: v0.2 training complete — fresh retrain on 8255 examples (5 tools), 1548 steps, 2h 10m; GGUF tokenizer fix (Unsloth strips added_tokens_decoder); 7/7 sanity tests pass
- 2026-03-09: v0.2 data generation complete — 9174 examples (5 tools, 3 categories), 14 agent runs, 0 invalid
- 2026-03-09: v0.2 pipeline infra — merge_dataset.py, updated benchmark.py for per-category reporting, tool schemas for write_file/create_directory
- 2026-03-09: Step 5 complete — MCP client bridge + interactive CLI demo, v0.1 MVP complete
- 2026-03-09: Step 4 complete — 4-model benchmark (temp=0): 90.8% specialist vs 23.3% GPT-OSS-120B vs 18.1% FunctionGemma vs 13.3% raw Gemma
- 2026-03-09: Step 3 complete — merged LoRA + GGUF Q8_0 (272 MB), registered in Ollama
- 2026-03-09: Step 2 complete — fine-tuned Gemma 3 270M-IT with LoRA r=128 (3 epochs, 609 steps, 55 min)
- 2026-03-09: Step 1 complete — 3600 clean training examples generated (1200/tool)
- 2026-03-09: v0.1 scaffolded, concept finalized
