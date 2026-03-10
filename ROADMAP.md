# Roadmap

## Completed

### Specialist Fine-Tune — 14 Tools, 99.5% Accuracy

Proved that a 270M specialist model (tools baked in weights) beats generalist function callers across 14 tools with zero accuracy degradation as tool count scales.

- [x] Define 14 tool schemas: 8 filesystem + 6 git (2 MCP servers)
- [x] Build extraction-only data generation standard (`docs/generation-standard.md`)
- [x] Generate 15,601 training examples via Claude Code agents (Sonnet 4.6)
- [x] Fine-tune Gemma 3 270M with LoRA r=128 (BF16, 3 epochs, ~2h on RTX 4060)
- [x] Merge adapter + convert to Q8_0 GGUF (291 MB) for Ollama
- [x] Scaling benchmark: 4 models × 4 subsets (3/5/8/14 tools)
- [x] MCP client bridge (JSON → MCP tools/call, 14 tools, 2 servers, arg translation)
- [x] Interactive CLI demo with batch mode and dual-server support

**Result**: 99.5% combined accuracy at 14 tools, 20 tokens/request, 153ms latency. Beats GPT-OSS 120B (41.7%), FunctionGemma (13.6%), and raw Gemma 3 (11.4%). No scaling wall found — the 270M model handles 14 tools as well as 3. See [`docs/benchmark-scaling-results.md`](docs/benchmark-scaling-results.md).

## What's Next

### Agentic Chains
Multi-step tool calling: list → read → decide → act.
- Train on multi-turn tool calling sequences
- Implement agentic loop in demo

### Packaged Edge Product
Drop-in local MCP tool caller, packaged for distribution.
- Ollama model registry publication
- Docker image for edge deployment
- Documentation for custom tool training
- Benchmark on BFCL leaderboard

### Domain-Specific Deployments
Apply the specialist approach to real-world edge scenarios:
- Smart home hub (3-5 tools, offline, $30 hardware)
- Factory floor sensor monitor (ruggedized tablet, no WiFi)
- Medical bedside monitor (HIPAA on-device processing)

See [`docs/real-world-use-cases.md`](docs/real-world-use-cases.md) for deployment scenarios.
