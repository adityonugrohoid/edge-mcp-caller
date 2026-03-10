# Competitive Landscape: Small Model Specialist Tool Calling

## Our Position

Edge MCP Caller: 270M parameter specialist, 14 tools, 99.5% combined accuracy, 20 tokens/request (zero schema), MCP-native.

**No other project combines all three: sub-1B + zero schema tokens + MCP integration.**

---

## Direct Competitors

### Octopus v2 (Nexa AI, April 2024)
- **Size**: 2B (Gemma 2B base) — 7.4x larger than us
- **Approach**: "Functional tokens" (`<nexa_0>` to `<nexa_N>`) baked into vocabulary via fine-tuning. Zero schema tokens at inference.
- **Results**: 99.5% accuracy, 0.38s latency. Surpasses GPT-4 in accuracy and latency. 36x faster than Llama-7B + RAG.
- **Difference**: Same accuracy but 7.4x larger, Android-specific APIs (not MCP), uses special tokens instead of standard JSON output.
- **Links**: [Paper](https://arxiv.org/abs/2404.01744), [HuggingFace](https://huggingface.co/NexaAI/Octopus-v2)

### FunctionGemma (Google, December 2025)
- **Size**: 270M (same base — Gemma 3 270M)
- **Approach**: Generalist function-calling foundation model. Designed to be further fine-tuned. Still passes tool schemas in prompt.
- **Results**: Baseline 58%, fine-tuned to 85% on Google Mobile Actions benchmark. We benchmarked it at 13.6% on our 14-tool eval.
- **Difference**: Generalist (schemas in prompt) vs our specialist (tools in weights). Same model size, wildly different accuracy (13.6% vs 99.5% on our eval).
- **Links**: [Blog](https://blog.google/technology/developers/functiongemma/), [HuggingFace](https://huggingface.co/google/functiongemma-270m-it)

### TinyAgent (UC Berkeley, EMNLP 2024)
- **Size**: 1.1B / 7B
- **Approach**: Fine-tuned small models + ToolRAG (retrieval to reduce prompt length). Still passes some tool context.
- **Results**: 1.1B: 80.06% success rate. 7B: 84.95%. Both surpass GPT-4-Turbo (79.08%).
- **Difference**: 4x larger at 1.1B, still uses tool retrieval (doesn't eliminate schema tokens), lower accuracy (80% vs 99.5%), no MCP.
- **Links**: [Paper](https://arxiv.org/abs/2409.00608), [BAIR Blog](https://bair.berkeley.edu/blog/2024/05/29/tiny-agent/), [GitHub](https://github.com/SqueezeAILab/TinyAgent)

### Hammer (MadeAgents, ICLR 2025 Spotlight)
- **Size**: 0.5B to 7B (Qwen 2.5 Coder series)
- **Approach**: "Function masking" during training — masks irrelevant schemas for robustness. Still passes schemas at inference.
- **Results**: 0.5B achieves ~68% on single-turn function calling (Monday8am benchmarks).
- **Difference**: Close in size (0.5B), but generalist (schemas in prompt). 31pp below our accuracy. Trained on APIGen 60K.
- **Links**: [GitHub](https://github.com/MadeAgents/Hammer), [HuggingFace 0.5B](https://huggingface.co/MadeAgents/Hammer2.1-0.5b), [Paper](https://openreview.net/pdf?id=yVQcr4qjD6)

### xLAM-1B (Salesforce AI Research, July 2024)
- **Size**: 1B (DeepSeek-Coder base)
- **Approach**: Generalist "Large Action Model" fine-tuned on APIGen 60K dataset. Schemas in prompt.
- **Results**: 78.94% on BFCL. Only sub-2B model on BFCL leaderboard at release. Surpasses GPT-3.5-Turbo.
- **Difference**: 3.7x larger, generalist, still requires 647+ schema tokens. 20.5pp below our accuracy.
- **Links**: [HuggingFace](https://huggingface.co/Salesforce/xLAM-1b-fc-r), [Blog](https://www.salesforce.com/blog/xlam-large-action-models/)

### Juniper (RidgeRun.ai, 2025)
- **Size**: 2B (Gemma-2-2B, 4-bit quantized)
- **Approach**: Specialist fine-tuning for embedded/edge function calling. Domain-specific JSON datasets.
- **Results**: Beat GPT-4o in function precision on their benchmark.
- **Difference**: Similar thesis (specialist for edge), but 7.4x larger. Targets embedded systems, not MCP.
- **Links**: [Blog](https://www.ridgerun.ai/post/introducing-juniper-fine-tuned-small-local-model-for-function-calling)

### Mercedes-Benz In-Vehicle (NeurIPS 2024/2025)
- **Size**: ~3.8B (structured pruning of Phi-3 mini)
- **Approach**: Specialized tokens mapped to gRPC vehicle functions. Baked into weights. Deployed via llama.cpp on automotive hardware.
- **Results**: 11 tokens/second without hardware acceleration on automotive chips.
- **Difference**: Very similar thesis (zero schema, specialist, edge). But starts from 3.8B and prunes down, vs our 270M from scratch. No MCP.
- **Links**: [Paper](https://arxiv.org/abs/2501.02342)

---

## Summary Comparison Table

| Project | Size | Schema Tokens | Accuracy | MCP | Specialist |
|---------|------|--------------|----------|-----|-----------|
| **Edge MCP Caller (ours)** | **270M** | **0** | **99.5%** | **Yes** | **Yes** |
| Octopus v2 | 2B | 0 | 99.5% | No | Yes |
| FunctionGemma | 270M | In prompt | 85%/13.6%* | No | No |
| TinyAgent | 1.1B | Reduced | 80% | No | Partial |
| Hammer 0.5B | 0.5B | In prompt | ~68% | No | No |
| xLAM-1B | 1B | In prompt | 79% | No | No |
| Juniper | 2B | Specialist | Beat GPT-4o | No | Yes |
| Mercedes in-vehicle | ~3.8B | 0 | N/A | No | Yes |

*FunctionGemma: 85% on Google's benchmark, 13.6% on our 14-tool eval.

---

## Academic Research

### TinyLLM Study (November 2025)
- Systematic evaluation of sub-1B models on BFCL with multiple optimization strategies.
- **Key finding**: Sub-1B generalist models top out at ~65.74% accuracy. Our 99.5% at 270M breaks this ceiling via specialist fine-tuning.
- [Paper](https://arxiv.org/abs/2511.22138)

### "Less is More" (DATE 2025)
- Reducing visible tools improves accuracy, execution time (up to 70%), and power consumption (up to 40%).
- We go further: eliminate schemas entirely.
- [Paper](https://arxiv.org/abs/2411.15399)

### APIGen (Salesforce, NeurIPS 2024)
- Automated pipeline for generating verified function-calling datasets. 60K entries, 3,673 APIs.
- Validates our focus on data quality (extraction-only rules, validation pipeline).
- [Paper](https://arxiv.org/abs/2406.18518)

### ToolLLM / ToolBench (OpenBMB, ICLR 2024 Spotlight)
- Framework for 16,464 real-world APIs. Generalist approach (opposite of ours).
- [Paper](https://arxiv.org/abs/2307.16789)

### When2Call (Harvard/NVIDIA, NAACL 2025)
- Benchmark for when LLMs should/shouldn't call tools. Complementary problem to ours.
- [Paper](https://aclanthology.org/2025.naacl-long.174/)

---

## Industry Validators

### AWS Semantic MCP Server
- Fine-tuned SLMs deployed on AWS Outposts for telco operations, exposed via MCP.
- Production enterprise validation of our thesis at scale.
- [Blog](https://aws.amazon.com/blogs/industries/architecting-the-semantic-mcp-server-edge-deployment-of-fine-tuned-slms-to-solve-the-data-ingestion-problem-for-telco-operations/)

### Qualcomm AI Hub
- Published on MCP for edge devices. Infrastructure ready for models like ours.
- [Blog](https://www.qualcomm.com/developer/blog/2025/10/how-mcp-simplifies-tool-integration-across-cloud-edge-real-world-devices)

### Microsoft Phi Series
- Explicitly acknowledges baking function definitions into weights reduces latency.
- Smallest function-calling model (Phi-4-mini) is 3.8B — 14x larger than us.
- [Blog](https://techcommunity.microsoft.com/blog/educatordeveloperblog/function-calling-with-small-language-models/4472720)

### Google AI Edge
- Full stack: FunctionGemma + AI Edge FC SDK + LiteRT-LM + Gallery app.
- Most comprehensive edge FC ecosystem, but still uses schemas in prompt.
- [Blog](https://developers.googleblog.com/on-device-function-calling-in-google-ai-edge-gallery/)

### Block/Goose Toolshim
- Fine-tuned model as MCP translator layer. Different architecture (post-processor vs direct caller).
- [Blog](https://block.github.io/goose/blog/2025/04/11/finetuning-toolshim/)

### LLMWare SLIM Models
- 1-3B specialist models for classification tasks. Same philosophy (single-purpose, baked into weights, CPU).
- Not for tool routing, but validates the specialist model pattern.
- [Blog](https://medium.com/@darrenoberst/slims-small-specialized-models-function-calling-and-multi-model-agents-8c935b341398)

### Monday8am EdgeLab
- Android testing platform for SLM agents. Benchmarked sub-1B generalists at ~68%.
- Validates that our specialist approach breaks the sub-1B ceiling.
- [Blog](https://monday8am.com/blog/2025/12/10/function-calling-edge-ai.html)

---

## Key Datasets in the Space

| Dataset | Size | Source |
|---------|------|--------|
| xLAM/APIGen 60K | 60K examples, 3,673 APIs | Salesforce |
| Glaive Function Calling v2 | ~113K examples | Glaive AI |
| Hermes Function Calling v1 | N/A | NousResearch |
| ToolBench | 16,464 APIs | OpenBMB |
| ToolMind | 360K samples, 20K functions | Nanbeige |
| **Edge MCP Caller (ours)** | **15,601 examples, 14 tools** | **Claude Code agents** |

---

## Our Differentiators (Messaging)

1. **Smallest specialist with near-perfect accuracy** — 270M, 99.5%. Next closest: Octopus v2 at 2B (7.4x larger).
2. **Zero schema tokens** — 20 tokens/request vs 647-1,092 for schema-based approaches. Only Octopus v2 and Mercedes-Benz share this, both at much larger sizes.
3. **MCP-native** — first sub-1B model with MCP integration. Standardized protocol, not proprietary APIs.
4. **Specialist breaks the sub-1B ceiling** — academic studies show generalist sub-1B models top out at ~65-68%. We achieve 99.5% by being a specialist.
5. **Data quality over data quantity** — 15K extraction-only examples vs 60-360K generalist datasets. Extraction-only rules prevent hallucinated arguments.
