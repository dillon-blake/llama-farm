# llama-farm

Train LoRA adapters (SFT / DPO / GRPO) on **frozen, quantized GGUF models**,
locally, on llama.cpp's ggml backend — CPU, Metal, CUDA, and Vulkan — with the
training step GPU-resident when a GPU is available. Trained adapters load
directly in stock llama.cpp, llama-server, and ollama.

**Status: planning complete, implementation starting.** This repository
currently contains the full engineering plan:

- [`PLAN.md`](PLAN.md) — the implementation plan overview (stages, milestones,
  effort, risks).
- [`tickets/`](tickets/) — the complete backlog: **84 one-PR tickets** across
  five stages (CPU groundwork → Metal → CUDA → Vulkan, plus a trigger-gated
  backlog). Start with [`tickets/README.md`](tickets/README.md) — the guide for
  agents picking up tickets (ordering, claim protocol, definition of done, CI).
- [`docs/GGUF-LORA-TRAINING-BLUEPRINT.md`](docs/GGUF-LORA-TRAINING-BLUEPRINT.md)
  — library architecture blueprint (design decisions, gap analysis, methods).
- [`docs/KERNEL-ROADMAP.md`](docs/KERNEL-ROADMAP.md) — per-backend kernel plan
  (OUT_PROD, sparse CE, flash-attention backward, MoE/SSM, licensing audit).

Both design documents are grounded in llama.cpp @ `4f37f51` and were
adversarially fact-checked; ticket citations were re-verified independently.

**One correction since:** the docs asserted that training graphs bypass the KV
cache. They do not — causal-arch training routes K/V through the cache, which
severs the autodiff edge and makes `ggml_build_backward_expand` abort
(`ggml.c:7093`). Upstream's own `llama-finetune` aborts before printing a loss.
This is now BLUEPRINT gap **G16** and ticket **S1-00**, a root of the stage-1
dependency graph; five tickets that had stated the no-KV-cache property as fact
now cite S1-00 as what establishes it.
