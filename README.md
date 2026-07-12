# learning-llamas

Train LoRA adapters (SFT / DPO / GRPO) on **frozen, quantized GGUF models**,
locally, on llama.cpp's ggml backend — CPU, Metal, CUDA, and Vulkan — with the
training step GPU-resident when a GPU is available. Trained adapters load
directly in stock llama.cpp, llama-server, and ollama.

**Status: stage 0 (groundwork) landing.** The build, the bindings, the adapter
format, and the test harness exist; the training core is stage 1.

## Getting started

```bash
git clone --recurse-submodules https://github.com/dillon-blake/llama-farm.git
cd llama-farm
python3 -m venv .venv && source .venv/bin/activate
pip install scikit-build-core cmake ninja pytest numpy
pip install -e . --no-build-isolation        # builds vendored llama.cpp + the C shim
pip install -e vendor/llama.cpp/gguf-py
pytest tests/ -m "not slow"
```

See [`docs/dev/`](docs/dev/) — the [build guide](docs/dev/building.md), the
[testing guide](docs/dev/testing.md), and the per-backend
[VM playbooks](docs/dev/vm-playbooks.md).

## The plan

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
- [`docs/adr/`](docs/adr/) — the decisions that bind every kernel PR:
  [ADR-0001](docs/adr/ADR-0001-vendor-lineage.md) (vendor lineage, rebase
  cadence, two-repo flow) and
  [ADR-0002](docs/adr/ADR-0002-numerics-determinism-parity.md) (F32 gradient
  accumulation, determinism by default, the parity criterion).
- [`docs/PROVENANCE.md`](docs/PROVENANCE.md) — what may be copied from where.

Both design documents are grounded in llama.cpp @ `4f37f51` and were
adversarially fact-checked; ticket citations were re-verified independently.

**One correction since:** the docs asserted that training graphs bypass the KV
cache. They do not — causal-arch training routes K/V through the cache, which
severs the autodiff edge and makes `ggml_build_backward_expand` abort
(`ggml.c:7093`). Upstream's own `llama-finetune` aborts before printing a loss.
This is now BLUEPRINT gap **G16** and ticket **S1-00**, a root of the stage-1
dependency graph; five tickets that had stated the no-KV-cache property as fact
now cite S1-00 as what establishes it.
