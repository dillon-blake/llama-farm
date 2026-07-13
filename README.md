# learning-llamas

Train LoRA adapters (SFT / DPO / GRPO) on **frozen, quantized GGUF models**,
locally, on llama.cpp's ggml backend — CPU, Metal, CUDA, and Vulkan — with the
training step GPU-resident when a GPU is available. Trained adapters load
directly in stock llama.cpp, llama-server, and ollama.

**Status: stage 1 (CPU training core), 27 of 37 tickets landed or in review.**
SFT, DPO and GRPO all train today on CPU, on quantized bases, with gradient
checkpointing and a chunked lm_head. What remains is MoE (S1-26, S1-28), SSM
(S1-30, S1-31), the convergence gate (S1-12), and chunked attention (S1-24).

The **flash-attention backward** family (S1-21/22/23) is **deferred out of stage 1**:
flash attention is force-disabled during training and none of those tickets turns it
back on, so the whole family — six to eight weeks — would leave the training path
unchanged. Its real product is a CPU oracle for *GPU* flash-attention kernels, which
belongs with whichever stage first starts a GPU backend. See the tickets for the
reasoning. Stages 2–4 (Metal, CUDA, Vulkan) have not started.

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

Then fine-tune something — four steps, and the shape never changes:

```python
from learning_llamas import Model, create_zero_adapter, libraries, read_adapter, save_adapter
from learning_llamas.train import SFTConfig, train_sft

create_zero_adapter("base.gguf", "adapter.gguf", r=16)      # a provable no-op at step 0

with Model("base.gguf", n_ctx=512, n_ubatch=512, training=True) as model:
    model.attach_adapter("adapter.gguf")
    train_sft(libraries(), model, samples, SFTConfig(lr=1e-4, seq_len=512))

    info = read_adapter("adapter.gguf")
    save_adapter(libraries(), model.adapter, "trained.gguf",
                 architecture=info.architecture, alpha=info.alpha)
```

```bash
llama-cli -m base.gguf --lora trained.gguf -p "..."     # no merge step required
```

**[`docs/quickstart.md`](docs/quickstart.md)** is the full version — where `samples`
comes from, GRPO, and the three things that will bite you.

For working on the library itself, see [`docs/dev/`](docs/dev/) — the
[build guide](docs/dev/building.md), the [testing guide](docs/dev/testing.md), the
[backward coverage](docs/dev/backward-coverage.md) table, the
[carried llama.cpp changes](docs/dev/fork-changes.md), and the per-backend
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
adversarially fact-checked; ticket citations were re-verified independently. Where
implementation has since proved one of them wrong, the document says so at the
point of the claim rather than being quietly patched — the corrections are worth
more than the appearance of having been right.

**The first such correction, now fixed.** The docs asserted that training graphs
bypass the KV cache. They did not: causal-arch training routed K/V through the
cache, which severs the autodiff edge and made `ggml_build_backward_expand` abort
(`ggml.c:7093`) — upstream's own `llama-finetune` aborts before printing a loss.
That became BLUEPRINT gap **G16** and ticket **S1-00**, the root of the stage-1
dependency graph, and it is what the training-mode graph bypass now establishes.
Training works.

The llama.cpp changes this carries — and which of them belong upstream — are
inventoried in [`docs/dev/fork-changes.md`](docs/dev/fork-changes.md).
