# Developer documentation

This is the documentation for working **on** the library. To work **with** it, start at
[`../quickstart.md`](../quickstart.md).

| Document | What it covers |
|---|---|
| [`building.md`](building.md) | Prerequisites, the editable install, per-backend CMake flags, building the vendored test binaries |
| [`testing.md`](testing.md) | The pytest suite and its markers; `test-backend-ops` including MODE_GRAD; where the tolerances come from |
| [`vm-playbooks.md`](vm-playbooks.md) | Copy-pasteable, idempotent provisioning for each backend's VM, plus self-hosted GitHub Actions runners |
| [`backward-coverage.md`](backward-coverage.md) | **Measured** MODE_GRAD coverage per op — which gradients are actually checked today, which abort, and which were wrong |
| [`fork-changes.md`](fork-changes.md) | Every change this project carries in its vendored llama.cpp, why, and which belong upstream — the map a rebase needs |
| [`gradient-checkpointing.md`](gradient-checkpointing.md) | Trading recompute for activation memory: the segmented backward, and what it costs (S1-17) |
| [`grpo.md`](grpo.md) | The GRPO update, and the several ways one can look like it is training when it is not (S1-15, S1-16) |
| [`cpu-throughput.md`](cpu-throughput.md) | What one CPU training step actually costs, measured (S1-32) |

Two architecture decisions bind every kernel PR. Read them before writing one:

- [`../adr/ADR-0001-vendor-lineage.md`](../adr/ADR-0001-vendor-lineage.md) — the pinned llama.cpp
  commit, the rebase cadence, and the two-repo PR flow.
- [`../adr/ADR-0002-numerics-determinism-parity.md`](../adr/ADR-0002-numerics-determinism-parity.md)
  — F32 gradient accumulation, determinism by default, and the parity criterion your kernel is
  measured against.

The work itself is planned in [`../../tickets/`](../../tickets/); start with
[`tickets/README.md`](../../tickets/README.md).

## Shipping a trained adapter (S1-08)

Two artefacts come out of a training run, and they are for different audiences.

**The adapter** is what you keep. `learning_llamas.adapter.save_adapter` reads the live adapter out
of the training context and writes a GGUF that stock llama.cpp loads:

```python
from learning_llamas.adapter import save_adapter

save_adapter(libs, model.adapter, "out.gguf", architecture="llama", alpha=float(rank))
```

It round-trips bit for bit — what goes in comes back out — and `llama-cli --lora out.gguf` reads it.
There is no CLI for this, because it operates on a *live* adapter; a command-line version could only
copy a file.

**The merged model** is what you ship. It folds the delta back into the weights, so there is one
file, no adapter, and no extra matmuls per adapted layer:

```
python -m learning_llamas.export BASE.gguf ADAPTER.gguf MERGED.gguf [--scale 1.0]
```

The merged tensors go back to the base tensor's **original type**, so merging a Q4_K model gives a
Q4_K model of the same size that runs on the same hardware. That needs libggml's own quantizer —
gguf-py can dequantize every type but can only *write* F32, F16, Q8_0 and the legacy Q4_0 family, so
it cannot round-trip a K-quant at all. See `learning_llamas/quant.py`.

The IQ quants are refused rather than approximated: they require an importance matrix computed from
calibration data, and there is nothing honest to substitute for one. Export to F16 and run
`llama-quantize` yourself with your own imatrix.


## Before you push

Both of these run in CI, and both are cheap to run first:

```
ruff format src/ tests/ && ruff check src/ tests/
clang-format -i csrc/*.cpp csrc/*.h
pytest -q
```

`clang-format` is **pinned to 19.1.7** in both `pyproject.toml`'s dev extra and
`.pre-commit-config.yaml`. Its output changes between major versions, so formatting `csrc/` with
whatever your distro ships will produce a diff CI rejects. `pip install -e .[dev]` gets the right one.
