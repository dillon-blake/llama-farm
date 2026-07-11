# llama-farm — Implementation Plan

**llama-farm** is a Python library for training LoRA adapters (SFT, DPO, GRPO) on
**frozen, quantized GGUF models** using llama.cpp's ggml backend. Base weights
stay quantized and mmap'd; only F32 LoRA A/B tensors train; the whole training
step (forward + backward + AdamW) runs **GPU-resident when a GPU is available**
on four backends — **CPU, Metal, CUDA, Vulkan** — with CPU as both a first-class
target and the correctness oracle for every kernel.

This plan turns two adversarially fact-checked design documents into an
agent-executable backlog:

- `docs/GGUF-LORA-TRAINING-BLUEPRINT.md` — library architecture (4 layers:
  vendored llama.cpp → C shim `libllamafarm` → ctypes binding → Python library),
  design decisions D1–D7, gap analysis G1–G15.
- `docs/KERNEL-ROADMAP.md` — the per-backend kernel plan (OUT_PROD, sparse CE,
  flash-attention backward, MoE and SSM ops), numerics/determinism policy,
  licensing audit of reusable unsloth ideas.

The backlog itself lives in `tickets/` — **83 tickets, each scoped to one pull
request**, with `tickets/README.md` as the guide agents read before picking up
work (ordering, claim protocol, definition of done, CI model).

## Staging (project decision: CPU → Metal → CUDA → Vulkan)

All groundwork and correctness work lands on CPU first; GPU backends follow in
the order Metal, then CUDA, then Vulkan. Each stage is developed and tested on a
backend-appropriate VM, and GitHub Actions runs detailed per-backend suites on
every PR (quick lane) plus nightly (full lane).

### Stage 0 — Groundwork (9 tickets)

Repo scaffolding and packaging (scikit-build-core), llama.cpp vendored as a
pinned submodule (`4f37f51`) against a project fork with a patch queue, the CMake
build of vendored llama.cpp + the C shim skeleton, the ctypes `_ffi` layer with
version locking, pure-Python zero-init adapter GGUF creation via gguf-py, the
pytest harness with tiny fixture models, CPU CI, developer/VM playbooks, and the
two binding ADRs (numerics + determinism policy; fork lineage).

**Exit:** CPU CI green; a zero-B adapter attached to a tiny model provably
changes nothing (the no-op smoke test).

### Stage 1 — CPU training core (33 tickets)

Everything needed to train **correctly** end-to-end on CPU:

- **Shim/library:** `ggml_set_param` wiring on adapter A/B (the whole trick —
  BLUEPRINT D2), the forked per-ubatch training loop with pluggable losses,
  SFT/DPO/GRPO trainers, chat-template + masking + packing data layer, adapter
  save/merge, name-keyed optimizer-state checkpoints, grad clipping, trainability
  preflight, gradient checkpointing, chunked no-grad logprob passes.
- **New ggml ops (CPU reference = oracle):** sparse cross-entropy loss (fixes the
  mathematically wrong masked-CE gradients; gate G-A ABI decided here),
  F16/BF16 `OUT_PROD` fix, TANH/SIGMOID/CLAMP VJPs, `SOFT_MAX_BACK` ALiBi lift,
  flash-attention backward (LSE-emitting forward ABI + modernized CPU kernel),
  the kernel-free chunked-attention fallback (unlocks 2–4k ctx everywhere),
  MoE backward (`MUL_MAT_ID` → `OUT_PROD_ID` + `OUT_PROD_ID_GRP`), and SSM
  backward (`SSM_CONV_BACK`, `SSM_SCAN_BACK` with chunk-recompute).

**Exit:** the tiny-model convergence gate (loss curve vs a recorded PEFT
reference) passes on CPU; tiny MoE and Mamba models train; trained adapters load
in stock `llama-cli`.

### Stage 2 — Metal (13 tickets)

Metal has only ROPE_BACK + optimizer kernels today, so this stage writes the
backward suite (SOFT_MAX/RMS_NORM/SILU/REPEAT backward), F32 and quantized
`OUT_PROD` (the critical path — simdgroup tiling with the native transpose flag),
sparse CE, then the MoE/SSM ports and flash-attention backward. Metal CI runs on
macOS arm64 runners from the start.

**Exit (S2-10 milestone):** convergence gate green on `--device metal` with zero
CPU-fallback ops in the dense path; perf/memory snapshot published.

### Stage 3 — CUDA (10 tickets)

CUDA is one op away from GPU-resident dense training: quantized `OUT_PROD` via
dequant-to-F16 + `cublasGemmEx` pinned to F32 accumulation. Then sparse CE, the
ALiBi gate lift, and the flagship: deterministic three-pass flash-attention
backward on the tile family (the difference between "8B LoRA at 4k context on a
24 GB GPU" and "not possible"), plus MoE/SSM ports. CI: per-PR compile lane +
GPU test lane (hosted GPU runner or self-hosted from the CUDA VM).

**Exit (S3-10 milestone):** fully GPU-resident training including FA; upstream
PRs filed for the upstream-early kernel set.

### Stage 4 — Vulkan (9 tickets)

Vulkan already has every `*_BACK` op the training set needs; this stage adds
`OUT_PROD` (quantized case reuses the entire tuned `mul_mm` pipeline via the
`out_prod(a,b) = mul_mat(contᵀa, contᵀb)` reformulation — zero-copy for the
frozen-weight case), sparse CE, a constraint audit, MoE/SSM ports, and FA
backward on the portable scalar base. CI: Mesa lavapipe (software Vulkan) on
hosted runners for correctness + native GPU lane self-hosted.

**Exit (S4-09 milestone):** cross-backend parity report — the same tiny-model
training run on all four backends within the ADR-0002 tolerance. This is the
plan's end state: a complete training system, quantized weights frozen, LoRA
trained, GPU used when available.

## Correctness spine

Three rules hold the whole plan together:

1. **CPU is the oracle.** Every new op lands on CPU first with finite-difference
   (`test-backend-ops` MODE_GRAD) coverage; every GPU port must match it within
   max-abs gradient error ≤ 0.05 at fp16 (ADR-0002).
2. **F32 accumulation on every gradient matmul** — CUDA forced to
   `CUBLAS_COMPUTE_32F`, Vulkan f16acc variants forbidden on grad paths.
3. **Deterministic backward kernels by default** (gate G-B); atomics only as
   measured, opt-in variants.

Two decide-first gates block families of tickets until settled: **G-A** (sparse-CE
cross-backend ABI, settled inside S1-04) and **G-B** (determinism default,
settled in S0-09).

## Effort picture

| Stage | Tickets | Rough effort (from ROADMAP sizing) |
|---|---|---|
| 0 — Groundwork | 9 | ~3–4 eng-weeks |
| 1 — CPU core | 33 | ~14–18 eng-weeks (parallel lanes: shim/python vs kernels) |
| 2 — Metal | 13 | ~8–10 eng-weeks (M2 quantized OUT_PROD + FA7 are the long poles) |
| 3 — CUDA | 10 | ~10–14 eng-weeks (FA5 backward is the flagship XL) |
| 4 — Vulkan | 9 | ~6–9 eng-weeks |
| Backlog | 9 stubs | deferred, trigger-gated |

Sizing legend: S ≤ 2 days · M ≤ 1 week · L 2–3 weeks · XL 4+ weeks. Stages
overlap safely along the dependency graph (e.g. backend CI tickets land right
after S0-07; Metal kernel work can start while stage-1 Python tickets finish).

## Risks the plan already hedges

- **Long poles** (Metal quantized OUT_PROD, CUDA FA backward): the chunked
  attention fallback (S1-24) and CPU fallback via `ggml_backend_sched` keep
  training functional everywhere while kernels land.
- **Upstream churn** against the pinned llama.cpp: monthly rebase cadence, full
  MODE_GRAD rerun per vendor bump, fork-local op enums at table tails, and an
  upstream-early PR set (small VJPs, CUDA OUT_PROD, ALiBi lift) to shrink the
  carried diff.
- **Licensing:** unsloth reuse is math/design only, from verified Apache-2.0
  files; AGPL/LGPL components are explicitly excluded (ROADMAP §13); per-file
  provenance headers enforced from S0-01.
