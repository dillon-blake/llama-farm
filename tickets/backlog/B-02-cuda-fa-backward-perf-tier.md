---
id: B-02
title: "CUDA FA backward perf tier: mma family + opt-in atomic-dQ + quantized-KV dequant + MLA/sink grads"
stage: backlog
track: kernels
size: XL
deps: [S3-06, S3-07]
status: open
pr: null
---

# B-02 — CUDA FA backward perf tier: mma family + opt-in atomic-dQ + quantized-KV dequant + MLA/sink grads

**One-line outcome:** **DEFERRED** — the K5 perf tier over the tile-family flash-attention
backward: mma-family backward, a measured opt-in atomic-dQ single pass, a quantized-KV dequant
path, MLA DKQ=576, and sink gradients.

**Activation trigger:** each sub-item is gated individually on S3-06/S3-07 profiling plus real
demand (ROADMAP §8 phasing, §11 K5): mma/atomic-dQ when the tile backward is a measured step-time
bottleneck; MLA-576 when a DeepSeek-style model is requested for FA training; quantized-KV when a
KV-storage-constrained flow materializes; sink grads only if sinks unfreeze (become trainable).

## Why (context)

S3-06/S3-07 deliver the v1 CUDA FA backward on the **tile family**
(`vendor/llama.cpp/ggml/src/ggml-cuda/fattn-tile.cuh:794`, kernel `flash_attn_tile`) with three
deterministic passes and no tensor cores — correct first, per ROADMAP §8 FA5. This ticket is the
inventory of what the forward's fast path has beyond that: the tensor-core mma family
(`vendor/llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh`) for backward throughput, and a
single-pass dQ via atomicAdd — which violates the determinism default (ADR-0002 / gate G-B) and is
therefore only ever an **opt-in** flag, adopted only if measured faster.

Two capability items ride in the same tier. MLA: the forward dispatches DeepSeek's DKQ=576/DV=512
head shape exclusively through the mma family (`vendor/llama.cpp/ggml/src/ggml-cuda/fattn.cu:183`,
`case 576:`), so MLA-576 backward unlocks DeepSeek-style archs for FA training and presupposes the
mma backward. Quantized-KV: training graphs cast K/V F32→F16 before `FLASH_ATTN_EXT`
(`vendor/llama.cpp/src/llama-graph.cpp:2416-2422`), so S3-06 legitimately scoped quantized-KV out;
a backward that dequantizes quantized K/V tiles removes that restriction for flows keeping KV
storage quantized. Sink gradients are dormant by construction — sinks are frozen in LoRA training
(ROADMAP §8) — and matter only if a product decision unfreezes them.

## What to do

1. Per activated sub-item, attach the gating evidence (profile share or named model demand).
2. **mma-family backward:** port the S3-06 three-pass scheme onto the mma kernel structure,
   preserving determinism (exclusive writes per pass); parity vs the CPU oracle (S1-23) and the
   tile backward.
3. **Opt-in atomic-dQ:** single-pass variant with atomicAdd on dQ behind an explicit flag (never
   default — gate G-B); adopt only if measurably faster; document the nondeterminism.
4. **Quantized-KV:** dequantize quantized K/V tiles in the backward with the existing per-type
   machinery; extend `supports_op`.
5. **MLA DKQ=576 + sink grads:** extend the (mma) backward to 576/512; add d_sinks emission
   through the FA autograd case only when sinks are trainable.
6. Extend the `test-backend-ops` FA grad cases per sub-item (quantized-KV types, 576 shape,
   sinks-with-grad) against the S1-23 CPU oracle.

## Out of scope

- Tile-family backward, head sizes 64/128/256, occupancy tuning — S3-06/S3-07.
- Vulkan/Metal FA backward tiers (S4-08/S2-13); forward changes beyond LSE emission (S3-05).

## Acceptance criteria

- [ ] Per activated sub-item: `test-backend-ops` MODE_GRAD FA cases pass vs the CPU oracle within
      the ADR-0002 tolerance (ROADMAP §12 Q4 recomputed-P numerics).
- [ ] Default path stays deterministic: bitwise-identical dQ/dK/dV across reruns with atomic-dQ
      off; the flag defaults off.
- [ ] Benchmark report (mma-vs-tile; atomic-vs-split dQ) at S3-10 profile shapes in `benches/`.
- [ ] MLA sub-item: a DeepSeek-style tiny model trains with FA on (loss falls) on `ci-cuda`.
- [ ] `ci-cuda` green: targeted FA grad sweep per-PR, full sweep nightly.

## Testing & verification

Vendored `tests/test-backend-ops` MODE_GRAD FA cases vs the S1-23 CPU oracle, on the fork branch
and learning-llamas `ci-cuda` (targeted per-PR, full nightly). Determinism checks and benches run on
the CUDA VM; e2e reuses the S3-10 convergence-gate config with FA enabled.

## PR notes

- Branch: `ticket/B-02-cuda-fa-backward-perf-tier` (split per sub-item if activated separately).
- Two-repo flow per S0-02: fork PR + learning-llamas submodule bump.
- Upstreaming disposition: **upstream-later** — builds on the fork-local `emit_lse` /
  `ggml_flash_attn_ext_back` ABI (S1-21); rides that RFC once stable (ROADMAP §11 triage b).
