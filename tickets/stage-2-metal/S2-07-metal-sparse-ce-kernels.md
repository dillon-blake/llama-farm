---
id: S2-07
title: "Metal M7/M8: sparse CE forward + backward kernels"
stage: 2
track: kernels
size: M
deps: ["S2-01", "S1-04"]
status: open
pr: null
---

# S2-07 — Metal M7/M8: sparse CE forward + backward kernels

**One-line outcome:** `ggml_cross_entropy_loss_sparse` forward and backward run on Metal as
grid-stride row kernels with simd reductions — any vocab size in 128 B of threadgroup memory —
making the loss head fully GPU-resident on Apple Silicon.

## Why (context)

The sparse CE op (BLUEPRINT D4) is the project's canonical training loss: per-token
`w·(lse − x_label)` forward, backward `dloss·w·(exp(x−lse) − onehot)` that is exactly zero
where `w = 0`. S1-04 settled its cross-backend ABI (gate G-A, recorded as ADR-0003: lse
stash-vs-recompute, logits-buffer aliasing, softcap/logit-scale op-params) and landed the CPU
oracle patterned on the dense CE forward
(`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:11158`). This ticket is the Metal port (ROADMAP
§6 M7/M8): until it lands, every training step ships the full `[n_vocab, n_tokens]` logits
tensor to the CPU for the loss and its gradient, which defeats GPU residency at exactly the
widest tensors in the graph.

The kernel pattern is in-tree (ROADMAP §3 K-CE): `kernel_soft_max`
(`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:1896-1999`) processes one row per
threadgroup with grid-stride column loops and `simd_max`/`simd_sum` reductions, backed by a
32-float threadgroup buffer — 128 B at any vocab size (`res.smem = 32*sizeof(float)`,
`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.cpp:475`). Sparse CE is *simpler* than
softmax on the data side: the i32 label read is one scalar per row, and no `[vocab×tokens]`
one-hot tensor exists anywhere. The dense `CROSS_ENTROPY_LOSS(_BACK)` op is deliberately
**skipped on Metal** — once sparse CE is canonical on all backends, the dense op needs no
Metal/Vulkan port (ROADMAP §3).

## What to do

All code changes land in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **Read ADR-0003 first and implement its ABI exactly** — the lse stash decision (extra
   per-row F32 output vs recompute in backward), the aliasing decision, and the softcap
   `t·tanh(x/t)` / logit-scale op-params. No Metal-side deviation: S1-04's tests encode the
   contract.
2. **Forward kernel** (`kernel_cross_entropy_loss_sparse_f32`) in
   `vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`: one threadgroup per token row;
   two grid-stride passes per the `kernel_soft_max` pattern (`metal:1896-1999`) — parallel max
   via `simd_max`, then sum-exp via `simd_sum`, cross-simdgroup reduction through the 32-float
   buffer; loss `w·(lse − x_label)` with softcap/scale applied per the ADR; `w = 0` rows write
   0.0 loss. All row statistics in F32 per ADR-0002. If the ADR stashes lse, store it here.
3. **Backward kernel**: `dlogits = dloss·w·(exp(x − lse) − onehot)` with the softcap `1−tanh²`
   and scale chain factors; rows with `w = 0` write **bitwise 0.0** to every element,
   matching the CPU oracle's exact-zero guarantee (never computed-then-scaled).
4. **The five mechanical additions × two ops** (ROADMAP §6 preamble): kargs structs in
   `ggml-metal-impl.h`, pipeline getters in `ggml-metal-device.cpp` (set
   `smem = 32*sizeof(float)` as the soft_max getter does at `:475`), encoder cases in
   `ggml-metal-ops.cpp`, and supports_op cases in
   `vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m` (coverage switch
   `:1051-1368`) gated on `has_simdgroup_reduction` + contiguous rows, mirroring the
   `GGML_OP_SOFT_MAX` gate (`:1174-1180`).
5. **Document the dense-CE skip**: a comment at the supports_op default noting that
   `GGML_OP_CROSS_ENTROPY_LOSS(_BACK)` stays unsupported on Metal by design (ROADMAP §3),
   so nobody "fixes" it later.
6. **Tests**: enable S1-04's `CROSS_ENTROPY_LOSS_SPARSE(_BACK)` test-backend-ops cases
   (patterned on `test_cross_entropy_loss` / `_back`,
   `vendor/llama.cpp/tests/test-backend-ops.cpp:6735` and `:6783`) on Metal. Matrix: mixed
   `w=0`/`w>0` rows, an all-masked row, vocab sizes that cross the CPU oracle's chunked-lse
   threshold (Metal's single grid-stride pass must agree with CPU's chunked log-add-exp
   results at those boundary sizes), softcap on/off, logit scale ≠ 1. Use the harness's
   expected-value filtering where gradients are discontinuous
   (`vendor/llama.cpp/tests/test-backend-ops.cpp:319-321`). Extend S1-04's bitwise-zero
   masked-row unit assertion to read back Metal-produced buffers.
7. **Submodule bump PR** in learning-llamas per S0-02; the S2-01 lanes pick the cases up.

## Out of scope

- The op's ABI, CPU oracle, and autograd wiring — S1-04 (done; this ticket must not change
  the op contract).
- Dense `CROSS_ENTROPY_LOSS` Metal port — permanently skipped (ROADMAP §3).
- CUDA and Vulkan sparse-CE ports — their stage-3/stage-4 tickets (ROADMAP §5 C3, §7 V3).
- Trainer/loss-epilogue wiring in learning-llamas — S1-05; chunked lm_head host pattern — S1-13.

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops` mode `grad` passes on Metal for every
      `CROSS_ENTROPY_LOSS_SPARSE(_BACK)` case (per-op ADR-0002 tolerance), including the
      wide-vocab chunk-boundary sizes, softcap/scale variants, and mixed/all-masked rows.
- [ ] Metal-vs-CPU parity on identical inputs meets ADR-0002: max-abs gradient error
      ≤ 0.05 @ fp16.
- [ ] The masked-row unit test proves bitwise-zero Metal gradients where `w = 0`.
- [ ] supports_op: both sparse ops return true under `has_simdgroup_reduction`; dense CE
      still returns false and the skip comment is present (grep-verifiable).
- [ ] S2-01 nightly e2e fallback report for the tiny dense model no longer lists the sparse
      CE ops in the CPU-fallback set.
- [ ] learning-llamas submodule-bump PR is green in `ci-metal / build` and `ci-metal / grad`.

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` MODE_GRAD plus forward eval, Metal backend
vs the S1-04 CPU oracle, under ADR-0002 tolerances. Per-PR: targeted `ci-metal / grad` (S2-01
runs ops named in changed files); nightly: the full Metal op sweep plus the convergence-gate
e2e with `--device metal`, whose sched log provides the fallback-report evidence. The loss
head appearing GPU-resident here is a prerequisite for S2-10's zero-fallback milestone.

## PR notes

- Branch: `ticket/S2-07-metal-sparse-ce-kernels`.
- Two-repo flow per S0-02: fork PR (`learning-llamas-base`) + trivial learning-llamas submodule-bump
  PR, both referencing the ticket ID.
- Upstreaming disposition: **upstream-later** (ROADMAP §11 triage class b) — the sparse-CE op
  enums are fork-local; this Metal port joins the CPU oracle as evidence for the one-RFC-per-
  op-family upstream proposal once the design is proven.
- Provenance per S0-01: kernel header names the pattern source
  `ggml/src/ggml-metal/ggml-metal.metal` (`kernel_soft_max`, MIT, commit `4f37f51`); the op's
  math design is the S1-04 import from unsloth `kernels/cross_entropy_loss.py` (Apache-2.0,
  math only) — reference S1-04 rather than restating it.
