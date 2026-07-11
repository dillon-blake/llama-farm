---
id: S4-04
title: "Vulkan V3: sparse CE forward + backward shaders"
stage: 4
track: kernels
size: M
deps: ["S1-04", "S4-01"]
status: open
pr: null
---

# S4-04 — Vulkan V3: sparse CE forward + backward shaders

**One-line outcome:** `ggml_cross_entropy_loss_sparse` fwd+bwd runs on Vulkan — a
workgroup-per-row two-pass column-loop forward and a `soft_max_back.comp`-pattern backward,
implementing the ADR-0003 ABI exactly — killing the `[vocab × tokens]` one-hot tensor that
also pressures Vulkan's `maxStorageBufferRange` gates.

## Why (context)

The sparse CE op is the project's canonical loss (BLUEPRINT D4, ROADMAP §3 K-CE): SFT prompt
masking, one-hot elimination, and DPO/GRPO logprob gather in one op, with backward
`dlogits = dloss·w·(exp(x−lse) − onehot)` exactly zero where `w = 0`. S1-04 decided the
cross-backend ABI (ADR-0003, gate G-A: lse stash vs recompute; logits-buffer aliasing;
softcap/scale op-params) and landed the CPU oracle. This ticket is the Vulkan port; it must
implement the ADR-0003 ABI exactly — no backend-local variations. Vulkan today has neither
dense nor sparse CE (ROADMAP §2), so every loss node falls back to CPU, dragging the full
logits tensor across the device boundary each step.

Vulkan gains an extra, backend-specific benefit: the dense-CE path needs a
`[n_vocab × n_ubatch]` one-hot F32 labels tensor (128k vocab × 512 ubatch ≈ 262 MB —
BLUEPRINT G5), and the Vulkan supports_op prologue rejects any op whose src or dst exceeds
`maxStorageBufferRange` (or the max buffer size) unless BDA/64-bit-indexing applies
(`vendor/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:17165-17186`). Sparse labels are one
i32 per token, so the op relieves that gate class entirely (ROADMAP §7 V3). Consequently the
dense `CROSS_ENTROPY_LOSS` op is **permanently skipped on Vulkan** — sparse is canonical
(ROADMAP §3 K-CE).

The shader patterns are both in-tree. Forward: `soft_max.comp` — one workgroup per row, a
`BLOCK_SIZE`-strided column loop, and a shared-memory `vals[BLOCK_SIZE]` tree reduction
(`vendor/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/soft_max.comp`). Backward:
`soft_max_back.comp` — workgroup-per-row with a shared `sum_yg[BLOCK_SIZE]` reduction
(`vendor/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/soft_max_back.comp`) — adapts
near-verbatim: replace the `dot(y, dy)` reduction with the lse/label lookup per ADR-0003.
Both must use shared-memory tree reductions only — no subgroup arithmetic, which the backend
force-disables on MoltenVK+AMD
(`vendor/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:5983-5994`) — and keep all row
statistics in F32 per ADR-0002 (S0-09).

## What to do

All ggml changes land in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **Forward shader `vulkan-shaders/ce_sparse.comp`:** one workgroup per token row; two-pass
   column loop over the vocab dimension (pass 1: running max; pass 2: sum of `exp(x − max)`)
   in the `soft_max.comp` style with per-pass shared-memory tree reductions; then per-token
   loss `w·(lse − x_label)` with the label read as one i32 scalar per row. Apply the
   softcap (`t·tanh(x/t)`) and logit-scale op-params exactly as the S1-04 CPU oracle does.
   Emit/stash lse per the ADR-0003 decision. F32 accumulators throughout.
2. **Backward shader `vulkan-shaders/ce_sparse_back.comp`:** adapt `soft_max_back.comp`'s
   workgroup-per-row structure; compute `dlogits = dloss·w·(exp(x − lse) − onehot)` with the
   softcap `1−tanh²` and scale chain factors; obtain lse per ADR-0003 (read the stash, or
   recompute with the forward's two-pass loop). Rows with `w = 0` write bitwise 0.0 to every
   element. Honor (or reject) logits-buffer aliasing exactly as ADR-0003 specifies.
3. **Wide-vocab strategy:** a single workgroup loops columns in `BLOCK_SIZE` strides, so no
   `[vocab × tokens]` intermediate exists at any vocab size; verify the 128k-and-above vocab
   cases from the S1-04 test matrix run within Vulkan's per-buffer limits (logits are
   `[n_vocab, n_tokens]` and remain subject to the `:17165-17186` size gates — the op does
   not need anything *larger* than the logits it is given).
4. **Plumbing — the six mechanical touch points** (ROADMAP §7) for both new ops
   (`CROSS_ENTROPY_LOSS_SPARSE`, `CROSS_ENTROPY_LOSS_SPARSE_BACK`, enums from S1-04):
   pipelines, dispatch functions, graph-build cases, and supports_op cases in
   `ggml_backend_vk_device_supports_op`
   (`vendor/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:17158-17719`): logits F32,
   labels I32, weights F32, contiguous rows.
5. **Dense CE stays off:** add no `CROSS_ENTROPY_LOSS` case; leave a one-line comment at the
   supports_op switch recording that dense CE is intentionally skipped on Vulkan (sparse
   canonical, ROADMAP §3).
6. **Tests:** the S1-04 `test-backend-ops` cases (eval + MODE_GRAD: mixed `w=0`/`w>0` rows,
   all-masked row, wide vocab, softcap on/off, scale ≠ 1) run on Vulkan against the CPU
   oracle. Extend the S1-04 masked-row exact-zero unit assertion to the Vulkan backend
   (bitwise-zero grads where `w = 0` — MODE_GRAD tolerance cannot prove exactness). Run on
   lavapipe per-PR (targeted `-o CROSS_ENTROPY_LOSS_SPARSE,CROSS_ENTROPY_LOSS_SPARSE_BACK`)
   and native coopmat + scalar nightly. Append both ops to ci-vulkan's targeted default
   list and to the S1-12 `PROJECT_ADDED_OPS` wrapper's Vulkan coverage.
7. **Submodule bump PR** in llama-farm per S0-02.

## Out of scope

- The op ABI, op enums, autograd wiring, and CPU oracle — S1-04 (done; this ticket changes
  no ABI).
- CUDA / Metal sparse-CE ports — S3-03 / S2-07.
- A dense `CROSS_ENTROPY_LOSS` Vulkan port — permanently skipped (ROADMAP §3).
- The chunked lm_head / selective-logprob host pattern — S1-13 (backend-agnostic, uses
  this op).
- Loss-epilogue registry / trainer wiring — S1-05 (already backend-agnostic via sched).

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops test -b Vulkan0 -o CROSS_ENTROPY_LOSS_SPARSE` passes
      on lavapipe for every S1-04 forward case, including the wide-vocab and softcap/scale
      variants.
- [ ] Fork branch: `test-backend-ops grad -b Vulkan0` (MODE_GRAD) for the sparse-CE cases
      passes vs the CPU oracle within the ADR-0002 tolerances (per-op bound; ≤ 0.05 max-abs
      @ fp16 cross-backend parity criterion), on lavapipe and on the native runner in both
      coopmat and `GGML_VK_DISABLE_COOPMAT=1` scalar configurations.
- [ ] The masked-row unit test proves bitwise-zero Vulkan gradients where `w = 0`.
- [ ] Both shaders contain no subgroup-arithmetic extension usage (grep-verifiable in the
      fork diff); row statistics are F32.
- [ ] A reviewer can map every ABI-relevant choice (lse stash/recompute, aliasing, op-param
      handling) to an ADR-0003 clause — stated explicitly in the fork PR description.
- [ ] supports_op has no dense-CE case and carries the intentional-skip comment.
- [ ] llama-farm submodule-bump PR is green in `ci-vulkan / lavapipe` and `ci-cpu`.

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` — eval + MODE_GRAD on the Vulkan backend
vs the S1-04 CPU oracle, tolerances per ADR-0002 (S0-09). Per-PR: targeted ops in
`ci-vulkan / lavapipe` plus the kernel-gated native `ci-vulkan / gpu` job; nightly: full
sweeps on both lanes (S4-01). The S1-12 grad-check pytest wrapper picks the ops up via
`PROJECT_ADDED_OPS` with `--device vulkan`. End-to-end, the loss head leaving the CPU shows
up in the S4-01 nightly fallback report and is enforced at S4-09.

## PR notes

- Branch: `ticket/S4-04-vulkan-sparse-ce-shaders`.
- Two-repo flow per S0-02: fork PR (`llama-farm-base`) + trivial llama-farm submodule-bump
  PR, both referencing the ticket ID.
- Upstreaming disposition: **upstream-later** (ROADMAP §11 triage class b) — the sparse-CE
  op enums are fork-local at tail position; this port rides the S1-04 RFC once the CPU
  oracle plus one GPU backend prove the design (this may *be* that backend).
- Provenance headers per S0-01 policy: shaders patterned on
  `ggml/src/ggml-vulkan/vulkan-shaders/soft_max.comp` and `soft_max_back.comp` (MIT, commit
  `4f37f51`); math design from unsloth `kernels/cross_entropy_loss.py` (Apache-2.0, math
  import only, via S1-04).
