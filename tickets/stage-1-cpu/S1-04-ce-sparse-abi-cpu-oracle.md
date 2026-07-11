---
id: S1-04
title: "New ggml op: ggml_cross_entropy_loss_sparse — ABI (gate G-A) + CPU oracle fwd/bwd"
stage: 1
track: kernels
size: M
deps: ["S0-09"]
status: open
pr: null
---

# S1-04 — New ggml op: ggml_cross_entropy_loss_sparse — ABI (gate G-A) + CPU oracle fwd/bwd

**One-line outcome:** the project's canonical loss op is decided (ADR-0003, gate G-A) and
implemented on CPU as the oracle: `ggml_cross_entropy_loss_sparse(logits, i32_labels, f32_weights)`
returns per-token loss, with backward `dlogits = dloss·w·(exp(x−lse) − onehot)` that is exactly
zero where `w = 0`.

## Why (context)

One new op serves all three training methods (BLUEPRINT D4): SFT prompt masking, elimination of
dense one-hot labels, and DPO/GRPO per-token logprob gather (negate; weights select completion
tokens). It is required for correctness, not just speed. Masking on the existing dense CE op is
**mathematically wrong** (gap G4): the CPU backward computes `(softmax − labels)·d/nr`
unconditionally, so all-zero label rows still emit `softmax·d/nr ≠ 0` gradients, and `1/nr` counts
masked rows — verified at `vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:11284-11307` (inside
`ggml_compute_forward_cross_entropy_loss_back_f32`, `:11254`). Dense one-hot labels are also
O(n_vocab) per token (gap G5; 128k vocab × 512 ubatch ≈ 262 MB F32). The sparse op fixes both:
forward is per-row stable logsumexp with per-token loss `w·(lse − x_label)` and `w=0` encoding
ignore; backward is exactly zero on masked rows (ROADMAP §3 K-CE). This is the unsloth fused-CE
idea expressed as a ggml op.

Before any implementation, **gate G-A must be settled** (ROADMAP §11, open question Q3 in §12): it
is a cross-backend ABI decision that blocks every GPU CE port (S2-07 Metal, S3-03 CUDA, S4-04
Vulkan) as well as this CPU oracle. Two questions: (a) does forward stash the per-row lse as an
extra `n_tokens` F32 output so backward avoids re-reducing the vocab row, and (b) may backward
alias the logits buffer (unsloth's in-place trick) — the latter needs a prototype against ggml's
graph allocator (`ggml_gallocr`, `vendor/llama.cpp/ggml/src/ggml-alloc.c`) before it can be
answered. The decision is recorded as ADR-0003, following ADR-0002 (S0-09), which also supplies
the acceptance tolerances this ticket's MODE_GRAD cases run under.

Two op-params ship from day one because they are cheap now and painful to retrofit (ROADMAP §13
item 2): logit softcap `t·tanh(x/t)` with backward factor `1−tanh²` and a Cohere-style logit
scale. The forward/chunking math is documented in unsloth's Apache-2.0 CE kernel — softcap/scale
at `unsloth/kernels/cross_entropy_loss.py:84-85`, the chunked-logsumexp decomposition for wide
vocab at `:87-150`, and the backward chain factors at `:247-273`. This is a math import only
(Triton → C rewrite; ROADMAP §13 item 1); the CPU kernel itself is patterned on the in-tree dense
CE forward `ggml_compute_forward_cross_entropy_loss_f32`
(`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:11158`).

## What to do

All code changes land in the vendored llama.cpp fork (new op enums must live in ggml's op table),
via the S0-02 two-repo flow; the ADR lands in llama-farm.

1. **Settle gate G-A first.** Prototype backward-aliasing of the logits buffer against
   `ggml_gallocr`; decide lse stash (extra `n_tokens` F32 output, e.g. a packed or secondary
   output) vs recompute (2× vocab reads in backward). Write
   `docs/adr/ADR-0003-ce-sparse-abi.md` (llama-farm repo, format per ADR-0001/0002): the two
   decisions, the prototype evidence, and the binding on all backend ports (S2-07/S3-03/S4-04
   implement exactly this ABI). Get it reviewed before kernel code is written.
2. **Op ABI in the fork:** add `GGML_OP_CROSS_ENTROPY_LOSS_SPARSE` and
   `GGML_OP_CROSS_ENTROPY_LOSS_SPARSE_BACK` at the **tail** of the op enum, immediately before
   `GGML_OP_COUNT` (`vendor/llama.cpp/ggml/include/ggml.h:589`) — tail position minimizes rebase
   conflicts (ROADMAP §11). Constructor
   `ggml_cross_entropy_loss_sparse(ctx, logits [n_vocab, n_tokens], labels I32 [n_tokens],
   weights F32 [n_tokens]) → F32 per-token loss [n_tokens]` (no reduction; callers reduce via
   `GGML_OPT_LOSS_TYPE_SUM`). Op-params: `float softcap` (0 = off) and `float logit_scale`
   (1 = off), set at construction.
3. **Autograd wiring:** a `GGML_OP_CROSS_ENTROPY_LOSS_SPARSE` case in `ggml_compute_backward`
   emitting the `_BACK` op with per-token `dloss` (pattern: the dense CE case at
   `vendor/llama.cpp/ggml/src/ggml.c:6879`). Labels are I32 and thus auto-excluded from
   gradients; weights are constants — no grads to either.
4. **CPU forward** in `ggml-cpu/ops.cpp`, patterned on `:11158`: per-row stable logsumexp
   (`lse = max + log Σ exp(x−max)`) in F32-or-better accumulation per ADR-0002; loss
   `w·(lse − x_label)`; softcap/scale applied per the unsloth math. For wide vocab rows, chunked
   lse: per-chunk partial lse reduced by log-add-exp (decomposition per
   `unsloth/kernels/cross_entropy_loss.py:87-150`), with the chunk threshold an internal constant
   that tests can exercise.
5. **CPU backward:** `dlogits = dloss·w·(exp(x − lse) − onehot)` with the softcap `1−tanh²` and
   scale chain factors (`unsloth/kernels/cross_entropy_loss.py:247-273`); rows with `w = 0` write
   **bitwise 0.0** to every element (never computed-then-scaled). Honor the G-A decision for lse
   (stash vs recompute) and aliasing.
6. **Tests in the fork:** `test-backend-ops` cases patterned on `test_cross_entropy_loss` /
   `test_cross_entropy_loss_back` (`vendor/llama.cpp/tests/test-backend-ops.cpp:6735` and
   `:6783`): MODE_GRAD (finite differences vs analytic, using the harness's expected-value
   filtering where needed, `vendor/llama.cpp/tests/test-backend-ops.cpp:319-321`) plus forward
   eval cases. Matrix: mixed `w=0`/`w>0` rows, all-masked row, vocab wide enough to cross the
   chunked-lse threshold, softcap on/off, logit scale ≠ 1. Add a dedicated exact-zero unit
   assertion (masked-row grads are bitwise zero) — MODE_GRAD tolerance alone cannot prove it.
7. **supports_op:** CPU returns true; all other backends return false until their port tickets
   land (the sched will fall back to CPU meanwhile, ROADMAP §11 scheduler note).
8. **Submodule bump PR** in llama-farm referencing this ticket, per S0-02, so `ci-cpu` builds and
   runs against the new fork commit.

## Out of scope

- GPU ports of the op — S3-03 (CUDA), S2-07 (Metal), S4-04 (Vulkan); all blocked on this oracle
  and ADR-0003.
- Wiring `ce_sparse` into the llama-farm loss-epilogue registry and the SFT trainer — S1-05.
- The chunked lm_head / selective-logprob host pattern — S1-13 (uses this op, does not change it).
- Metal/Vulkan ports of the **dense** CE op — skipped permanently once sparse is canonical
  (ROADMAP §3).
- Upstream RFC submission — disposition below; not part of this ticket's acceptance.

## Acceptance criteria

- [ ] `docs/adr/ADR-0003-ce-sparse-abi.md` exists, Status "Accepted": lse stash-vs-recompute and
      backward-aliasing both decided with prototype evidence, and the binding on S2-07/S3-03/S4-04
      stated.
- [ ] Fork branch: `test-backend-ops` mode `grad` passes on CPU for every new
      `CROSS_ENTROPY_LOSS_SPARSE(_BACK)` case within the ADR-0002 per-op tolerance
      (`max_maa_err`, default bound 1e-4), including the wide-vocab chunked case and softcap/scale
      variants.
- [ ] The masked-row unit test proves bitwise-zero gradients where `w = 0`.
- [ ] Forward parity test: on random inputs with all `w = 1` and softcap/scale off, summed sparse
      loss matches the dense `ggml_cross_entropy_loss` value within F32 round-off tolerance.
- [ ] Both new enum values sit immediately before `GGML_OP_COUNT`; no existing op enum value
      changed (grep-verifiable in the fork diff).
- [ ] llama-farm submodule-bump PR is green in `ci-cpu`, which runs the vendored
      `test-backend-ops` grad cases for the new op (per-PR).

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` in MODE_GRAD plus forward-eval mode, run on
the fork branch CI and, after the submodule bump, in llama-farm's `ci-cpu` lane per-PR (S1-12
later adds the pytest wrapper that runs these cases for all project-added ops; do not wait for
it). CPU is the oracle: every later GPU port ticket (S2-07/S3-03/S4-04) validates against this
implementation under the ADR-0002 cross-backend parity criterion (max-abs gradient error ≤ 0.05 at
fp16), so the reference must itself be MODE_GRAD-clean first. Nightly `ci-cpu` re-runs the full
suite.

## PR notes

- Branch: `ticket/S1-04-ce-sparse-abi-cpu-oracle`.
- Two-repo flow per S0-02: (1) implementation PR against the fork's `llama-farm-base` branch with
  the ticket ID in the title; (2) trivial llama-farm PR bumping the `vendor/llama.cpp` gitlink and
  adding ADR-0003, referencing the same ticket ID. (Stage-0 vendor infrastructure is assumed in
  place; the frontmatter dep is S0-09 for ADR-0002 tolerances.)
- Upstreaming disposition: **upstream-later** — new op enums are fork-local at tail position
  first, proposed upstream as an RFC once the CPU oracle plus one GPU backend prove the design
  (ROADMAP §11 triage class b).
- Kernel files carry provenance headers per S0-01 policy: pattern source
  `ggml/src/ggml-cpu/ops.cpp` (MIT, commit `4f37f51`) and math design from unsloth
  `kernels/cross_entropy_loss.py` (Apache-2.0, math import only).
