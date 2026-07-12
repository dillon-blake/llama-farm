---
id: S3-03
title: "CUDA C3: sparse CE forward + backward"
stage: 3
track: kernels
size: M
deps: [S1-04, S3-01]
status: open
pr: null
---

# S3-03 — CUDA C3: sparse CE forward + backward

**One-line outcome:** `ggml_cross_entropy_loss_sparse` (fwd + bwd) runs on CUDA with
block-per-row kernels scaled to 128k-vocab rows — the loss head is GPU-resident.

## Why (context)

S1-04 created the project's canonical loss op (BLUEPRINT D4) — per-token
`w·(lse − x_label)` forward, `dlogits = dloss·w·(exp(x − lse) − onehot)` backward, exactly
zero where `w = 0` — decided its cross-backend ABI as ADR-0003 (gate G-A: LSE stash vs
recompute, logits aliasing, softcap/scale op-params), and shipped the CPU oracle. This
ticket is the CUDA port (ROADMAP §5 C3). Without it, every training step's loss head falls
back to CPU via `ggml_backend_sched`, dragging the logits tensor (n_vocab × n_ubatch) across
the PCIe boundary each step; with it, CUDA training is loss-to-optimizer GPU-resident.

The in-tree dense CE kernels are the pattern
(`vendor/llama.cpp/ggml/src/ggml-cuda/cross-entropy-loss.cu:8-92`): one block per row,
logits cached in shared memory when they fit under the device's shared-memory-per-block
opt-in limit, with a global-memory fallback branch when they do not — so there is **no hard
vocab limit**. Two things must change beyond sparsification. First, the existing kernels run
one warp per row (`blocks_dim(WARP_SIZE, 1, 1)`, `cross-entropy-loss.cu:116` and `:164`) —
fine at test sizes, underpowered for 128k-vocab rows; widen toward 1024 threads with a
two-level (intra-warp shuffle + cross-warp shared-memory) reduction, following the width
switch `rms_norm_back_f32_cuda` uses (`vendor/llama.cpp/ggml/src/ggml-cuda/norm.cu:410-417`:
`WARP_SIZE` block below 1024 columns, 1024-thread block above). Second, the dense backward
asserts a scalar incoming gradient (`GGML_ASSERT(ggml_is_scalar(grad))`,
`cross-entropy-loss.cu:147`) because the dense op pre-reduces its loss; the sparse op
returns per-token loss, so backward takes a **per-row grad vector** — drop that assert
pattern entirely.

Sparse is *simpler* than dense (ROADMAP §5 C3): there is no `[n_vocab]` label row to read —
per row, the kernel reads one i32 label and one f32 weight. That removes half the dense
kernel's memory traffic and its second shared-memory operand.

## What to do

All ggml changes land in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **New file `vendor/llama.cpp/ggml/src/ggml-cuda/cross-entropy-loss-sparse.cu` (+ `.cuh`)**
   patterned on `cross-entropy-loss.cu:8-92`, with a provenance header (MIT, commit
   `4f37f51`). Implement the ABI exactly as ADR-0003 specifies — operand order, LSE
   stash-vs-recompute, aliasing, and the `softcap`/`logit_scale` op-params. GPU ports never
   redefine semantics (tickets/README correctness policy).
2. **Forward kernel:** one block per token row; block width selected from vocab size per the
   `norm.cu:410-417` pattern (WARP_SIZE for narrow rows, up to 1024 threads with two-level
   reduction for wide rows); stable logsumexp (max, then Σexp(x−max)) in F32 per ADR-0002;
   shared-memory logits cache with the global fallback branch (smpbo check) as in the dense
   kernels; apply logit scale and softcap before the reduction; emit per-token
   `w·(lse − x_label)` (plus the LSE stash if ADR-0003 says so).
3. **Backward kernel:** per-row `dloss` read from the incoming grad vector (no
   `ggml_is_scalar` assert); `dlogits = dloss·w·(exp(x − lse) − onehot)` with the softcap
   `1−tanh²` and scale chain factors per ADR-0003; rows with `w = 0` write **bitwise 0.0**
   to every element. Honor the ADR-0003 LSE decision (reuse the stash or recompute the row
   reduction) and the aliasing decision.
4. **Dispatch + supports_op:** add the two op cases in `ggml-cuda.cu` next to the dense CE
   cases (dispatch at `vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:2227` and `:2245`;
   `supports_op` at `:4963-4964`), returning true for the shapes/types the ABI defines.
5. **Tests:** the S1-04 `test-backend-ops` cases run on CUDA once `supports_op` returns
   true — MODE_GRAD vs the CPU oracle plus forward eval, over the S1-04 matrix: mixed
   `w=0`/`w>0` rows, all-masked row, wide-vocab (128k-class, crossing both the block-width
   switch and the shared-memory fallback), softcap on/off, logit scale ≠ 1. Extend the
   exact-zero masked-row assertion to compare CUDA output bitwise. Append the op to
   `PROJECT_ADDED_OPS` device coverage if S1-12's wrapper needs a backend entry.
6. **Submodule bump PR** in learning-llamas per S0-02; ensure ci-cuda's targeted default op list
   includes `CROSS_ENTROPY_LOSS_SPARSE`.

## Out of scope

- The op ABI, ADR-0003, and the CPU oracle — decided/landed in S1-04; this ticket changes
  neither.
- Metal and Vulkan ports — S2-07, S4-04 (same ABI).
- Porting/repairing the **dense** CE op on any backend — superseded by sparse (ROADMAP §3).
- Chunked lm_head / selective-logprob host pattern — S1-13.
- Saved-LSE exploitation beyond what ADR-0003 already decided — backlog B-04.

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops grad -b CUDA0 -o CROSS_ENTROPY_LOSS_SPARSE` (MODE_GRAD)
      passes for every S1-04 case within the ADR-0002 tolerances (per-op bound; ≤ 0.05
      max-abs @ fp16 cross-backend parity vs the CPU oracle), including softcap/scale
      variants.
- [ ] Forward eval parity vs CPU passes for the wide-vocab case that exceeds the
      shared-memory limit (global-fallback branch exercised) and for a row count/width that
      crosses the 1024-thread block-width switch.
- [ ] The masked-row test proves bitwise-zero CUDA gradients where `w = 0`.
- [ ] The backward path accepts a non-scalar per-row grad (no `ggml_is_scalar` assert in the
      new code; grep-verifiable).
- [ ] The kernels implement the ADR-0003 ABI choices verbatim (reviewer checks against the
      ADR; no CUDA-only semantic deviation).
- [ ] learning-llamas submodule-bump PR is green in `ci-cuda` (compile + GPU lanes) and `ci-cpu`.

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` MODE_GRAD + forward eval on the CUDA
backend vs the S1-04 CPU oracle, under ADR-0002 tolerances. Runs on the fork branch CI,
then per-PR in learning-llamas's `ci-cuda` GPU lane (kernel-gated, targeted `-o`) and the nightly
full sweep (S3-01). The S1-12 grad-check pytest wrapper picks the op up via
`PROJECT_ADDED_OPS` with `--device cuda` in the nightly e2e job. End-to-end, the loss head's
residency shows up in ci-cuda's fallback report immediately and becomes a hard requirement
at S3-10.

## PR notes

- Branch: `ticket/S3-03-cuda-sparse-ce-kernels`.
- Two-repo flow per S0-02: fork PR (`learning-llamas-base`) + trivial learning-llamas submodule-bump
  PR, both carrying the ticket ID.
- Upstreaming disposition: **upstream-later** (ROADMAP §11 triage class b) — this CUDA port
  is the "one GPU backend" that, together with the S1-04 CPU oracle, makes the sparse-CE op
  family ready for its upstream RFC; coordinate the RFC with S1-04's disposition rather than
  filing independently.
- Provenance headers per S0-01 policy: pattern source `ggml/src/ggml-cuda/cross-entropy-loss.cu`
  and `ggml/src/ggml-cuda/norm.cu` (MIT, commit `4f37f51`); math design from unsloth
  `kernels/cross_entropy_loss.py` (Apache-2.0, math import only, via S1-04/ADR-0003).
