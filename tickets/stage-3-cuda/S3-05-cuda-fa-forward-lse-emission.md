---
id: S3-05
title: "CUDA FA4: flash-attention forward LSE emission"
stage: 3
track: kernels
size: S
deps: ["S1-21", "S3-01"]
status: open
pr: null
---

# S3-05 — CUDA FA4: flash-attention forward LSE emission

**One-line outcome:** all four CUDA flash-attention kernel families (vec/tile/wmma/mma)
optionally store per-row LSE into the S1-21 packed `O‖LSE` dst — the forward half of CUDA
FA training — with bit-identical behavior when `emit_lse` is off.

## Why (context)

FA backward (S3-06) recomputes the attention matrix P from Q/K/mask/LSE instead of
storing it, which is what turns the n_ctx² memory cliff into a per-row vector
(ROADMAP §8). That only works if the forward hands over the log-sum-exp per
(q-position × head) row. S1-21 (FA1) defined the cross-backend ABI — a packed `O‖LSE`
F32 dst with accessor views, `lse = M + log(S)` with sinks folded in and `-INFINITY`
for fully-masked rows — and implemented it on CPU. This ticket is the CUDA port
(ROADMAP §8 FA4), a prerequisite for S3-06 and deliberately small: the ingredients
already exist and are currently thrown away.

Every CUDA FA kernel already maintains the running max and sum: the shared kernel
signature carries a `float2 * dst_meta` output
(`vendor/llama.cpp/ggml/src/ggml-cuda/fattn-common.cuh:29`), and each family stores
`(KQ_max, KQ_sum)` at its epilogue — tile at
`vendor/llama.cpp/ggml/src/ggml-cuda/fattn-tile.cuh:1136-1138`, vec at
`vendor/llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh:513-515`, wmma at
`vendor/llama.cpp/ggml/src/ggml-cuda/fattn-wmma-f16.cu:495-502`, and mma as the
`KQ_cmr = (KQ_max, KQ_rowsum)` fixup metadata at
`vendor/llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh:1436-1449`. The catch: today the
meta is only materialized when multiple blocks must be combined — the tile/vec/wmma
stores are guarded by `gridDim.y != 1`, the pool buffer `dst_tmp_meta`
(`fattn-common.cuh:1010`) is only allocated on the stream-k and `parallel_blocks > 1`
paths (`fattn-common.cuh:1149` and `:1181-1183`), and the combine kernel
`flash_attn_combine_results` (`fattn-common.cuh:916-970`) consumes it and discards the
normalizer. Single-block launches and mma tiles that need no fixup never emit meta at
all.

The pre-scoped risk (ROADMAP §8 FA4, manifest): the CUDA backend **over-allocates the
FA dst** to stash F16 K/V conversion scratch past `dst->data + ggml_nbytes(dst)`
(`ggml_cuda_flash_attn_ext_get_f16_extra_data`, `fattn-common.cuh:47-85`, sized by
`ggml_cuda_flash_attn_ext_get_alloc_size`,
`vendor/llama.cpp/ggml/src/ggml-cuda/fattn.cu:546`, hooked at
`vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:834-838`). With the packed dst,
`ggml_nbytes(dst)` grows by the LSE region — the scratch computation stays consistent
only if every consumer uses the packed size, and any code that equates "dst bytes" with
"O bytes" (e.g. `dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV))` at
`fattn-common.cuh:1182-1183`) silently over- or mis-sizes.

## What to do

All code lands in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **Plumb `emit_lse` through `launch_fattn`**
   (`vendor/llama.cpp/ggml/src/ggml-cuda/fattn-common.cuh:973`): read the S1-21 op-param
   (i32 slot 4), derive the O and LSE region pointers/shapes via the FA1 accessor logic
   (never hand-computed offsets), and size internal transients from the **O region**, not
   `ggml_nelements/ggml_nrows(KQV)` (fix the `:1182-1183` sizing under emit_lse).
2. **Materialize meta unconditionally when `emit_lse` is set:** allocate `dst_tmp_meta`
   even for single-block, non-stream-k launches and relax the `gridDim.y != 1` epilogue
   guards (tile `:1136-1138`, vec `:513-515`, wmma `:495-502`) to also store when an
   `emit_lse` kernel argument (or template flag) is set. For mma, ensure non-fixup tiles
   emit their `(KQ_max, KQ_rowsum)` too (today only fixup tiles write meta,
   `fattn-mma-f16.cuh:1436-1449`).
3. **Write LSE at the reduction points:** a small epilogue kernel (or an extension of
   `flash_attn_combine_results`, `fattn-common.cuh:916-970`, and the stream-k fixup
   kernels at `:723` and `:807`) computes `lse = kq_max_combined + log(denominator)` in
   F32 and stores it into the packed LSE region; the single-block path converts its
   per-row `(KQ_max, KQ_sum)` directly. Honor the FA1 contract exactly: sinks are already
   folded into max/sum, fully-masked rows (`sum == 0`) store `-INFINITY`.
4. **Alloc-size audit:** verify `ggml_cuda_flash_attn_ext_get_alloc_size` (`fattn.cu:546`)
   and the runtime extra-data computation (`fattn-common.cuh:47-85`) both use
   `ggml_nbytes` of the packed dst, so the F16 K/V scratch lands past the LSE region;
   add an assert that the scratch base ≥ end of the LSE region.
5. **supports_op / dispatch:** accept the `emit_lse` variant in
   `ggml_cuda_flash_attn_ext_supported` (wired at
   `vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:4961-4962`) for every combination
   the plain forward already supports; dispatch is unchanged
   (`ggml_cuda_flash_attn_ext`, `fattn.cu:581`, over the four-family enum at
   `fattn.cu:332-337`).
6. **Tests (fork):** extend the vendored `test-backend-ops` FLASH_ATTN_EXT cases with
   `emit_lse` variants — `test` mode compares the full packed dst (O and LSE) against
   the S1-21 CPU forward. Choose shapes/types that reach each kernel family reachable on
   the CI GPU (vec/tile/wmma/mma selection is shape- and arch-driven,
   `fattn.cu:332-337`); log which family executed. Matrix: head sizes 64/128/256, GQA
   on/off, mask on/off, `max_bias > 0`, softcap, sinks, fully-masked row, and shapes
   forcing single-block, `parallel_blocks > 1`, and stream-k paths.
7. **No-behavior-change guard:** the full existing FLASH_ATTN_EXT suite must pass
   unmodified with the flag off — all new stores/allocs are gated on `emit_lse`.
8. **Submodule bump PR** in learning-llamas per S0-02, adding `FLASH_ATTN_EXT` to the
   ci-cuda kernel-gated targeted op list.

## Out of scope

- The CUDA FA backward kernel — S3-06 (consumes this LSE).
- MODE_GRAD coverage for FA on CUDA — arrives with S3-06 (until then the back op is
  unsupported on CUDA and grad cases skip; forward `test`-mode parity is the acceptance
  here).
- Metal/Vulkan forward LSE emission — stage-2/4 FA tickets (ROADMAP §8 FA6/FA7).
- Any change to the LSE definition or packed layout — fixed by S1-21; mismatches are a
  named cross-backend risk (ROADMAP §12 Q4), not a local design freedom.
- Perf tuning of the epilogue/combine path — revisit with S3-07's occupancy work if
  profiling flags it.

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT` passes for the
      full `emit_lse` case matrix of step 6 (O and LSE regions compared against the CPU
      oracle within ADR-0002 forward tolerance), including the `-INFINITY` fully-masked
      row convention.
- [ ] The test log records at least two distinct kernel families executing `emit_lse`
      cases on the CI GPU, and the case set includes single-block,
      `parallel_blocks > 1`, and stream-k shapes.
- [ ] With `emit_lse` off, the pre-existing FLASH_ATTN_EXT test set passes unchanged
      (no diffs to those cases in the PR).
- [ ] The scratch-vs-LSE assert of step 4 is present, and an `emit_lse` case with
      non-F16 K/V (forcing the F16 conversion scratch) passes.
- [ ] learning-llamas submodule-bump PR is green in `ci-cuda` (compile lane + GPU quick
      subset with `-o FLASH_ATTN_EXT`) and `ci-cpu`.

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` in `test` (forward-parity) mode,
CUDA vs the S1-21 CPU oracle, per ADR-0002. Runs on the fork branch CI and, after the
submodule bump, in learning-llamas's `ci-cuda` GPU lane per-PR (kernel-gated, targeted
`-o FLASH_ATTN_EXT`) plus the nightly full sweep (S3-01). MODE_GRAD acceptance for the
FA path lands with S3-06, which validates its backward against S1-23's CPU oracle using
the LSE this ticket emits.

## PR notes

- Branch: `ticket/S3-05-cuda-fa-forward-lse-emission`.
- Two-repo flow per S0-02: fork PR against `learning-llamas-base` (ticket ID in title) plus a
  trivial learning-llamas submodule-bump PR referencing the same ID.
- Upstreaming disposition: **upstream-later** (ROADMAP §11 triage class b) — this rides
  the FA-training op-family RFC with the S1-21 `emit_lse` ABI; propose upstream once
  S3-06 proves the design on CUDA.
- No copied external code; all edits are in-tree MIT CUDA sources at `4f37f51` — keep
  provenance notes in touched-function comments per S0-01 policy.
