---
id: S3-09
title: "CUDA SSM: SSM_CONV_BACK + SSM_SCAN_BACK ports"
stage: 3
track: kernels
size: L
deps: [S3-01, S1-30, S1-31]
status: open
pr: null
---

# S3-09 — CUDA SSM: SSM_CONV_BACK + SSM_SCAN_BACK ports

**One-line outcome:** Mamba-family LoRA training is GPU-resident on CUDA —
`SSM_CONV_BACK` and `SSM_SCAN_BACK` run as siblings of the existing forward kernels,
MODE_GRAD-parity-checked against the S1-30/S1-31 CPU oracles, and a tiny-Mamba model
trains end-to-end on `--device cuda`.

## Why (context)

All four LoRA-able Mamba projections are plain `MUL_MAT`, already covered by the base
`OUT_PROD` work; what blocks GPU-resident SSM training is gradient flow *through* the
SSM ops (ROADMAP §10). S1-29 wired the backward switch, S1-30/S1-31 delivered the CPU
kernels; until this ticket lands, every mamba backward node falls back to CPU via
`ggml_backend_sched`. Unlike MoE, CUDA already has SSM **forwards**, so each backward is
a same-shape sibling kernel (ROADMAP §10).

The forward structures to mirror: `ssm_conv_f32`
(`vendor/llama.cpp/ggml/src/ggml-cuda/ssm-conv.cu:6`) is row-parallel — grid over
(sequence, `d_inner` slices), one thread per channel row, the sliding window held in
registers; a long-token variant adds a token-block grid dimension
(`ssm-conv.cu:61`); op entry at `ssm-conv.cu:160`. `ssm_scan_f32`
(`vendor/llama.cpp/ggml/src/ggml-cuda/ssm-scan.cu:20`) runs one block per
(sequence x `splitD` slice) with per-thread state registers, CUB
`BlockLoad`/`BlockStore` staging (`ssm-scan.cu:60-71`, `USE_CUB` gate at `:1-8`) and a
sequential token loop (`:84`); the Mamba-2 grouped variant is `ssm_scan_f32_group`
(`:130`). The CUDA supports_op entries carry shape gates the backward must mirror —
Mamba-2 `d_state` 128/256 with `d_head % 16 == 0`, Mamba-1 `d_state == 16`, `SSM_CONV`
`d_inner % 128 == 0` (`vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:4882-4895`).

The hard part is `SSM_SCAN_BACK` (ROADMAP §10 S3): the forward overwrites recurrent
states in place (CPU reference: `s0 = s`,
`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:9770`), so the backward must recompute
states — checkpoint every K tokens, re-run the forward inside each chunk, then
reverse-scan — following the chunk-recompute schedule and reverse-pass math the S1-31
CPU reference established (`ddt` chains through `sigmoid(dt)` since the forward applies
softplus, `ops.cpp:9623`; `dB`/`dC` reduce over GQA-style head groups, `:9625`).
ROADMAP §12 Q7 pre-scopes the two CUDA-specific risks this ticket must answer: register
pressure in the **reversed** token loop (the backward keeps `ds`, recomputed states, and
the decay recomputation live where the forward kept one state register set), and where
the checkpoint buffer lives — `ggml_cuda_pool_alloc` transient vs an extra dst tensor —
a decision this ticket makes and records. Determinism is gate G-B (S0-09/ADR-0002): the
group reductions for `dB`/`dC` use fixed-order block/CUB reductions, no atomics.

## What to do

All ggml changes land in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **`SSM_CONV_BACK` kernel** in a new
   `vendor/llama.cpp/ggml/src/ggml-cuda/ssm-conv-back.cu` (or alongside the forward in
   `ssm-conv.cu`): the flipped-window correlation per the S1-30 contract — only `d_sx`
   is produced (conv weight frozen). Keep the forward's row-parallel structure
   (`ssm-conv.cu:6`): one thread per `d_inner` channel row gives exclusive dst-row
   ownership, hence determinism for free. Cover long `n_t` (mirror or subsume the
   long-token variant, `ssm-conv.cu:61`); zero-init owned rows before accumulating.
2. **`SSM_SCAN_BACK` kernel(s)** in a new `ssm-scan-back.cu`, consuming the S1-29
   constructor's srcs and writing the packed `dx‖ddt‖dB‖dC` dst, for both forward
   families: the `splitD` Mamba-1 pattern (`ssm-scan.cu:20`) and the grouped Mamba-2
   pattern (`:130`). Per chunk: pass 1 re-runs the forward recurrence from the previous
   checkpoint to materialize the chunk's states; pass 2 walks the chunk's tokens in a
   **reversed** loop applying the S1-31 reverse-pass math with F32 accumulation
   throughout. `dB`/`dC` are shared per head group — reduce per-block partials with
   fixed-order CUB/shared-memory reductions (pattern: the forward's CUB staging,
   `ssm-scan.cu:60-71`); no atomics per gate G-B.
3. **Checkpoint buffer decision (Q7):** implement checkpoints as a
   `ggml_cuda_pool_alloc` transient sized `ceil(n_t/K) x state` unless a concrete
   allocator constraint forces the extra-dst alternative; record the decision, the
   chosen default K, and the rationale in a code comment and the fork PR (the CPU
   reference resolved the same question as wdata — keep semantics identical, placement
   backend-appropriate).
4. **Register-pressure check (Q7):** measure occupancy of the reversed loop (nsight or
   `--ptxas-options=-v` register counts) at the shipped `splitD`/`N` template
   parameters; if spilling, split the reversed loop's working set (smaller `splitD`, or
   staging via shared memory) and record the measurements in the fork PR.
5. **Plumbing:** dispatch cases next to the forwards' (`ggml-cuda.cu:2212-2216`) and
   supports_op entries mirroring the forward shape gates (`:4882-4895`) so the backward
   never accepts shapes its sibling forward rejects; unsupported shapes keep falling
   back to the CPU oracle via sched.
6. **Tests:** re-run the S1-30/S1-31 MODE_GRAD case lists on CUDA — grad-enabled
   `test_ssm_conv` (`vendor/llama.cpp/tests/test-backend-ops.cpp:3753`, shapes at
   `:8477-8482` incl. the long-token cases) and `test_ssm_scan` (`:3819`, shapes at
   `:8507-8510`: Mamba-1, Mamba-2, Falcon-H1) plus S1-31's added cases (`n_group > 1`,
   multi-sequence, `n_seq_tokens` below/above K) — vs the CPU oracle within the
   ADR-0002 tolerance. Add a determinism check: two identical CUDA runs produce
   bitwise-identical `dx‖ddt‖dB‖dC`.
7. **e2e:** run S1-31's `tests/test_ssm_training.py` tiny-Mamba SFT with
   `--device cuda`; the S3-01 fallback report must show the SSM backward ops on GPU.
8. **Submodule bump PR** in llama-farm per S0-02, appending both ops to the ci-cuda
   targeted-op defaults.

## Out of scope

- Metal/Vulkan ports — S2-12 / S4-07 (same contracts, same CPU oracles).
- `ds0`, `dA`, `dD`, conv-weight and `dt_bias` grads; cross-ubatch BPTT — ROADMAP §10
  S5, deferred (truncation-at-ubatch semantics documented in S1-31).
- RWKV6/7, `GATED_LINEAR_ATTN`, `GATED_DELTA_NET` backward — backlog B-06.
- Perf tuning beyond the Q7 occupancy check (K5 territory); atomics-based variants —
  opt-in later per gate G-B, backlog.

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops grad -b CUDA0` passes for all grad-enabled
      `SSM_CONV` cases (incl. long-token) within the ADR-0002 tolerance vs the S1-30
      CPU oracle.
- [ ] Fork branch: `test-backend-ops grad -b CUDA0` passes for all grad-enabled
      `SSM_SCAN` cases (Mamba-1, Mamba-2, `n_group > 1`, multi-seq, chunking on/off
      boundaries) within the ADR-0002 tolerance vs the S1-31 CPU oracle
      (≤ 0.05 max-abs @ fp16 parity criterion).
- [ ] Determinism: two identical CUDA runs produce bitwise-identical packed grads
      (gate G-B).
- [ ] The Q7 record exists in the fork PR: checkpoint-buffer decision (pool-alloc vs
      extra dst) with rationale, default K, and the reversed-loop register/occupancy
      measurements.
- [ ] supports_op for both `*_BACK` ops mirrors the forward shape gates; a
      shape the forward rejects is rejected by the backward (unit-asserted).
- [ ] `tests/test_ssm_training.py --device cuda` passes; its fallback report shows
      `SSM_CONV_BACK`/`SSM_SCAN_BACK` executing on CUDA.
- [ ] llama-farm submodule-bump PR is green in `ci-cuda` (compile + GPU lanes) and
      `ci-cpu`.

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` MODE_GRAD on the CUDA backend vs the
S1-30/S1-31 CPU oracles under the ADR-0002 tolerances (S0-09), plus the fork-side
determinism check. Runs on the fork branch CI and, after the submodule bump, in
llama-farm's `ci-cuda` GPU lane per-PR (kernel-gated targeted op list) and the nightly
full sweep (S3-01). The tiny-Mamba e2e joins the nightly `ci-cuda` e2e job; S3-10 later
folds the SSM ops into the fallback-forbidden set.

## PR notes

- Branch: `ticket/S3-09-cuda-ssm-backward-ports`.
- Two-repo flow per S0-02: implementation PR against the fork's `llama-farm-base`
  branch with the ticket ID in the title (may be split fork-side into conv-back and
  scan-back commits for review), plus a trivial llama-farm submodule-bump PR
  referencing the same ticket ID.
- Upstreaming disposition: **fork-local first, upstream-later** — the `SSM_*_BACK` op
  enums ride the op-family RFC with S1-29/30/31 once the CPU oracle and one GPU backend
  prove the design (ROADMAP §11 triage b); this port likely anchors that RFC —
  coordinate with S2-12.
- Provenance: kernels adapt the in-tree MIT forward structures (`ssm-conv.cu`,
  `ssm-scan.cu`, llama.cpp `4f37f51`); reverse-pass math comes from this project's
  research (ROADMAP §10 S3), not from any external implementation.
