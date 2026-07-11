---
id: S2-12
title: "Metal SSM: SSM_CONV_BACK + SSM_SCAN_BACK ports"
stage: 2
track: kernels
size: L
deps: ["S2-01", "S1-30", "S1-31"]
status: open
pr: null
---

# S2-12 — Metal SSM: SSM_CONV_BACK + SSM_SCAN_BACK ports

**One-line outcome:** Mamba-family LoRA training is GPU-resident on Apple Silicon —
`SSM_CONV_BACK` and the chunk-recompute `SSM_SCAN_BACK` run on Metal with MODE_GRAD parity
against the S1-30/S1-31 CPU oracles, and the tiny-Mamba e2e trains with `--device metal`.

## Why (context)

All four LoRA-able Mamba projections are plain `MUL_MAT` through `build_lora_mm`, already
covered by the stage's base `OUT_PROD` work; what blocks SSM training on Metal is gradient
flow *through* the two SSM ops (ROADMAP §10). The CPU reference kernels exist: S1-30's
flipped-window correlation for `SSM_CONV_BACK` (S2) and S1-31's chunk-recompute
`SSM_SCAN_BACK` (S3, the flagship). Unlike MoE, every backend already has the SSM
**forwards**, so each port is a same-shape sibling kernel (ROADMAP §10) — Metal's are the
`kernel_ssm_conv_f32_f32` family
(`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:2118-2229`, four variants) and
`kernel_ssm_scan_f32` (`metal:2276`), gated on `has_simdgroup_reduction` in supports_op
(`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m:1267-1269`) and encoded by
`ggml_metal_op_ssm_conv`/`ggml_metal_op_ssm_scan`
(`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:1390`, `:1463`).

The structural problem the scan backward inherits from the CPU work: the forward
overwrites intermediate recurrent states in place (CPU: `s0 = s;`,
`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:9770`; the Metal kernel mirrors that
reference, per its own header comment), so backward must **recompute states** — checkpoint
every K tokens, then per chunk re-run the forward recurrence and reverse-scan (S1-31's
schedule; do not attempt the algebraic state inverse — `dA = exp(dt_softplus·A)`
underflows). ROADMAP §12 Q7 asks where the checkpoint buffer lives; on Metal the natural
answer is the backend's per-op fleeting-data mechanism:
`ggml_backend_metal_buffer_type_get_alloc_size` over-allocates the dst buffer for ops that
need scratch (`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal.cpp:213-227`), with FA's
`ggml_metal_op_flash_attn_ext_extra_tmp` as the precedent
(`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2620`). That is the "pool alloc"
resolution the manifest names — no allocator changes needed.

Determinism is gate G-B (S0-09/ADR-0002): `dB`/`dC` reduce over GQA-style head groups
(the forward's `g = h / (nh/ng)` repeat_interleave, CPU `ops.cpp:9625`), which invites
atomics; the port instead uses fixed-order simd reductions and thread-exclusive ownership,
matching the CPU oracle's scheme so parity checks are meaningful.

## What to do

All code lands in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **`SSM_CONV_BACK` kernel:** sibling of the conv forward family (`metal:2118-2229`),
   implementing S1-30's semantics — `d_sx(j, i1) = Σ_t dy(i1, t) · c(j−t, i1)` for
   `j−t ∈ [0, d_conv)`, only `d_sx` (conv weight frozen). Mirror the forward's
   partitioning so each thread owns disjoint `d_inner` rows — deterministic, no atomics.
   Start from the plain variant; add the `_4`/batched siblings only if the encoder's
   dispatch heuristics need them for the test shapes.
2. **`SSM_SCAN_BACK` kernel** patterned on `kernel_ssm_scan_f32` (`metal:2276`):
   implement S1-31's chunk-recompute schedule — pass 1 re-runs the forward recurrence
   storing every K-th state to the checkpoint scratch; pass 2 walks chunks in reverse,
   re-materializing the chunk's states then reverse-scanning to produce the packed
   `dx‖ddt‖dB‖dC` dst. Reverse-pass math and softplus/`sigmoid(dt)` chaining exactly per
   the S1-31 CPU reference (forward's `dt_soft_plus` at CPU `ops.cpp:9623`); handle both
   the Mamba-2 scalar-decay and Mamba-1 branches. K is the same op/config parameter as the
   CPU kernel (identical chunking so cross-backend comparisons align). F32 accumulation
   throughout (ADR-0002).
3. **Checkpoint scratch:** add a `GGML_OP_SSM_SCAN_BACK` case to
   `ggml_backend_metal_buffer_type_get_alloc_size` (`ggml-metal.cpp:213-227`) sizing
   `n_checkpoints × state-size` (+ per-chunk working states) after FA's `extra_tmp`
   precedent (`ggml-metal-ops.cpp:2620`); expose the sizing helper from
   `ggml-metal-ops.h` as the existing ones are.
4. **Determinism (gate G-B):** threads partition heads for `dx`/`ddt` as the forward does;
   `dB`/`dC` group reductions use `simd_sum` plus fixed-order combination across the
   `nh/ng` heads of a group (or group-ownership per threadgroup) — no atomics; document
   the write-ownership argument in a kernel comment.
5. **The five mechanical additions ×2 ops:** kargs structs, pipeline getters, encoder
   cases next to `ggml_metal_op_ssm_conv`/`_scan` (`ggml-metal-ops.cpp:1390`, `:1463`),
   and supports_op cases in `ggml-metal-device.m` next to the forwards (`:1267-1269`),
   gated on `has_simdgroup_reduction`.
6. **MODE_GRAD on Metal:** run the S1-30/S1-31 grad-enabled cases (`test_ssm_conv`,
   `vendor/llama.cpp/tests/test-backend-ops.cpp:3753`; `test_ssm_scan`, `:3819`, base
   shapes at `:8507-8509`) on Metal vs the CPU oracles: Mamba-1, Mamba-2, `n_group > 1`
   (exercises the group reduction), multi-sequence, and `n_seq_tokens` below/above K
   (exercises chunk boundaries). Cross-backend parity ≤ 0.05 @ fp16 per ADR-0002.
7. **Determinism check:** two Metal runs on identical inputs produce bitwise-identical
   packed grads (fork-side harness per the S1-31 pattern).
8. **e2e:** S1-31's tiny-Mamba SFT fixture with `--device metal` in the `ci-metal`
   nightly: loss falls; the S2-01 fallback report shows `SSM_CONV_BACK`/`SSM_SCAN_BACK`
   executing on Metal, with any remaining Mamba-path CPU fallbacks enumerated (reported,
   not forbidden — S2-10's dense set is unaffected).
9. **Submodule bump PR** in llama-farm per S0-02.

## Out of scope

- CUDA and Vulkan SSM backward ports — S3-09 / S4-07 (the CUDA reversed-loop
  register-pressure half of ROADMAP §12 Q7 belongs there).
- `ds0`, `dA`, `dD`, conv-weight and `dt_bias` grads; cross-ubatch BPTT — ROADMAP §10 S5,
  deferred (truncation-at-ubatch semantics documented by S1-31).
- RWKV / `GATED_DELTA_NET` linear-attention backward — ROADMAP §10 "Others", unscheduled.
- CPU kernel changes and K-parameter tuning studies — S1-31 owns the reference; tuning is
  perf work (K5).

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops` mode `grad` passes on Metal for all S1-30/S1-31 grad
      cases (conv shapes; scan: Mamba-1/Mamba-2/`n_group>1`/multi-seq/chunk-boundary)
      within ADR-0002 tolerances and ≤ 0.05 @ fp16 parity vs the CPU oracles.
- [ ] Metal and CPU produce matching grads with the same K on the chunk-boundary cases
      (identical chunking verified in at least one case log).
- [ ] Determinism: bitwise-identical `dx‖ddt‖dB‖dC` across two Metal runs.
- [ ] The checkpoint scratch is sized via `get_alloc_size` (code inspection + a wide-shape
      case that would overflow without it passes).
- [ ] `ci-metal` nightly tiny-Mamba e2e: loss falls with `--device metal`; fallback report
      lists both new ops on Metal.
- [ ] llama-farm submodule-bump PR green in `ci-metal / build` and `ci-metal / grad`.

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` MODE_GRAD, Metal vs the S1-30/S1-31 CPU
oracles — per-PR in the targeted `ci-metal / grad` lane, full sweep nightly. The
tiny-Mamba e2e joins the `ci-metal` nightly with the S2-01 fallback report. Determinism
checks run as fork-side tests in the same lanes. Local dry-run: any Apple Silicon machine
per the S0-08 playbook, using the exact CI invocations.

## PR notes

- Branch: `ticket/S2-12-metal-ssm-backward-ports`.
- Two-repo flow per S0-02: fork PR against `llama-farm-base` (conv-first then scan commits
  recommended for review), plus a trivial llama-farm submodule-bump PR.
- Upstreaming disposition: **fork-local first, upstream-later** — rides the `SSM_*_BACK`
  op-family RFC with S1-29/30/31 once the CPU oracle plus one GPU backend prove the design
  (ROADMAP §11 triage b).
- Provenance per S0-01: kernels carry headers naming their pattern sources
  (`kernel_ssm_conv_f32_f32`/`kernel_ssm_scan_f32`, `ggml-metal.metal`, MIT, `4f37f51`);
  the reverse-pass math is this project's own derivation (ROADMAP §10 S3), not translated
  code.
