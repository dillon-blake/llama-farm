---
id: S4-06
title: "Vulkan MoE: OUT_PROD_ID + OUT_PROD_ID_GRP ports"
stage: 4
track: kernels
size: L
deps: [S4-03, S1-26, S1-27]
status: open
pr: null
---

# S4-06 — Vulkan MoE: OUT_PROD_ID + OUT_PROD_ID_GRP ports

**One-line outcome:** MoE LoRA training is GPU-resident on Vulkan — expert-compacted
quantized outer products (`OUT_PROD_ID`) and grouped F32 expert grads (`OUT_PROD_ID_GRP`)
execute on the Vulkan backend, deterministic by default, MODE_GRAD-parity-checked against
their S1-26/S1-27 CPU oracles.

## Why (context)

MoE training is exactly one missing backward case (`MUL_MAT_ID`) that decomposes into
these two ops (ROADMAP §0 finding 3, §9 E2/E3), and both are required even for LoRA-only
training: `build_lora_mm_id` computes `mul_mat_id(B, mul_mat_id(A, cur, ids), ids)` — the
trainable LoRA A/B stacks are themselves the 3D expert operand of `mul_mat_id`
(`vendor/llama.cpp/src/llama-graph.cpp:1438-1442`). S1-25/26/27 delivered the wiring and
the CPU reference kernels; this ticket ports both ops to Vulkan so the MoE backward stops
falling back to CPU via `ggml_backend_sched` (ROADMAP §11 scheduler note).

No new matmul shader is needed. `OUT_PROD_ID` reuses the S4-03 (V2) composition machinery
wholesale — dequant the expert matrix to F16 with the existing `pipeline_dequant[type]`
pipelines (`vendor/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:4996-5018`, staged in a
`prealloc_x`-style transient exactly as `ggml_vk_mul_mat_q_f16` does, `:8600`), transpose
via `copy_transpose.comp`, and run the tuned `mul_mm` pipeline — extended here with
per-expert segmenting. Note the segmenting has no in-backend precedent: Vulkan's
`mul_mat_id` forward gathers row ids inside the `mul_mm` shader
(`ggml_vk_mul_mat_id_q_f16`, `ggml-vulkan.cpp:9505`) rather than compacting first, so the
compaction pass is new backend-side work; its vocabulary (`expert_bounds` prefix array
over compacted (token, slot) pairs) deliberately matches CUDA's `mm_ids_helper`
(`vendor/llama.cpp/ggml/src/ggml-cuda/mmid.cu:28-60`, bounds written at `:109-115`) and
the S1-26/S1-27 CPU workspace (`vendor/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c:1574-1637`).
`OUT_PROD_ID_GRP` is pure F32 segmented `mul_mm` launches and is not gated on the
quantized machinery at all (ROADMAP §11 hard dependencies; §9 E3).

Two project rules bind the design. Numerics (ADR-0002/S0-09): gradient matmuls force the
F32-accumulation `mul_mm` variants — the tuned pipelines carry inference-only `f16acc`
twins (`vk_matmul_pipeline2`, `ggml-vulkan.cpp:216`) that are forbidden on grad paths.
Determinism (gate G-B, decided in S0-09): ragged expert segments make atomic scatter
nondeterministic, so both ops execute per-expert segments in fixed order with exclusive
writes; atomics only as a later measured opt-in.

## What to do

All ggml changes land in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **Expert-compaction pass:** a small compute shader over `ids` producing
   `ids_compact`/`expert_bounds` buffers in transient device memory (no subgroup
   arithmetic — shared-memory scans only, per the MoltenVK/AMD force-disable rule,
   `vendor/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:5983-5994`). Because Vulkan
   records command buffers before results exist, per-expert dispatch sizes cannot be read
   on the host: choose between (a) fixed worst-case per-expert dispatches whose shaders
   read `expert_bounds` on-device and early-exit empty/short segments, and
   (b) `vkCmdDispatchIndirect` driven from the compaction output. Record the decision and
   rationale in a code comment and the fork PR. Either way, segments execute in fixed
   expert order — the gate G-B determinism argument.
2. **`OUT_PROD_ID(as, grad, ids) → dB`** (semantics and `s' = s % ne_b1` broadcast rule
   per the S1-26 contract): per expert segment, dequant expert `e` to F16 via
   `pipeline_dequant[type]` (`:4996-5018`), transpose via `copy_transpose.comp`, gather
   the segment's grad columns, run the tuned `mul_mm` pipeline pinned to **F32-acc
   variants** (never the `f16acc` twins, `:216`), and accumulate into the zero-initialized
   F32 dst in fixed segment order. Type coverage: every type with a non-null
   `pipeline_dequant` entry, plus F32/F16 expert stacks.
3. **`OUT_PROD_ID_GRP(b, grad, ids, n_expert) → dAs`** (semantics per S1-27; dst
   `[n_in, n_out, n_expert]`, F32-only assert): segmented F32 `mul_mm` launches per
   `expert_bounds`, one accumulating GEMM per expert into its `dAs[:, :, e]` slab —
   exclusive writes, deterministic by construction. Zero-fill the whole dst first so
   empty experts are exactly zero. This op may land first within the PR sequence (not
   blocked on the dequant path).
4. **Plumbing:** the six mechanical touch points per new op (ROADMAP §7) for both ops,
   including `supports_op` cases in `ggml_backend_vk_device_supports_op`
   (`ggml-vulkan.cpp:17158`; `MUL_MAT_ID` precedent at `:17234-17235`): `OUT_PROD_ID`
   accepts dequant-supported/F16 `as` with F32 grad/dst; `OUT_PROD_ID_GRP` F32 only.
   Unsupported cases keep falling back to the CPU oracles via sched.
5. **Tests:** re-run the S1-26/S1-27 MODE_GRAD case lists on Vulkan — the grad-enabled
   `test_mul_mat_id` cases (`vendor/llama.cpp/tests/test-backend-ops.cpp:4248`,
   live instantiations at `:8722-8753` including the all-types sweep at `:8735-8737`; the
   emplacements at `:8693-8696` sit in a disabled `#if 0` block — do not count on them):
   quantized type sweep (pattern: the quantized `test_out_prod` cases,
   `:8780-8806`), broadcast on/off, ragged assignment, empty
   experts, and S1-27's nested `build_lora_mm_id`-shaped two-level case — vs the CPU
   oracles within the ADR-0002 tolerance, on lavapipe and on the native lane's scalar
   **and** coopmat drivers (the composition must be correct under both `mul_mm` paths).
   Add a fork-side determinism check: two identical runs produce bitwise-identical
   dB/dAs.
6. **e2e:** run S1-28's tiny-MoE training test with `--device vulkan` — correctness on
   lavapipe, perf numbers only from the native lane (S4-01 split); the fallback report
   must show both ops executing on Vulkan.
7. **Submodule bump PR** in llama-farm per S0-02, appending both ops to the ci-vulkan
   targeted-op defaults.

## Out of scope

- Metal/CUDA ports — S2-11 / S3-08 (same op contracts, same CPU oracles).
- Fused per-quant-type expert outer-product tile shaders — backlog B-01
  (profiling-triggered, ROADMAP §12 Q1).
- Atomics-based opt-in variants (gate G-B allows them later, measured) — backlog.
- Per-expert bias grads, router full training, quantized-expert full FT — ROADMAP §9 E8,
  deferred.
- The CPU kernels and backward wiring — S1-25/26/27 (consumed here as oracles).

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops grad -b Vulkan0` passes for the `OUT_PROD_ID`
      (`test_mul_mat_id` activation-grad) cases across the quantized type sweep plus
      F32/F16, broadcast on/off, ragged and empty-expert cases, within the ADR-0002
      tolerance vs the S1-26 CPU oracle (≤ 0.05 max-abs @ fp16), on lavapipe and on the
      native lane (scalar and coopmat drivers both recorded).
- [ ] Fork branch: the `OUT_PROD_ID_GRP` (F32 `as`-as-param) cases pass on Vulkan,
      including the nested `build_lora_mm_id`-shaped case; empty-expert slabs are exactly
      zero.
- [ ] Determinism: two identical Vulkan runs produce bitwise-identical dB and dAs.
- [ ] No `f16acc` pipeline is selectable on either op's path (code-review assertion or
      runtime check in the fork diff).
- [ ] The segmenting-mechanism decision (worst-case dispatch vs indirect dispatch) is
      recorded in the fork PR with rationale.
- [ ] Tiny-MoE e2e passes with `--device vulkan` on the lavapipe lane; the native-lane
      run's fallback report shows `OUT_PROD_ID`/`OUT_PROD_ID_GRP` executing on Vulkan.
- [ ] llama-farm submodule-bump PR is green in `ci-vulkan / lavapipe` (per-PR) and
      `ci-vulkan / gpu` (label-gated), plus `ci-cpu`.

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` MODE_GRAD on the Vulkan backend vs the
S1-26/S1-27 CPU oracles under the ADR-0002 tolerances (S0-09), plus the fork-side
determinism check. Runs per-PR on `ci-vulkan / lavapipe` (targeted op list) with the full
sweep nightly, and on the label-gated `ci-vulkan / gpu` native lane (S4-01), which also
records driver caps for the scalar-vs-coopmat coverage claim. The tiny-MoE e2e joins the
nightly ci-vulkan e2e job; S4-09 later folds both ops into the fallback-forbidden set.

## PR notes

- Branch: `ticket/S4-06-vulkan-moe-out-prod-ports`.
- Two-repo flow per S0-02: implementation PR against the fork's `llama-farm-base` branch
  with the ticket ID in the title, plus a trivial llama-farm submodule-bump PR
  referencing the same ticket ID.
- Upstreaming disposition: **fork-local first, upstream-later** — the op enums ride the
  E2/E3 op-family RFC once the CPU oracle plus one GPU backend prove the design (ROADMAP
  §11 triage b); coordinate with S2-11/S3-08 on which backend anchors the RFC.
- Provenance headers per S0-01 policy: adapted in-tree MIT code (S4-03 composition,
  `copy_transpose.comp`, dequant pipelines; compaction vocabulary from `mmid.cu`;
  llama.cpp `4f37f51`).
