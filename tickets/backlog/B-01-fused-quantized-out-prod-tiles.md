---
id: B-01
title: "Fused quantized OUT_PROD tile kernels (CUDA/Vulkan) — only if profiling triggers"
stage: backlog
track: kernels
size: XL
deps: [S3-02, S4-03]
status: open
pr: null
---

# B-01 — Fused quantized OUT_PROD tile kernels (CUDA/Vulkan) — only if profiling triggers

**One-line outcome:** **DEFERRED** — fused dequant+rank-k `OUT_PROD` kernels that eliminate the
dequant round-trip shipped in S3-02 (CUDA) and S4-03 (Vulkan), built only if profiling triggers.

**Activation trigger:** profiling from the S3-10 (CUDA) or S4-09 (Vulkan) milestone runs shows the
dequant round-trip dominating `OUT_PROD` wall time at realistic training shapes, **or** the dequant
transients are the binding memory constraint on a target config (ROADMAP §12 Q1 / risk R2).

## Why (context)

The shipped quantized `OUT_PROD` is deliberately unfused: CUDA dequantizes src0 to F16 and runs
`cublasGemmEx` (S3-02, ROADMAP §5 C1); Vulkan dequantizes and reformulates onto the tuned `mul_mm`
pipeline (S4-03, ROADMAP §7 V2). Per §5 C1, a custom quantized kernel cannot reuse int8 mmq/mmvq —
the `OUT_PROD` reduction axis (src0 rows, ne01) is orthogonal to the quant-block axis (ne00), and
every dp4a/int-mma tile design assumes blocks lie along the reduction axis
(`vendor/llama.cpp/ggml/src/ggml-cuda/mmq.cuh:17-45`) — and a fused dequant+rank-k kernel gets no
tensor cores and is expected to lose to `GemmEx` at training batch sizes. **Re-verify that by
measurement before building anything** (ROADMAP §12 Q1).

Two payoffs can activate this: throughput (conversion overhead dominating) and memory (transient
elimination — S3-02 chunks the lm_head case precisely because whole-tensor dequant is ≈ 1 GB F16
at 128k×4096; a fused kernel needs no F16 transient). The Vulkan fused alternative is pre-scoped
L/XL: per-quant-type `out_prod` tile shaders — roughly one per supported quant type (see the
`pipeline_dequant` table, `vendor/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:4996-5018`) times
warptile variants, the ~25×N shader explosion §7 V2 defers.

## What to do

1. Attach the trigger evidence: `OUT_PROD` wall-time breakdown (dequant / src1 convert / GEMM) and
   transient sizes from the S3-10/S4-09 profiles at 7–8B dense-LoRA shapes.
2. **CUDA prototype first:** one quant type (Q4_K) fused dequant+rank-k tile kernel vs the S3-02
   path (`ggml_cuda_out_prod`, `vendor/llama.cpp/ggml/src/ggml-cuda/out-prod.cu:27`) at training
   batch sizes. If `GemmEx` wins and the memory win is immaterial, close with numbers recorded —
   a valid outcome.
3. If it wins: generalize across quant types via the per-type dequant device functions;
   deterministic accumulation only (no atomics — ADR-0002 / gate G-B); preserve S3-02's chunking
   semantics and `supports_op` surface (path selection internal).
4. Vulkan: per-quant-type `out_prod` tile shaders patterned on the `mul_mm` warptiles, F32
   accumulation, generated for the `pipeline_dequant` types; keep S4-03 as the fallback path.
5. Benchmark fused vs round-trip per quant type/shape/backend; commit the report under `benches/`;
   default to the fused path only where it wins.

## Out of scope

- Metal: S2-06 already fuses dequant into its tiled kernel — no round-trip to eliminate.
- MoE variants `OUT_PROD_ID(_GRP)` (S3-08/S4-06); any `OUT_PROD` semantic or ABI change.

## Acceptance criteria

- [ ] Trigger report committed (or ticket closed with prototype numbers showing the round-trip wins).
- [ ] `test-backend-ops` MODE_GRAD `-o OUT_PROD` passes vs the CPU oracle within the ADR-0002
      tolerance on CUDA and Vulkan with the fused path forced, for every covered quant type.
- [ ] Determinism: bitwise-identical dst across reruns (no atomics).
- [ ] Benchmark report in `benches/`; default-path selection matches it.
- [ ] `ci-cuda` and `ci-vulkan` green, including targeted `OUT_PROD` sweeps.

## Testing & verification

Vendored `tests/test-backend-ops` `test` + `grad` vs the CPU oracle (ADR-0002), fused path forced:
`ci-cuda` GPU lane and `ci-vulkan` native lane per-PR (targeted), nightly full sweeps incl. the
coopmat/MoltenVK tiers. Benchmarks in `benches/` on the stage VMs.

## PR notes

- Branch: `ticket/B-01-fused-quantized-out-prod-tiles`.
- Two-repo flow per S0-02: fork PR against `learning-llamas-base` + learning-llamas submodule bump.
- Upstreaming disposition: **upstream-later** — pure perf kernels; propose upstream once
  benchmarks prove wins across ≥2 GPU generations (ROADMAP §11 triage b).
