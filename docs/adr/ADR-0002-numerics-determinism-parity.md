# ADR-0002 — Numerics policy, determinism default (gate G-B), and the parity criterion

- **Status:** Accepted
- **Date:** 2026-07-12
- **Ticket:** S0-09
- **Decides:** ROADMAP gate **G-B** (determinism default)
- **Binds:** every kernel ticket in stages 1-4

## Context

Stages 1-4 add roughly forty new or ported kernels across four backends. Without one
authoritative numerics document, each PR re-argues precision from scratch, and "is this gradient
correct?" gets answered forty different ways. ROADMAP §3 states the policy; this ADR freezes it,
so an acceptance criterion reading "MODE_GRAD parity within the ADR-0002 tolerance" is
well-defined.

It also settles **gate G-B** (ROADMAP §9, §11, open question Q9). `atomicAdd` scatter over ragged
expert segments, and the atomic-dQ formulation of flash-attention backward, both make gradients
nondeterministic run to run. That choice constrains the *design* of the E2/E3-class MoE ops and
of FA5's dQ strategy — so it has to be recorded before those tickets start, not discovered
inside them.

## Decision 1 — F32 accumulation on every gradient path

- **Gradient matmuls accumulate in F32 on every backend**, without exception.
- **Row statistics, log-sum-exp, and loss values are computed in F32.**
- **Elementwise math runs in F32**, with casts only at storage boundaries.

Storage may be F16/BF16/quantized; *accumulation* may not. The forward pass is free to use
whatever the backend's inference path uses — this ADR governs gradients only.

## Decision 2 — Two forbidden patterns, named

Both were verified in the vendored tree at the pinned commit. Both are *inference* optimizations
that are silently wrong on a gradient path, and both are reachable by simply reusing the existing
matmul entry points, which is exactly what a kernel author will reach for first.

### (a) CUDA: cuBLAS with the default F16 traits

`ggml_cuda_mul_mat_cublas_impl` (`ggml/src/ggml-cuda/ggml-cuda.cu:1325`) is templated on the
compute type, and the F16 specialization sets **F16 accumulation**:

```cpp
template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;   // <-- ggml-cuda.cu:1313
    ...
};
```

There *is* an F32-output path, but it is a **per-architecture heuristic**, not a policy
(`ggml-cuda.cu:1426-1431`):

```cpp
    bool prefer_f32_output = false;
    if (compute_type == GGML_TYPE_F16) {
        prefer_f32_output = cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);
    } else if (compute_type == GGML_TYPE_BF16) {
        prefer_f32_output = !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_CDNA(cc);
    }
```

Read that carefully: for F16 it is true **only** on Volta, RDNA4, and CDNA. On Ampere, Ada, and
Hopper — the GPUs people actually train on — `prefer_f32_output` is **false**, and an F16 GEMM
accumulates in F16.

> **Normative:** gradient GEMMs on CUDA must pin **`CUBLAS_COMPUTE_32F`** unconditionally, on
> every architecture. It is not sufficient to rely on `prefer_f32_output`; a kernel that does so
> is correct on a V100 and quietly wrong on an A100. (BF16's traits already use
> `CUBLAS_COMPUTE_32F`, `ggml-cuda.cu:1299`, and are safe.)

### (b) Vulkan: the `f16acc` `mul_mm` pipeline variants

Vulkan's tuned matmul pipelines come in two accumulation flavors, and the struct carries both
(`ggml/src/ggml-vulkan/ggml-vulkan.cpp:216`):

```cpp
    vk_matmul_pipeline f16acc;
```

> **Normative:** the **`f16acc`** variants are inference-only and must never be selected on a
> gradient path. Vulkan gradient matmuls select the F32-accumulation variant.

## Decision 3 — Gate G-B: deterministic by default, atomics as measured opt-in

**Deterministic segmented / exclusive-write schemes are the project default for every backward
kernel.**

Atomics-based variants are permitted **only** when all three hold:

1. they are opt-in behind an explicit flag (never the default path);
2. a benchmark in the PR shows a *measured* win on real shapes;
3. the deterministic implementation still exists and is still tested.

**Bound tickets.** Every backward kernel PR must state, in its description, which scheme it
implements. The ops this decision was made *for*:

| Op / family | Ticket(s) | Why it is bound |
|---|---|---|
| `OUT_PROD_ID` (E2-class) | S1-26, S2-11, S3-08, S4-06 | `atomicAdd` scatter over ragged expert segments |
| `OUT_PROD_ID_GRP` (E3-class) | S1-27, S2-11, S3-08, S4-06 | same |
| Flash-attention backward, **dQ strategy** | S1-23, S2-13, S3-06/S3-07, S4-08 | atomic-dQ is the classic formulation and is nondeterministic |

**Consequence, stated as a promise:** with a fixed backend, a fixed build, and a fixed seed,
learning-llamas training runs are **bit-reproducible** by default. A user who cannot reproduce a
loss curve has found a bug, not a floating-point fact of life. That property is worth real
performance, and this ADR spends it deliberately.

## Decision 4 — The acceptance harness and two tolerance layers

The vendored `test-backend-ops` harness in mode **`grad`** (MODE_GRAD) is the acceptance harness
for every new or ported kernel, with the **CPU implementation as the oracle**. GPU ports never
redefine semantics; they match CPU within tolerance.

The two layers are different things and are routinely conflated:

**(a) The per-op finite-difference bound.** Within one backend, MODE_GRAD compares the analytic
gradient against a finite-difference estimate, using the harness's mean-abs-asymmetry machinery.
The default bound is `max_maa_err() = 1e-4` (`tests/test-backend-ops.cpp:1158-1160`), overridable
per test case. Discontinuous gradients (ReLU-like kinks, where a finite-difference estimate
straddling the kink is simply wrong) use the harness's expected-value filtering
(`tests/test-backend-ops.cpp:319-321`) rather than a loosened bound — the right fix for a
discontinuity is to *not sample across it*, not to accept a larger error.

**(b) The cross-backend parity criterion.** Between backends, on identical inputs:

> **max-abs gradient error ≤ 0.05 at fp16**, GPU backend vs the CPU oracle.

That is the bar for declaring a backend's kernel "at parity", and it is what every backend
milestone ticket (S2-10, S3-10, S4-09) measures. The threshold follows the external precedent
cited in ROADMAP §13 item 8: unsloth's own kernel self-tests accept the same 0.05 max-abs
gradient error at fp16. (Prior art for a threshold — no unsloth code is used; see
docs/PROVENANCE.md.)

Tightening either bound later requires amending this ADR.

## Consequences

**Every kernel ticket must include:**

- a `test-backend-ops` MODE_GRAD case for the new op, passing against the CPU oracle;
- an explicit statement of which scheme it implements (Decision 3) — "deterministic segmented"
  or "atomic, opt-in, benchmarked";
- no instance of either forbidden pattern (Decision 2) on a gradient path.

**Every vendor bump re-runs the full MODE_GRAD suite** before its submodule-bump PR merges
(ADR-0001, decision 4). This is what keeps "CPU is the oracle" from quietly decaying: an upstream
change to a shared kernel can move the oracle itself, and the finite-difference suite is the only
thing that would notice.

**We accept a performance cost** for determinism (Decision 3) and for F32 accumulation
(Decision 1). Both are recoverable later behind measured, opt-in flags. Neither is recoverable if
we ship nondeterministic gradients first and try to add reproducibility afterwards.

## Out of scope

- **Gate G-A** — the sparse-CE cross-backend ABI (LSE stash vs recompute, logit-buffer aliasing,
  softcap/scale op-params). Decided in **ADR-0003**, inside S1-04.
- **FA-backward numerics details** — the forward FTZ threshold, the KQ max-offset, and the
  definition of LSE-with-sinks. Constrained by this ADR but decided in the FA ticket family
  (ROADMAP §12 item 4).
- **Per-op tolerance overrides.** Chosen inside individual kernel tickets, within the bounds set
  here.
