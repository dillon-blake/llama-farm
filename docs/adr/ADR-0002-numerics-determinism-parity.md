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

### A MODE_GRAD case is vacuous unless the test calls `ggml_set_param`

*(Amended by S0-10, which measured this rather than assuming it.)*

`test_case::eval_grad` checks gradients **only if** the test's `build_graph` marked something as
a parameter. If nothing is a parameter, it prints `not supported [<OP>]`, checks nothing, and the
run still ends in `Backend CPU: OK`.

At the pinned commit, **52 of the 100 `test_case` classes never call `ggml_set_param`** — and the
list includes `test_out_prod`, `test_flash_attn_ext`, `test_mul_mat_id`, `test_ssm_scan`,
`test_ssm_conv`, `test_glu`, and `test_clamp`. That is, very nearly, the exact set of ops this
project exists to implement. `test-backend-ops grad -o OUT_PROD` reports `Backend CPU: OK` while
checking **zero** gradients.

> **Normative:** a kernel ticket does not satisfy its MODE_GRAD acceptance criterion by adding a
> test case. It must ensure the `test_case` **calls `ggml_set_param` on the input whose gradient
> it means to check**, and it must state in the PR how many cases were actually grad-checked.
> An acceptance criterion that passes without checking a gradient is worse than none, because it
> looks like evidence.

Also: `test-backend-ops` prints `N/M tests passed` where **M is the global case count of the
entire binary, not of the `-o` filter** — `grad -o ADD`, `grad -o OUT_PROD`, and
`grad -o CROSS_ENTROPY_LOSS` all print the same denominator. Do not quote that number as evidence.
The signal is the `Backend CPU: OK` / `FAIL` verdict and the per-case lines.

The measured state of every op is in
[`docs/dev/backward-coverage.md`](../dev/backward-coverage.md).

### The two tolerance layers

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

### A finite difference through a quantized base measures nothing

*(Amended by S1-03, which measured this rather than assuming it. It invalidates that ticket's own
stated acceptance criterion, so it is recorded here rather than in a PR description.)*

llama.cpp does not implement a quantized matmul by dequantizing the weights and multiplying in
float. It **quantizes the activations** to the weight type's `vec_dot_type` — Q8_K for a Q4_K
weight, Q8_0 for a Q8_0 one — and dot-products in the integer domain.

So a small perturbation of a trainable weight makes a small perturbation of the activations, which
*usually changes their 8-bit codes not at all*, and *occasionally flips one by a whole quantum*.
**The forward pass is a step function of the weights**, with steps of order 1e-3 in the loss.
Measured on the tiny fixture, sweeping one element of a LoRA `B`:

```
  B[0]      Q4_K base        F32 base
 -0.040   6.3610424995    6.2726659775
 -0.020   6.3612122536    6.2726297379
  0.000   6.3611059189    6.2725958824
 +0.020   6.3608088493    6.2725639343
 +0.040   6.3606944084    6.2725348473
```

The F32 column is a straight line. The Q4_K column has **no trend at all**: the true signal — a
slope of ~1e-3 across the entire sweep — is buried under the quantization steps. No choice of `eps`
recovers it, because shrinking `eps` shrinks the signal and leaves the steps exactly where they are.

The backward, meanwhile, differentiates the **smooth dequantized** function: the MUL_MAT gradient is
`ggml_out_prod(W, grad)`, which dequantizes `W`. That is the correct and intended behaviour — the
standard straight-through treatment of a non-differentiable quantizer. The analytic gradient and the
finite difference are therefore computing **different functions**, and neither is wrong.

> **Normative:** an end-to-end finite-difference gradient check must run on an **F32 base**. A ticket
> that reports one on a quantized base is reporting noise, and a tolerance loose enough to let it
> pass is loose enough to hide a genuinely wrong gradient. (The S1-02 backward bug — a masked-CE
> loss whose gradient was wrong for every non-unit weight — sat at a relative error of 0.4–1.4,
> which is *inside* the noise floor of a Q4_K finite difference.)

The quantized backward is instead validated **against the F32 backward**, and the evidence is the
*trend*, not any single threshold: a backward that dequantizes correctly must degrade gracefully and
monotonically with bit width, because bit width is the only thing that changed. Measured on
`blk.0.attn_q`, same adapter, same batch:

```
  q8_0   cosine 0.99991    0.8° from the F32 gradient   magnitude x0.998
  q4_k   cosine 0.97791   12.1° from the F32 gradient   magnitude x1.019
```

A backward that dequantized the wrong way, or transposed, has no reason to be three orders of
magnitude tighter at 8 bits than at 4. That ordering is the assertion; see
`tests/test_p0_gradient.py::test_the_quantized_backward_tracks_the_f32_backward`.

### Determinism is per-shape: a BLAS build changes kernel at 32 tokens

*(Amended by S1-07/S1-05, found by CI: a test passed on Linux and failed on macOS.)*

ggml dispatches `MUL_MAT` to a BLAS backend **only when the ubatch has at least 32 tokens** —
`min_batch = 32` in `ggml/src/ggml-blas/ggml-blas.cpp`. macOS enables Accelerate by default, so on
that build a 16-token batch and a 32-token batch run **different matmul kernels**, and the *same
token's* logits come out slightly differently.

Measured: the per-valid-token loss of one sample, evaluated at two paddings.

```
                seq_len 16     seq_len 32     relative
  macOS/BLAS    6.259182       6.258213       1.5e-4
  Linux/no BLAS   bit-identical across 16, 24, 32, 48, 64
```

Neither number is wrong. The invariance being violated is exact in real arithmetic and exact within
one kernel; what breaks it is the *dispatch*, not the batching.

> **Normative:** the determinism guarantee of Decision 3 is **per shape**. Two runs of the same data
> at the same ubatch size are bit-identical; two runs at different ubatch sizes are not comparable
> bit-for-bit on a build with BLAS, and a test that compares across a batch size of 32 is measuring
> kernel selection rather than whatever it meant to measure.
>
> A test that must compare across paddings therefore keeps both lengths on the **same side of 32**,
> which lets it assert *exact equality* — a far stronger claim than a tolerance, and one that a real
> masking bug (a graded pad, a denominator that counts pads) breaks by a **factor**, not by parts in
> ten thousand. See `tests/test_sft.py::test_padding_does_not_change_the_loss`.

### A MODE_GRAD case is also vacuous if the objective conserves the output's sum

*(Amended by S1-34, which found it the hard way.)*

S0-10 recorded one way for a MODE_GRAD case to check nothing: never calling `ggml_set_param`. There
is a second, and it is subtler, because the case *is* a parameter and the harness *does* compare a
gradient.

**MODE_GRAD differentiates `sum(out)`.** For an op whose output has a **conserved sum**, that
objective has no gradient at all. A softmax is the obvious one: its rows sum to one by construction,
so `sum(out)` is identically 1 whatever the input, and `d(sum(out))/dx` is *exactly zero*. The
analytic gradient is zero, the finite difference is zero, they agree, and the case reports `OK`.

`test-backend-ops grad -o SOFT_MAX` had been reporting `OK` on every case without an attention sink,
and every one of them was comparing zero against zero. It had never exercised `SOFT_MAX_BACK`.

(It was *failing* on the sink cases — and only those — because a sink takes part in the
normalization but produces no output, so the rows sum to `1 - p_sink`, which does depend on the
input. Those were the only cases in the test with a nonzero gradient, hence the only ones that could
fail. The kernel was right all along: verified against a float64 finite difference to an MAA of
2.1e-7, against a harness bound of 1e-4.)

> **Normative:** a kernel whose output has a conserved sum — a softmax, a normalization, anything
> that divides by its own total — must override `test_case::grad_loss` with an objective that
> actually depends on its input. A weighted sum with unequal weights suffices. The fork adds that
> hook; `grad_loss` defaults to `sum(out)`, which remains correct for everything else.
>
### ...and a case is vacuous if the objective cannot distinguish a wrong answer

*(Amended by S1-29.)*

The third way, and it is the same disease as the second.

`sum(out)` makes `dL/d(out)` **1 everywhere**. For an op that merely **routes** its input to its
output — concat, view, permute, transpose, cpy — the gradient is "hand each source back its own slab
of `dL/d(out)`", and **every slab of an all-ones tensor is all ones**. Swap the slabs, hand both
sources the same one, offset them wrongly: the gradient is *identical*, and the case reports `OK`.

Measured on `CONCAT`: with `sum(out)` and both sources flagged as parameters, deliberately setting
the backward rule's `src1` offset to **zero** — a straightforwardly wrong gradient — still reported
`Backend CPU: OK`. It only fails under a weighted objective.

> **Normative:** the three known ways for a MODE_GRAD case to be green while checking nothing are:
>
> 1. it never calls `ggml_set_param` — nothing is a parameter *(S0-10)*;
> 2. its tensors exceed `grad_nmax()` and the case is **skipped for speed**, which still prints `OK`
>    *(S1-29)*;
> 3. **the objective cannot distinguish a wrong answer** — because it conserves the output's sum
>    *(S1-34)*, or because the op only routes and `sum(out)` gives every route the same gradient
>    *(S1-29)*.
>
> A kernel PR must state which of these it ruled out, and the honest way to rule out (3) is to
> **break the kernel on purpose and watch the test fail.** If it does not, the test is decoration.

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
