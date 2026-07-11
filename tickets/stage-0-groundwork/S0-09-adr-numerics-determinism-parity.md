---
id: S0-09
title: "ADR: numerics policy + determinism default (gate G-B) + parity criterion"
stage: 0
track: docs
size: S
deps: [S0-01]
status: open
pr: null
---

# S0-09 — ADR: numerics policy + determinism default (gate G-B) + parity criterion

**One-line outcome:** ADR-0002 exists and fixes, project-wide: F32 accumulation on all gradient paths, determinism-by-default with atomics as measured opt-in (gate G-B decided), max-abs gradient error ≤ 0.05 @ fp16 vs the CPU oracle as the cross-backend parity criterion, and `test-backend-ops` MODE_GRAD as the acceptance harness.

## Why (context)

Every kernel ticket in stages 1-4 must cite one authoritative numerics document instead of
re-arguing precision per PR. ROADMAP §3 states the policy; this ticket freezes it as
ADR-0002 so acceptance criteria like "MODE_GRAD parity within the ADR-0002 tolerance" are
well-defined. ROADMAP §13 item 8 supplies the external precedent: unsloth's self-tests
accept max-abs gradient error ≤ 0.05 at fp16 (see the assert at unsloth
`unsloth/kernels/rms_layernorm.py:326` in the research checkout) — we adopt the same
threshold as our cross-backend parity criterion.

The ADR must also settle **decide-first gate G-B** (ROADMAP §9, §11, and open question Q9
in §12): atomicAdd scatter over ragged expert segments and atomic-dQ FA backward make
gradients nondeterministic run-to-run. The ROADMAP recommendation — deterministic
segmented schemes as the project default, atomic variants only as measured opt-in — blocks
the design of the E2/E3-class MoE ops and FA5's dQ strategy, so it must be recorded now,
in Stage 0, before any of those tickets start.

Finally, the ADR names the two concrete forbidden patterns from ROADMAP §5 C1, both
verified in the vendored tree: (a) CUDA's `ggml_cuda_mul_mat_cublas_impl`
(`vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:1324-1536`) defaults to F16 traits
that set `CUBLAS_COMPUTE_16F` — F16 accumulation
(`vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:1311-1323`); gradient-path reuse must
pin its `prefer_f32_output` / `CUBLAS_COMPUTE_32F` variant
(`vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:1425-1440`) on all arches. (b) Vulkan's
tuned `mul_mm` pipelines carry f16acc variants
(`vendor/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:216`) that are inference-only and
must never be selected on gradient paths.

## What to do

1. **Create `docs/adr/ADR-0002-numerics-determinism-parity.md`** (Status / Context /
   Decision / Consequences format, consistent with ADR-0001 from S0-02).
2. **Decision 1 — numerics policy** (reproduce ROADMAP §3 verbatim, then add rationale):
   gradient matmuls use F32 accumulation on every backend; row statistics, logsumexp, and
   loss values are computed in F32; elementwise math runs in F32 with casts only at
   storage boundaries.
3. **Decision 2 — forbidden patterns:** on gradient paths, Vulkan f16acc `mul_mm`
   variants and CUDA GemmEx with the default F16 traits (`CUBLAS_COMPUTE_16F`) are
   forbidden; CUDA gradient GEMMs must use `CUBLAS_COMPUTE_32F` / the `prefer_f32_output`
   configuration. Cite the four anchors from the Why section.
4. **Decision 3 — gate G-B, determinism default:** deterministic segmented/exclusive-write
   schemes are the project default for every backward kernel; atomics-based variants are
   permitted only as opt-in, behind an explicit flag, and only with a benchmark showing a
   measured win. Record the downstream bindings: E2/E3-class ops (`OUT_PROD_ID`,
   `OUT_PROD_ID_GRP`) and the FA-backward dQ strategy (FA5) must conform; every backward
   kernel ticket states which scheme it implements. State the consequence: with a fixed
   backend, build, and seed, training runs are bit-reproducible by default.
5. **Decision 4 — parity criterion and harness:** `test-backend-ops` MODE_GRAD (mode
   `grad`) is the acceptance harness for every new/ported kernel, with CPU reference
   implementations as the oracle. Define the two tolerance layers explicitly:
   (a) per-op finite-difference MODE_GRAD checks use the harness's mean-abs-asymmetry
   machinery with expected-value filtering for discontinuous gradients
   (`vendor/llama.cpp/tests/test-backend-ops.cpp:319-321`) and its default per-op bound
   `max_maa_err() = 1e-4` (`vendor/llama.cpp/tests/test-backend-ops.cpp:1158-1160`),
   overridable per test; (b) the **cross-backend parity criterion** — max-abs gradient
   error ≤ 0.05 at fp16, GPU backend vs CPU oracle on identical inputs — is the bar for
   declaring a backend's kernel "at parity", enforced wherever a kernel ticket compares
   backend gradients against CPU-produced gradients. Tightening either bound later
   requires amending this ADR.
6. **Consequences section:** what kernel tickets must include (a MODE_GRAD case, the
   scheme declaration from Decision 3, no forbidden patterns), and that every vendor bump
   re-runs the full MODE_GRAD suite (per ADR-0001/S0-02 cadence).
7. **Wire-up:** link ADR-0002 from the docs index and from `docs/dev/testing.md` if S0-08
   has merged (soft coordination; otherwise S0-08 adds the link).

## Out of scope

- Gate G-A (sparse-CE ABI: lse stash vs recompute, logits aliasing) — a separate
  decide-first ADR owned by the sparse-CE ticket track (ROADMAP §11).
- Any code, CI, or test changes — this is a docs-only decision record; enforcement lands
  with each kernel ticket.
- Per-op tolerance overrides for specific kernels — chosen inside those tickets within
  the bounds this ADR sets.
- FA-backward numerics details (FTZ thresholds, LSE-with-sinks definition; ROADMAP §12
  item 4) — the FA ticket family, constrained by but not decided in this ADR.

## Acceptance criteria

- [ ] `docs/adr/ADR-0002-numerics-determinism-parity.md` exists with Status "Accepted"
      and the four Decision subsections above.
- [ ] The ADR contains the exact strings "CUBLAS_COMPUTE_32F", "f16acc", and
      "max-abs gradient error ≤ 0.05" (grep-verifiable), each in a normative statement.
- [ ] Gate G-B is recorded as **decided** (deterministic default, atomics opt-in) with the
      explicit list of bound tickets/ops (E2/E3-class, FA-backward dQ).
- [ ] The ADR names `test-backend-ops` mode `grad` (MODE_GRAD) as the acceptance harness
      and distinguishes the per-op finite-difference bound from the 0.05 cross-backend
      parity criterion.
- [ ] All four vendored-code anchors cited in the ADR resolve at commit `4f37f51`
      (reviewer spot-check).
- [ ] The docs index links to ADR-0002.

## Testing & verification

Docs-only: no pytest or test-backend-ops changes. Verification is review — the grep-able
acceptance strings above, plus a reviewer spot-check of the cited anchors against the
pinned vendor commit. Once S0-07's `ci-cpu` lane exists, the ADR rides its markdown/link
check (per-PR); no dedicated CI lane. Downstream enforcement: every kernel ticket's
MODE_GRAD acceptance criterion references this ADR by ID.

## PR notes

- Branch: `ticket/S0-09-adr-numerics-determinism-parity`.
- One PR; documentation only — no vendored llama.cpp changes, no two-repo flow.
  Upstreaming disposition: **fork-local** (the policy governs our fork; individual kernel
  PRs carry its consequences upstream per ROADMAP §11 triage).
- Soft coordination: ADR numbering follows ADR-0001 (S0-02); if S0-02 has not merged,
  agree on the `docs/adr/` layout with its open PR rather than inventing a second format.
