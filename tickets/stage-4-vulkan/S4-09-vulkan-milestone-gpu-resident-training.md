---
id: S4-09
title: "Vulkan milestone: GPU-resident training on NVIDIA/AMD/Intel via Vulkan"
stage: 4
track: python
size: M
deps: [S4-02, S4-03, S4-04, S4-05, S4-06, S4-07, S1-12]
status: open
pr: null
---

# S4-09 — Vulkan milestone: GPU-resident training on NVIDIA/AMD/Intel via Vulkan

**One-line outcome:** Stage-4 exit is proven and locked in: the S1-12 convergence gate
plus the MoE and SSM e2e tests are green on `--device vulkan` with zero CPU-fallback ops
on the native lane, cross-vendor coverage is recorded, `ci-vulkan / e2e` is
fallback-forbidden from now on, `docs/perf/vulkan.md` is published, and the final
cross-backend parity report — the same tiny-model run on cpu/metal/cuda/vulkan within
the ADR-0002 tolerance — closes out the plan's "complete training system" deliverable.

## Why (context)

ROADMAP §11 defines phase K1 as the Vulkan OUT_PROD + CE work (V1-V5) whose completion
unlocks "GPU-resident training on NVIDIA/AMD/Intel via Vulkan", with each phase's exit
criterion being its CI-matrix column going green — the e2e tier being the tiny-model
convergence gate vs the recorded PEFT reference (S1-12, per-phase exit). This project's
stage 4 additionally folds in the MoE (S4-06) and SSM (S4-07) ports, so the milestone
covers every training path except FA, which joins only if S4-08 has landed (it is
deliberately not a dependency — ROADMAP §8 FA8's chunked fallback remains the documented
long-context story otherwise). Vulkan is also the last backend stage: ROADMAP §0's goal —
the entire training step GPU-resident on CUDA, Metal, and Vulkan with CPU as oracle — is
provable only now, which is why this milestone owns the plan's end-state deliverable, the
four-backend parity report.

The auditable criterion mirrors S2-10/S3-10: `ggml_backend_sched` silently executes
unsupported nodes on CPU (ROADMAP §11 scheduler note) and exposes assignments via
`GGML_SCHED_DEBUG` (getenv at `vendor/llama.cpp/ggml/src/ggml-backend.cpp:1740-1741`;
printer `ggml_backend_sched_print_assignments`, `:945`). S4-01's e2e lane already parses
that into a fallback report but tolerates fallbacks; this ticket flips the **native**
lane to fallback-forbidden. The lavapipe lane stays fallback-tolerated-but-reported: it
proves driver-semantics correctness, but "GPU-resident" and all perf claims are
native-lane statements only (S4-01 split).

Vulkan is the one backend where a single green run does not prove the claim: the backend
selects materially different code paths per driver — coopmat/coopmat2 vs scalar `mul_mm`,
subgroup vs shared-memory reductions (ROADMAP §11 CI matrix: coopmat NV, scalar
AMD/Intel, MoltenVK). The milestone therefore records a cross-driver matrix of which
vendor/driver/path combinations actually ran, via the S4-01 caps probe, per available
hardware — honest gaps recorded, not glossed.

## What to do

1. **Run the gates on Vulkan:** `pytest tests/test_convergence.py --device vulkan -m
   slow` on both S1-12 fixtures (F32-base and Q8_0-base), plus the tiny-MoE e2e (S1-28)
   and tiny-Mamba e2e (S1-31 `tests/test_ssm_training.py`) with `--device vulkan`:
   correctness on the lavapipe lane, the milestone-proving runs on the S4-01 native GPU
   runner. Archive all JSON report artifacts (device + fallback indicator).
2. **Extend the shared op allowlist** (S2-10's `.github/scripts/dense-lora-ops.txt`
   format and parser assertion mode — shared, not forked): add the Vulkan-scoped
   dense set (S4-02/S4-03 `OUT_PROD`, S4-04 sparse CE fwd+bwd, the S4-05-audited
   `*_BACK` set), MoE (`OUT_PROD_ID`, `OUT_PROD_ID_GRP`), and SSM (`SSM_CONV_BACK`,
   `SSM_SCAN_BACK`) ops. Include FA ops only if S4-08 has landed; otherwise document
   the FA8 fallback configuration as the supported long-context path. Document every
   exception with evidence — e.g. Mamba-1 `SSM_SCAN` shapes fall back by design (the
   Vulkan forward accepts Mamba-2 only,
   `vendor/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:17647-17683`, mirrored by
   S4-07); prefer Mamba-2-class fixture configs over allowlist holes.
3. **Assert zero fallback on the native lane:** run every gate/e2e step with
   `GGML_SCHED_DEBUG=2`, parse assignments via the shared parser in assertion mode, and
   fail on any allowlisted op assigned to CPU.
4. **Cross-driver matrix:** execute the gate on each available native driver per the
   ROADMAP §11 matrix — coopmat NVIDIA (the S4-01 VM), scalar AMD/Intel, and optionally
   MoltenVK (ROADMAP §12 Q10 stopgap note) — recording for each run the S4-01 caps
   probe output (coopmat/coopmat2/subgroup) and gate result in a committed table
   (`docs/perf/vulkan.md` appendix). Where hardware is unavailable, force the scalar
   path on the NVIDIA runner via the existing `GGML_VK_DISABLE_COOPMAT`/
   `GGML_VK_DISABLE_COOPMAT2` env toggles
   (`vendor/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:5816-5824`) and record that
   as path coverage, explicitly distinct from vendor coverage.
5. **Flip `ci-vulkan / e2e` (native lane) to fallback-forbidden** (nightly +
   `workflow_dispatch`): the job fails on any allowlisted-op fallback and still fails
   when the report artifact is missing (S4-01 semantics retained). Lavapipe e2e stays
   report-required, fallback-tolerated.
6. **Perf snapshot `docs/perf/vulkan.md`** (mirror `docs/perf/metal.md` / `cuda.md`):
   tok/s for the gate config on native Vulkan vs CPU on the same host, peak memory,
   full provenance (GPU, driver version, Vulkan SDK, llama-farm + vendor commits), the
   fallback report alongside, and an explicit **lavapipe-vs-native note** stating that
   lavapipe numbers are correctness evidence only and never perf claims. One snapshot,
   not a benchmark suite.
7. **Final cross-backend parity report** (the plan's end-state deliverable,
   ROADMAP §0/§11): run the identical S1-12 tiny-model configuration on
   `--device cpu`, `metal`, `cuda`, and `vulkan` (each on its stage's CI runner or VM,
   collated from the per-stage artifacts where re-running is impractical) and publish
   `docs/perf/cross-backend-parity.md`: per-backend loss curves vs the PEFT reference
   bands, max pairwise divergence, and confirmation that every backend sits within the
   ADR-0002 cross-backend criterion; plus a coverage table (dense/MoE/SSM/FA ×
   backend) with each cell's status and owning ticket. This document is the project's
   "complete training system" evidence.
8. **Update the BLUEPRINT §8 platform table** in
   `docs/GGUF-LORA-TRAINING-BLUEPRINT.md`: add/refresh the Linux/Windows Vulkan row to
   GPU-resident dense/MoE/SSM training, with pointers to remaining gaps (FA — S4-08 if
   not landed; fused variants — backlog B-01).
9. **File follow-up issues** for every tolerance outlier, driver-specific failure, or
   band pressure the milestone runs surface; link them from the PR, or state
   explicitly that none were found.

## Out of scope

- The kernels themselves (S4-02..S4-07) — a gate failure goes to the owning ticket and
  this milestone waits.
- FA on Vulkan (S4-08) — folded into the forbidden set by that ticket's landing, not
  gated on here; the FA8 fallback documentation covers the interim.
- MoltenVK as a supported platform — one optional matrix row (ROADMAP §12 Q10), not an
  exit criterion; native Metal (S2-10) owns Apple Silicon.
- Performance optimization beyond the snapshot (K5), fused/atomic backlog items
  (B-01 etc.), and RWKV/delta-net coverage.
- Upstream-early PR filing audits — completed at the S3-10 milestone; stage-4 ops are
  all fork-local-first RFC material (ROADMAP §11 triage b).

## Acceptance criteria

- [ ] S1-12 convergence gate passes with `--device vulkan` for F32-base and Q8_0-base
      fixtures within their documented bands on the native lane; tiny-MoE and
      tiny-Mamba e2e tests pass with `--device vulkan`; all JSON report artifacts
      archived on the CI run.
- [ ] The native-lane fallback reports contain **zero** allowlisted ops on CPU across
      dense, MoE, and SSM paths; the extended allowlist file exists with documented
      exceptions (each with evidence); FA status (S4-08 landed / FA8 fallback) is
      stated explicitly.
- [ ] `ci-vulkan / e2e` (native) is fallback-forbidden: one deliberately induced
      fallback (or a synthetic report fixture) turns the job red — exercised once and
      linked in the PR; the lavapipe lane remains report-required, fallback-tolerated.
- [ ] The cross-driver matrix table exists with caps-probe output per run: coopmat
      NVIDIA plus at least one scalar-path run (real AMD/Intel hardware, or
      `GGML_VK_DISABLE_COOPMAT*` on NVIDIA recorded as path-only coverage); MoltenVK
      row present or explicitly marked not-run.
- [ ] `docs/perf/vulkan.md` exists with native-vs-CPU tok/s, peak memory, provenance,
      and the lavapipe-vs-native note.
- [ ] `docs/perf/cross-backend-parity.md` exists: the same tiny-model run on
      cpu/metal/cuda/vulkan, all within the ADR-0002 cross-backend criterion, with the
      per-backend coverage table.
- [ ] The BLUEPRINT §8 platform table row for Vulkan is updated in `docs/`.
- [ ] Follow-up issues exist for all outliers found (or the PR states none were).

## Testing & verification

The verification *is* the gate suite under `ci-vulkan / e2e` (nightly + one
`workflow_dispatch` for the PR evidence): S1-12 with `--device vulkan`, the MoE/SSM e2e
tests, fallback enforcement via the shared report parser in assertion mode on the native
lane, and the cross-driver matrix runs. Per-PR `ci-vulkan / lavapipe` MODE_GRAD lanes are
unchanged. The perf snapshot and any off-runner matrix rows require manual runs on the
project Vulkan VM / available hardware (S0-08 playbook); record which hardware produced
every published figure. The parity report collates the S1-12 artifacts from all four
backend milestones (S2-10, S3-10, this).

## PR notes

- Branch: `ticket/S4-09-vulkan-milestone-gpu-resident-training`.
- Single llama-farm PR (workflow flip + allowlist extension + docs + perf snapshot +
  parity report); no vendored llama.cpp changes expected, so no two-repo flow.
- Upstreaming disposition: **fork-local** (project CI, docs, milestone evidence).
- Soft coordination: reuses S2-10's allowlist format and parser assertion mode and
  S3-10's milestone shape (memory-cliff bound stays CUDA-owned); consumes S1-12's
  `--device` interface and S4-01's lane split, caps probe, and report semantics. If
  S4-08 lands while this ticket is open, extend the allowlist and parity-report
  coverage table in the same PR rather than a follow-up.
