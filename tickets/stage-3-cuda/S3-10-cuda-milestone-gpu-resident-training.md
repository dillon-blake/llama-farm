---
id: S3-10
title: "CUDA milestone: fully GPU-resident training including FA"
stage: 3
track: python
size: M
deps: [S3-02, S3-03, S3-04, S3-06, S3-08, S3-09, S1-12]
status: open
pr: null
---

# S3-10 — CUDA milestone: fully GPU-resident training including FA

**One-line outcome:** Stage-3 exit is proven and locked in: the S1-12 convergence gate
plus the MoE and SSM e2e tests are green on `--device cuda` with **zero CPU-fallback
ops** (dense, MoE, SSM, and FA paths), `ci-cuda / e2e` enforces that invariant plus a
memory-cliff regression bound from now on, a perf/memory snapshot is published in
`docs/perf/cuda.md`, and every upstream-early ticket has its mainline PR filed.

## Why (context)

ROADMAP §11 phases K0 ("fully GPU-resident dense-LoRA training on CUDA — the 'GPU used
when available' milestone for the primary platform") and K3 (FA training completed by
FA5 on CUDA) both exit when their CI-matrix column goes green, the e2e tier being the
tiny-model convergence gate (S1-12, per-phase exit). Because this project runs Metal
before CUDA, stage 3 collapses those exits into one milestone that goes beyond S2-10's
dense-only scope: with S3-02 (quantized `OUT_PROD` — ROADMAP §0 calls CUDA "one op
away" from this milestone), S3-03 (sparse CE), S3-04 (ALiBi), S3-06 (FA backward),
S3-08 (MoE), and S3-09 (SSM) landed, every training path this project supports is
CUDA-resident. This ticket proves it end-to-end, converts the proof into permanent CI
invariants, and closes the stage's paperwork.

The auditable criterion mirrors S2-10: `ggml_backend_sched` silently executes
unsupported nodes on CPU (ROADMAP §11 scheduler note), and exposes assignments via
`GGML_SCHED_DEBUG` (getenv at `vendor/llama.cpp/ggml/src/ggml-backend.cpp:1740-1741`;
printer `ggml_backend_sched_print_assignments`, `:945`). S3-01's e2e lane already
parses that into a GPU-vs-CPU fallback report but deliberately tolerates fallbacks;
this ticket flips the lane to **fallback-forbidden** for the full stage-3 op set. The
FA path additionally carries a quantitative invariant: ROADMAP §8's memory cliff (the
naive path's simultaneously-live attention matrices reach 128-192 GiB at 4k ctx for an
8B-class model — infeasible; FA backward replaces the n_ctx² term with LSE vectors).
S3-06 demonstrated the win once; this milestone wires it as a regression check so a
future graph change cannot silently revive the cliff.

Last, ROADMAP §11's upstreaming triage names an **upstream-early** set whose PRs should
be sent as soon as they are stable — K-F16OP (S1-18), K-TANH et al. (S1-19), K-SMB
(S1-20), C1/C2 (S3-02), C4 (S3-04) — because mainline training benefits directly and
tests exist. Stage 3 is the natural audit point: everything in that set is now landed
and soaking in CI, and the longer the fork carries divergences the worse the rebase
cadence hurts (ROADMAP §12 risk 11).

## What to do

1. **Run the gates on CUDA:** `pytest tests/test_convergence.py --device cuda -m slow`
   on both S1-12 fixtures (F32-base and Q8_0-base), plus the tiny-MoE e2e (S1-28) and
   tiny-Mamba e2e (S1-31 `tests/test_ssm_training.py`) with `--device cuda`, on the
   S3-01 GPU runner. Archive all JSON report artifacts (device + fallback indicator).
2. **FA-on long-context run:** execute the long-context training config from S3-06's
   e2e (tiny model, 2-4k ctx) twice — FA backward on, and the FA8 chunked fallback
   (S1-24) for comparison — recording peak allocation and tok/s for both (the ROADMAP
   §8 memory-cliff evidence, now reproducible in CI rather than a one-off).
3. **Extend the shared op allowlist** (S2-10's `.github/scripts/dense-lora-ops.txt`
   format and parser assertion mode — keep them shared, not forked): add the
   CUDA-scoped MoE (`OUT_PROD_ID`, `OUT_PROD_ID_GRP`), SSM (`SSM_CONV_BACK`,
   `SSM_SCAN_BACK`), and FA (`FLASH_ATTN_EXT` with LSE, its backward op) sets on top of
   the dense set. Run every gate/e2e step with `GGML_SCHED_DEBUG=2` and fail on any
   allowlisted op assigned to CPU. Document explicit exceptions with evidence (e.g.
   shapes outside S3-09's mirrored forward gates, if any tiny-model config hits them —
   preferably fix the config instead).
4. **Flip `ci-cuda / e2e` to fallback-forbidden** (nightly + `workflow_dispatch`): the
   job now fails on any allowlisted-op fallback and still fails when the report
   artifact is missing (S3-01 semantics retained).
5. **Memory-cliff regression check:** the nightly e2e job asserts peak allocation of
   the FA-on 4k-ctx run stays under a documented bound derived from step 2's
   measurement plus stated headroom; exceeding it turns the job red. Record the bound
   and its derivation next to the allowlist.
6. **Perf/memory snapshot `docs/perf/cuda.md`** (mirror `docs/perf/metal.md`): tok/s
   for the gate config on CUDA vs CPU on the same host, the Metal figures from S2-10
   cross-referenced (different host — say so), peak memory at 512/2k/4k ctx (FA on,
   plus the FA8 comparison at 4k), full provenance (GPU model, driver, CUDA toolkit,
   llama-farm + vendor commits), and the fallback report alongside. One snapshot, not a
   benchmark suite.
7. **Upstream-early audit:** verify each of S1-18, S1-19, S1-20, S3-02, S3-04 has a
   mainline llama.cpp PR filed per ROADMAP §11 triage (a); file any missing ones
   (cherry-picks from the fork; acceptance is *filed*, not merged — upstream review
   timelines are not ours). Record a table of ticket → upstream PR link in the
   llama-farm PR description and in `docs/dev/` next to the S0-02 fork notes.
8. **Update the platform table** in `docs/GGUF-LORA-TRAINING-BLUEPRINT.md` §8: the
   Linux CUDA row moves from "backward through quantized weights falls to CPU" to
   fully GPU-resident training including FA/MoE/SSM, with pointers to the remaining
   CUDA gaps (FA D=256 — S3-07 if not yet landed; mma-family/atomic-dQ/quantized-KV —
   backlog B-02).
9. **File follow-up issues** for every tolerance outlier or band pressure the milestone
   runs surface; link them from the PR, or state explicitly that none were found.

## Out of scope

- The kernels themselves (S3-02..S3-09) — a gate failure goes to the owning ticket and
  this milestone waits.
- FA head-size 256 and occupancy tuning — S3-07 (not a milestone gate; ROADMAP §8
  scopes FA5 v1 to D=64/128).
- ROCm/HIP tier, Vulkan milestone (S4-09), performance optimization beyond the
  snapshot (K5), and the fused/atomics backlog items (B-01/B-02).
- Landing upstream PRs — this ticket files and links them; merges happen on upstream's
  schedule.

## Acceptance criteria

- [ ] S1-12 convergence gate passes with `--device cuda` for F32-base and Q8_0-base
      fixtures within their documented bands; tiny-MoE and tiny-Mamba e2e tests pass
      with `--device cuda`; all JSON report artifacts archived on the CI run.
- [ ] The fallback reports for those runs contain **zero** allowlisted ops on CPU
      across dense, MoE, SSM, and FA paths; the extended allowlist file exists with
      documented exceptions (each with evidence).
- [ ] `ci-cuda / e2e` is fallback-forbidden: one deliberately induced fallback (or a
      synthetic report fixture) turns the job red — exercised once and linked in the PR.
- [ ] The FA-on vs FA8 4k-ctx comparison (peak alloc + tok/s) is recorded, and the
      nightly memory-cliff regression check is wired with a documented bound.
- [ ] `docs/perf/cuda.md` exists with CUDA-vs-CPU tok/s, memory at 512/2k/4k ctx, and
      full provenance.
- [ ] The upstream audit table exists and every upstream-early ticket (S1-18, S1-19,
      S1-20, S3-02, S3-04) links a filed mainline PR.
- [ ] The BLUEPRINT §8 platform table row for Linux CUDA is updated in `docs/`.
- [ ] Follow-up issues exist for all outliers found (or the PR states none were).

## Testing & verification

The verification *is* the gate suite under `ci-cuda / e2e` (nightly + one
`workflow_dispatch` for the PR evidence): S1-12 with `--device cuda`, the MoE/SSM e2e
tests, and the FA long-context run, all with fallback enforcement via the shared report
parser in assertion mode and the new peak-alloc bound. Per-PR `ci-cuda` compile and
kernel-gated MODE_GRAD lanes are unchanged. The perf snapshot requires one manual run on
the project CUDA VM (S0-08 playbook); record which hardware produced the published
figures.

## PR notes

- Branch: `ticket/S3-10-cuda-milestone-gpu-resident-training`.
- Single llama-farm PR (workflow flip + allowlist extension + docs + perf snapshot +
  audit table); no vendored llama.cpp changes expected, so no two-repo flow. The
  upstream-early filings happen in the upstream llama.cpp repo and are linked, not
  vendored.
- Upstreaming disposition: **fork-local** (project CI, docs, milestone evidence); the
  audited kernel tickets carry their own upstream-early dispositions.
- Soft coordination: reuses S2-10's allowlist format and parser assertion mode (keep
  backend-agnostic — S4-09 mirrors this shape next); consumes S1-12's `--device`
  interface and S3-01's report/parser semantics.
