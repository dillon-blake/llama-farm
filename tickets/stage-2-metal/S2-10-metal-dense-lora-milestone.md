---
id: S2-10
title: "Metal milestone: GPU-resident dense-LoRA training on Apple Silicon"
stage: 2
track: python
size: M
deps: ["S2-02", "S2-03", "S2-04", "S2-05", "S2-06", "S2-07", "S2-08", "S2-09", "S1-12"]
status: open
pr: null
---

# S2-10 — Metal milestone: GPU-resident dense-LoRA training on Apple Silicon

**One-line outcome:** Stage-2 exit is proven and locked in: the S1-12 tiny-model convergence
gate is green with `--device metal` with **zero CPU-fallback ops in the dense-LoRA path**, the
`ci-metal` e2e lane enforces that invariant from now on, and a perf + memory snapshot is
published in `docs/perf/metal.md`.

## Why (context)

ROADMAP §11 defines phase K2 as the Metal kernel suite (M3-M5 → M1 → M2 → M7/M8 → M6/M10)
whose completion unlocks "GPU-resident training on Apple Silicon", and sets each K phase's
exit criterion as its CI-matrix column going green — the e2e tier being the tiny-model
convergence gate vs the recorded PEFT reference (S1-12, "per-phase exit" on Metal). With
S2-02..S2-09 landed, every op a dense-transformer LoRA training step emits (ROADMAP §1) has a
Metal kernel. This ticket is the milestone that *proves* the claim end-to-end, converts the
proof into a permanent CI invariant, and updates the project's public platform story.

The auditable criterion comes from the ROADMAP §11 scheduler note: `ggml_backend_sched`
transparently executes unsupported nodes on CPU, so "training works on Metal" was true before
stage 2 started — what stage 2 changes is the fallback set going empty for the dense path.
The sched exposes assignments via the `GGML_SCHED_DEBUG` env variable (getenv at
`vendor/llama.cpp/ggml/src/ggml-backend.cpp:1740-1741`; printer
`ggml_backend_sched_print_assignments`, `:945`), which S2-01's e2e job already parses into a
GPU-vs-CPU fallback report. Until now that lane was deliberately fallback-tolerated
(report-required, fallbacks allowed); this ticket flips it to **fallback-forbidden** for the
dense op set — the concrete meaning of the project goal "the GPU is used when available"
(ROADMAP §0).

The docs still tell the pre-stage-2 story: BLUEPRINT §8's platform table records
"macOS Metal — forward on GPU only … backward is CPU-bound; Metal kernels post-P2", written
when Metal had only `ROPE_BACK` and the optimizer steps (ROADMAP §6 preamble). This milestone
updates the table so users and the stage-3/4 milestone tickets (which mirror this shape)
inherit an accurate baseline, and it publishes one honest perf/memory snapshot so the
milestone claim has numbers attached, not just a green check.

## What to do

1. **Run the gate on Apple Silicon:** `pytest tests/test_convergence.py --device metal
   -m slow` on both S1-12 fixtures (F32-base and Q8_0-base), on the `ci-metal` runner (hosted,
   or the S0-08 self-hosted fallback if the paravirtual GPU probe fails). Archive the S1-12
   JSON report artifacts (device + fallback indicator) for both runs.
2. **Define the dense-LoRA op allowlist** in one shared location (e.g.
   `.github/scripts/dense-lora-ops.txt`, consumed by S2-01's fallback-report parser): the ops
   of the dense training step per ROADMAP §1 — `MUL_MAT`, `OUT_PROD`, `SOFT_MAX_BACK`,
   `RMS_NORM_BACK`, `ROPE_BACK`, `SILU_BACK`, `REPEAT_BACK`, sparse CE fwd+bwd,
   `OPT_STEP_ADAMW`, plus the ordinary forward/elementwise/reduction/view ops. Document
   explicit exceptions with evidence (e.g. `ADD1`/`DIAG_MASK_ZERO` if S2-09 closed them as
   not-emitted, citing its graph dump).
3. **Assert zero dense fallback:** run the e2e training step with `GGML_SCHED_DEBUG=2`, parse
   assignments, and fail if any allowlisted op executed on CPU. Wire this as an assertion mode
   of S2-01's existing report parser, not a second parser.
4. **Flip `ci-metal / e2e` to fallback-forbidden** for the dense op set (nightly + dispatch):
   the job now fails on a dense-set fallback, and still fails when the report artifact is
   missing (S2-01 semantics retained). MoE/SSM/FA ops stay outside the allowlist — S2-11/
   S2-12/S2-13 tighten their own sets when they land.
5. **Perf + memory snapshot** in `docs/perf/metal.md`: tok/s for the gate config on Metal vs
   CPU on the *same* Apple Silicon host, peak-memory figures for both, hardware/OS/commit
   provenance, and the fallback report alongside. One snapshot, not a benchmark suite
   (throughput audits are ROADMAP §4 P4 / K5 territory).
6. **Update BLUEPRINT §8's platform table** in `docs/GGUF-LORA-TRAINING-BLUEPRINT.md`: the
   macOS Metal row moves from "forward on GPU only" to full dense-LoRA training, with pointers
   to the remaining Metal gaps (FA — S2-13; MoE — S2-11; SSM — S2-12).
7. **File follow-up issues** for every tolerance outlier the milestone runs surface (per-op
   MODE_GRAD cases needing overrides, Q8_0 band pressure, hosted-vs-real-hardware deltas);
   link them from the PR. If none, say so explicitly in the PR description.

## Out of scope

- The kernels themselves (S2-02..S2-09) — if a gate failure needs a kernel fix, it goes to
  the owning ticket and this milestone waits.
- MoE/SSM/FA on Metal (S2-11/S2-12/S2-13) — the dense milestone neither gates on them nor
  adds their ops to the forbidden set.
- CUDA/Vulkan milestones — stage-3/4 tickets mirror this shape.
- Performance optimization — snapshot only (K5 owns perf work).

## Acceptance criteria

- [ ] S1-12 convergence gate passes with `--device metal` for the F32-base and Q8_0-base
      fixtures within their documented bands; both JSON report artifacts are archived on the
      CI run.
- [ ] The fallback reports for those runs contain **zero** allowlisted dense-LoRA ops on CPU;
      `dense-lora-ops.txt` exists with documented exceptions (each with evidence).
- [ ] `ci-metal / e2e` is fallback-forbidden: one deliberately induced dense fallback (or a
      synthetic report fixture) turns the job red — exercised once and linked in the PR.
- [ ] `docs/perf/metal.md` exists with Metal-vs-CPU tok/s on the same host, peak memory, and
      full provenance (hardware, OS, learning-llamas + vendor commits).
- [ ] The BLUEPRINT §8 platform table row for macOS Metal is updated in `docs/`.
- [ ] Follow-up issues exist for all tolerance outliers found (or the PR states none were).

## Testing & verification

The verification *is* the S1-12 gate run under `ci-metal / e2e` (nightly + one
`workflow_dispatch` for the PR evidence), with fallback enforcement via S2-01's report parser
in its new assertion mode. Per-PR `ci-metal / grad` MODE_GRAD lanes are unchanged. The perf
snapshot requires one manual run on real Apple Silicon hardware (S0-08 playbook) — hosted
paravirtual-GPU numbers are not representative; record which hardware produced the published
figures.

## PR notes

- Branch: `ticket/S2-10-metal-dense-lora-milestone`.
- Single learning-llamas PR (workflow flip + allowlist + docs + perf snapshot); no vendored
  llama.cpp changes expected, so no two-repo flow.
- Upstreaming disposition: **fork-local** (project CI, docs, milestone evidence).
- Soft coordination: stage-3/4 milestone tickets copy this structure — keep the allowlist
  file format and the parser's assertion mode backend-agnostic; S2-11/S2-12/S2-13 later
  extend the forbidden set they own.
