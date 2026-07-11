---
id: S1-12
title: "Convergence gate: tiny-model SFT vs recorded PEFT reference"
stage: 1
track: python
size: M
deps: ["S1-05"]
status: open
pr: null
---

# S1-12 — Convergence gate: tiny-model SFT vs recorded PEFT reference

**One-line outcome:** the reusable end-to-end correctness gate exists — a tiny-model SFT
run whose loss curve must match a recorded PEFT/transformers reference within documented
tolerance bands, runnable per-backend via one flag — plus the pytest wrapper that runs the
vendored `test-backend-ops` MODE_GRAD cases for every op this project added.

## Why (context)

BLUEPRINT §9 P1 names the "tiny-model convergence test with loss-curve tolerance vs a
recorded PEFT reference" as part of making SFT real: unit tests prove pieces, but only an
end-to-end run against an independent implementation (PEFT/transformers on identical
weights, data, and hyperparameters) proves the whole training stack — graph build, adapter
param wiring, `ce_sparse` loss, masking, optimizer — computes the right thing. This is the
project's defense against the class of bug that passes every per-op check and still trains
wrong.

ROADMAP §11 then makes this gate load-bearing far beyond Stage 1: the CI matrix's e2e tier
is "tiny-model e2e convergence (loss-curve tolerance vs recorded PEFT reference — blueprint
P1 test)", and its column going green is the **phase-exit criterion for every backend
stage** (per-phase exit on CUDA/CPU/Metal/Vulkan, nightly elsewhere). Hence the gate must
be backend-parameterized from day one. The ROADMAP §11 scheduler note sets the fallback
semantics: until a backend's kernel gaps close, `ggml_backend_sched` transparently runs
unsupported ops on CPU — training *works* everywhere at reduced speed — so a non-CPU run
with CPU fallbacks is **acceptable but must be reported**, never silently passed off as
GPU-resident.

The second deliverable closes a harness gap: S1-04 (and every later kernel ticket) adds
`test-backend-ops` MODE_GRAD cases that today run only as ad-hoc CI steps. A pytest wrapper
that invokes the vendored binary (`test-backend-ops [mode] [-o <op,..>] [-b <backend>]`,
usage at `vendor/llama.cpp/tests/test-backend-ops.cpp:10076`; grad mode `MODE_GRAD`,
`:479`) for the project-added op list makes the ADR-0002 acceptance harness a first-class,
named check that backend lanes reuse unchanged.

## What to do

1. **Twin fixtures.** Extend the S0-06 fixture tooling with
   `tests/convergence/gen_reference_model.py`: from one seeded RNG, emit the same tiny
   llama-arch weights both as an HF-transformers checkpoint (for PEFT) and as GGUF — F32
   base and a Q8_0-quantized variant (same numeric source weights; gguf-py numpy
   quantizer). Deterministic and cached like S0-06 fixtures; never committed.
2. **One-time reference recorder** `tests/convergence/record_reference.py` (requires
   `torch`+`peft`, which are *not* runtime deps — document the recording env): run PEFT
   LoRA SFT on the HF twin with the exact config the gate uses (rank/alpha/targets, AdamW
   hyperparams, LR schedule, fixed seed, committed token/mask dataset), record the
   per-step loss curve, and write `tests/convergence/reference_curve.json` (committed)
   embedding the full config, library versions, and generator hash.
3. **The gate** `tests/test_convergence.py` (pytest marker `slow`): run S1-05
   `train_sft` on the F32-base and Q8_0-base fixtures with the identical config and
   compare loss curves against the reference using **tolerance bands per step window, not
   per-step exact match** (windowed means over e.g. steps 1-10/11-25/26-50). The F32 run
   gets tight bands; the Q8_0 run compares against the same F32 PEFT reference with wider,
   separately documented bands (quantizing the base perturbs logits, so trajectories
   diverge in the small — PEFT cannot run the Q8_0 GGUF itself). Document band rationale
   and the regeneration procedure in `tests/convergence/README.md`.
4. **Backend parameterization:** a pytest `--device {cpu,metal,cuda,vulkan}` option
   (default `cpu`) that selects the compute backend for the run; unavailable devices skip
   cleanly. Fallback reporting per ROADMAP §11: the gate records the requested device and
   a fallback indicator (at minimum the sched split count via an `_ffi` accessor, plus the
   documented expectation per stage) into a JSON report artifact attached to the CI run —
   acceptable-but-reported, never hidden.
5. **Nightly wiring:** the `slow` marker puts the gate in S0-07's nightly `ci-cpu` run.
   Backend CI tickets (S2-01/S3-01/S4-01) reuse this test with `--device` as their
   phase-exit criterion — prose contract; keep the test free of CPU-only assumptions.
6. **Grad-check pytest wrapper** `tests/test_backend_ops_grad.py`: locate the vendored
   `test-backend-ops` binary from the build tree (built per S0-07), run mode `grad` with
   `-o` set from a single registry constant `PROJECT_ADDED_OPS` (starts with
   `CROSS_ENTROPY_LOSS_SPARSE` / `CROSS_ENTROPY_LOSS_SPARSE_BACK` from S1-04; later kernel
   tickets append), and fail on any reported failure. Tolerances are the harness's own,
   governed by ADR-0002 (per-op `max_maa_err` bound; expected-value filtering for
   discontinuous gradients, `vendor/llama.cpp/tests/test-backend-ops.cpp:319-321`). Runs
   per-PR (fast at current op count); takes the same `--device` option for backend lanes.
7. **Failure ergonomics:** on band violation, print the offending window, both curves, and
   the reference config hash; store the run's curve JSON as a CI artifact for diffing.

## Out of scope

- Recording DPO/GRPO references (S1-14/S1-16 own their own correctness stories).
- The backend CI lanes themselves (S2-01/S3-01/S4-01) — this ticket only guarantees the
  gate is reusable by them.
- Throughput/perf benchmarking (`benches/`, ROADMAP §4 P4) — this gate checks correctness
  only.
- Any change to trainers or kernels; if the gate fails, the fix belongs to the responsible
  ticket.

## Acceptance criteria

- [ ] `tests/convergence/reference_curve.json` is committed with embedded config,
      versions, and generator hash; `record_reference.py` regenerates it byte-stably
      (modulo recorded version strings) per the documented procedure.
- [ ] `pytest tests/test_convergence.py --device cpu -m slow` passes on the Linux CPU VM:
      F32-base and Q8_0-base runs both fall inside their documented bands.
- [ ] Band definitions (windows + widths, per variant) and their rationale exist in
      `tests/convergence/README.md`.
- [ ] `--device` exists for all four values; non-CPU devices skip cleanly on the CPU VM;
      the JSON report artifact records device + fallback indicator for every run.
- [ ] `pytest tests/test_backend_ops_grad.py` passes per-PR in `ci-cpu`, running the
      vendored binary in mode `grad` for every op in `PROJECT_ADDED_OPS`.
- [ ] The nightly `ci-cpu` workflow includes the `slow`-marked gate (workflow diff or a
      triggered `workflow_dispatch` nightly run shows it executed).

## Testing & verification

This ticket *is* a test deliverable. The convergence gate runs in nightly `ci-cpu`
(marker `slow`, S0-07 two-tier convention); the grad-check wrapper runs per-PR in
`ci-cpu / test` and later in `ci-metal` / `ci-cuda` / `ci-vulkan` via `--device`
(those lanes are S2-01/S3-01/S4-01). MODE_GRAD parity for project ops is enforced through
the wrapper under the ADR-0002 tolerances (per-op bound on CPU now; the ≤ 0.05 @ fp16
cross-backend criterion applies when GPU lanes compare against the CPU oracle). Local:
run the recorder once in a torch venv, then the gate twice to confirm determinism of the
llama-farm side.

## PR notes

- Branch: `ticket/S1-12-convergence-gate-peft-reference`.
- Single llama-farm PR (tests + tools + committed reference JSON + workflow tweak); no
  vendored llama.cpp changes, so no two-repo flow.
- Upstreaming disposition: **fork-local** (project test infrastructure).
- Soft coordination: S2-01/S3-01/S4-01 consume `--device` and the report artifact as
  their phase-exit evidence — keep both interfaces stable; `torch`/`peft` stay out of
  `pyproject.toml` runtime/test deps (recording env documented in
  `tests/convergence/README.md`).
