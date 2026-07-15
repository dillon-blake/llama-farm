---
id: B-11
title: "CPU throughput audit + published tok/s sizing (the S1-32 remainder)"
stage: backlog
track: docs
size: M
deps: [S1-12, S1-13, S1-17, S1-32]
status: open
pr: null
---

# B-11 — CPU throughput audit + published tok/s sizing

**One-line outcome:** finish the S1-32 audit — a machine-readable benchmark of the dense-LoRA
backward across CPU classes, a maintained memory-sizing worksheet with unit tests, and a published
`docs/perf/cpu.md` so a GPU-less user can size a run.

**Activation trigger:** any time perf reporting or user-facing sizing guidance is needed for the CPU
target; otherwise low-priority (report-only, gates nothing). Not a Stage-1 blocker.

## Why (context)

S1-32 shipped a runnable `benches/cpu_train_step.py` and a one-box recorded pass
(`docs/dev/cpu-throughput-audit.md`), but the bulk of its acceptance criteria never landed — see the
`## Status` block in `tickets/stage-1-cpu/S1-32-cpu-throughput-audit-toks-sizing.md` and the
2026-07-15 audit's ci-docs major finding. This ticket carries the remainder whole. The full context,
the BLUEPRINT §10 risk-5 sizing formula, and the per-op/eval-callback measurement mechanisms are all
in the retained S1-32 body; read it first.

## What to do (the carried S1-32 deliverables)

1. **Machine-readable report.** `benches/cpu_train_step.py` and the per-op perf runner emit JSON
   (thread sweep, tok/s median, fwd/bwd split, per-op breakdown) alongside the human tables, with a
   documented one-command invocation that reproduces both.
2. **Per-op microbenchmarks.** A `benches/` runner invoking the vendored `test-backend-ops` perf mode
   for the training-critical ops (`OUT_PROD` quantized + F32, `SOFT_MAX_BACK`, `RMS_NORM_BACK`, the
   ce_sparse op, and the Stage-1 backward kernels), reproducible outside a full train step.
3. **Thread-scaling analysis on a realistic model.** Add a ~1B-class Q4_K llama-arch fixture so the
   sweep's scaling-cliff is meaningful (on the tiny CI fixture it is at one thread — see the recorded
   pass). Compute per-op and end-to-end scaling efficiency; flag any anomalous op as the ROADMAP §4 P3
   SIMD-attention queue (possibly empty, stated explicitly).
4. **Memory worksheet.** `src/learning_llamas/sizing.py` implementing the BLUEPRINT §10 risk-5 formula
   as a function of (hparams, quant type, n_ctx, n_ubatch, rank), with worked examples for ≥3 named
   configs and a predicted-vs-measured peak-RSS delta from the bench runs. Note where S1-13 (chunked
   lm_head) and S1-17 (checkpointing) change the formula's terms. Unit tests (formula vs hand-computed
   examples) run in the per-PR `ci-cpu` lane.
5. **`examples/training/finetune` baseline.** Run the vendored finetune binary on a comparable F32
   config and report its tok/s and peak memory next to the quantized-base LoRA numbers, config deltas
   stated (BLUEPRINT §9 P1).
6. **`docs/perf/cpu.md`.** tok/s tables for ≥2 CPU classes (x86 + arm64 via the S0-08 playbooks), the
   scaling analysis, the per-op breakdown, the SIMD flag list, the worksheet's worked examples, and
   plain "what can I train on N cores / M GB" guidance, with the measurement protocol stated.
7. **Nightly CI.** A report-only `ci-cpu` job running the bench on the fixed runner and uploading the
   report artifact — no thresholds, no gate.

## Acceptance criteria

- [ ] Both runners emit JSON + human tables; one documented command reproduces them.
- [ ] `docs/perf/cpu.md` exists with tok/s for ≥2 CPU classes, thread-scaling, per-op breakdown, and
      the SIMD flag list (possibly empty, stated).
- [ ] The `examples/training/finetune` baseline comparison is reported with config deltas.
- [ ] `src/learning_llamas/sizing.py` exists with ≥3 worked examples and a predicted-vs-measured
      peak-memory delta; its unit tests pass per-PR.
- [ ] The nightly `ci-cpu` report-only job runs and uploads the artifact (one completed run shown).

## Out of scope

- Kernel/SIMD changes (each becomes its own ticket under the ROADMAP §4 P3 policy).
- GPU benchmarks and cross-backend parity (S4-09); perf-regression *gating* (needs dedicated
  hardware); MoE/SSM throughput (cheap follow-on once needed).
