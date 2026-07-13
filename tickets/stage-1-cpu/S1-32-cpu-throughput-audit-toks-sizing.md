---
id: S1-32
title: "CPU throughput audit + published tok/s sizing (P4)"
stage: 1
track: docs
size: S
deps: ["S1-12"]
status: pr-open
pr: https://github.com/dillon-blake/llama-farm/pull/29
---

# S1-32 — CPU throughput audit + published tok/s sizing (P4)

**One-line outcome:** one benchmark pass over the dense-LoRA backward on representative
CPUs confirms the dequant-per-row `out_prod` path and threadpool scaling are adequate, and
`docs/perf/cpu.md` publishes expected tok/s classes plus a memory-sizing worksheet so
GPU-less users can size runs.

## Why (context)

CPU is not just the oracle — it is a first-class training target: for small models and
LoRA ranks, CPU-only training is viable, with all ops threaded through the existing ggml
threadpool (ROADMAP §4). ROADMAP §4 P4 asks for exactly one audit pass before Stage 1
closes: confirm that the dequant-per-row `out_prod` path
(`ggml_compute_forward_out_prod_q_f32`,
`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:4363-4501` — the op hit on every linear layer,
every microbatch) and the threadpool scaling of the backward ops are adequate, and publish
expected tok/s classes. The audit also enforces the scalar-first kernel policy (ROADMAP §4
P3): SIMD variants of the new reference kernels are written only if this profiling
justifies them, so the audit's flag list is the sanctioned trigger for any later SIMD work.

The sizing half comes from BLUEPRINT §10 risk 5: activation memory dominates at long
context (the `n_head·n_ctx` attention-matrix term under softmax attention), and the
blueprint gives the back-of-envelope formula — mmap'd quantized base + activations
(~`n_layers × n_ubatch × (c₁·n_embd + n_head·n_ctx)` × 4 B) + logits forward+grad
(`2 × n_vocab × n_ubatch × 4 B` until chunked CE) + LoRA params × 12 B (grad + AdamW m/v)
+ F32 KV cache — with the instruction to produce worked examples per config. This ticket
turns that formula into a maintained worksheet.

One correction discovered during ticket authoring: ggml at the pinned commit has no
built-in per-op perf instrumentation (no `GGML_PERF`-style hooks exist in the vendored
tree). Per-op timing therefore uses the two mechanisms that do exist: the vendored
`test-backend-ops` perf mode (`vendor/llama.cpp/tests/test-backend-ops.cpp:478`) for
isolated per-op microbenchmarks, and
`ggml_backend_sched_set_eval_callback`
(`vendor/llama.cpp/ggml/include/ggml-backend.h:352`; callback type at `:314`) for
in-graph per-node wall-clock attribution during a real train step.

## What to do

1. **`benches/cpu_train_step.py`** (the `benches/` directory is part of the BLUEPRINT §4
   repo layout): drive a dense-LoRA SFT train step through the S1-12 harness on a tiny and
   a small dense model (reuse the S1-12 fixtures; add one Q4_K llama-arch model in the
   ~1B class for a realistic dequant load), reporting tokens/s (median over N steps,
   warmup excluded) and a per-op time breakdown captured via a
   `ggml_backend_sched_set_eval_callback` timer shim in the bindings. Parameters: thread
   count sweep (1, 2, 4, ..., n_physical), n_ctx/n_ubatch, LoRA rank.
2. **Per-op microbenchmarks:** a `benches/` runner invoking the vendored
   `test-backend-ops` in perf mode for the training-critical ops (`OUT_PROD` quantized and
   F32, `SOFT_MAX_BACK`, `RMS_NORM_BACK`, the ce_sparse op, and the new Stage-1 backward
   kernels) so per-op numbers are reproducible outside a full train step.
3. **Threadpool-scaling verification:** from the thread sweep, compute per-op and
   end-to-end scaling efficiency; flag any op whose scaling or absolute share is
   anomalous (expected suspects: dequant-per-row `out_prod` at large `n_vocab`). The flag
   list, with numbers, goes in the report as the ROADMAP §4 P3 SIMD-attention queue —
   this ticket only flags; it changes no kernels.
4. **Baseline vs the in-tree finetune example** (BLUEPRINT §9 P1 "benches vs
   `examples/training/finetune`"): run the vendored `examples/training/finetune` binary on a
   comparable F32-base config and report its tok/s and peak memory next to learning-llamas's
   quantized-base LoRA numbers, with the config deltas stated (full-FT F32 + F32 KV cache vs
   frozen-quantized + LoRA — the comparison contextualizes, it does not race like-for-like).
5. **Memory worksheet:** `src/learning_llamas/sizing.py` (or equivalent) implementing the
   BLUEPRINT §10 risk-5 formula as a function of (model hparams, quant type, n_ctx,
   n_ubatch, rank), plus worked examples for 2-3 named configs; cross-check predictions
   against measured peak RSS from the benchmark runs and report the delta. Note in the
   worksheet where S1-13 (chunked lm_head) and S1-17 (checkpointing) change the formula's
   terms once active.
6. **`docs/perf/cpu.md`:** results tables (tok/s per CPU class on 2-3 machines — e.g. the
   `ci-cpu` ubuntu x86 runner, an AVX2/AVX-512 desktop, and an Apple Silicon arm64 box per
   the S0-08 VM playbooks), the scaling analysis, the SIMD flag list, the sizing
   worksheet's worked examples, and plain guidance ("what can I train on N cores / M GB").
   State the measurement protocol so numbers are reproducible.
7. **Nightly CI wiring:** add a report-only job to the nightly `ci-cpu` schedule running
   the benchmark on the fixed CI runner and uploading the report as an artifact — no
   thresholds, no gate (CI-runner variance makes gating noise; the artifact gives trend
   data for free).

## Out of scope

- Any kernel changes — SIMD work triggered by the flag list becomes new tickets under the
  ROADMAP §4 P3 policy.
- GPU benchmarks and CPU-vs-GPU comparisons (each backend stage owns its own perf
  reporting; the cross-backend parity report is S4-09).
- Perf regression *gating* in CI (report-only here; a gate needs stable dedicated
  hardware, which is a backlog concern).
- MoE/SSM throughput (this audit is the ROADMAP §4 P4 dense-LoRA pass; rerunning the
  harness for those archs is cheap follow-on work once S1-28/S1-31 land).

## Acceptance criteria

- [ ] `benches/cpu_train_step.py` and the per-op perf runner exist and produce a
      machine-readable report (JSON) plus the human-readable tables; a documented
      one-command invocation reproduces them.
- [ ] `docs/perf/cpu.md` exists with: tok/s classes for at least two distinct CPU classes
      (x86 and arm64), the thread-scaling analysis, the per-op breakdown of a dense-LoRA
      train step, and the SIMD-attention flag list (possibly empty, stated explicitly).
- [ ] The report includes the `examples/training/finetune` baseline comparison with config
      deltas stated (BLUEPRINT §9 P1).
- [ ] The sizing worksheet exists with worked examples for at least three configs, and the
      doc reports predicted-vs-measured peak memory deltas for the benchmarked runs.
- [ ] Nightly `ci-cpu` runs the benchmark job report-only and uploads the artifact; a
      completed nightly run demonstrates it.
- [ ] The unit tests for the sizing worksheet (formula vs hand-computed examples) pass in
      the per-PR `ci-cpu` lane.

## Testing & verification

The benchmark itself is the deliverable, not a test subject: verification is
reproducibility (documented protocol, one-command run, JSON artifact) plus the sizing
worksheet's unit tests in `tests/`. Runs: per-PR `ci-cpu` executes only the worksheet
tests and a smoke invocation of the bench script (1 step, tiny model); the full benchmark
runs in the nightly `ci-cpu` job as report-only. Off-CI machines (desktop x86, Apple
Silicon) run via the S0-08 playbooks; their reports land in `docs/perf/cpu.md`, not CI.

## PR notes

- Branch: `ticket/S1-32-cpu-throughput-audit-toks-sizing`.
- Single-repo PR (learning-llamas only): no vendored-code changes, so no fork PR or submodule
  bump — the eval-callback timer lives in the bindings layer.
- Upstreaming disposition: **fork-local** (project docs and benches; nothing to upstream).
- No copied external code. Cite measured machines' CPU models and flags in the report for
  provenance of the numbers.
