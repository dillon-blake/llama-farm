---
id: S1-03
title: "P0 proof-of-gradient: stopgap composite CE, loss falls, FD check, llama-cli loads adapter"
stage: 1
track: python
size: M
deps: ["S1-02", "S0-06"]
status: pr-open
pr: https://github.com/dillon-blake/llama-farm/pull/19
---

# S1-03 — P0 proof-of-gradient: stopgap composite CE, loss falls, FD check, llama-cli loads adapter

**One-line outcome:** end-to-end proof on CPU that the whole stack trains: SFT with the stopgap
composite CE on a tiny Q4_K llama-arch model where loss decreases, a finite-difference gradient
check on A/B passes, and the saved adapter loads in stock `llama-cli --lora`.

## Why (context)

This is the P0 exit milestone from BLUEPRINT §9: "loss falls; trained adapter loads in stock
`llama-cli --lora`; finite-difference grad check on A/B passes." Everything below it exists after
S1-01/S1-02 (param wiring, forked step loop) and S0-05/S0-06 (adapter writer, fixture models), but
nothing yet proves that gradients flowing through a **quantized** base — dequantizing `OUT_PROD`
on the backward path — are numerically right end-to-end. BLUEPRINT §10 (risk 2) flags exactly
this: backward through quantized weights is engine-supported but lightly exercised, so validate
early with a finite-difference test. Passing this ticket closes the Stage-0→1 gate: it is the
first demonstration that learning-llamas's core promise (train LoRA on a frozen quantized GGUF) holds.

The loss is the bring-up stopgap from BLUEPRINT §6.1, registered in S1-02's epilogue registry:
softmax → one-hot mul → sum_rows → log, with select-then-log ordering so masked/near-zero
probabilities never produce `0·(−inf)` NaNs. All constituent VJPs exist upstream. It materializes
one-hot tensors and is O(n_vocab) per token, which is acceptable only on tiny fixtures — S1-04's
`ce_sparse` op and the S1-05 trainer replace it, but keeping it working (behind a debug flag,
per S1-05) preserves a permanent cross-check for the new op.

The interchange test matters as much as the math: BLUEPRINT D3 makes the adapter GGUF format the
checkpoint format, so a freshly trained adapter must load in stock tooling with zero conversion.
`llama-cli` exposes `--lora` (`vendor/llama.cpp/common/arg.cpp:2661`), and the stock loader/attach
path is public C API (`vendor/llama.cpp/include/llama.h:657` `llama_adapter_lora_init`, `:690`
`llama_set_adapters_lora`). The in-tree `examples/training/finetune.cpp` remains a smoke-test
reference only (full-parameter F32, forces mmap off) — this ticket is its LoRA-shaped replacement
proof, run with `use_mmap=true` throughout.

## What to do

1. Minimal debug accessors in the shim (explicitly superseded by S1-08's real enumeration API;
   mark them `ll_debug_*` and exclude them from API-stability promises):
   `ll_debug_get_tensor` / `ll_debug_set_tensor` addressing adapter A/B by `ab_map` name, and
   `ll_debug_grad_acc` wrapping `ggml_opt_grad_acc`
   (`vendor/llama.cpp/ggml/include/ggml-opt.h:156`) for a named param tensor. Bind via `_ffi`.
2. Python driver `tests/test_p0_gradient.py` (plus a reusable helper under `src/learning_llamas/` only
   if trivially shared): load the Q4_K fixture (S0-06) with `use_mmap=true`, create and attach a
   rank-4 zero-init adapter (S0-05 writer), `ll_opt_init_lora`, then run N=32 `ll_train_step`
   calls with `sft_ce_stopgap` on one fixed tiny batch (fixed seed, constant shapes per D1).
3. Loss-decrease assertion (monotonic-ish, not per-step): `mean(loss[-4:]) < 0.8 * mean(loss[:4])`
   and at least 75% of consecutive 4-step-window means decrease. Record the curve in the test log.
4. Finite-difference gradient check: after one training-mode forward/backward on the fixed batch,
   compare `ll_debug_grad_acc` values against central finite differences for ≥16 sampled A and B
   elements across ≥2 layers (perturb via `ll_debug_set_tensor`, re-evaluate loss forward-only,
   restore). Acceptance tolerance: relative error `|g_an − g_fd| / max(|g_an|, |g_fd|, 1e-8)`
   ≤ 5e-2 per element, documented in the test docstring (the quantized forward is deterministic,
   so FD is well-defined; this is a full-graph Python-level check, distinct from the per-op
   MODE_GRAD bounds of ADR-0002).
5. NaN-robustness test for the stopgap composite: a batch whose mask zeroes prompt positions and
   whose logits are pushed to extremes must yield finite loss and finite A/B grads (validates the
   select-then-log ordering claim from BLUEPRINT §6.1).
6. Save the trained adapter with the S0-05 writer (pull live A/B via the debug accessors; write
   real `adapter.lora.alpha` from the training config — never 0, per the
   `vendor/llama.cpp/src/llama-adapter.h:48-88` scale caveat). Then verify interchange two ways:
   (a) stock reload via `llama_adapter_lora_init` + `llama_set_adapters_lora` in-process — logits
   on a fixed prompt with the trained adapter differ from base logits (adapter shifts logits) and
   the load emits no errors; (b) run the vendor-built `llama-cli` (S0-03 keeps
   `LLAMA_BUILD_TOOLS` available) with `--lora <adapter>` on a fixed prompt, assert exit code 0
   and the loader's "loading lora adapter from" log line
   (`vendor/llama.cpp/src/llama-adapter.cpp:150`).
7. Write `docs/dev/p0-proof-of-gradient.md`: the exact recipe (fixture, adapter config, step
   count, expected curve shape, how to rerun each check), and state that this is the Stage-0→1
   gate recipe that later backends re-run conceptually via S1-12's convergence gate.

## Out of scope

- The `ce_sparse` op and the real SFT trainer/normalization (S1-04, S1-05).
- Data layer (templating, masking round-trip, packing) — S1-06/S1-07; this ticket uses a
  hand-built fixed batch.
- Full adapter enumeration/save API surface (S1-08) and merged-model export (S1-08).
- Convergence-vs-PEFT reference tolerance testing (S1-12) — this ticket only proves "loss falls",
  not curve parity.
- Per-quant-type FD sweep beyond Q4_K (+F32 sanity) — broadened by S1-12 and backend stages.

## Acceptance criteria

- [ ] `pytest tests/test_p0_gradient.py` passes on a Linux x86_64 CPU-only build: loss-decrease
      assertion holds on the Q4_K fixture with `use_mmap=true` (and a fast F32-fixture sanity
      variant of the same test also passes).
- [ ] The finite-difference check passes for all sampled A/B elements at the documented ≤ 5e-2
      relative tolerance, on both F32 and Q4_K bases.
- [ ] The NaN-robustness test passes (finite loss and grads with masked extreme logits).
- [ ] The saved adapter loads via stock `llama_adapter_lora_init` with no error and measurably
      shifts logits vs base on a fixed prompt (exact comparison of logit vectors differing).
- [ ] `llama-cli --lora` run on the saved adapter exits 0 with the "loading lora adapter from" log
      line (captured in the test via subprocess).
- [ ] `docs/dev/p0-proof-of-gradient.md` exists and documents the full recipe.
- [ ] `ci-cpu` green with the new tests in the per-PR selection (if total runtime exceeds the
      per-PR budget, only the 32-step loss-curve test may move behind the `slow` marker; the FD
      and interchange tests stay per-PR).

## Testing & verification

This ticket *is* a test package: `tests/test_p0_gradient.py` in the S0-06 harness, per-PR in
`ci-cpu` (with the nightly lane running everything including any `slow`-marked pieces). No new
ggml ops, so no `test-backend-ops` MODE_GRAD cases here — the Python FD check is deliberately a
full-graph, quantized-base complement to MODE_GRAD, per BLUEPRINT §10 risk 2. The `llama-cli`
interchange check runs as a subprocess against the vendor-built binary in the same CI job.

## PR notes

- Branch: `ticket/S1-03-p0-proof-of-gradient`.
- One learning-llamas PR: shim debug accessors + Python tests + docs. No vendored llama.cpp changes,
  so no two-repo flow.
- Upstreaming disposition: **fork-local**.
- Soft coordination: S1-05 will hide `sft_ce_stopgap` behind a debug flag rather than delete it —
  keep the epilogue name stable; S1-08 replaces the `ll_debug_*` accessors with the real G2 API
  and should remove them in its PR.
