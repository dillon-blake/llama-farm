---
id: S1-16
title: "GRPO training step: clip-via-relu graph, k3 KL, self-verification harness"
stage: 1
track: python
size: M
deps: ["S1-15", "S1-13"]
status: open
pr: null
---

# S1-16 — GRPO training step: clip-via-relu graph, k3 KL, self-verification harness

**One-line outcome:** the GRPO update works end-to-end on CPU — importance ratio
`exp(logp_new − logp_old)`, PPO clip composed from RELU identities, per-token advantage
weighting, optional k3 KL to the reference — plus the project's self-verification
harness (first-use compare vs the naive path, permanent fallback on mismatch).

## Why (context)

BLUEPRINT §6.3 lays out the whole update in terms of ops that already have VJPs.
`logp_new` comes from the S1-04 `ce_sparse` op negated, on the training graph's logits.
The importance ratio is `exp(logp_new − logp_old_const)` — the EXP backward case exists
(`vendor/llama.cpp/ggml/src/ggml.c:6857`), and `logp_old` arrives as a constant named
input from S1-15's sample-time capture. The PPO clip cannot use `ggml_clamp`: CLAMP has
no backward case in `ggml_compute_backward` and falls into the op-level default abort
(`vendor/llama.cpp/ggml/src/ggml.c:6904-6907`). Instead the clip is composed from RELU
identities — `clip(r,lo,hi) = lo + relu(r−lo) − relu(r−hi)` and `min(a,b) = b −
relu(b−a)` — using the existing RELU VJP (`vendor/llama.cpp/ggml/src/ggml.c:6847`) plus
SUB/SCALE (`:6493`, `:6631`). A CLAMP VJP composite may land later (S1-19) and can
simplify the graph, but the relu composite stays in-tree permanently as the reference
form. The optional KL penalty uses the low-variance k3 estimator `exp(Δ) − Δ − 1` with
`Δ = logp_ref − logp_new`, from precomputed ref logprobs (BLUEPRINT D6) via EXP/SUB.

The step shape is the three-pass pattern of BLUEPRINT §6.3: (1) generate with the
adapter on (S1-15), (2) no-grad chunked logp passes for old/ref (S1-13; ref with the
adapter disabled per D6), (3) the grad pass with the loss above through S1-02's
`lf_train_step`. Per-token weighting is a plain `mul` by an `advantage·mask` input
tensor (MUL VJP at `vendor/llama.cpp/ggml/src/ggml.c:6501`), and the per-sequence/batch
reduction folds into the single `outputs` scalar under `GGML_OPT_LOSS_TYPE_SUM` (extra
loss nodes rejected, `vendor/llama.cpp/ggml/src/ggml-opt.cpp:343`). Fixed ubatch shapes
across steps (BLUEPRINT D1) mean rollout batches are padded to one layout.

Licensing shapes the harness work. ROADMAP §13 marks unsloth's GRPO chunked-logprob
orchestration as AGPL — the function-level marker sits inside
`_get_per_token_logps_and_entropies` (unsloth `unsloth/models/rl_replacements.py:1191`
in the research checkout) — so that code must never be read as implementation reference;
the underlying math is fully available on the Apache side (unsloth
`unsloth/kernels/cross_entropy_loss.py:87-150`, already consumed by S1-04/S1-13). What
we *do* copy is unsloth's discipline, not its code: any exactness-preserving
optimization first-use-compares against the naive path and permanently falls back on
mismatch (BLUEPRINT §6.3, Appendix A). This ticket builds that harness because GRPO is
its first real customer (sample-time logp capture vs chunked recompute).

## What to do

1. `csrc/farm_train.cpp`: register a `grpo` epilogue in the S1-02 loss registry. Named
   constant inputs: `labels` (I32), `adv_mask` (F32, per-token `advantage·mask`
   precomputed host-side), `logp_old` (F32 per token), optional `logp_ref` (F32 per
   token). Graph: `logp_new = −ce_sparse(logits, labels, ones)` → `r = exp(logp_new −
   logp_old)` → clipped surrogate `min(r·adv_mask, clip(r, 1−ε, 1+ε)·adv_mask)` via the
   relu identities above (correct elementwise for negative advantages by construction of
   `min`) → negate → optional `+ kl_coef · (exp(Δ) − Δ − 1)` masked to completion tokens
   → `sum_rows`/sum into `outputs` under `GGML_OPT_LOSS_TYPE_SUM`. `ε` and `kl_coef` are
   epilogue parameters, fixed per run (topology-stable, D1).
2. `src/llama_farm/train/grpo.py`: `train_grpo(model, adapter, prompts, reward_fn,
   config)` orchestrating the three-pass step: S1-15 rollouts → S1-13 chunked no-grad
   passes for `logp_old` cross-check and `logp_ref` (adapter disabled per D6/S1-14's
   helper; skipped when `kl_coef == 0`) → pad rollouts to the fixed batch layout →
   `lf_train_step` with the `grpo` epilogue; host-normalize by valid tokens; log mean
   reward, ratio stats, clip fraction, KL estimate.
3. Self-verification harness `src/llama_farm/verify.py`: `SelfVerified(fast_fn,
   naive_fn, tolerance, name)` — on first invocation (and optionally every N-th) runs
   both, compares within tolerance; on mismatch logs the divergence and permanently
   routes to `naive_fn` for the rest of the process. Wire the first customer:
   sample-time `logp_old` capture (fast) vs S1-13 chunked recompute (naive). Clean-room
   note in the module docstring: interface and math re-derived; the unsloth GRPO
   orchestration function is AGPL and was not consulted (ROADMAP §13).
4. Numpy reference implementations (`tests/reference_grpo.py`) of ratio/clip/min/k3 and
   the full per-token loss for the equality tests below.
5. Tests `tests/test_grpo.py` on the S0-06 fixture models (see Acceptance criteria),
   including the toy-task run: reward = closeness to a target output length, assert mean
   group reward increases over N updates on CPU with a fixed seed.

## Out of scope

- Rollout generation mechanics (S1-15) and the chunked logp pass (S1-13).
- The CLAMP VJP composite itself (S1-19); this ticket must work without it.
- GRPO variants (DAPO/GSPO-style clipping, token-level KL schedules, entropy bonuses)
  — `tickets/backlog/` once the core converges.
- Any GPU execution or kernel work; backend stages inherit this via the S1-12 gate.
- Packing/prefix-sharing optimizations beyond what S1-15 already does (future
  self-verification customers, not this ticket).

## Acceptance criteria

- [ ] `pytest tests/test_grpo.py` passes on the Linux CPU VM.
- [ ] Composite-equality: relu-composed `clip` and `min` match numpy elementwise on
      random inputs spanning both clip regions and negative advantages (exact on CPU);
      the k3 term matches numpy within documented tolerance.
- [ ] Epilogue-vs-numpy equality: the graph's loss on a hand-built rollout batch matches
      `tests/reference_grpo.py` within documented tolerance, with and without the KL
      term.
- [ ] Gradient sanity: with `kl_coef = 0`, a batch with all-zero advantages produces
      zero A/B updates; no gradient accumulator exists for `logp_old`/`logp_ref`/
      `adv_mask` (named-input constants contract, S1-02), and a step mutates only
      adapter A/B tensors.
- [ ] Toy task: mean group reward strictly increases over the test window
      (fixed seed, CPU) on the output-length target task.
- [ ] Self-verification: a test with an artificially divergent fast path proves
      first-use comparison, mismatch logging, and permanent fallback.
- [ ] `ci-cpu / test` per-PR green; the toy-task run is in the nightly ci-cpu job if it
      exceeds the per-PR budget (decide by measured runtime, record in the PR).

## Testing & verification

- `tests/test_grpo.py` + `tests/reference_grpo.py` (new), pytest on the S0-06 fixture
  models; equality/gradient/self-verification tests per-PR in `ci-cpu / test` (S0-07),
  toy-task convergence per-PR if fast enough, else nightly ci-cpu.
- No new ops or kernels — no new `test-backend-ops` MODE_GRAD cases (ADR-0002 coverage
  for `ce_sparse` lives in S1-04; EXP/RELU/MUL/SUB VJPs are upstream-covered).

## PR notes

- Branch: `ticket/S1-16-grpo-step-clip-relu-kl`.
- Single llama-farm PR (Python + `csrc/` epilogue registration); no vendored llama.cpp
  changes, so no two-repo flow.
- Upstreaming disposition: **fork-local** (product training code).
- Provenance: `verify.py` and the epilogue carry a comment citing the unsloth
  *discipline* as design inspiration with the ROADMAP §13 clean-room statement (no AGPL
  code consulted), per S0-01 policy.
- Soft coordination: S1-19 (optional graph simplification behind a flag, relu composite
  kept as reference) and S1-14 (shared ref-logprob precompute helper).
