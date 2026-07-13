---
id: S1-14
title: "DPO trainer"
stage: 1
track: python
size: M
deps: ["S1-05", "S1-13"]
status: pr-open
pr: https://github.com/dillon-blake/llama-farm/pull/28
---

# S1-14 — DPO trainer

**One-line outcome:** `learning_llamas.train.dpo` runs pairwise DPO with offline-precomputed
reference logprobs (same weights, adapter disabled), a softplus-composite loss, and
converges on a tiny synthetic preference dataset on CPU.

## Why (context)

DPO needs four ingredients on top of the SFT machinery, and BLUEPRINT §6.2 shows every
one is already available. Policy per-token logprobs of realized tokens are the S1-04
`ce_sparse` op negated (per-token `−w·log_softmax(logits)[y]`, so `−ce_sparse` *is* the
masked logprob). Per-sequence sums with a prompt mask are `mul` + `sum_rows`, whose VJPs
exist (`vendor/llama.cpp/ggml/src/ggml.c:6501`, `:6551`). The pairwise loss
`−log σ(β·Δ)` cannot be built on `ggml_sigmoid` because SIGMOID has no backward — it
falls into the unary default abort (`vendor/llama.cpp/ggml/src/ggml.c:6872-6876`) — so
the canonical form is the identity `−log σ(x) = softplus(−x)`, whose VJP exists
(`vendor/llama.cpp/ggml/src/ggml.c:6867`). A SIGMOID backward composite may exist by the
time this ticket runs (S1-19), but the softplus identity stays canonical (BLUEPRINT
§6.2): it is one node and numerically stable at large `|x|`.

Reference logprobs follow BLUEPRINT D6: the reference model is the identical frozen base
with LoRA off — never a second model, zero extra weight memory. LoRA off is
`llama_set_adapters_lora` with an empty adapter set or scale 0
(`vendor/llama.cpp/include/llama.h:690`). D6 explicitly prefers precomputing ref
logprobs *offline* (ordinary decode plus the S1-13 chunked selective-logprob gather) and
feeding them into the training graph as constant F32 inputs, so the training step needs
no second forward at all; BLUEPRINT §10 risk 4 adds the operational reason — toggling the
adapter set forces a graph rebuild, so ref passes must be batched up front, not
interleaved per step.

Pairing is a data-layout problem, not a graph problem: the Python batcher packs
chosen/rejected sequences into the batch dimension, and the per-pair loss values become
the `outputs` tensor reduced under `GGML_OPT_LOSS_TYPE_SUM` — the single-scalar
constraint from S1-02 (extra loss nodes are rejected,
`vendor/llama.cpp/ggml/src/ggml-opt.cpp:343`). Fixed ubatch shapes across steps remain
mandatory (BLUEPRINT D1), so pairs are padded to a fixed layout by the collator.

## What to do

1. `src/learning_llamas/train/dpo.py`: `train_dpo(model, adapter, pref_dataset, config)` with
   `config.beta`; consumes the S1-06 data layer for templating/tokenization/masks and
   drives the S1-05 `train/loop.py` step loop.
2. Offline ref-logprob precompute (`precompute_ref_logprobs(model, pref_dataset)`):
   disable adapters via `llama_set_adapters_lora` with an empty set
   (`vendor/llama.cpp/include/llama.h:690`), run the S1-13 chunked no-grad
   selective-logprob pass over every chosen and rejected sequence, gather per-token
   logprobs of the realized completion tokens, re-enable the adapter once at the end
   (one rebuild each way, per BLUEPRINT §10 risk 4). Cache results keyed by sample id so
   repeated epochs skip the pass.
3. Pair collator: pack chosen/rejected into the batch dimension in a fixed layout
   (chosen rows first half, rejected second half of each ubatch), padded to identical
   shapes every step (D1); pad and prompt tokens get weight 0.
4. `csrc/farm_train.cpp`: register a `dpo` epilogue in the S1-02 loss registry. Named
   inputs: `labels` (I32), `mask` (F32), `ref_logp_sum` (F32, per sequence, constant).
   Graph: per-token `logp = −ce_sparse(logits, labels, mask)` (S1-04) → per-sequence
   sums via `mul` by mask + `sum_rows` → `Δ = (logp_pol_c − logp_pol_r) −
   (ref_logp_c − ref_logp_r)` from the fixed pair layout → per-pair loss
   `softplus(−β·Δ)` → `outputs` under `GGML_OPT_LOSS_TYPE_SUM`. Named inputs are
   constants, never params (S1-02 contract), so gradients flow only through the policy
   logits.
5. Logging via the S1-05 loop hooks: mean implicit-reward margin
   `β·((logp_pol_c − ref_c) − (logp_pol_r − ref_r))`, pair accuracy (margin > 0
   fraction), loss per pair.
6. Tests `tests/test_dpo.py` on the S0-06 fixture models: (a) epilogue-vs-numpy equality
   — the graph's loss on one hand-built pair batch matches a numpy DPO reference within
   documented tolerance; (b) margin increases and pair accuracy reaches 1.0 over N steps
   on a tiny separable synthetic preference set; (c) grads flow only through the policy
   pass — no gradient accumulator exists for any named input (the S1-02
   constants-never-params contract, asserted through the shim), and a DPO step mutates
   only adapter A/B tensors (base weights and named-input buffers byte-identical
   before/after); (d) fully-masked pairs contribute zero gradient.

## Out of scope

- The chunked selective-logprob pass itself (S1-13 owns it; this ticket is a consumer).
- The `ce_sparse` op (S1-04) and SFT trainer/loop mechanics (S1-05).
- GRPO rollouts and training step (S1-15, S1-16).
- SIGMOID/CLAMP/TANH VJP composites (S1-19); this ticket must not depend on them.
- DPO variants (IPO, KTO, label smoothing, length normalization) — `tickets/backlog/`
  candidates once the core converges.
- Online/interleaved reference passes — D6 rules them out for v1.

## Acceptance criteria

- [ ] `pytest tests/test_dpo.py` passes on the Linux CPU VM: numpy-equality, margin/
      accuracy convergence, policy-only-gradient, and masked-pair tests all green.
- [ ] `precompute_ref_logprobs` output matches a naive full-logits decode gather within
      documented tolerance on a fixture model (equality test against the S1-13 path).
- [ ] A run log (committed test artifact or CI output) shows implicit-reward margin
      strictly increasing over the test window on the synthetic preference set.
- [ ] Changing pair-batch shape mid-run raises the S1-02 shape-drift error, tested.
- [ ] `ci-cpu / test` is green with the new tests in the per-PR selection.

## Testing & verification

- `tests/test_dpo.py` (new), pytest on the S0-06 fixture models (F32 + Q8_0 tiny
  llama-arch); runs in `ci-cpu / test` per-PR (S0-07). The synthetic set is tiny enough
  for per-PR budgets; if step count pushes past it, split the convergence case into the
  nightly ci-cpu job and keep equality/gradient tests per-PR.
- No new ops or kernels, so no new `test-backend-ops` MODE_GRAD cases; the op-level
  MODE_GRAD coverage this trainer relies on (ADR-0002 harness) lives in S1-04.

## PR notes

- Branch: `ticket/S1-14-dpo-trainer-offline-ref`.
- Single learning-llamas PR (Python + the small `csrc/` epilogue registration); no vendored
  llama.cpp changes, so no two-repo flow.
- Upstreaming disposition: **fork-local** (product training code).
- Soft coordination: if S1-19's SIGMOID composite has landed, add a cross-check test
  (softplus form vs sigmoid form agree) but keep softplus as the shipped loss; S1-16
  reuses this ticket's ref-precompute helper — keep it importable from a shared module.
