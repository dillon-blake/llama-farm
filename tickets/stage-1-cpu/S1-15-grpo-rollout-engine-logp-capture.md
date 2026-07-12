---
id: S1-15
title: "GRPO rollout engine: generation, logp_old capture, rewards, advantages"
stage: 1
track: python
size: M
deps: ["S1-05"]
status: open
pr: null
---

# S1-15 — GRPO rollout engine: generation, logp_old capture, rewards, advantages

**One-line outcome:** `train/rollout.py` generates G rollouts per prompt via ordinary
llama.cpp decoding (KV cache, C-API samplers, parallel sequences, adapter active),
captures per-token `logp_old` at sample time, runs pluggable reward functions on decoded
text, and returns group-normalized advantages in numpy.

## Why (context)

BLUEPRINT §6.3's first claim is the one that makes GRPO cheap here: rollouts are
ordinary llama.cpp inference — KV cache, samplers, parallel sequences, adapter attached —
so the generation side needs zero new native code. Everything this ticket touches is
public C API: batched decode (`llama_decode`,
`vendor/llama.cpp/include/llama.h:966-968`) with multiple `seq_id`s per batch (each token
can carry up to `n_seq_max` sequence ids, `vendor/llama.cpp/include/llama.h:930`;
`llama_batch_init` at `:936`, `llama_n_seq_max` at `:548`), sampler chains
(`llama_sampler_chain_init`/`_add`, `vendor/llama.cpp/include/llama.h:1306`, `:1309`;
`llama_sampler_sample` at `:1498`), and the adapter kept active on the context via
`llama_set_adapters_lora` (`vendor/llama.cpp/include/llama.h:690`).

`logp_old` — the behavior-policy logprob of each sampled token — must be captured at
sample time, without an extra pass: after each decode step the raw logits row for every
live sequence is reachable via `llama_get_logits_ith`
(`vendor/llama.cpp/include/llama.h:1017`), and a host-side stable log-softmax at the
sampled id gives the logprob. Two correctness details: it is computed from the *raw*
logits (the policy distribution), not from the sampler-warped distribution (top-p and
temperature reshape sampling but do not change the policy the ratio is defined against);
and the fallback when capture is unavailable is the S1-13 chunked no-grad pass over the
finished rollout (soft coordination — S1-13 is not a frontmatter dep of this ticket; the
capture path must work standalone).

Prompt-group batching exploits llama.cpp's native KV sharing: decode the prompt once on
one sequence, then copy its KV to the other G−1 group members with `llama_memory_seq_cp`
(`vendor/llama.cpp/include/llama.h:734`) before divergent sampling. This is deliberately
the sanctioned replacement for unsloth's prefix-grouper, which is AGPL-excluded — ROADMAP
§13 notes llama.cpp's native KV-cache prompt sharing achieves the same effect. Rewards
and group-normalized advantages are plain numpy on decoded text (BLUEPRINT §6.3);
BLUEPRINT §10 risk 4 makes rollout throughput the GRPO bottleneck, so the engine carries
counters from day one.

## What to do

1. `src/learning_llamas/train/rollout.py`: `RolloutEngine(model, adapter, sampler_config,
   n_rollouts_per_prompt, max_new_tokens, seed)` producing `RolloutBatch{prompt_tokens,
   completion_tokens, completion_mask, logp_old, rewards, advantages, group_index}`
   (numpy arrays, ragged lengths carried explicitly for the S1-16 collator to pad).
2. Generation loop: tokenize prompts via the S1-06 data layer; decode each prompt once
   on the group's first `seq_id`; `llama_memory_seq_cp`
   (`vendor/llama.cpp/include/llama.h:734`) to the remaining G−1 sequences; then step
   all live sequences in one `llama_batch` per token (one decode call, G logits rows),
   adapter active throughout. Respect `llama_n_seq_max`
   (`vendor/llama.cpp/include/llama.h:548`) when sizing groups; `llama_memory_clear`
   (`:716`) or per-seq removal between prompt groups.
3. Samplers via the C API through `_ffi`: build a chain per sequence —
   top-p/temperature/dist (`vendor/llama.cpp/include/llama.h:1336`, `:1345`, `:1329`) —
   with per-sequence seeds derived from `config.seed` (determinism knob); a greedy chain
   (`:1326`) for tests. Stop on EOS or `max_new_tokens`.
4. `logp_old` capture at sample time: per decode step, read each live sequence's logits
   row (`llama_get_logits_ith`, `vendor/llama.cpp/include/llama.h:1017`), compute
   host-side stable logsumexp (numpy) and record `logits[y] − lse` for the sampled `y`.
   No extra forward pass. Provide `recompute_logp(engine, rollouts)` as the documented
   fallback/cross-check that calls the S1-13 chunked pass when available.
5. Reward plug-in interface: `reward_fn(prompt_text, completion_text, sample_meta) →
   float`, applied to `llama_detokenize`d completions
   (`vendor/llama.cpp/include/llama.h:1170`); rewards evaluated per group after
   generation. Ship two built-ins for tests: target-output-length reward and
   substring-match reward.
6. Advantages: per-group normalization in numpy — `adv = (r − mean_g)/(std_g + ε)` —
   with a documented degenerate-group rule (all-equal rewards → zero advantages).
7. Determinism + throughput: fixed-seed rollouts are reproducible run-to-run on CPU;
   counters for generated tokens/s, decode calls, prompt-KV-reuse hits, and wall time
   per group, surfaced through the S1-05 logging hooks.
8. Tests `tests/test_rollout.py` on the S0-06 fixture models (see Acceptance criteria).

## Out of scope

- The GRPO loss graph, three-pass step orchestration, and self-verification harness
  (S1-16).
- The chunked no-grad logprob pass implementation (S1-13) — referenced as fallback only.
- Reward-model training, LLM-judge rewards, and any server/distributed rollout backend
  (`tickets/backlog/` candidates).
- Prefix-grouper-style attention sharing beyond native KV copy (AGPL-excluded, ROADMAP
  §13; native `llama_memory_seq_cp` is the v1 mechanism).
- Speculative decoding or batched-server throughput work (BLUEPRINT §10 risk 4 revisit).

## Acceptance criteria

- [ ] `pytest tests/test_rollout.py` passes on the Linux CPU VM.
- [ ] Determinism: two runs with the same seed produce identical token ids and bitwise
      identical `logp_old` arrays on CPU.
- [ ] `logp_old` correctness: sample-time capture matches a no-grad full-logits
      recompute of the same rollouts within documented tolerance (and exactly for
      greedy sampling).
- [ ] KV-reuse correctness: greedy rollouts with the seq-copy path equal greedy rollouts
      generated per-sequence from scratch, token for token.
- [ ] Advantages: per-group mean ≈ 0 and std ≈ 1 on random rewards; degenerate groups
      yield zero advantages (tested).
- [ ] Throughput counters (tokens/s, decode calls, KV-reuse hits) are populated and
      asserted nonzero in tests.
- [ ] `ci-cpu / test` is green with the new tests in the per-PR selection.

## Testing & verification

- `tests/test_rollout.py` (new), pytest on the S0-06 fixture models; runs in
  `ci-cpu / test` per-PR (S0-07). Generation lengths stay tiny (≤ 32 new tokens) to fit
  the per-PR budget.
- No new ops, kernels, or training-graph changes — no `test-backend-ops` MODE_GRAD cases
  and no convergence-gate interaction (S1-12 untouched). The engine's consumer-side
  correctness is exercised end-to-end by S1-16's toy-task test.

## PR notes

- Branch: `ticket/S1-15-grpo-rollout-engine-logp-capture`.
- Single learning-llamas PR (pure Python over existing `_ffi` bindings; extend `_ffi` with
  any missing sampler/memory symbols per the S0-04 pattern). No vendored llama.cpp
  changes, so no two-repo flow.
- Upstreaming disposition: **fork-local** (product training code).
- Soft coordination: S1-13 (`recompute_logp` fallback wiring once it merges) and S1-16
  (consumes `RolloutBatch` — keep its field layout documented and stable).
