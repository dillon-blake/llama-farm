---
id: S1-07
title: "Data layer: sample packing with seq_ids, boundary masking, fixed-shape collators"
stage: 1
track: python
size: M
deps: ["S1-06"]
status: open
pr: null
---

# S1-07 — Data layer: sample packing with seq_ids, boundary masking, fixed-shape collators

**One-line outcome:** a packing collator that fills fixed-shape ubatches with multiple
samples under distinct `seq_id`s, masks pack-boundary tokens, always pads to identical
shapes, and proves packed-vs-unpacked loss equality by test.

## Why (context)

Padding-free packing is one of the highest-value throughput techniques for SFT on short
samples, and llama.cpp gives the hard part away: attention isolation between packed
samples is **native**. Assigning each packed sample a distinct `seq_id` in the
`llama_batch` (`vendor/llama.cpp/include/llama.h:250-251`) makes the graph's mask input
block cross-sequence attention — `fill_mask` explicitly skips positions whose seq_id differs
("mask different sequences", `vendor/llama.cpp/src/llama-graph.cpp:424-427`, inside
`llm_graph_input_attn_no_cache::set_input` at `llama-graph.cpp:406-453`). So the Python
collator only decides placement and weights; no attention code changes (BLUEPRINT D7).

**Depends on S1-00 for this to hold.** That `fill_mask` belongs to the *no-cache* attention
input, which serves embedding/non-causal archs — not the causal archs trained here, which today
route through the KV cache and get a `[n_kv, n_tokens]` mask instead. S1-00 makes the training
graph bypass the cache and emit the uncached-shaped mask with seq_id isolation preserved; that
is what makes packing-by-seq_id work. Verify the isolation property against the post-S1-00
graph, not by assumption.

Two correctness rules define the collator. First, **boundary masking**: next-token CE at
the last token of packed sample k would predict the first token of sample k+1, so that
position's loss weight must be 0 — `boundary = cumsum(lengths) - 1 → weight 0`. This rule
must be **re-derived independently**: the unsloth packing implementation is LGPL-3.0+
(header at `unsloth/utils/packing.py:1-14`) and is concept-only per ROADMAP §13 item 9;
the one-sentence spec above is the entire permissible import. Second, **fixed shapes
always**: ggml-opt's dynamic-graph mode keys gradient/optimizer state by forward-graph
node index (`vendor/llama.cpp/ggml/src/ggml-opt.cpp:458-486`) and asserts on varying
batch sizes (`ggml-opt.cpp:851`), so every ubatch is padded to the same shape every step
(BLUEPRINT D1), pad tokens carrying weight 0.

Packing is an exactness-preserving optimization, so it falls under the project's
self-verification discipline (BLUEPRINT §6.3): the masked loss over a packed batch must
equal the loss over the same samples unpacked, and this ticket lands that as a property
test rather than trusting the construction.

## What to do

1. `src/learning_llamas/data/packing.py`: `PackingCollator(n_ubatch, max_samples_per_pack,
   strategy)` consuming S1-06 sample records (`tokens`, `weights`, length) and yielding
   fixed-shape batches: token buffer `[n_ubatch]`, weight buffer `[n_ubatch]` F32,
   `seq_id` per slot, and per-sample position values restarting at 0 within each sample
   (positions and seq_ids feed the S1-02 batch construction; llama.cpp derives the
   isolation mask from them). Start with deterministic first-fit-decreasing over a
   fetch window; the strategy is pluggable but only one implementation ships.
2. Boundary masking, re-derived: after placement, set weight 0 at each packed sample's
   last token (`cumsum(lengths) - 1` within the pack). Keep this independent of any
   sample-level masking from S1-06 (a sample whose final tokens are already weight-0 is
   unaffected).
3. Padding: fill remaining slots with the pad token (fall back to BOS/EOS when the vocab
   lacks a pad id, per the S1-06 vocab accessors), weight 0, a dedicated pad `seq_id`.
   Assert every emitted batch has the identical shape tuple; expose it so `loop.py`
   (S1-05) can enforce the D1 fixed-topology error early.
4. `NoPackCollator` with the same interface (one sample per ubatch, padded) — the
   reference path for the equality test and the fallback for long-sample datasets.
5. Property tests `tests/test_packing.py`, collator-level (no native code): boundary
   indices correct for randomized length distributions (hypothesis-style or seeded
   fuzz); all batches shape-identical; every non-pad token's `(seq_id, pos)` is
   consistent with its source sample; total weight-1 count is conserved between packed
   and unpacked collation of the same dataset.
6. Packed-vs-unpacked **loss equality** test: identical data through `PackingCollator`
   and `NoPackCollator`, forward-only masked loss via the S1-05 eval path, assert
   equality within F32 summation tolerance on a fixture model. Soft coordination: S1-05
   is not a frontmatter dep — if it has not merged when this ticket is otherwise ready,
   land this test in the same PR gated on S1-05's merge order (the criterion below still
   binds).
7. Throughput counters surfaced to the S1-05 logging hooks: tokens/step, valid-token
   fraction, pad fraction, samples/pack.

## Out of scope

- Attention-kernel or mask changes in llama.cpp (native seq_id isolation is used as-is).
- The trainer/step loop itself (S1-05) and DPO pair packing (S1-14 packs chosen/rejected
  into the batch dimension itself).
- Cross-ubatch sample splitting (a sample longer than `n_ubatch` errors with guidance;
  long-context handling is the gradient-checkpointing/chunking track, S1-17/S1-13).
- GRPO prefix sharing and any shared-prefix packing (AGPL prefix-grouper is excluded;
  llama.cpp KV-cache prompt sharing covers generation — ROADMAP §13).

## Acceptance criteria

- [ ] `pytest tests/test_packing.py` passes on the Linux CPU VM, including the seeded
      fuzz over length distributions.
- [ ] The packed-vs-unpacked loss-equality test passes on a fixture model with the
      documented tolerance.
- [ ] A batch stream from `PackingCollator` over a ragged dataset yields byte-identical
      shape tuples for every batch (tested).
- [ ] Boundary positions carry weight 0 in every emitted batch (direct assertion in the
      fuzz test, not only via loss equality).
- [ ] Counters (tokens/step, pad fraction) appear in the logging-hook payload (unit
      test).
- [ ] `ci-cpu / test` green per-PR with the new tests.

## Testing & verification

- `tests/test_packing.py` (new): collator-level property tests run pure-Python; the loss
  equality test uses S0-06 fixture models + the S1-05 forward-only eval; all in
  `ci-cpu / test` per-PR (S0-07), fuzz with more iterations nightly.
- No MODE_GRAD applicability: no ops/kernels; correctness is the equality property.

## PR notes

- Branch: `ticket/S1-07-packing-seq-ids-collators`.
- Single learning-llamas PR; no vendored llama.cpp changes.
- Upstreaming disposition: **fork-local** (product data layer).
- Provenance: the boundary-mask rule is re-derived from its public one-sentence spec;
  per S0-01 policy add a comment in `packing.py` noting the LGPL exclusion of
  `unsloth/utils/packing.py` and that no code was consulted or copied from it.
