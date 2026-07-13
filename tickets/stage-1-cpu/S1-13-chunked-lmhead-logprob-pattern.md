---
id: S1-13
title: "Chunked lm_head / selective-logprob host pattern for no-grad passes"
stage: 1
track: python
size: M
deps: ["S1-04"]
status: done
pr: 31
---

# S1-13 — Chunked lm_head / selective-logprob host pattern for no-grad passes

**One-line outcome:** a host-side chunked lm_head + selective log-softmax module that
returns per-token logprobs of realized tokens without ever materializing the full
`[n_tokens, n_vocab]` logits tensor — the no-grad logprob engine for DPO/GRPO passes and
long-context evaluation.

## Why (context)

BLUEPRINT §7 item 3: even with the sparse-CE op landed, long contexts and the DPO/GRPO
no-grad logprob passes must chunk the lm_head matmul over token rows — when only the
logprobs of realized tokens are needed, materializing full logits is pure waste. The
numbers are decisive: at 128k vocab, full F32 logits cost 512 KB *per token row* — 4096
tokens ≈ 2 GB for one forward — while a 256-row chunk holds a 128 MB transient and emits
only `n_tokens` floats. BLUEPRINT §6.3 builds GRPO's three-pass step shape on exactly this
("no-grad chunked logp passes for old/ref"), and D6 prefers precomputing DPO reference
logprobs offline and feeding them as constant inputs — this module is that precompute.

The pattern needs **no new native code** (BLUEPRINT §7 item 3: "all existing ops, host
loop"): per chunk, matmul hidden-states×lm_head, then the S1-04 `ce_sparse` op with
`w = 1` on completion tokens, negated, *is* the selective log-softmax — per-token
`logp = −(lse − x_label)` — with stable-lse math and softcap/logit-scale handled inside
the op. The inputs exist today: with embeddings output enabled (`llama_set_embeddings`,
`vendor/llama.cpp/include/llama.h:983`) a normal decode exposes the post-final-norm hidden
states (`llama_get_embeddings` / `_ith`, `vendor/llama.cpp/include/llama.h:1026` and
`:1033`) — verified to be taken after `output_norm` and immediately before the lm_head
projection (`res->t_embd` set at `vendor/llama.cpp/src/models/llama.cpp:236`, lm_head
`build_lora_mm(model.output, ...)` at `:240`).

Licensing shapes the design (ROADMAP §13): the chunked-logsumexp math is fully documented
on unsloth's Apache-2.0 side (`unsloth/kernels/cross_entropy_loss.py:87-150`; softcap and
logit-scale handling at `:84-85`) and is already imported into the S1-04 kernel. But the
GRPO chunked-logprob **orchestration** function is AGPL-marked
(`unsloth/models/rl_replacements.py:1191`), and its `autotune_batch_and_chunks` helper is
pulled from the unavailable/unaudited `unsloth_zoo` package
(`unsloth/models/rl.py:31`, `:367`) and invoked inside that AGPL function
(`unsloth/models/rl_replacements.py:1236`). Everything at the orchestration level —
chunk-size autotuning included — must therefore be a clean-room re-derivation of the
*idea*, never a translation (ROADMAP §13 exclusions; BLUEPRINT §7 licensing note).

## What to do

1. **Module `src/learning_llamas/logprobs.py`** with two layers: a low-level
   `chunked_token_logprobs(hidden, lm_head, labels, *, chunk_rows, softcap=0.0,
   logit_scale=1.0) → f32[n_tokens]`, and a high-level
   `sequence_logprobs(ctx, tokens, seq boundaries, masks) → per-token/per-sequence sums`
   that runs the decode and gathers hidden states.
2. **Hidden-state capture:** enable embeddings output (`llama_set_embeddings`,
   `vendor/llama.cpp/include/llama.h:983`), decode the batch normally (adapter attached or
   not — caller's choice, matching D6 ref-pass semantics), and read post-final-norm hidden
   states via `llama_get_embeddings(_ith)` (`vendor/llama.cpp/include/llama.h:1026`,
   `:1033`) through `_ffi`.
3. **lm_head binding:** read the output-projection tensor from the base GGUF via gguf-py's
   zero-copy `GGUFReader` mmap, handling tied embeddings (no `output.weight` → use
   `token_embd.weight`). Bind it — still quantized — as `src0` of a `MUL_MAT` in a small
   CPU ggml graph built through `_ffi` (in-repo ctypes-over-libggml precedent:
   `vendor/llama.cpp/gguf-py/tests/test_quants.py:112`), reused across chunks with fixed
   chunk shape (last chunk padded, pad rows weighted 0).
4. **Per-chunk pipeline:** `logits_chunk = mul_mat(W, hidden_chunk)` →
   `ce_sparse(logits_chunk, labels_chunk, weights_chunk)` (S1-04) → negate → write into
   the output logp buffer. Weights carry the caller's mask (prompt tokens 0, completion
   tokens 1), so masking falls out for free. Peak transient = one chunk of logits; assert
   no `[n_tokens, n_vocab]` allocation anywhere.
5. **Adapter-on-lm_head correctness:** if the attached adapter targets the output
   projection, the base-weight matmul alone is the wrong policy — add the rank-r delta
   (`scale·B(A·h_chunk)`, two small F32 matmuls from the adapter's A/B) in the chunk
   graph; when the adapter does not target `output`, skip. Cover both in tests.
6. **Chunk-size heuristic:** `choose_chunk_rows(n_tokens, n_vocab, n_embd, mem_budget)` —
   an autotune-style interface (idea per BLUEPRINT §7 item 3), re-derived clean-room per
   the licensing analysis above: pick the largest chunk whose logits transient fits the
   budget, with an explicit override. Document the provenance boundary in the module
   docstring (math: Apache CE kernel; orchestration: original).
7. **Equality test vs the unchunked path** (self-verification discipline, BLUEPRINT §6.3):
   on small shapes, compare against full-logits decode (`llama_get_logits`) + numpy
   log-softmax + gather, across chunk sizes {1, ragged, ≥ n_tokens}, F32 tolerance; also
   compare tied- vs untied-embedding fixtures and adapter-on/off. Design the API so
   S1-16's first-use-compare harness can wrap it unchanged (a pure function of visible
   inputs).
8. **Consumers (soft coordination, prose only):** S1-14 DPO ref/old logprob precompute,
   S1-16 GRPO old/ref passes, and S1-05's long-context masked eval can adopt it; keep the
   returned layout (per-token logp + n_valid) aligned with S1-05's normalization
   convention.

## Out of scope

- The `ce_sparse` op itself and any kernel change (S1-04; GPU ports S2-07/S3-03/S4-04).
- DPO/GRPO trainers and the runtime self-verification harness (S1-14, S1-16) — this
  ticket ships the equality *test*; the permanent first-use-compare mechanism is S1-16's.
- Grad-pass chunking of the lm_head (training-graph memory work; BLUEPRINT §7 item 3's
  no-grad scope only — the grad-side story is gradient checkpointing, S1-17, and chunked
  CE inside the training graph, later).
- GPU placement of the chunk graph — Stage 1 is CPU; the graph is sched-agnostic by
  construction.

## Acceptance criteria

- [ ] `pytest tests/test_logprobs.py` passes on the Linux CPU VM: chunked-vs-unchunked
      equality within documented F32 tolerance on the S0-06 fixture models (F32 and Q8_0
      bases), across chunk sizes {1, ragged, ≥ n_tokens}.
- [ ] Tied-embedding fixture (no `output.weight`) and untied fixture both pass; a test
      asserts correct fallback to `token_embd.weight`.
- [ ] Adapter-on-lm_head test: with an adapter targeting `output`, chunked logprobs match
      the full-logits decode with the same adapter attached; with a non-output-targeting
      adapter, the delta path is skipped and results still match.
- [ ] A memory assertion (allocator stats or instrumented allocation hook) proves peak
      transient scales with `chunk_rows`, not `n_tokens`, on a long synthetic input.
- [ ] `choose_chunk_rows` unit tests pass (budget respected, override honored), and the
      module docstring carries the clean-room provenance note.
- [ ] `ci-cpu / test` is green per-PR with the new tests.

## Testing & verification

`tests/test_logprobs.py` in the S0-06 harness, per-PR in `ci-cpu / test` (S0-07). No new
ops or kernels, so no new `test-backend-ops` MODE_GRAD cases; the op-level correctness this
module leans on (`ce_sparse` forward incl. softcap/scale and wide-vocab chunked lse) is
S1-04's, already enforced in `ci-cpu`. The equality test doubles as the module's standing
self-verification fixture for S1-16. Manual: paste peak-transient numbers for a 4k-token
synthetic run at two chunk sizes into the PR description.

## PR notes

- Branch: `ticket/S1-13-chunked-lmhead-logprob-pattern`.
- Single learning-llamas PR (Python + `_ffi` additions + tests); no vendored llama.cpp changes,
  so no two-repo flow.
- Upstreaming disposition: **fork-local** (host-side product code).
- Provenance per S0-01: module docstring states the math source (unsloth
  `kernels/cross_entropy_loss.py`, Apache-2.0, already imported via S1-04) and that the
  orchestration/autotune layer is original work re-derived from the idea only — no AGPL or
  unsloth_zoo code consulted for implementation.
