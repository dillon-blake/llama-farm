---
id: S1-24
title: "FA8: graph-level chunked-attention backward fallback (kernel-free long-context path)"
stage: 1
track: shim
size: M
deps: ["S1-00", "S1-19", "S1-20"]
status: pr-open
pr: https://github.com/dillon-blake/llama-farm/pull/43
---

# S1-24 — FA8: graph-level chunked-attention backward fallback (kernel-free long-context path)

> **✅ DONE — re-scoped exactly as the banner below recommended, and the re-scope was right.**
>
> Measured (tiny fixture, peak compute buffer, one training step):
>
> | n_ctx | naive | chunk only | ckpt only | **ckpt+chunk** | vs naive |
> |---:|---:|---:|---:|---:|---:|
> | 1024 | 79 MiB | 66 MiB | 47 MiB | **44 MiB** | 1.80x |
> | 2048 | 254 MiB | 196 MiB | 166 MiB | **128 MiB** | 1.98x |
> | 4096 | 904 MiB | 652 MiB | 620 MiB | **416 MiB** | **2.17x** |
>
> The acceptance bar (>=2x at 4096) is met. Losses are **bit-identical** to naive; gradients agree
> to ~1e-6. The `chunk only` column is the "you must pair this with S1-17" warning, measured.
>
> **Two things the plan did not anticipate**, both now carrying comments in the fork:
> 1. **Slice the token axis BEFORE the permute.** Slicing the permuted `q` takes a view of a
>    non-contiguous tensor and ggml's autodiff dies in `ggml_scale`'s `is_padded_1d` assert — inside
>    the *LoRA scale's* backward, three ops from the mistake, naming a function you never called.
> 2. **A balanced concat tree, not a left fold.** A fold's accumulators grow `1/C .. C/C` of the
>    output and hand the saved memory straight back: peak got **worse** past 4 chunks.
>
> **NOT DONE:** softcap (gemma2) and ALiBi have no fixture. `attn_soft_cap` is set in *code* by
> `models/gemma2.cpp`, not by a GGUF key, so it cannot be enabled on a llama-arch fixture. Both
> branches are written and are chunk-invariant by construction (ALiBi's slope is a function of the
> HEAD index; softcap is elementwise) — an argument, not a test, and recorded as one. A gemma2
> fixture is the follow-up.
>
> ---
>
> **⚠️ The original justification was stale and the original plan did not work. (Kept for the record.)**
>
> **The memory figures are stale.** The "Why" quotes 128-192 GiB at 4k/8B on the grounds that the
> attention matrices are live across *all layers simultaneously*. **S1-17 (gradient checkpointing)
> landed** and killed the `x n_layers` factor: at `segment_len=1` only one layer's matrices are live.
> The real numbers are roughly 4-8 GiB with S1-17 alone, and 0.5-1 GiB with S1-17 + this ticket. The
> two are **multiplicative, not alternatives** — S1-17 removes the depth factor, S1-24 removes the
> `n_ctx^2` factor *within* a layer — so the ticket is still worth doing, for a smaller stated win.
>
> **"No vendored llama.cpp changes" is FALSE.** The shim cannot reach the backward: the forward graph
> is built inside `llm_graph_context::build_attn_mha`, the backward inside `ggml_opt_build`, and the
> shim's `build_loss` callback can only *append forward nodes* to `gf`. The only injection point is
> `opt_ctx->checkpoints` — which is precisely the hook S1-17 had to add **to the vendor**.
>
> **Recommended re-scope, which is also cheaper:** chunk the **forward** attention into Q-chunks
> inside `build_attn_mha` behind a cparam, and let S1-17's existing recompute machinery produce the
> chunked backward *for free* — each chunk's `kq_soft_max` becomes a segment-interior node,
> recomputed just before the backward node that reads it, then dies. No hand-emitted VJPs, no new
> backward hook. One small vendor change instead of a graph rewriter plus a new vendor API.
>
> **This is the only member of the FA family that changes the path training actually uses** — FA is
> force-disabled during training (`llama-context.cpp`, `set_training`), and S1-21/22/23 all place
> "turn FA back on" out of scope. Prioritize it above them.

**One-line outcome:** chunked attention backward built entirely from existing ops
(per-Q-chunk `soft_max_ext` recompute + `SOFT_MAX_BACK`/`MUL_MAT`/`OUT_PROD`): peak
attention memory drops by the chunk factor at ~+50% attention FLOPs, unblocking 2-4k
ctx training on every backend before FA kernels land.

## Why (context)

The naive attention path materializes 2-3 `[n_kv, n_q, n_head]` F32 tensors per layer,
live across all layers simultaneously because the stock backward references them — for
a Llama-3.1-8B-class model that is 2-3 GiB at n_ctx 512, 8-12 GiB at 1024, 32-48 GiB
at 2048, and an infeasible 128-192 GiB at 4096 (ROADMAP §8 memory-cliff table; the
`n_head·n_ctx` term in the BLUEPRINT §10 risk-5 sizing formula is exactly this).
Real FA backward kernels (FA5-FA7) are the long poles of the whole roadmap, so FA8 is
the designated standing fallback while they are in flight (ROADMAP §12 risk 12): if
the backward never references the full-size attention matrices — instead recomputing
softmax per Q-chunk and consuming it immediately — peak attention memory drops by the
chunk factor at the cost of roughly +50% attention FLOPs (the S/P recompute).

This is kernel-free by design (ROADMAP §8 FA8): every emitted op exists on CPU today.
The gradient set is the same one ggml's own backward uses for the naive path —
`ggml_soft_max_ext_back` (constructor `vendor/llama.cpp/ggml/src/ggml.c:4128`, emitted
by the SOFT_MAX backward case at `vendor/llama.cpp/ggml/src/ggml.c:6762-6773`) and the
MUL_MAT VJP pair `mul_mat`/`out_prod` (`vendor/llama.cpp/ggml/src/ggml.c:6578-6630`).
Two coverage items ride on the frontmatter deps: softcap graphs (gemma2/3) put a
`tanh` node inside the attention subgraph
(`vendor/llama.cpp/src/llama-graph.cpp:2470-2477`), so the chunked backward needs the
TANH VJP (S1-19); ALiBi models put `max_bias > 0` on `soft_max_ext`, so the recompute's
backward needs `SOFT_MAX_BACK` with `max_bias > 0` (S1-20). Both must be documented as
covered once this lands. This is shim-track graph work in learning-llamas — no vendored
llama.cpp changes — and it must respect the fixed-topology constraint of ggml-opt's
node-index-keyed optimizer state (BLUEPRINT D1): the chunk factor is fixed for the
lifetime of a training run.

## What to do

1. **Attention-subgraph identification** (new `csrc/farm_attn_chunk.cpp`): in the
   S1-02 forked-loop graph build, locate each layer's naive attention subgraph —
   `kq = mul_mat(k, q)` → optional softcap scale/tanh/scale →
   `soft_max_ext(kq, mask, scale, max_bias)` → `kqv = mul_mat(v, kq)`
   (`vendor/llama.cpp/src/llama-graph.cpp:2450-2494`; nodes are named "kq",
   "kq_soft_max", "kqv" by the graph callback). Error clearly on graphs where the
   pattern is not found (MLA/FA-forward variants), leaving those on the stock path.
2. **Chunked backward construction:** for each identified subgraph, keep the forward
   result but build the backward manually per Q-chunk instead of letting
   `ggml_build_backward_expand` derive it: for chunk `c` take view `Qc` (and the
   matching mask-row view), recompute `Sc = mul_mat(K, Qc)` (+ softcap chain),
   `Pc = soft_max_ext(Sc, mask_c, scale, max_bias)`, then emit the gradient ops —
   `dPc` from `dOc` and V (MUL_MAT VJP pattern), `dSc = soft_max_ext_back(dPc, Pc,
   scale, max_bias)`, the softcap `(1−tanh²)` chain via the S1-19 TANH VJP, and
   `dQc` / `dK +=` / `dV +=` via `mul_mat`/`out_prod` on chunk views. Mechanism
   choice — hand-emitting VJP ops vs a scoped `ggml_build_backward_expand` over the
   per-chunk recompute subgraph — is the implementer's, recorded with rationale in the
   PR description; either way the emitted set is
   `SOFT_MAX_BACK`/`MUL_MAT`/`OUT_PROD` (+ TANH backward for softcap), and node
   topology must be identical every step (D1).
3. **Make the memory win real:** the full-size `kq`/`P` forward tensors must not be
   kept live for backward (that is the entire point). Verify via peak-alloc
   measurement that `ggml_gallocr` reuses them once the stock backward no longer
   references them; if forward residency still dominates, chunk the forward attention
   too (store only `O` and the Q/K/V inputs). This integrates naturally with S1-17
   gradient checkpointing (attention as a recompute segment) — soft coordination, not
   a dependency; the off-mode of each feature must compose with the other.
4. **Configuration** in `csrc/farm_api.h` + `_ffi`:
   `ll_set_chunked_attention(ctx, mode, chunk_q)` — `off` (default) | `on` | `auto`.
   `auto` picks the chunk factor from n_ctx/n_head and a memory budget via a
   documented heuristic implementing the BLUEPRINT §10 risk-5 back-of-envelope.
   Reject mode/factor changes after the first opt-graph build (S1-02-style error;
   topology).
5. **Memory measurement (required artifact):** on a small multi-layer fixture model
   (S0-06), reproduce the ROADMAP §8 cliff-table *shape* at n_ctx 512/1024/2048/4096:
   report peak compute-buffer allocation for naive vs chunked at ≥2 chunk factors,
   as a table in `docs/dev/` plus the S1-17-style step-result stats. Expected: naive
   grows ~quadratically with n_ctx; chunked cuts the attention term by ~the chunk
   factor.
6. **Correctness tests** `tests/test_chunked_attention.py` (self-verification
   discipline, BLUEPRINT §6.3): on fixture models, N identical steps chunked vs naive
   from identical initial state — losses, gradient accumulators, and post-step A/B
   tensors match within a documented tight tolerance (chunking reorders reductions,
   so bitwise equality is not guaranteed; state the bound and why). Matrix: softcap
   config (S1-19 path), ALiBi config with `max_bias > 0` (S1-20 path), plain causal;
   mode-change-after-build rejected; `auto` picks a sane factor; composition with
   S1-17 checkpointing on/off (gate the case if S1-17 is unlanded when this merges).

## Out of scope

- FA kernels and their ABI (FA1-FA7) — S1-21/S1-22/S1-23 and stage-2/3/4 tickets; FA8
  stays the fallback until those land and remains useful on backends whose FA backward
  hasn't shipped.
- Vendored llama.cpp changes — none; the needed VJPs land in S1-19/S1-20.
- GPU-resident measurement — each backend stage reruns the memory table once its
  `OUT_PROD` port lands (FA8 needs OUT_PROD per backend; CPU already has it).
- Autotuned/dynamic chunk factors beyond the documented heuristic, and
  chunked-attention perf tuning — `tickets/backlog/` once profiles exist.
- MLA-style attention subgraph variants — stock path with a clear preflight note.

## Acceptance criteria

- [ ] `pytest tests/test_chunked_attention.py` passes on the Linux CPU VM:
      chunked-vs-naive parity within the documented tolerance over N ≥ 3 steps,
      including the softcap and ALiBi (`max_bias > 0`) configurations.
- [ ] The memory table exists in `docs/dev/` with peak-alloc numbers for naive vs ≥2
      chunk factors at n_ctx 512-4096 on the fixture model, reproduced by CI output,
      and chunked peak at n_ctx 4096 is lower than naive by at least 2× for the
      measured config.
- [ ] Chunking works through the unchanged S1-02 `ll_train_step` ABI; existing S1-02
      tests stay green with chunking on.
- [ ] Mode/factor change after first build raises the documented error (tested);
      `off` mode is byte-identical to today's path.
- [ ] `_ffi` symbol-table test resolves `ll_set_chunked_attention`.
- [ ] `ci-cpu / test` per-PR green; the n_ctx-sweep memory run lands in nightly
      `ci-cpu` if it exceeds the per-PR budget.

## Testing & verification

`tests/test_chunked_attention.py` (new), pytest on the S0-06 fixture models, per-PR in
`ci-cpu / test` (S0-07); the n_ctx 512-4096 memory sweep in nightly `ci-cpu` if
needed. No new ops or kernels, so no `test-backend-ops` MODE_GRAD cases here — the
constituent VJPs are MODE_GRAD-verified in their own tickets (S1-19 TANH, S1-20
SOFT_MAX_BACK max_bias>0); this ticket's correctness evidence is the
chunked-vs-naive equivalence above. GPU stages re-verify parity under the ADR-0002
0.05 fp16 criterion when they adopt the fallback.

## PR notes

- Branch: `ticket/S1-24-chunked-attention-backward-fallback`.
- Single learning-llamas PR (shim + `_ffi` + tests + docs table); no fork PR — no vendored
  changes. Requires a vendor commit containing S1-19/S1-20 (frontmatter deps).
- Upstreaming disposition: **fork-local** — shim graph construction, not llama.cpp
  code; nothing to upstream.
- This is the standing long-context fallback while FA5/FA6/FA7 are in flight
  (ROADMAP §12 risk 12) — keep the builder isolated so it can be retired per-backend
  as FA backward lands.

## Accepted deviation (S1-50): chunked softcap/ALiBi branches are covered by argument, not a fixture

- **What.** The chunked path (`llama-graph.cpp`) emits a softcap branch (`scale`/`tanh`/`scale`, under
  `hparams.attn_soft_cap`) and an ALiBi branch (`soft_max_ext` with `hparams.f_max_alibi_bias`). No
  fixture toggles either, so the parity matrix's softcap and ALiBi rows are unmet for the *composed*
  chunked path; only the no-softcap/no-ALiBi row is fixture-tested.
- **Why implementing from this box is disproportionate.** Neither branch can be turned on for a
  `llama`-arch fixture. `attn_soft_cap` is set in `gemma2.cpp` code, not from a GGUF key; and the
  `llama` loader never reads `%s.attention.max_alibi_bias` into `f_max_alibi_bias` (verified: no
  `get_key(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, ...)` in `llama-model.cpp`), so it stays 0 on a `llama`
  GGUF regardless of the key. Exercising the composed branches live would require authoring a
  gemma2 or ALiBi-arch *training* fixture — a different architecture with its own tensor set and
  loader path — to cover two branches whose constituent ops are already independently grad-checked:
  `test_softcap` (scale/tanh/scale), `test_soft_max` (`max_bias` in `{0, 8}`), and `test_flash_attn_ext`
  (`logit_softcap` in `{0, 10}` x `max_bias` in `{0, 8}`) in the vendored `test-backend-ops`. The chunked
  path is the same `ggml_scale`/`ggml_tanh`/`ggml_soft_max_ext` primitives as the non-chunked path,
  which uses the identical softcap + `soft_max_ext(f_max_alibi_bias)` calls.
- **What would change the decision.** A committed gemma2 or ALiBi *training* fixture arriving for
  another ticket (at which point the chunked composition can be toggled on it directly), or the
  `llama` loader gaining a GGUF-key toggle for these hparams so a `llama` fixture could set them.
