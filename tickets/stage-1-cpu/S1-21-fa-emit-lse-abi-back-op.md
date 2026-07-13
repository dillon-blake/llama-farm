---
id: S1-21
title: "FA1: emit_lse ABI on FLASH_ATTN_EXT + ggml_flash_attn_ext_back op + CPU forward LSE"
stage: 1
track: kernels
size: M
deps: ["S1-00", "S0-09"]
status: deferred
pr: null
---

# S1-21 — FA1: emit_lse ABI on FLASH_ATTN_EXT + ggml_flash_attn_ext_back op + CPU forward LSE

> **⏸️ DEFERRED OUT OF STAGE 1 — project decision, 2026-07.**
>
> This ticket does **not** change how learning-llamas trains. Flash attention is force-disabled for
> training (`llama_context::set_training` logs *"disabling flash attention for training (no backward
> pass)"*), and **none of S1-21 / S1-22 / S1-23 turns it back on** — all three place that explicitly
> out of scope. Completing the whole family, ~6-8 weeks, would leave the training path byte-for-byte
> identical.
>
> Its real product is a **CPU correctness oracle for GPU flash-attention backward kernels** (FA5/6/7,
> stages 2-4). That payoff is entirely deferred on a CPU-only target, so this family moves to the
> front of whichever stage first starts a GPU backend.
>
> The memory argument does not rescue it either: S1-17 (gradient checkpointing) already removed the
> `x n_layers` factor, and **S1-24** — re-scoped as forward Q-chunking over S1-17's existing recompute
> — removes the `n_ctx^2` factor within a layer, using ops that already have backward rules. S1-24
> stays in stage 1; S1-21/22/23 do not.
>
> **A landmine for whoever picks this up.** The tiled FA forward path never updates `M[tq]` when a
> sink raises the running max (`ggml-cpu/ops.cpp`, the tiled sink fold — compare the one-chunk path,
> which does `M = s;`). Harmless today, because the forward normalizes by `S` and throws `M` away.
> **Silently wrong the moment anyone computes `lse = M + log(S)`** — which is exactly what S1-21
> exists to do. Fix it first, and test sinks x tiled, or the oracle ships wrong to every GPU backend.

**One-line outcome:** the FA-training ABI exists: `FLASH_ATTN_EXT` gains an `emit_lse` variant
producing a packed `O‖LSE` dst with accessor views, a new
`ggml_flash_attn_ext_back(q,k,v,mask,sinks,o,dO,lse,…) → dq‖dk‖dv` constructor replaces the
aborted legacy API, and the CPU forward emits LSE on all three of its code paths.

## Why (context)

Without FA backward, training materializes 2-3 `[n_kv, n_q, n_head]` F32 tensors per layer,
live across all layers — 128-192 GiB at 4k context for an 8B model (ROADMAP §8 memory-cliff
table). FA backward replaces the n_ctx² term with a per-row log-sum-exp vector. FA1 is the ABI
piece of that program (ROADMAP §8): S1-22 (autograd wiring), S1-23 (CPU backward kernel — the
GPU oracle), and every later GPU FA ticket implement against the contract this ticket defines,
so the packed layout and the LSE definition fixed here are load-bearing across stages 2-4.

The forward already computes the LSE ingredients and throws them away (ROADMAP §8, "what
already exists"). On CPU, the per-row running sum `S` and max `M` are declared at
`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:8496-8497` and discarded at the `V /= S`
normalization. There are **three** CPU forward paths, each with its own epilogue that must
store LSE: the vec/one-chunk path (`ggml_compute_forward_flash_attn_ext_f16_one_chunk`,
`:8403`, normalization at `:8626-8628`), the tiled path (`:8641`, normalization at
`:8912-8913`), and the split-KV path whose partials (`[M, S, VKQ]` per chunk) are combined by
`ggml_flash_attn_ext_reduce_partials` (`:8931`, final `S_inv` at `:8993`); dispatch between
them lives in `ggml_compute_forward_flash_attn_ext_f16` (`:9001`). Attention sinks are folded
into `M`/`S` before normalization (`:8600-8616`), so `lse = M + log(S)` naturally includes the
sink denominator — no special-casing (ROADMAP §8).

The legacy backward API is dead but instructive: `ggml_flash_attn_back`'s constructor aborts
("TODO: adapt to ggml_flash_attn_ext() changes", `vendor/llama.cpp/ggml/src/ggml.c:5470`), yet
its packed output — `dq‖dk‖dv` as `GGML_PAD`-aligned regions of one 1D F32 tensor
(`:5501-5529`) — is the precedent for both packed layouts here. The replacement signature must
honor (mask, scale, max_bias/slope, softcap, sinks) because llama.cpp bakes causal/padding/
SWA/ALiBi into one additive mask (`fill_mask`,
`vendor/llama.cpp/src/llama-graph.cpp:406-453`): one signature covers all model variants.
Training graphs bypass the KV cache **once S1-00 lands** (not before — see that ticket; today
the causal training graph goes through the cache and backward-graph construction aborts). After
S1-00, K/V arrive as F32→F16 casts (`vendor/llama.cpp/src/llama-graph.cpp:2416-2422`), so
quantized-KV backward is out of scope by construction. Current constructor facts to build on: op_params floats
`{scale, max_bias, logit_softcap}` (`ggml.c:5414-5415`), precision at i32 slot 3
(`:5426-5443`), sinks in `src[4]` (`:5445-5459`).

## What to do

All code changes land in the vendored llama.cpp fork via the S0-02 two-repo flow (stage-0
vendor infrastructure assumed in place; the frontmatter dep is S0-09 for the ADR-0002 numerics
policy that fixes LSE/row-stat precision at F32).

1. **`emit_lse` forward API** in `vendor/llama.cpp/ggml/include/ggml.h` (next to `:2411`) and
   `ggml.c`: a constructor variant that builds a packed dst — one 1D F32 tensor with two
   `GGML_PAD`-aligned regions, O first (identical element layout to the plain
   `FLASH_ATTN_EXT` dst, so kernel store code changes are offset-only), then LSE
   (`n_head · n_q · n_batch` F32). Record `emit_lse` as an i32 op-param at slot 4 (slot 3 is
   taken by precision). Follow the `:5501-5529` offset pattern.
2. **Accessor views:** `ggml_flash_attn_ext_get_o(ctx, t)` and
   `ggml_flash_attn_ext_get_lse(ctx, t)` returning correctly-shaped views (byte offset +
   strides) into the packed dst, so graph code and tests never hand-compute offsets.
3. **Define LSE once, in a header comment, as the cross-backend contract:** per (q-position ×
   head) row, `lse = M + log(S)` in F32, sinks included (they are already folded into `M`/`S`
   at `ops.cpp:8600-8616`); for a fully-masked row (`S == 0`), `lse = -INFINITY`, matching the
   forward's zero-output convention (`S_inv = 0`, `:8627`). S1-23's backward and every GPU
   port must match this definition exactly — mismatched LSE-with-sinks definitions are a named
   risk (ROADMAP §12 Q4).
4. **CPU forward LSE emission:** when `emit_lse` is set, store `lse` at each of the three
   epilogue sites (one-chunk `:8626-8628`, tiled `:8912-8913`, split-KV reduction `:8993`) and
   write O at its packed offset. Guard: identical numeric behavior for O with the flag on/off.
5. **New op constructor** `ggml_flash_attn_ext_back(ctx, q, k, v, mask, sinks, o, dO, lse,
   scale, max_bias, logit_softcap) → packed dq‖dk‖dv` (F32, legacy-pattern offsets): shape
   asserts mirroring the forward's, op_params as in the forward, mask/sinks optional (NULL).
   Repurpose the existing `GGML_OP_FLASH_ATTN_BACK` enum value
   (`vendor/llama.cpp/ggml/include/ggml.h:561`) — the legacy op is unreachable (constructor
   aborts) and reusing the slot avoids a mid-table enum insertion; delete the legacy
   `ggml_flash_attn_back` constructor (`ggml.c:5463-5530`) and its `ggml.h:2432-2433`
   declaration. Do **not** delete the legacy CPU kernel
   (`ggml_compute_forward_flash_attn_back_f32`, `ops.cpp:9156-9488`) — S1-23 modernizes it in
   place; instead guard its dispatch entry with a clear abort message referencing S1-23.
6. **supports_op:** the back op returns false on every backend (execution arrives with S1-23);
   the `emit_lse` forward returns true on CPU only, false elsewhere for now (sched falls back
   to CPU, ROADMAP §11 scheduler note). Add accessor/packing dq‖dk‖dv view helpers for the
   back op's output at the same time (S1-22 needs them to route src grads).
7. **Tests in the fork:** (a) accessor unit tests — offsets, strides, round-trip through the
   views; (b) forward-parity: O from the `emit_lse` variant matches plain `FLASH_ATTN_EXT` on
   identical inputs, with shapes chosen to hit all three CPU paths (split-KV requires
   `n_q == 1` and `n_kv ≥ 512`); (c) LSE-reference: LSE matches a naive
   double-precision logsumexp of `scale·QᵀK + slope·mask` (softcap applied when set) for the
   matrix: mask on/off, `max_bias > 0`, softcap > 0, sinks present, fully-masked row (−INF),
   F16 and F32 K/V; (d) a shape-only test that `ggml_flash_attn_ext_back` constructs a
   correctly-sized node. MODE_GRAD is **not applicable** to this ticket — no backward
   computation exists until S1-22/S1-23; grad acceptance lands there.
8. **Submodule bump PR** in learning-llamas referencing this ticket, per S0-02.

## Out of scope

- Autograd wiring (`FLASH_ATTN_EXT` case in `ggml_compute_backward`) — S1-22.
- The CPU backward kernel (modernizing `ops.cpp:9156-9488`) — S1-23, which is also where
  MODE_GRAD coverage for the FA path lands.
- GPU forward-LSE emission and GPU backward kernels — stage-2/3/4 FA tickets (ROADMAP §8
  FA4-FA7).
- Quantized-KV backward — excluded by construction (training graphs cast K/V; ROADMAP §8).
- The graph-level chunked-attention fallback — S1-24 (independent of this ABI).

## Acceptance criteria

- [ ] Fork branch: accessor unit tests and the `ggml_flash_attn_ext_back` shape test pass.
- [ ] Fork branch: O-parity tests pass on CPU (emit_lse vs plain) for shapes exercising all
      three CPU forward paths; existing `FLASH_ATTN_EXT` tests remain green (flag-off behavior
      unchanged).
- [ ] Fork branch: LSE-reference tests pass on CPU within the ADR-0002 forward tolerance for
      the full variant matrix (mask/ALiBi/softcap/sinks/masked-row/F16-KV), including the
      documented `-INFINITY` convention for fully-masked rows.
- [ ] The LSE definition (with sinks, with the S==0 convention) is recorded as a comment block
      in `ggml.h` next to the new API — the stage-2/3/4 contract text.
- [ ] Legacy `ggml_flash_attn_back` constructor and declaration are gone; the legacy CPU
      kernel body remains, guarded, for S1-23 (grep-verifiable in the fork diff).
- [ ] learning-llamas submodule-bump PR is green in `ci-cpu` (per-PR).

## Testing & verification

Primary harness: new test cases in the vendored `tests/test-backend-ops` (forward-eval mode)
plus small dedicated unit tests in the fork's test tree for the packed-view accessors; run on
the fork branch CI and in learning-llamas's `ci-cpu` lane per-PR after the submodule bump; nightly
`ci-cpu` re-runs the full suite. MODE_GRAD acceptance for FA is explicitly deferred to
S1-22/S1-23 (this ticket ships no backward computation); when S1-23 lands, its MODE_GRAD cases
run against the LSE this ticket emits, which is why the reference tests here compare against
an independent naive softmax implementation rather than the FA kernel itself.

## PR notes

- Branch: `ticket/S1-21-fa-emit-lse-abi-back-op`.
- Two-repo flow per S0-02: fork PR (`learning-llamas-base`) + trivial learning-llamas submodule-bump PR,
  both referencing the ticket ID.
- Upstreaming disposition: **upstream-later** (ROADMAP §11 triage class b) — the `emit_lse`
  ABI change and the repurposed back-op are fork-local first; propose upstream as one RFC for
  the FA-training op family once the CPU oracle (S1-23) plus one GPU backend prove the design.
  Fallback if upstream rejects the ABI: keep `emit_lse` behind a fork-local op flag and carry
  a small rebase patch (ROADMAP §11).
- No copied external code; the packed-layout pattern is in-tree (`ggml.c:5501-5529`), covered
  by fork history.
