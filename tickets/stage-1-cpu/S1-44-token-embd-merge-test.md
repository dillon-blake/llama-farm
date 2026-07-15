---
id: S1-44
title: "token_embd flipped-convention merge + Q8_0 output-type preservation tests"
stage: 1
track: python
size: S
deps: [S1-08]
status: done
pr: null
---

# S1-44 — token_embd flipped-convention merge test

**One-line outcome:** the merge path that folds a LoRA delta into `token_embd.weight` — the one
target that uses the flipped, A-transposed convention — is pinned by a numpy oracle and a
runs-and-agrees logits check; and a Q8_0 base is proven to survive a merge without falling back to
F16.

## Why (context)

The Stage 0+1 audit (2026-07-15, harness-adapter section) rated this a **major test-gap**:
`export._delta()` has a distinct `token_embd` branch — `a @ b.T` instead of `b @ a`
(`src/learning_llamas/export.py:104-111`) — and `merge()` derives its rank differently there
(`rank = a.shape[1]`, not `a.shape[0]`, `export.py:172`). This is the loader's flipped convention:
`token_embd.weight` is validated as `model.ne[0] == b.ne[1] && model.ne[1] == a.ne[1]`
(`vendor/llama.cpp/src/llama-adapter.cpp:356-368`) and applied A-transposed in `llm_build_inp_embd`
(`vendor/llama.cpp/src/llama-graph.cpp:2268-2273`). **Every** merge test in `test_export.py` builds
its adapter with `create_zero_adapter(...)` under the default preset, which sets
`include_token_embd=False` — so the flipped branch had *zero* coverage. A wrong transpose there
produces a merged model whose embeddings are garbage, and nothing would have said so.

Separately (confirmed **minor**): S1-08's acceptance criterion names Q8_0 explicitly — "merged
output tensors have the base's original quant types for at least Q4_K **and Q8_0** fixture bases
(not F16)" — but every merge test used `tiny_q4_k` only. "The merged model keeps Q8_0 rather than
falling back to F16" was asserted for Q4_K and nothing else.

## What to do

- `tests/test_export.py`: three tests, oracle discipline (numeric ground truth, not "loads and
  runs"), all on the existing S0-06 fixtures — no new fixtures, no HF downloads.
  - **token_embd numeric oracle** — a `token_embd`-only adapter with nonzero A and B, merged into
    the F32 base; the merged tensor is checked bit-tight against `base + (alpha/rank) * (A @ B.T)`,
    where `A @ B.T` is transcribed from the graph (`llama-graph.cpp:2268-2273`), **not** by calling
    `export._delta`. F32 base ⇒ dequantize/quantize are identities ⇒ strict equality.
  - **merged-model fidelity** — load the merged GGUF and base+adapter (`attach_adapter` on the
    unmerged base); their logits must agree to float32 round-off while both sit far from the bare
    base. The fixture is non-tied, so `output.weight` is untouched and the comparison is clean.
  - **Q8_0 output-type preservation** — merge a real (nonzero) delta into `tiny_q8_0` and assert
    every base tensor's quant type is reproduced in the output (Q8_0 stays Q8_0, F32 norms stay
    F32), with `FALLBACK_TYPE` (F16) appearing nowhere and the Q8_0 re-quantize path shown to have
    actually run.
- Mutation-proof the token_embd oracle in the assertion structure: the fixture is non-square
  (`n_vocab=512 != n_embd=256`), so a transposed delta does not fit the base tensor and `merge`
  raises; and the delta's square block is asserted strongly non-symmetric, so a same-shape
  transpose would miss the oracle by ~1e5x the tolerance.

## Out of scope

- Merging an adapter that targets both `token_embd` and `output` at once (output uses the *normal*
  convention — the loader's flip is `token_embd.weight`-only, `llama-adapter.cpp:356-368` — so it
  is already covered by the default-preset merge tests).
- Q4_K token_embd numeric equality (Q4_K adds a quantization round-trip, so the oracle would be
  measuring its own tolerance; the F32 fixture is the exact-oracle vehicle, and Q4_K type
  preservation is already pinned by `test_merging_at_scale_zero...`).

## Acceptance criteria

- [x] A `token_embd` merge into the F32 base reproduces `base + (alpha/rank)*(A @ B.T)` to within
      1e-6 (observed 0.0, bit-exact), with the oracle written from first principles rather than
      from `export._delta`.
- [x] The merged model's logits agree with base+adapter to float32 round-off (observed 8.2e-05,
      against a bare-base distance of ~0.89) and pick the same next token.
- [x] A Q8_0 base merge preserves every tensor's quant type — Q8_0 stays Q8_0, no F16 fallback —
      with the re-quantize path shown to be non-vacuously exercised.
- [x] The token_embd oracle demonstrably can fail: a transposed delta raises on shape, and the
      asserted square-block asymmetry (~0.15) is ≥ 1e4x the equality tolerance.
- [x] Full suite green; ruff clean.

## Testing & verification

Measured on this host (F32 base, r=4, alpha=8 ⇒ effective scale 2.0): merged token_embd vs the
`A @ B.T` oracle **0.0** (bit-exact); merged-model logits vs base+adapter **1.49e-07** at
alpha/rank=1, **8.2e-05** at alpha/rank=2 (the committed config), both against a ~0.89 adapter
effect; square-block asymmetry **0.149**. Mutation check (reverted): forcing the token_embd rank to
`a.shape[0]` makes the oracle test fail and the logits diverge by 1.15; transposing the delta makes
`merge` raise on shape. The convention was confirmed **correct** — no production bug found; these
tests pin it against regression.
