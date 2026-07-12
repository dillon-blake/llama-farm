---
id: S1-00
title: "Training attention path: bypass the KV cache so gradients reach K/V (unblocks all backward)"
stage: 1
track: kernels
size: M
deps: ["S0-02", "S0-03"]
status: pr-open
pr: https://github.com/dillon-blake/llama-farm/pull/11
---

# S1-00 — Training attention path: bypass the KV cache so gradients reach K/V

**One-line outcome:** the training graph routes attention through `k_cur`/`v_cur` directly
instead of through the KV cache, so `ggml_build_backward_expand` stops aborting and gradients
actually reach the K/V projections — establishing the "training graphs have no KV cache"
property that S1-07, S1-21, S1-22, S1-23 and S3-06 already assume as fact.

## Why (context)

**This ticket exists because the plan assumed its outcome was already true.** Five tickets state
"training graphs have no KV cache" as established background and build on it. It is not true at
the pinned commit `4f37f51`. Nothing in the plan schedules the work that makes it true, and
S1-03 (P0 proof-of-gradient) is the first ticket that would execute this code — it would abort
before printing a single loss value.

### The defect

`build_attn` (KV variant, `vendor/llama.cpp/src/llama-graph.cpp:2629`) stores K/V into the cache
and then **re-reads them from the cache** (`:2667-2677`):

```c
ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));  // ggml_set_rows
ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));

ggml_tensor * q = q_cur;                        // q flows in directly
ggml_tensor * k = mctx_cur->get_k(ctx0, il);    // a VIEW of the cache leaf — not k_cur
ggml_tensor * v = mctx_cur->get_v(ctx0, il);    // likewise
ggml_tensor * cur = build_attn_mha(q, k, v, ...);
```

`cpy_k` is `ggml_set_rows` (`vendor/llama.cpp/src/llama-kv-cache.cpp:1329`); `get_k` returns
`ggml_view_4d` of the cache leaf (`:1243`). The forward pass works by **memory aliasing** — but
the autodiff graph has **no edge from `k_cur` to attention**. Q's gradient path is intact; K's
and V's are severed.

It does not fail quietly. `ggml_set_rows` returns `ggml_view_tensor(ctx, a)` with
`op = GGML_OP_SET_ROWS` (`vendor/llama.cpp/ggml/src/ggml.c:3917`), and
`ggml_build_backward_expand` refuses view-ops it cannot differentiate
(`vendor/llama.cpp/ggml/src/ggml.c:7093`):

```c
// inplace operations are currently not supported
GGML_ASSERT(!node->view_src || node->op == GGML_OP_CPY || node->op == GGML_OP_VIEW ||
    node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE);
```

So the moment any node upstream of `k_cur` needs a gradient, backward-graph construction
**hard-aborts**.

### It is unconditional — there is no target-selection workaround

`k_cur = build_lora_mm(wk, cur)`. Even with **no** adapter on `wk`/`wv`, `cur` (the layer input)
requires a gradient as soon as any adapter exists in any earlier layer — so `k_cur` requires a
gradient, and the assert fires. Every multi-layer LoRA configuration hits this. Restricting
LoRA targets to `{q, o, ffn}` does not avoid it.

### Reproduced against upstream

llama.cpp's own `llama-finetune`, unmodified, on a valid tiny llama-arch GGUF at `4f37f51`:

```
main: force changing k cache type to f32 due to a lack of f16 support for OUT_PROD
-optimizer adamw -lr0 1e-05 -wd 0 -lr-min -1 -min-epochs -1 -epochs 1 -period 1 -val 0.05
ggml.c:7093: GGML_ASSERT(!node->view_src || node->op == GGML_OP_CPY || ...) failed
[abort]
```

Upstream full-parameter finetuning is therefore **broken at this commit**, and never reaches a
loss print. `examples/training/` has **no CI coverage**, which is how it went unnoticed.

### Do not "fix" this by adding a SET_ROWS backward

`GGML_OP_SET_ROWS` has zero cases in `ggml_compute_backward` (`ggml.c:6430-6910`). Lifting the
view assert without a VJP would abort at the op-level default (`:6906`); adding neither would hit
the `if (!grad) return;` early-out (`:6435`) and **silently produce zero gradients for every K/V
projection** — a trainer whose loss still falls (Q/O/FFN train) while K/V never move. The correct
fix is to keep the cache out of the training graph entirely. Training has the whole sequence; it
does not need a KV cache.

### Where the premise came from

The five tickets cite `llm_graph_input_attn_no_cache::set_input`
(`vendor/llama.cpp/src/llama-graph.cpp:406-453`) as "the training path". That code is real and
does exactly what they describe — but it serves the **11 embedding/non-causal archs** that call
`build_attn_inp_no_cache()`. The **80 causal archs** you would actually train call
`build_attn_inp_kv()`, and no model builder branches on training to pick the other one. Real
code, wrong path.

## What to do

All changes land in the vendored llama.cpp fork (two-repo flow per S0-02).

1. **Add a training flag** to the graph context — `bool training` on `llm_graph_params` /
   `cparams`, defaulted **off** so inference is bit-identical.
2. **Bypass the cache in `build_attn`** (KV variant, `llama-graph.cpp:2629`): when `training`,
   skip the `cpy_k`/`cpy_v` stores and the `get_k`/`get_v` reads, and pass `k_cur`/`v_cur`
   straight into `build_attn_mha` — exactly as the no-cache overload already does at
   `:2544-2580` (`k = k_cur; v = v_cur;`). This is the whole functional change.
3. **Fix the mask shape.** `llm_graph_input_attn_kv::set_input` (`:467`) builds a
   `[n_kv, n_tokens]` mask against padded cache slots; the uncached path builds
   `[n_tokens, n_tokens]` via `llm_graph_input_attn_no_cache::set_input` (`:406-453`). In
   training mode `build_attn_inp_kv()` must emit the uncached-shaped mask (`n_kv = n_tokens`)
   while preserving causal, SWA and **seq_id isolation** semantics — reuse `fill_mask`, do not
   fork it. The seq_id isolation property is what S1-07's packing collator depends on.
4. **Do not touch `src/models/*.cpp`.** The change stays inside `llm_graph_context` so all 80
   archs inherit it unchanged (preserves BLUEPRINT D5, "inherit by construction"). This is a
   hard acceptance criterion, not a preference.
5. **Scope the other `build_attn` overloads.** There are six (`:2544, 2629, 2733, 2791, 2867,
   2970` — no-cache, kv, k, iswa, kv-iswa, cross). v1 covers the plain `kv` path; the iSWA,
   cross-attention and DSA variants are either given the same bypass or are **rejected at
   preflight with an actionable error** (S1-11 owns the rejection). State which, in the PR.
6. **Upstream it.** This is an upstream bug — `llama-finetune` is broken for everyone. File the
   issue with the repro, and the PR against upstream, independently of the fork carry.

## Out of scope

- A `SET_ROWS` VJP — explicitly rejected above; the cache stays out of the training graph.
- Flash-attention forward/backward — S1-21, S1-22, S1-23 (they depend on this ticket's outcome).
- The chunked-attention fallback — S1-24 (depends on the training mask shape settled here).
- Preflight rejection of unsupported attention variants — S1-11.
- CI coverage for the upstream `examples/training/` target — note it in the upstream issue; not
  this project's ticket.

## Acceptance criteria

- [ ] **Before-fix repro recorded in the PR:** unmodified `llama-finetune` on a tiny llama-arch
      GGUF aborts at `ggml.c:7093`.
- [ ] **After-fix:** the same command completes ≥1 epoch and prints a **falling loss**.
- [ ] **The actual bug is proven fixed:** a gradient test shows `dL/d(wk)` and `dL/d(wv)` are
      nonzero and finite-difference-correct. A loss that merely falls is **not** sufficient
      evidence — it falls even when K/V gradients are identically zero.
- [ ] **Zero diff in `src/models/*.cpp`** (grep-verifiable in the fork diff).
- [ ] **Inference is unchanged:** the training flag defaults off; `llama-cli` / perplexity output
      is byte-identical to the base commit and the existing KV-cache tests pass.
- [ ] **Mask semantics:** in training mode, two samples packed with distinct `seq_id`s cannot
      attend to each other (the property S1-07 relies on), verified by a test.
- [ ] The five tickets asserting "training graphs have no KV cache" (S1-07, S1-21, S1-22, S1-23,
      S3-06) are updated to cite S1-00 as what establishes it.
- [ ] Upstream issue filed with the repro; PR link recorded here.

## Testing & verification

Reproduction recipe (also the regression test): build `llama-finetune` from the vendored tree,
generate a tiny llama-arch F32 GGUF (2 layers, n_embd 64, n_head 4, tokenizer KVs copied from
`models/ggml-vocab-llama-bpe.gguf`), and run:

```
llama-finetune -m tiny-llama.gguf -f train.txt -c 64 -b 64 -ub 64 --epochs 1 -ngl 0
```

Pre-fix this aborts at `ggml.c:7093`; post-fix it trains. Add a gradient test in the fork
asserting nonzero, finite-difference-correct `dL/d(wk)` and `dL/d(wv)` — the zero-gradient
failure mode is silent and a loss curve will not catch it. Runs in `ci-cpu` per-PR. Because this
gates every backward pass in the project, it is worth a dedicated CI case that would fail loudly
if a future vendor bump reintroduces a cache write into the training graph.

## PR notes

- Branch: `ticket/S1-00-training-attn-bypass-kv-cache`.
- Two-repo flow per S0-02: implementation PR against the fork's `learning-llamas-base` branch, then a
  trivial learning-llamas PR bumping the `vendor/llama.cpp` gitlink.
- Upstreaming disposition: **upstream-early** — this is a straight bug fix to a broken upstream
  example, and carrying it fork-local indefinitely fights every rebase (ROADMAP §11 triage
  class a).
- This ticket is a **root of the stage-1 dependency graph**: S1-01 (and transitively the whole
  shim/trainer chain), S1-21 and S1-24 depend on it. Nothing that builds a backward graph can
  land before it.
