# ADR-0003 — Sparse cross-entropy: the cross-backend ABI (gate G-A)

- **Status:** Accepted
- **Date:** 2026-07-13
- **Ticket:** S1-04
- **Decides:** ROADMAP gate **G-A**
- **Binds:** every backend CE port — S2-07 (Metal), S3-03 (CUDA), S4-04 (Vulkan)

## Context

`ggml_cross_entropy_loss` takes a **dense** label matrix the same shape as the logits, and
mean-reduces over all rows. Both properties are wrong for training a language model.

The dense labels are the obvious waste: an `[n_vocab, n_tokens]` F32 matrix that is zero
everywhere except one entry per row. At a 128k vocab and 2048 tokens that is **1 GB of zeros**,
materialized, uploaded, and multiplied through — to express one integer per token.

The mean-over-all-rows is the subtle one, and it is worse. It gives no way to *mask* a token, and
a masked token is not a nicety: an instruction-tuned model must not take loss on its prompt. S1-02
worked around this by abusing the dense op — putting a weight `wᵢ` where the one-hot 1 would go,
and pre-scaling by `n_rows/Σw` on the host so the built-in `1/nr` cancels. It works, and it is a
stopgap. It still materializes the 1 GB of zeros.

So learning-llamas adds `ggml_cross_entropy_loss_sparse`. Because every backend must implement the
same op, its ABI has to be fixed **before** any kernel is written — that is decide-first gate G-A
(ROADMAP §11), and this ADR is it.

## Decision 1 — The op signature

```c
// logits:  F32 [n_vocab, n_tokens]
// labels:  I32 [n_tokens]        -- the target token id per position
// weights: F32 [n_tokens]        -- the per-token loss weight; 0 masks the token out
// returns: F32 [n_tokens]        -- the per-token loss. NOT reduced.
ggml_tensor * ggml_cross_entropy_loss_sparse(
        ggml_context * ctx,
        ggml_tensor  * logits,
        ggml_tensor  * labels,
        ggml_tensor  * weights,
        float          logit_scale,   // 1.0 = off
        float          softcap);      // 0.0 = off
```

**No reduction.** The op returns a per-token vector, and the caller reduces it — with
`GGML_OPT_LOSS_TYPE_SUM`, or by a `ggml_sum` / weighted mean of its own choosing. Baking a
reduction into the op is what made the dense version unusable: a mean over *all* rows is simply
the wrong statistic when some rows are masked, and no op-param can retrofit that. Returning the
vector costs `n_tokens` floats and makes every reduction expressible.

**`logit_scale` and `softcap` are op-params from day one.** They are nearly free now
(two multiplies and a `tanh` in a kernel that is already memory-bound) and painful to retrofit
into four backends later. Gemma-2/3 need the softcap; several architectures scale logits.

The mathematics, in order:

```
u = x * logit_scale
z = (softcap > 0) ? softcap * tanh(u / softcap) : u
lse_i = logsumexp_j(z_ij)
loss_i = w_i * (lse_i - z_i[label_i])
```

## Decision 2 — Backward recomputes the LSE; it is not stashed

The forward could stash its per-row `lse` (an extra `n_tokens` F32 output) so the backward does
not have to re-reduce the vocab row. **It does not. The backward recomputes it.**

Two reasons, and the first is decisive:

**ggml has no clean multi-output op.** A ggml node has one `dst`. Stashing the LSE means either
widening `dst` to `[2, n_tokens]` — which breaks the "per-token loss" contract this ADR just
fixed, and would make `GGML_OPT_LOSS_TYPE_SUM` sum the LSEs into the loss — or introducing a
side-channel output tensor, which the graph allocator has no way to reason about. Both are worse
than the cost they avoid.

**The cost they avoid is smaller than it looks.** The backward must write `dlogits`, which is
`[n_vocab, n_tokens]` — it is *already* streaming the whole logits row, and already writing a row
of the same size. Recomputing the LSE adds one more read pass over a row that the write pass will
touch anyway. It is one extra pass over data that is in cache, not an extra materialization.

**Consequence for the ports:** every backend's `_BACK` kernel recomputes `lse` from `logits`. No
backend may assume a stashed LSE is available, and none may add one unilaterally — that would
fork the ABI. If profiling later shows the recompute dominating, the stash is
[`tickets/backlog/B-04`](../../tickets/backlog/B-04-saved-lse-ce-rmsnorm-invvar.md), and it
amends this ADR.

## Decision 3 — The backward does not alias the logits buffer

unsloth writes `dlogits` **in place** over `logits`, saving an `[n_vocab, n_tokens]` allocation.
learning-llamas does not.

In ggml, aliasing means the `_BACK` node's result is a *view* of its source — and a view with a
non-whitelisted op is exactly what `ggml_build_backward_expand` refuses ("inplace operations are
currently not supported", `ggml.c:7093`). S0-10 and S1-19 both hit that assert from different
directions; it is not a rule to be clever around.

It also buys less than it appears to. `ggml_gallocr` already reuses buffers whose lifetimes have
ended: once the forward loss is computed, the logits buffer *is* a reuse candidate for `dlogits`,
and the allocator can make that choice itself without an alias that the autodiff pass has to be
taught to tolerate.

**Consequence for the ports:** every backend's `_BACK` kernel writes to a distinct `dst`. No
backend implements an in-place variant.

## Decision 4 — A masked row's gradient is written as bitwise zero

For any row with `wᵢ == 0`, the backward writes **exactly `0.0f`** to every element of that row —
it does not compute `(p - onehot) · 0` and rely on the multiply.

This is a correctness requirement, not an optimization. `p - onehot` can contain values that are
`inf` or `NaN` when the logits are extreme (a saturated `exp`, an all-`-inf` masked row), and
`0 * NaN` is `NaN`, not `0`. A single masked row with a degenerate logit would poison the whole
gradient. The dense op has this bug latent; the sparse op must not inherit it.

MODE_GRAD's tolerance cannot prove this — an exact-zero assertion can, and S1-04 ships one.

## Consequences

- **S2-07, S3-03, S4-04 implement this ABI exactly.** Same signature, same op-params, same
  math-in-this-order, backward recomputes LSE, no aliasing, bitwise-zero masked rows. A port that
  deviates is not a port.
- The **dense** `CROSS_ENTROPY_LOSS` op needs no Metal or Vulkan port (ROADMAP §3): once sparse is
  canonical, nothing in learning-llamas calls the dense one.
- S1-02's weighted-dense-label stopgap is retired by S1-05, which switches the trainer to this op.
- The op enums go at the **tail** of ggml's op table (ROADMAP §11): inserting in the middle
  renumbers every op after it and turns every rebase into a conflict across the whole backend
  matrix.

## Out of scope

- The chunked-LSE decomposition for very wide vocabularies. On CPU a two-pass (max, then
  sum-exp) over a row is stable and simple, and the row is in cache. Chunking exists to fit a GPU
  threadgroup's shared memory, so it is a *port* concern and each backend ticket chooses its own —
  it does not change this ABI, because the value computed is identical.
- Upstreaming the op. Disposition is **in-fork-first** (ROADMAP §11): the shape is settled here,
  but a new ggml op with four backend implementations is not a first upstream PR.
