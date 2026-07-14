# Chunked attention (S1-24) — the kernel-free long-context path

The attention matrix is `[n_kv, n_q, n_head]`. It is **quadratic in context**, and it is the reason
long-context training runs out of memory. Chunked attention splits the **query** axis and softmaxes
each chunk on its own, so only `[n_kv, chunk_q, n_head]` is live at a time.

```
ll_set_chunked_attention(ctx, chunk_q)   # 0 = off (the default)
ll_set_grad_checkpointing(ctx, 1)        # you want both — see below
```

## Why chunking queries is exact, and chunking keys is not

Softmax is row-wise over the **key** axis. An output row depends on its own row of scores and
nothing else — so partitioning the **query** axis is *exact*, not an approximation. The losses come
out **bit-identical** to the naive path; the gradients differ only by float32 reduction order
(~1e-6).

Chunking the **key** axis is a different problem: a partial softmax has to be rescaled when a later
chunk raises the running maximum. That is what flash attention does, and it is why FA needs a
kernel. This needs none — every op here (`MUL_MAT`, `SOFT_MAX`, `CONCAT`, `OUT_PROD`) already exists
on every backend. It is the standing fallback until FA5/FA7's backward kernels land, and it stays
useful on any backend whose FA backward has not shipped.

## ⚠️ Chunking alone does not shrink the backward. Pair it with S1-17.

This is the part that is easy to get wrong, and it is easy to get wrong *silently* — the feature
appears to work, computes the right answer, and saves nothing.

**`SOFT_MAX_BACK` reads the softmax's own output.** So without recompute, every chunk's `P` stays
live from the forward until the backward consumes it. All the memory is still there; it is just
spread across more tensors. Under `ggml_build_backward_expand_checkpointed` (S1-17) each `P` becomes
a **segment-interior** node: rebuilt immediately ahead of the backward node that reads it, and dead
again straight after. Only one is live at a time.

**The two features are multiplicative, not alternatives.** S1-17 removes the *depth* factor
(`n_layer` attention matrices live at once); S1-24 removes the `n_ctx²` factor *within* a layer.
`tests/test_chunked_attention.py::test_chunking_needs_checkpointing_to_shrink_the_backward` pins
this, so a refactor cannot quietly break the pairing and leave the feature dead but green.

## The measured cliff

Tiny fixture (`n_layer=2, n_embd=256, n_head=4, n_vocab=512`), CPU, F32 base, LoRA r=4, one training
step, peak compute-buffer allocation. `chunk_q = n_ctx/8`; `ckpt` is `segment_len=1`.

| n_ctx | naive | chunk only | ckpt only | **ckpt + chunk** | vs naive |
|------:|------:|-----------:|----------:|-----------------:|---------:|
| 512   | 25.0 MiB | 25.0 MiB | 17.0 MiB | **17.0 MiB** | 1.47× |
| 1024  | 79.0 MiB | 66.0 MiB | 47.0 MiB | **44.0 MiB** | 1.80× |
| 2048  | 254.1 MiB | 196.1 MiB | 166.0 MiB | **128.0 MiB** | 1.98× |
| 4096  | 904.1 MiB | 652.1 MiB | 620.1 MiB | **416.1 MiB** | **2.17×** |

The shape is the point. Naive grows quadratically; the win grows with it, because the term being
removed is the quadratic one. At 512 chunking buys nothing at all — the attention matrix is not yet
the dominant term — and that is not a defect, it is the honest scale of the effect.

Note the **`chunk only`** column: it tracks naive far more closely than `ckpt + chunk` does. That
column *is* the warning in the section above, measured.

### Choosing `chunk_q`: the knee is around 8 chunks

At n_ctx 2048 with `ckpt(1)`:

| chunk_q | 1024 | 512 | 256 | 128 | 64 | 32 |
|---|---|---|---|---|---|---|
| peak | 136.0 | 128.6 | 128.0 | 127.1 | 130.0 | 130.3 MiB |

It **saturates** at ~4-8 chunks and then flattens. Past the knee the attention term is no longer the
peak, so a smaller chunk buys nothing and costs more graph nodes. **`n_ctx/8` is a sound default.**

> Earlier this got *worse* past the knee — 151.5 MiB at `chunk_q=64` — because the per-chunk outputs
> were reassembled with a **left fold** (`acc = concat(acc, next)`), whose accumulators grow
> `1/C, 2/C, … C/C` of the output and hand the saved memory straight back. It is now a **balanced
> concat tree**: `log2(C)` levels, only ~2 ever live. If you touch that loop, the left fold is the
> mistake waiting for you.

## Cost

Roughly **+50% attention FLOPs** — the `S`/`P` recompute — plus the concat tree. Attention is not
the whole model, so the end-to-end step cost is well under that. You are buying context length with
arithmetic, which is the trade the whole feature exists to make.

## Not covered

**The softcap (gemma2) and ALiBi (`max_bias > 0`) branches are written but have no fixture.**

`hparams.attn_soft_cap` is set *in code* by `models/gemma2.cpp` — it is **not** a GGUF key — so it
cannot be switched on for a llama-arch fixture, and covering it means building a gemma2 fixture.
Both operations are chunk-invariant by construction (ALiBi's slope is a function of the **head**
index; softcap is elementwise on the scores — neither mixes query rows), and both mirror the naive
path line for line.

That is an argument, not a test, and it is recorded as one. A gemma2 fixture is the follow-up.

## Constraints

* **Fixed for the lifetime of a run.** Chunking changes the graph's node *count*, and ggml-opt keys
  its gradient accumulators and AdamW momenta by **node index** (BLUEPRINT D1). Changing the factor
  mid-run would land one parameter's momentum on another, and it would not crash.
  `ll_set_chunked_attention` returns `LL_ERR_ALREADY_INIT` after the first step.
* **Training only.** Inference keeps no attention matrix alive, so there is nothing to win and the
  concat would be pure cost.
* **Not the flash-attention path.** FA is force-disabled during training anyway
  (`llama-context.cpp`, `set_training`) because `ggml_flash_attn_back` is a stub whose first
  statement is a `GGML_ABORT`.

## The trap, for whoever touches `build_attn_mha` next

Slice the token axis **before** the permute, not after.

`build_attn_mha` permutes `q` to `[head_dim, n_tokens, n_head, n_stream]`. Slicing *that* means
taking a view of a **non-contiguous** tensor, and ggml's autodiff does not survive it: the gradient
comes back through `PERMUTE` as a non-contiguous view and dies in `ggml_scale`'s
`GGML_ASSERT(ggml_is_padded_1d(a))` inside the LoRA scale's backward — an abort three ops away from
the actual mistake, naming a function you did not call.

So the chunked path keeps `q_bhd` (`[head_dim, n_head, n_tokens, n_stream]`, token axis = `ne[2]`,
rows contiguous), slices there, and permutes each slice. The backward chain is then the same shape
the naive path already proves works.
