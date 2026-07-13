# Gradient checkpointing (S1-17)

Training runs out of memory on **activations**, not weights. LoRA on a quantized base is already
tiny in weights — a few megabytes of A/B tensors over a frozen 4-bit model. What actually fills the
machine is the forward pass's intermediates, because the backward pass reads them: every activation
stays live from the moment it is produced until the moment its gradient is taken, and for layer 0
that is the entire graph.

So the footprint grows with **depth × context**, and neither of those is something you wanted to
give up.

Checkpointing keeps only the layer boundaries and **recomputes** everything between them, on demand,
immediately before the backward node that reads it — and drops it again straight after. One extra
forward pass of arithmetic, in exchange for an activation footprint that stops growing with depth.

```python
libs.farm.ll_set_grad_checkpointing(model.ctx, segment_len)   # 0 = off (default)
```

`segment_len` is layers per segment. `1` checkpoints every layer boundary: the most memory saved,
the most recomputed.

## Measured

`python benches/grad_checkpointing.py --n-layer 12 --seq 512`, on an Intel N100. Tiny-llama fixture,
12 layers, n_embd 256, Q4_K base, LoRA rank 4, sequence length 512:

| segment | peak activations | vs off | step | gradients |
|--------:|-----------------:|-------:|-----:|:----------|
| off | 136.86 MB | 1.00× | 949 ms | — |
| 4 | 51.39 MB | 0.38× | 1287 ms | **identical** |
| 2 | 30.94 MB | 0.23× | 1337 ms | **identical** |
| 1 | **23.08 MB** | **0.17×** | 1303 ms | **identical** |

**~6× less activation memory for ~35% more time.** On a machine where the alternative is not
training at all, that is not a trade — it is the whole game.

Note the *shape* of it: most of the saving arrives at the first split (0.38× at segment 4), and the
rest comes slowly. Segment 4 recomputes a quarter as often as segment 1 and still gives back nearly
two thirds of the memory. If you are tuning this, start at 2 or 4 rather than 1.

## "identical" is a bitwise claim

Not "within tolerance" — **the same bits**.

Recomputing an activation means running the same ops on the same inputs on the same backend, so it
produces the same bytes, so the gradients are the same bytes, so the trained weights are the same
bytes. The CPU backend is deterministic per shape (ADR-0002), which is what makes this an equality
rather than a hope.

That matters more than it sounds. A tolerance would hide exactly the bug worth finding: a recompute
that does not reproduce the forward. `tests/test_grad_checkpointing.py` asserts bitwise equality of
both the losses and the trained A/B tensors, at every segment length.

And it asserts the memory falls — separately, because **bitwise equality would also hold for an
implementation that did nothing at all.** That is the one bug that matters here and it is invisible
to every correctness test. It gets its own assertion, and a fixture deep enough (12 layers) for the
assertion to have teeth; at the 2-layer default fixture there is nothing between the boundaries to
recompute, and a do-nothing implementation would pass.

## How it works

`ggml_build_backward_expand_checkpointed` (ggml.c, this fork). ggml had this once and lost it in the
2024 backward refactor; this restores it against the current API.

The gradient *rules* are not touched — `ggml_build_backward_expand` already computes them correctly.
All that changes is **which tensor each backward node reads**: the original activation, or a
recomputed stand-in holding the same numbers. The rewrite:

1. build the ordinary backward into a scratch graph;
2. copy the forward into the output graph, **in the same order** — ggml-opt indexes its gradient
   accumulators and AdamW momenta by forward node index, so a moved forward prefix would apply one
   parameter's momentum to another;
3. append each backward node with its reads redirected to stand-ins, pulling each stand-in's
   recompute chain in immediately ahead of it. That ordering is what confines a segment's interior
   to that segment's backward, and it is the whole trick — `ggml_gallocr`'s liveness analysis does
   the rest for free.

Boundaries come from the fork's `t_layer_inp` record, falling back to the `"l_out-<il>"` naming
convention that ~108 model builders follow. An architecture that exposes neither gets a warning and
runs without checkpointing, rather than silently paying for nothing.

### The view-op trap

A view op (`RESHAPE`, `VIEW`, `PERMUTE`, …) computes *nothing*: its kernel is a literal no-op and
the data is read through `view_src`. A recomputed view with a null `view_src` therefore hands the
backward pass a buffer of uninitialized memory — silently, only under checkpointing, and only
sometimes. The clone points its view at the **recomputed** base, which is the entire point.

This is not hypothetical: removing that handling makes the bitwise test fail immediately, which is
why the test is bitwise.

## What it does not fix

The `n_ctx²` attention term inside each layer. That is flash-attention's backward (S1-21–S1-24), and
the two are complementary: FA kills the per-layer quadratic, checkpointing kills the
across-all-layers residency. Each matters without the other.

## A bug this found

`llama_context::graph_max_nodes()` sizes the compute graph for a **forward** pass
(`max(1024, 8 × n_tensors)`). A training graph is the forward *and* its backward, and the backward is
the bigger half. The fixture is two layers deep, so nobody had noticed — but at twelve layers,
forward + backward overflows the budget and ggml aborts inside `ggml_build_forward_expand` on
`n_nodes < size`, saying nothing about sizing.

Training graphs now get 4× the nodes, and `set_training()` re-sizes the graph buffers and the
scheduler when it flips the context into training mode (they were built for an inference context).
This is fixed independently of checkpointing: **it broke deep-model training with checkpointing
off**, which is to say it broke deep-model training.
