---
id: S1-05
title: "SFT trainer on sparse CE + masked validation eval"
stage: 1
track: python
size: M
deps: ["S1-03", "S1-04", "S1-06"]
status: open
pr: null
---

# S1-05 — SFT trainer on sparse CE + masked validation eval

**One-line outcome:** `llama_farm.train.sft` runs real SFT on `ce_sparse` with prompt
masking and host-side normalization by valid-token count, plus a masked validation loss
on a held-out split computed forward-only.

## Why (context)

S1-03 proved gradients flow end-to-end, but with the stopgap composite CE (softmax →
one-hot mul → sum_rows → log) that materializes dense one-hot tensors and exists only for
bring-up (BLUEPRINT §6.1). The real SFT objective is `sum(ce_sparse(logits, labels,
mask))` under `GGML_OPT_LOSS_TYPE_SUM`, normalized host-side by the number of valid
(unmasked) tokens — this is exactly what the S1-04 op was designed for (BLUEPRINT D4):
per-token loss with backward exactly zero where the weight is zero, no dense labels.
The SUM reduction is the officially sanctioned custom-loss escape hatch: everything must
fold into one scalar because extra `GGML_TENSOR_FLAG_LOSS` nodes are rejected
(`vendor/llama.cpp/ggml/src/ggml-opt.cpp:343`; loss types at
`vendor/llama.cpp/ggml/include/ggml-opt.h:29-32`).

The training loop mechanics come nearly free from ggml-opt (BLUEPRINT §1.1): gradient
accumulation is `opt_period`, and LR schedules work by mutating the Python-owned
`ggml_opt_optimizer_params` struct between steps — the S0-04 policy of passing
`ggml_opt_get_constant_optimizer_params` (`vendor/llama.cpp/ggml/include/ggml-opt.h:108`)
with the struct as userdata, never a Python callback. Two hard constraints shape the loop:
ubatch shapes must be identical every step because dynamic-graph optimizer state is keyed
by node index (BLUEPRINT D1; varying batch sizes assert at
`vendor/llama.cpp/ggml/src/ggml-opt.cpp:851`) — padding is the data layer's job (S1-06/
S1-07, pad tokens get weight 0 — soft coordination with S1-07, which is not a dependency);
and validation must reuse the same graph forward-only, which is how the stock epoch loop
already does it: `ggml_opt_alloc(opt_ctx, backward)` takes the train/eval toggle
(`vendor/llama.cpp/ggml/include/ggml-opt.h:186`; precedent
`vendor/llama.cpp/src/llama-context.cpp:3341`, `ggml_opt_alloc(opt_ctx, train)`).

Masked validation loss on a held-out split is the BLUEPRINT §6.1 evaluation story for v1;
generation-based eval waits for the GRPO rollout engine (S1-15). The tiny-model
convergence gate that consumes this trainer is S1-12, not this ticket.

## What to do

1. `csrc/farm_train.cpp`: register the `sft_ce_sparse` loss epilogue in the S1-02
   epilogue registry — `outputs = ggml_cross_entropy_loss_sparse(logits, labels_i32,
   weights_f32)` (S1-04 op), reduced via `GGML_OPT_LOSS_TYPE_SUM`. `labels` and `weights`
   are S1-02 extra named inputs filled per ubatch. Return `n_valid` (count of nonzero
   weights in the step) in the `lf_train_step` result so Python can normalize.
2. `src/llama_farm/train/loop.py`: the generic step loop used by SFT now and DPO/GRPO
   later — iterates collated fixed-shape batches, calls `lf_train_step`, implements
   gradient accumulation via `opt_period`, applies an LR schedule by mutating the
   Python-owned optimizer-params struct each step (warmup + cosine and constant to start),
   and fires logging callbacks (step, loss/valid-token, LR, tokens/s). Checkpoint hooks
   are stubs wired later by S1-09.
3. `src/llama_farm/train/sft.py`: `train_sft(model, adapter, dataset, config)` — consumes
   the S1-06 data layer (tokenized samples with loss masks), pads to fixed ubatch shapes
   with weight-0 pad tokens, selects the `sft_ce_sparse` epilogue, and reports loss
   normalized by valid tokens host-side.
4. Keep the S1-03 stopgap composite epilogue selectable behind a debug flag
   (`loss="sft_ce_stopgap"`); add a cross-check test asserting stopgap and `ce_sparse`
   losses match within tolerance on the same tiny batch with a nontrivial mask.
5. Masked validation: `evaluate(model, adapter, val_split)` runs the same graph
   forward-only (backward toggle off through the shim's train-step entry; expose an
   `lf_train_step` eval mode if S1-02 did not already) and returns masked loss per valid
   token on the held-out split. No optimizer state may change during eval (assert loss
   unchanged when eval runs between two identical train steps).
6. Unit tests `tests/test_sft.py` on the S0-06 fixture models: loss decreases over N
   steps on a fixed tiny batch; a fully-masked batch yields zero gradient contribution
   (A/B tensors unchanged after a step, using `ggml_opt_grad_acc`-backed accessors or
   tensor snapshots); normalization: doubling pad-token count leaves per-valid-token loss
   unchanged; LR schedule: recorded per-step LR matches the schedule closed-form.

## Out of scope

- The `ce_sparse` op itself, its ABI, and its kernels (S1-04 owns the op; GPU ports are
  S2-07/S3-03/S4-04).
- Packing collators and the packed-vs-unpacked equality test (S1-07).
- Checkpoint/resume wiring (S1-09) and gradient clipping (S1-10) — `loop.py` leaves named
  hook points.
- The convergence gate vs the recorded PEFT reference (S1-12).
- DPO/GRPO trainers (S1-14/S1-16) — they reuse `loop.py`.

## Acceptance criteria

- [ ] `pytest tests/test_sft.py` passes on the Linux CPU VM: loss decrease, fully-masked
      zero-gradient, pad-invariance, and LR-schedule tests all green.
- [ ] The stopgap-vs-`ce_sparse` cross-check test passes with the documented tolerance.
- [ ] `evaluate()` returns a masked validation loss and a test proves it mutates no
      optimizer or adapter state.
- [ ] A shape-mismatch batch (wrong ubatch size on step 2) raises the S1-02 clear error,
      tested — not the raw ggml assert.
- [ ] `ci-cpu / test` is green with the new tests in the per-PR selection.

## Testing & verification

- `tests/test_sft.py` (new), pytest on the S0-06 fixture models (F32 and Q8_0 tiny
  llama-arch); runs in `ci-cpu / test` per-PR (S0-07). The heavier convergence run is
  S1-12's nightly gate.
- No new ops/kernels, so no new MODE_GRAD cases; the op-level MODE_GRAD coverage this
  trainer relies on lives in S1-04 and already runs in ci-cpu.

## PR notes

- Branch: `ticket/S1-05-sft-trainer-sparse-ce`.
- Single llama-farm PR (Python + the small `csrc/` epilogue registration); no vendored
  llama.cpp changes, so no two-repo flow.
- Upstreaming disposition: **fork-local** (product training code).
- Soft coordination: S1-07's packing collator plugs into the same `loop.py` batch
  interface — keep the collator boundary (`iterable of fixed-shape batches with weights`)
  explicit and documented.
