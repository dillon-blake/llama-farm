---
id: S1-48
title: "Full-finetune gradients: the base weights against the float64 oracle"
stage: 1
track: python
size: M
deps: [S1-00, S1-12, S1-38]
status: done
pr: null
---

# S1-48 — Full-finetune gradients checked against the float64 oracle

**One-line outcome:** every trainable base weight's gradient — the token embedding, the output
head, and every RMSNorm weight and attention/FFN projection in every block — is compared per
tensor against `tests/reference_llama.py`, one step and over a short trajectory, instead of against
nothing.

## Why (context)

The Stage 0+1 audit (2026-07-15) rated this **major** (seeded finding #7): the only thing that
exercised full fine-tuning was `tests/test_training_graph.py`, which proves the backward *builds*
and the loss *falls*. The base-weight gradients themselves were compared to nothing. That is
exactly the S1-03 bug class — a correct forward with a wrong backward — reproduced at the level of
the base weights: a gradient wrong by a fraction of a percent still makes the loss fall, still
converges (somewhere slightly else), and passes a loss-falls test in silence. The LoRA path had its
oracle (S1-12's one-step all-28-gradients check); the base weights had none.

There was also no full-finetune *training path* in the shim at all: `ll_opt_init_lora` flags only
adapter tensors, and `test_training_graph.py` reaches the base weights through llama.cpp's own
`llama_opt_init`, which hardcodes an unmasked dense cross-entropy and offers no per-tensor gradient
readback. So the gradients could not be inspected even if one wanted to.

## What to do

- `csrc/farm_train.cpp` + `csrc/farm_api.h`: `ll_opt_init_full` — the mirror of `ll_opt_init_lora`,
  flagging the base model's own F32 leaf weights (`ggml_set_param`) and reusing the whole existing
  machinery unchanged (the SUM-loss opt context, the masked-CE `ll_train_step`, the
  gradient/momentum capture). Debug accessors `ll_debug_base_grad` / `ll_debug_base_n_elements`
  address a flagged base weight by its exact ggml name (the full-finetune analogue of the adapter
  accessors' `base_name + is_b`).
- `src/learning_llamas/_ffi/farm.py`: bind them, with an `opt_init_full` wrapper that keeps the
  `ll_opt_params` struct alive exactly as `opt_init_lora` does.
- `tests/reference_llama.py`: extend `backward` to emit base-weight gradients when `base_grads=True`
  — the output head (`dlogits.T @ xf`), every RMSNorm weight (`sum_rows dy * x_hat`), every
  projection (`dy.T @ x`), and the embedding table (a scatter-add of the residual-stream gradient,
  `get_rows`'s VJP). `rms_norm_back` / `linear_back` gain optional accumulation with defaults that
  leave every existing LoRA caller (including `reference_moe.py` and `reference_mamba.py`) untouched.
- `tests/test_convergence.py`: extend the reference self-audit to finite-difference a sample of the
  new base-weight gradients too (same `h`, same tolerance); the LoRA assertions are untouched.
- `tests/test_full_finetune_grads.py`: the trainable-set assertion, the one-step all-base-gradients
  comparison at `lr = 1e-30`, and a 15-step full-finetune trajectory with ggml's fused AdamW moving
  the base weights, each within a **measured** band.

## Which tensors train

`ll_opt_init_full` flags, for the F32 llama fixture (21 tensors): `token_embd.weight`,
`output_norm.weight`, `output.weight`, and per block `attn_norm`, `attn_q`, `attn_k`, `attn_v`,
`attn_output`, `ffn_norm`, `ffn_gate`, `ffn_up`, `ffn_down`. `rope_freqs` (a precomputed constant)
is skipped by name; non-F32, non-leaf, and already-flagged (tied) tensors are skipped.

**One deliberate difference from stock llama.cpp:** `llama_context::opt_init` FIXMEs the token
embedding out of training (`//llama_set_param(model->tok_embd ...)`). `ll_opt_init_full` trains it —
its gradient is `get_rows`'s VJP (`GET_ROWS_BACK`), which this fork implements and the CPU backend
schedules — and this ticket's test is what proves that inclusion is *correct*: the embedding's
scatter-add gradient matches the float64 reference to the same tolerance as every dense projection.

## Out of scope

Quantized-base full fine-tuning (a quantized weight cannot receive a gradient; the base must be F32,
which is why full fine-tuning already forces mmap off and is the smooth model anyway). Tied-embedding
full fine-tuning (the F32 fixture is untied; the shim dedups a tied `output == tok_embd` but no test
drives it). Wiring full fine-tuning into `Trainer` / `train_sft` (which are adapter-centric); the
trajectory drives `ll_train_step` directly.

## Acceptance criteria

- [x] `ll_opt_init_full` flags exactly the F32 base weights the forward reaches (21 on the fixture),
      asserted against a spelled-out expected set including `token_embd.weight`.
- [x] One step at `lr = 1e-30`: the loss and **every** base-weight gradient match the float64
      reference, compared tensor by tensor so a failure names the tensor.
- [x] The reference's own base-weight backward is finite-differenced against its own forward (the
      oracle audits itself), one sample of each gradient kind.
- [x] A ≥ 10-step full-finetune trajectory tracks the reference per step within a measured band, and
      the loss falls.
- [x] Every band quotes the observed measurement beside it.
- [x] No fork change (the shim is main-repo `csrc/`); the existing 366 tests stay green.

## Testing & verification

Measured on this host (F32 fixture, 2 threads):

- Reference self-audit, base weights: worst **4.1e-08** relative vs a central finite difference of
  the reference's own forward (`h = 1e-4`).
- One step at `lr = 1e-30`: loss agrees to **6.9e-09** relative; worst base-weight gradient
  **1.05e-06** over all 21 tensors (`blk.1.attn_norm.weight`), token embedding's scatter included at
  **8.7e-07**. Band `1e-3` (~900x margin).
- 15-step trajectory (`lr = 1e-3`, AdamW): worst per-step **2.1e-06**, loss `6.278 → 5.273`. Band
  `1e-4` (~48x margin).

Finding: **ggml's full-finetune base-weight gradients are correct** — no graph aliasing bug of the
S1-41/S1-47 class here. Notably the token embedding, which upstream llama.cpp declines to train,
gradients correctly through `GET_ROWS_BACK`.
