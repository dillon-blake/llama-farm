---
id: S1-47
title: "SSM (Mamba-1) training oracle — which caught the scan backward reading the overwritten state cache"
stage: 1
track: python
size: L
deps: [S1-29, S1-30, S1-31, S1-38, S1-41]
status: done
pr: null
---

# S1-47 — Mamba-1 training oracle, and the live bug it caught

**One-line outcome:** a Mamba-1 model trains through the real stack (Model(training=True) +
create_zero_adapter + train_sft), and the composed SSM gradient (ssm_in -> causal conv ->
selective scan -> D skip -> SiLU gate -> ssm_out, LoRA on all four projections) is verified per
tensor against a self-audited float64 reference — which exposed and fixed a real fork bug that had
silently corrupted the scan gradient of every Mamba layer but the last. This is the tiny-Mamba e2e
that S1-31's acceptance criteria named and never shipped (the audit's `moe-ssm` major).

## The bug (fork commit on `ticket/S1-24-chunked-attention`, `src/models/mamba-base.cpp`)

`ggml_ssm_scan`'s initial-state input (`ssm`) is a **view of the recurrent-state cache**
(`ssm_states_all`), and the forward's `ggml_cpy(final_state -> cache)` overwrites that cache in
place (mamba-base.cpp:127-130). `ggml_ssm_scan_back` recomputes the intermediate states from that
initial state (the forward destroys them in place, so they must be recomputed for the `dA` path of
`ddt`). By backward time the cache holds the **final** state, so the recompute starts from the
wrong point and every scan-derived gradient (`dx`, `ddt`, `dB`, `dC`) is wrong — which corrupts
`ssm_in`/`ssm_x`/`ssm_dt` while leaving `ssm_out` (downstream of the scan) correct. It presents as
**layer-0-wrong, last-layer-right**: the last layer's `ssm` buffer survives to the backward, the
earlier layers' do not (their slot is reused). Fix: `ssm = ggml_cont(ctx, ssm)` before the scan, in
both the Mamba-1 and Mamba-2 paths — the initial state gets its own buffer, immune to the in-place
cache update. The kernel is untouched, so `test-backend-ops grad -o SSM_SCAN` still passes.

Why the existing testing missed it: the scan kernel is MODE_GRAD-correct in isolation (verified at
fixture-scale dims, d_state=16/n_head=16/n_seq_tokens=16 — OK), because test-backend-ops has no
in-place cache overwrite aliasing the scan's input. The e2e path had literally never run: a
non-contiguous ACC in the `GGML_OP_VIEW` backward aborted every Mamba training graph before this
ticket (fixed here too — see below).

## The other fixes this ticket carried (fork, `ggml/src/ggml.c`)

- **VIEW-backward non-contiguous ACC.** The Mamba conv splits `ssm_in`'s output with a view and
  feeds it through `ggml_transpose`; the transpose backward hands the view backward a
  non-contiguous gradient, and `ggml_acc` asserts `nb[0] == sizeof(float)`. Every Mamba training
  graph aborted at backward. Fix: `ggml_cont` the grad before `ggml_acc_or_set`, matching the guard
  the neighbouring `RESHAPE` backward already applies.
- **n_group > 1 refused loudly.** The `SSM_SCAN` backward switch now asserts `ssm_B->ne[1] == 1`.
  The kernel routes heads to per-group `dB`/`dC` slabs via `g = h/(nh/ng)`, and that routing has
  **no** finite-difference oracle (every grad-enabled `test_ssm_scan` case is `n_group == 1`; the
  `n_group > 1` shapes exceed `grad_nmax` and are skipped). Mamba-1 always builds `n_group == 1`, so
  this ticket's fixture is squarely inside the verified boundary; Mamba-2 / Falcon-H1 with
  `n_group > 1` now abort with a message pointing at the follow-up rather than training on an
  unproven gradient. See `tickets/backlog/B-10`.

## Support boundary established

- **Verified (this ticket + MODE_GRAD):** Mamba-1 — per-state `A` (`A->ne[0] == d_state`),
  `head_dim == 1`, `n_group == 1`. Trains end-to-end; every LoRA gradient matches float64 at
  ~1e-6.
- **Kernel-verified but no e2e fixture:** the scalar-`A` (Mamba-2, `head_dim > 1`) branch is
  grad-checked at `n_group == 1` in test-backend-ops. The `ggml_cont` fix is applied to that path
  too, but there is no Mamba-2 GGUF fixture here.
- **Refused loudly (no oracle):** `n_group > 1` group-index routing. B-10.
- **Refused (by design):** `A`-matrix and conv-weight gradients (frozen; ROADMAP §10 S5 / B-09).

## What to do

- `tests/fixtures/gen_tiny_mamba.py`: a tiny Mamba-1 GGUF (arch `mamba`, `n_embd=64`,
  `d_inner=128`, `d_conv=4`, `d_state=16`, `dt_rank=16`, 2 layers). F32 only — the conv weight's
  four-wide rows cannot be K-quantized.
- `tests/reference_mamba.py`: float64 forward/backward derived from the Mamba S6 maths and the
  block architecture (NOT transcribed from `ggml_compute_forward_ssm_scan`), self-audited against a
  finite difference of its own forward.
- `tests/test_ssm_training.py`: the oracle self-audit, one step x all 16 LoRA-gradient comparisons
  at effective scale 2.0 (`alpha != rank`, the S1-38 lesson), a 24-step `train_sft` trajectory, and
  the e2e gate (preflight reports the arch trainable, loss falls).
- `src/learning_llamas/adapter.py`: extend `DEFAULT_PRESET` with `ssm_in`/`ssm_x`/`ssm_dt`/`ssm_out`
  (a mamba adapter is empty otherwise; non-mamba models have no tensor with these names).
- Fork: the `ggml_cont` state-preservation fix, the VIEW-backward contiguity fix, the `n_group > 1`
  refusal.

## Acceptance criteria

- [x] Oracle self-audit < 1e-5 vs FD of its own forward (observed 5.1e-08).
- [x] One-step: loss matches at < 1e-5 rel (observed 3.9e-08); all 16 tensor grads < 1e-3 rel
      (observed 3.7e-06, post-fix; pre-fix layer-0 tensors were ~1e-2 and a finite difference of
      the *real* ggml forward sided with the float64 reference, S1-41-style).
- [x] 24-step Mamba trajectory per-step < 1e-4 (observed 4.5e-07).
- [x] e2e: preflight reports the mamba arch trainable, SFT loss falls.
- [x] `test-backend-ops grad` for `SSM_SCAN`/`SSM_CONV`/`CONCAT`/`VIEW`/`GET_ROWS` still green
      (kernels untouched).
- [x] Dense + MoE paths untouched (`test_moe_gradients.py`, `test_convergence.py`,
      `test_adapter.py` green; the VIEW-backward `cont` only fires on non-contiguous grads).

## Out of scope

- Mamba-2 / `n_group > 1` gradient oracle and e2e fixture (B-10).
- Cross-ubatch BPTT, `A`/`D`/conv-weight/`dt_bias` grads (ROADMAP §10 S5 / B-09).
- RWKV / gated-delta-net linear-attention backward (B-06).
