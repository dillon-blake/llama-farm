# P0 proof-of-gradient — the recipe

This is the Stage-0→1 exit gate from BLUEPRINT §9: end-to-end proof on CPU that the whole stack
trains, and that the gradient reaching the adapter is the **right** gradient, not merely *a*
gradient. A gradient wrong by a constant factor, transposed, or missing a term still makes a loss
curve fall — it just converges somewhere else, slowly, and nothing complains. (This gate found
exactly such a bug: `build_masked_ce` in `csrc/farm_train.cpp`.)

Every check below lives in `tests/test_p0_gradient.py` and runs per-PR in `ci-cpu`. Later backends
re-run this conceptually through S1-12's convergence gate.

## Fixture and adapter

- **Base:** the tiny llama-arch fixture from `tests/fixtures/gen_tiny_llama.py`, in both **F32** and
  **Q4_K**, loaded with `use_mmap=true` throughout.
- **Adapter:** a rank-4 zero-init adapter (`create_zero_adapter`, fixed seed) — `A ~ N(0, sigma)`,
  `B == 0`, so training starts from an exact no-op delta.
- **Batch:** one hand-built, fixed tiny batch (fixed seed, constant shapes per D1).
- **Steps:** N = 32 `ll_train_step` calls with the S1-02 stopgap composite CE
  (`softmax -> one-hot mul -> sum_rows -> log`, select-then-log ordering per BLUEPRINT §6.1 so a
  masked/near-zero probability never produces a `0·(-inf)` NaN).

## The checks

1. **Loss falls** — `mean(loss[-4:]) < 0.8 · mean(loss[:4])`, on both F32 and Q4_K bases. The full
   curve is recorded in the test log. This is necessary, not sufficient — hence everything below.

2. **Finite-difference gradient check (on the F32 base).** For sampled elements of the adapter's `A`
   and `B` across ≥2 layers, compare ggml's analytic gradient (`ll_debug_grad`) against a **central
   finite difference of the whole graph**: perturb one weight (`ll_debug_set_tensor`), re-run the
   real forward, measure how the real loss moved, restore. Tolerance: relative error
   `|g_an − g_fd| / max(|g_an|, |g_fd|, 1e-8)` ≤ **3e-2** for `dL/dB`, ≤ **5e-2** for `dL/dA`. A
   companion test pins the structure the LoRA math implies: `dL/dA` is exactly zero while `B == 0`
   and becomes nonzero only once `B` moves.

   **Why the FD runs on F32 and not the quantized base.** llama.cpp does not dequantize weights for a
   matmul — it quantizes the *activations* to the weight's `vec_dot_type` and dots in the integer
   domain. So the forward is a **step function** of the adapter weights (steps ~1e-3 in the loss),
   and a finite difference of it measures quantization edges, not a derivative — no choice of `eps`
   recovers the ~1e-3 true slope buried under the steps. The backward, correctly, differentiates the
   *smooth dequantized* function (`MUL_MAT`'s backward is `ggml_out_prod(W, grad)`, which dequantizes
   `W` — the standard straight-through treatment of a non-differentiable quantizer). Analytic and FD
   are computing genuinely different things on a quantized base, and neither is wrong. This is a
   deliberate deviation from the S1-03 ticket, which named the FD on a quantized base; it is replaced
   by check 3.

3. **The quantized backward tracks the F32 backward.** Rather than finite-difference the staircase,
   the Q4_K gradient is validated by direct equivalence against the F32 gradient it is the
   straight-through image of.

4. **NaN robustness.** A batch whose mask zeroes prompt positions and whose logits are pushed to
   extremes must yield a finite loss and finite `A`/`B` gradients — the check on the select-then-log
   ordering of §6.1.

5. **Gradients do not accumulate across steps** — a regression guard for the dynamic-graph
   accumulator-reset bug (fork commit `79e37988c`).

6. **Interchange.** The trained adapter is saved with the S0-05 writer (real `adapter.lora.alpha`
   from the training config, never 0), then reloaded in-process via stock
   `llama_adapter_lora_init` + `llama_set_adapters_lora`: the load emits no error and the adapter
   *measurably shifts the logits* on a fixed prompt versus base. This is the D3 promise — the adapter
   GGUF is the checkpoint format and loads in stock tooling with zero conversion.

## Rerun

```bash
.venv/bin/python -m pytest tests/test_p0_gradient.py -q
```

The fixtures build themselves on first run. The FD and interchange checks stay per-PR; only the
32-step loss-curve test may move behind the `slow` marker if the per-PR budget is exceeded.
