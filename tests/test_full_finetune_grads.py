"""Full fine-tuning (S1-48): the BASE weights' gradients, checked against the float64 oracle.

The convergence gate (S1-12) proved the LoRA path computes the right thing: one step, every one of
the 28 adapter gradients, against ``tests/reference_llama.py``. Full fine-tuning was the hole — the
only thing that exercised it was ``test_training_graph.py``, which proves the backward *builds* and
the loss *falls*, and compares the base-weight gradients to **nothing**. A base gradient wrong by a
fraction of a percent would still make the loss fall, converge somewhere slightly else, and pass
that test in silence — exactly the S1-03 class of bug (a correct forward with a wrong backward).

So this is the LoRA gate's twin for the base weights. The shim's ``ll_opt_init_full`` flags every
F32 leaf weight the forward reaches — the token embedding, the output head, and every RMSNorm
weight and attention/FFN projection in every block — and a single step at ``lr = 1e-30`` reads each
one's gradient back through ``ll_debug_base_grad`` and compares it, tensor by tensor, against a
backward derived by hand in float64 rather than transcribed from the graph. A short trajectory then
lets ggml's fused AdamW actually move the base weights and tracks the reference for fifteen steps.

Which tensors train, and one difference from stock llama.cpp
------------------------------------------------------------
``llama.cpp``'s own ``llama_opt_init`` FIXMEs the token embedding out of training
(``llama-context.cpp``: ``//llama_set_param(model->tok_embd ...)``). ``ll_opt_init_full`` trains it:
its gradient is ``get_rows``'s VJP (``GET_ROWS_BACK``), which this fork implements and the CPU
backend schedules. This test is what proves that inclusion is *correct* and not merely accepted —
the embedding's scatter-add gradient matches the float64 reference to the same tolerance as every
dense projection.
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np

from learning_llamas import Model, _ffi

from . import reference_llama as ref
from .convergence import config

# opt_init asserts n_ctx_train % n_batch == 0; a requested 64 is what the LoRA harness uses too.
N_CTX = 64

# One full-finetune step at lr=1e-30, ggml vs the float64 reference, worst over all 21 base tensors.
#   observed: 1.1e-06 worst (token_embd's scatter included), loss agreeing to 6.9e-09 relative.
# The margin is for another host's SIMD reduction order (ADR-0002), which cannot be measured here.
# 1e-3 is ~900x the observed gap and still catches a gradient wrong in its 4th significant figure.
BASE_GRAD_TOL = 1e-3
LOSS_TOL = 1e-5

# Fifteen full-finetune steps, ggml's fused AdamW moving the base weights, per step vs the float64
# reference. observed: 2.1e-06 worst |diff| over the run.
# 1e-4 is ~48x the observed gap; a band wide enough to absorb kernel drift would hide a real bug.
TRAJ_TOL = 1e-4
TRAJ_STEPS = 15
TRAJ_LR = 1e-3


def _open(libs: _ffi.Libraries, path: pathlib.Path) -> Model:
    """A training-mode context over ``path`` with mmap off, so AdamW can write the base weights."""
    return Model(
        path, libs=libs, n_ctx=N_CTX, n_ubatch=config.SEQ_LEN, full_finetune=True, n_threads=2
    )


def _expected_trainable(hp: ref.RefHParams) -> set[str]:
    """The base tensors full fine-tuning should flag, spelled out rather than read back.

    The token embedding, the output norm, the (untied) output head, and per block: the two RMSNorm
    weights and the four attention plus three FFN projections. rope_freqs is a precomputed constant,
    not a learnable, and the fixture has no bias tensors — so this is the whole trainable set.
    """
    names = {"token_embd.weight", "output_norm.weight", "output.weight"}
    for il in range(hp.n_layer):
        for tensor in (
            "attn_norm",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_norm",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ):
            names.add(f"blk.{il}.{tensor}.weight")
    return names


def _read_base_grad(
    libs: _ffi.Libraries, ctx: int, name: str, shape: tuple[int, ...]
) -> np.ndarray:
    """Read one flagged base weight's gradient back out of ggml-opt's accumulator, as float64."""
    size = int(np.prod(shape))
    buf = (ctypes.c_float * size)()
    got = libs.farm.ll_debug_base_grad(ctx, name.encode(), buf, size)
    assert got == size, f"ll_debug_base_grad({name}) returned {got}, expected {size}"
    return np.frombuffer(buf, dtype=np.float32, count=got).reshape(shape).astype(np.float64)


def _train_step(libs: _ffi.Libraries, ctx: int, tokens, targets, weights) -> float:
    """One masked-CE full-finetune step over one sample. Returns the loss."""
    n = len(tokens)
    ct = (ctypes.c_int32 * n)(*[int(t) for t in tokens])
    cg = (ctypes.c_int32 * n)(*[int(t) for t in targets])
    cw = (ctypes.c_float * n)(*[float(w) for w in weights])
    loss = ctypes.c_float()
    _ffi.check(
        libs.farm.ll_train_step(ctx, ct, cg, cw, None, None, n, True, ctypes.byref(loss)),
        "ll_train_step",
    )
    return loss.value


def test_full_finetune_flags_exactly_the_base_weights(
    tiny_f32: pathlib.Path, libs: _ffi.Libraries
) -> None:
    """``ll_opt_init_full`` flags every F32 base weight the forward reaches, and only those.

    The list is the answer to "what does full fine-tuning train?", asserted rather than assumed —
    including the token embedding, which stock ``llama_opt_init`` leaves out.
    """
    _, hp = ref.load_model(tiny_f32)
    expected = _expected_trainable(hp)

    model = _open(libs, tiny_f32)
    try:
        params = _ffi.ll_opt_params(alpha=1e-30, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0)
        n_flagged = _ffi.opt_init_full(
            libs, model.ctx, model.model, params, opt_period=1, grad_clip=0.0
        )

        assert n_flagged == len(expected), f"flagged {n_flagged} tensors, expected {len(expected)}"
        assert libs.farm.ll_opt_n_params(model.ctx) == len(expected)

        # And every expected tensor is genuinely a flagged parameter (n_elements resolves it).
        for name in expected:
            n_el = libs.farm.ll_debug_base_n_elements(model.ctx, name.encode())
            assert n_el > 0, f"{name} is not a flagged parameter (ll_debug_base_n_elements={n_el})"
    finally:
        _ffi.opt_free(libs, model.ctx)
        model.close()


def test_one_full_finetune_step_matches_the_reference(
    tiny_f32: pathlib.Path, libs: _ffi.Libraries, conv_data
) -> None:
    """The loss and **every base-weight gradient**, one step at lr=1e-30, against float64.

    A matching loss proves the forward and says nothing about the backward (the S1-03 bug had a
    correct forward). So the gradients are compared tensor by tensor, and a failure names the
    tensor. No weight needs perturbing off any special value the way a fresh LoRA's B does: the base
    weights are generic, and every one of them carries a nonzero gradient on a generic batch.
    """
    tokens, targets, weights = conv_data
    base, hp = ref.load_model(tiny_f32)
    expected = _expected_trainable(hp)
    assert set(base) == expected, "the fixture's base weights are not the expected trainable set"

    model = _open(libs, tiny_f32)
    try:
        params = _ffi.ll_opt_params(alpha=1e-30, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0)
        _ffi.opt_init_full(libs, model.ctx, model.model, params, opt_period=1, grad_clip=0.0)

        got_loss = _train_step(libs, model.ctx, tokens[0], targets[0], weights[0])

        # The reference: no adapter (empty loras), so the scale is irrelevant.
        logits, cache = ref.forward(base, hp, {}, 1.0, tokens[0])
        want_loss, dlogits = ref.loss_from_logits(logits, targets[0], weights[0])
        grads = ref.backward(base, hp, {}, 1.0, cache, dlogits, base_grads=True)

        loss_rel = abs(got_loss - want_loss) / abs(want_loss)
        assert loss_rel < LOSS_TOL, (
            f"the full-finetune loss disagrees with the reference by {loss_rel:.2e} relative"
        )

        by_tensor: dict[str, float] = {}
        for name in expected:
            want = grads[name]
            actual = _read_base_grad(libs, model.ctx, name, want.shape)
            by_tensor[name] = float(np.abs(actual - want).max() / max(np.abs(want).max(), 1e-30))

        assert len(by_tensor) == len(expected)
        worst = max(by_tensor, key=by_tensor.__getitem__)
        assert by_tensor[worst] < BASE_GRAD_TOL, (
            f"{worst}: ggml's full-finetune gradient disagrees with the float64 reference by "
            f"{by_tensor[worst]:.2e} (relative to the tensor's largest element)"
        )
    finally:
        _ffi.opt_free(libs, model.ctx)
        model.close()


def _reference_trajectory(base_path: pathlib.Path, conv_data) -> list[float]:
    """The float64 reference run: forward, base-weight backward, AdamW on the base weights."""
    tokens, targets, weights = conv_data
    base, hp = ref.load_model(base_path)
    opt = ref.AdamW(lr=TRAJ_LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    curve: list[float] = []
    for step in range(TRAJ_STEPS):
        i = step % config.N_SAMPLES
        logits, cache = ref.forward(base, hp, {}, 1.0, tokens[i])
        loss, dlogits = ref.loss_from_logits(logits, targets[i], weights[i])
        grads = ref.backward(base, hp, {}, 1.0, cache, dlogits, base_grads=True)
        opt.step(base, grads)  # updates base in place, exactly as ggml writes the weights back
        curve.append(loss)
    return curve


def _ggml_trajectory(libs: _ffi.Libraries, base_path: pathlib.Path, conv_data) -> list[float]:
    """The real stack: full_finetune=True, ll_opt_init_full, TRAJ_STEPS masked-CE steps."""
    tokens, targets, weights = conv_data
    model = _open(libs, base_path)
    try:
        params = _ffi.ll_opt_params(alpha=TRAJ_LR, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0)
        _ffi.opt_init_full(libs, model.ctx, model.model, params, opt_period=1, grad_clip=0.0)

        curve: list[float] = []
        for step in range(TRAJ_STEPS):
            i = step % config.N_SAMPLES
            curve.append(_train_step(libs, model.ctx, tokens[i], targets[i], weights[i]))
        return curve
    finally:
        _ffi.opt_free(libs, model.ctx)
        model.close()


def test_full_finetune_trajectory_matches_the_reference(
    tiny_f32: pathlib.Path, libs: _ffi.Libraries, conv_data
) -> None:
    """Fifteen full-finetune steps, ggml's fused AdamW moving the base weights, against float64.

    This is the whole stack under motion: the graph, the base-weight backward, and AdamW's bias
    correction, fifteen times over on weights that actually change between steps. Anything wrong by
    a fraction of a percent compounds here. The loss must also fall — a disconnected or wrong-signed
    base gradient would leave it flat or push it up.
    """
    got = _ggml_trajectory(libs, tiny_f32, conv_data)
    want = _reference_trajectory(tiny_f32, conv_data)

    assert len(got) == len(want) == TRAJ_STEPS

    diff = np.abs(np.array(got) - np.array(want))
    worst = int(diff.argmax())
    assert diff.max() < TRAJ_TOL, (
        f"the full-finetune loss curve left the float64 reference at step {worst}: "
        f"{got[worst]:.8f} vs {want[worst]:.8f} (worst |diff| {diff.max():.2e} > {TRAJ_TOL:.0e})"
    )

    assert got[-1] < got[0], f"the full-finetune loss did not fall over {TRAJ_STEPS} steps: {got}"
