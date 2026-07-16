"""Chunked attention (S1-24): the same answer, in less memory.

The attention matrix is ``[n_kv, n_q, n_head]`` — quadratic in context, and the thing that stops
long-context training. Chunking the QUERY axis and softmaxing each chunk on its own leaves only
``[n_kv, chunk_q, n_head]`` live at a time. Softmax is row-wise over the KEY axis, so this is
**exact**: each output row depends on its own row of scores and nothing else. (Chunking the KEY
axis is what flash attention does, and *that* needs a running-max rescale — which is why it needs
a kernel and this does not.)

Two claims, and both are tested here, because either one alone is worthless:

1. **It computes the same thing.** Not "close enough" — the losses come out *bit-identical*, and
   the gradients agree to ~1e-6 (float32, and the reduction order changes).
2. **It actually saves memory.** A rewrite that is exact and saves nothing is a rewrite that costs
   +50% attention FLOPs for fun.

And a third thing that is easy to get wrong: **chunking alone does not shrink the backward.**
SOFT_MAX_BACK reads the softmax's own output, so without recompute every chunk's ``P`` stays live
from the forward until the backward consumes it, and the peak is unchanged. It needs gradient
checkpointing (S1-17) to become a segment-interior node. The two features are *multiplicative*, and
:func:`test_chunking_needs_checkpointing_to_shrink_the_backward` pins exactly that, because a
future refactor that quietly breaks the pairing would leave both tests passing and the feature dead.

**Not covered, and said out loud:** the softcap (gemma2) and ALiBi (``max_bias > 0``) branches of
the chunked path are written but have no fixture. ``hparams.attn_soft_cap`` is set *in code* by
``models/gemma2.cpp`` — it is not a GGUF key — so it cannot be switched on for a llama-arch fixture,
and covering it means building a gemma2 fixture. Both operations are provably chunk-invariant
(ALiBi's slope is a function of the HEAD index, softcap is elementwise on the scores; neither mixes
query rows), and both mirror the naive path line for line. That is an argument, not a test. See
docs/dev/chunked-attention.md.
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np
import pytest

from learning_llamas import _ffi, create_zero_adapter
from learning_llamas.train import TrainConfig, Trainer
from learning_llamas.train.loop import Batch

SEQ = 128
N_STEPS = 3

# ggml computes in float32 and chunking changes the reduction order, so the gradients are not
# bitwise equal. Observed worst: 1.4e-06 relative. 1e-4 leaves two orders of margin without leaving
# room for an actually-wrong gradient (a dropped chunk misses by tens of percent).
GRAD_TOL = 1e-4


def _batch(seq: int = SEQ) -> Batch:
    rng = np.random.default_rng(0)
    return Batch(
        tokens=[int(t) for t in rng.integers(3, 512, size=seq)],
        targets=[int(t) for t in rng.integers(3, 512, size=seq)],
        # A masked prompt, so the loss normalization is exercised rather than degenerate.
        weights=[0.0] * (seq // 4) + [1.0] * (seq - seq // 4),
    )


def _run(
    libs: _ffi.Libraries,
    load_model,  # noqa: ANN001
    base: pathlib.Path,
    adapter: pathlib.Path,
    chunk_q: int,
    ckpt: int = 0,
    steps: int = N_STEPS,
    seq: int = SEQ,
) -> tuple[list[float], dict[tuple[str, bool], np.ndarray], int]:
    """N steps, returning the losses, the final gradients, and the peak compute buffer."""
    targets = create_zero_adapter(base, adapter, r=4, seed=3)

    model = load_model(base, n_ctx=seq, n_ubatch=seq, training=True)
    model.attach_adapter(adapter, scale=1.0)

    batch = _batch(seq)
    losses: list[float] = []
    grads: dict[tuple[str, bool], np.ndarray] = {}

    with Trainer(libs, model, TrainConfig(lr=1e-3)) as trainer:
        # Both settings need the training state (ll_opt_init_lora made it) and must land before the
        # first step: they change the graph, and ggml-opt keys optimizer state by node index.
        if ckpt:
            _ffi.check(libs.farm.ll_set_grad_checkpointing(model.ctx, ckpt), "ckpt")
        if chunk_q:
            _ffi.check(libs.farm.ll_set_chunked_attention(model.ctx, chunk_q), "chunk")

        for _ in range(steps):
            losses.append(trainer.step(batch).loss)

        for target in targets:
            for is_b in (False, True):
                n = libs.farm.ll_debug_n_elements(model.ctx, target.name.encode(), is_b)
                buf = (ctypes.c_float * n)()
                assert libs.farm.ll_debug_grad(model.ctx, target.name.encode(), is_b, buf, n) == n
                grads[(target.name, is_b)] = np.frombuffer(buf, dtype=np.float32, count=n).copy()

        peak = libs.farm.ll_compute_buffer_bytes(model.ctx)

    return losses, grads, peak


# ---------------------------------------------------------------------------
# It computes the same thing.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_q", [128, 64, 32, 48])
def test_chunked_attention_is_exact(tiny_f32, tmp_path, libs, load_model, chunk_q) -> None:
    """Chunked vs naive: identical losses, and every LoRA gradient within float32 noise.

    ``chunk_q=128`` is the whole sequence — one chunk, no concat — which isolates the view/permute
    rewrite from the chunking. ``48`` does not divide 128, so the last chunk is ragged (32 tokens):
    an off-by-one in the tail would leave the last 32 queries unattended, and the loss would still
    fall.
    """
    naive_losses, naive_grads, _ = _run(libs, load_model, tiny_f32, tmp_path / "n.gguf", 0)
    losses, grads, _ = _run(libs, load_model, tiny_f32, tmp_path / "c.gguf", chunk_q)

    assert losses == naive_losses, (
        f"chunk_q={chunk_q} changed the loss: {losses} vs {naive_losses}. Chunking the QUERY axis "
        "is exact — a softmax row depends on its own scores and nothing else — so this is a bug, "
        "not a tolerance."
    )

    worst, worst_name = 0.0, ""
    for key, expected in naive_grads.items():
        scale = max(float(np.abs(expected).max()), 1e-30)
        rel = float(np.abs(grads[key] - expected).max()) / scale
        if rel > worst:
            worst, worst_name = rel, f"{key[0]}.lora_{'b' if key[1] else 'a'}"

    assert worst < GRAD_TOL, f"chunk_q={chunk_q}: {worst_name} gradient off by {worst:.2e}"


def test_off_is_the_naive_path_exactly(tiny_f32, tmp_path, libs, load_model) -> None:
    """chunk_q=0 must not perturb anything — it is the default, and every other test rests on it."""
    a_losses, a_grads, _ = _run(libs, load_model, tiny_f32, tmp_path / "a.gguf", 0)
    b_losses, b_grads, _ = _run(libs, load_model, tiny_f32, tmp_path / "b.gguf", 0)

    assert a_losses == b_losses
    for key in a_grads:
        assert np.array_equal(a_grads[key], b_grads[key])


def test_chunking_composes_with_gradient_checkpointing(
    tiny_f32, tmp_path, libs, load_model
) -> None:
    """S1-17 and S1-24 together, still exact. They have to compose: that is the whole design."""
    naive_losses, naive_grads, _ = _run(libs, load_model, tiny_f32, tmp_path / "n.gguf", 0)
    losses, grads, _ = _run(libs, load_model, tiny_f32, tmp_path / "c.gguf", 32, ckpt=1)

    assert losses == naive_losses
    for key, expected in naive_grads.items():
        scale = max(float(np.abs(expected).max()), 1e-30)
        assert float(np.abs(grads[key] - expected).max()) / scale < GRAD_TOL


# ---------------------------------------------------------------------------
# It actually saves memory. (An exact rewrite that saves nothing is +50% FLOPs for fun.)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_chunking_needs_checkpointing_to_shrink_the_backward(
    tiny_f32, tmp_path, libs, load_model
) -> None:
    """The claim the whole design rests on, pinned so a refactor cannot quietly break it.

    SOFT_MAX_BACK reads the softmax's own output. So with chunking alone, every chunk's ``P`` stays
    live from the forward until the backward consumes it, and the peak barely moves — the memory is
    all still there, just in more tensors. Only under ``ggml_build_backward_expand_checkpointed``
    does each ``P`` become a segment-interior node, rebuilt immediately ahead of the backward node
    that reads it and dead again straight after.

    A long context is required for this to be visible at all: the attention term is *quadratic*, so
    at 128 tokens it is 256 KiB and invisible. Measured at n_ctx 1024 (see
    docs/dev/chunked-attention.md for the full cliff):

        naive 79 MiB   chunk-only 66 MiB   ckpt-only 47 MiB   ckpt+chunk 44 MiB
    """
    seq, chunk = 1024, 128
    _, _, naive = _run(libs, load_model, tiny_f32, tmp_path / "n.gguf", 0, steps=1, seq=seq)
    _, _, ckpt = _run(libs, load_model, tiny_f32, tmp_path / "k.gguf", 0, ckpt=1, steps=1, seq=seq)
    _, _, both = _run(
        libs, load_model, tiny_f32, tmp_path / "b.gguf", chunk, ckpt=1, steps=1, seq=seq
    )

    assert both < ckpt < naive, (
        f"the features are supposed to be multiplicative: naive={naive / 2**20:.1f} MiB, "
        f"ckpt={ckpt / 2**20:.1f} MiB, ckpt+chunk={both / 2**20:.1f} MiB"
    )
    assert naive / both > 1.5, f"only {naive / both:.2f}x off naive at n_ctx={seq}"


@pytest.mark.slow
def test_the_acceptance_bar_of_2x_at_n_ctx_4096(tiny_f32, tmp_path, libs, load_model) -> None:
    """S1-24's headline: ckpt+chunk peak at n_ctx 4096 beats naive by at least 2x, from CI output.

    The 2.17x figure lived only in a hand-measured table (docs/dev/chunked-attention.md); the
    per-PR memory gate asserts 1.5x at n_ctx 1024, the same claim on a smaller context. This
    runs the actual acceptance context and asserts the actual bar, so the number is reproduced by a
    machine and not just recorded by a human. It is @slow: n_ctx 4096 is a 4096-token ubatch in one
    graph, and the naive arm alone is ~0.9 GiB of compute buffer -- fine for the nightly box, too
    much to pay on every PR. Observed on the reference box: naive ~904 MiB, ckpt+chunk ~414 MiB.
    """
    seq, chunk = 4096, 256
    _, _, naive = _run(libs, load_model, tiny_f32, tmp_path / "n.gguf", 0, steps=1, seq=seq)
    _, _, both = _run(
        libs, load_model, tiny_f32, tmp_path / "b.gguf", chunk, ckpt=1, steps=1, seq=seq
    )

    assert naive / both >= 2.0, (
        f"chunked+checkpointed peak at n_ctx={seq} is only {naive / both:.2f}x below naive "
        f"(naive={naive / 2**20:.1f} MiB, ckpt+chunk={both / 2**20:.1f} MiB); the acceptance bar "
        f"is 2x"
    )


# ---------------------------------------------------------------------------
# The contract.
# ---------------------------------------------------------------------------


def test_the_symbol_resolves(libs) -> None:
    assert callable(libs.farm.ll_set_chunked_attention)


def test_changing_the_chunk_factor_mid_run_is_refused(tiny_f32, tmp_path, libs, load_model) -> None:
    """Topology is fixed for the life of a run, and this one is fixed harder than most.

    Chunking emits ``n_chunks`` softmaxes and ``n_chunks - 1`` concats per layer where the naive
    path emits one softmax. ggml-opt keys its gradient accumulators and AdamW momenta by NODE INDEX
    (BLUEPRINT D1), from the first graph it is shown. Change the factor mid-run and one parameter's
    momentum lands on another — and it would not crash.
    """
    adapter = tmp_path / "a.gguf"
    create_zero_adapter(tiny_f32, adapter, r=4, seed=3)

    model = load_model(tiny_f32, n_ctx=SEQ, n_ubatch=SEQ, training=True)
    model.attach_adapter(adapter, scale=1.0)

    with Trainer(libs, model, TrainConfig(lr=1e-3)) as trainer:
        _ffi.check(libs.farm.ll_set_chunked_attention(model.ctx, 64), "chunk")
        trainer.step(_batch())

        assert libs.farm.ll_set_chunked_attention(model.ctx, 32) == -3  # LL_ERR_ALREADY_INIT
        assert libs.farm.ll_set_chunked_attention(model.ctx, 0) == -3


def test_a_negative_chunk_is_rejected(tiny_f32, tmp_path, libs, load_model) -> None:
    adapter = tmp_path / "a.gguf"
    create_zero_adapter(tiny_f32, adapter, r=4, seed=3)

    model = load_model(tiny_f32, n_ctx=SEQ, n_ubatch=SEQ, training=True)
    model.attach_adapter(adapter, scale=1.0)

    with Trainer(libs, model, TrainConfig(lr=1e-3)):
        assert libs.farm.ll_set_chunked_attention(model.ctx, -1) == -1  # LL_ERR_INVALID_ARG


def test_setting_it_without_a_training_context_is_refused(
    tiny_f32, tmp_path, libs, load_model
) -> None:
    """It needs the training state, which ll_opt_init_lora creates. Say so, rather than segfault."""
    model = load_model(tiny_f32, n_ctx=SEQ, n_ubatch=SEQ, training=True)
    assert libs.farm.ll_set_chunked_attention(model.ctx, 64) == -6  # LL_ERR_NOT_INITIALIZED


def test_a_chunk_larger_than_the_sequence_is_one_chunk(
    tiny_f32, tmp_path, libs, load_model
) -> None:
    """Clamped, not an error: `chunk_q=4096` on a 128-token batch is simply "do not chunk"."""
    naive_losses, _, _ = _run(libs, load_model, tiny_f32, tmp_path / "n.gguf", 0)
    losses, _, _ = _run(libs, load_model, tiny_f32, tmp_path / "c.gguf", 4096)

    assert losses == naive_losses


def test_it_still_trains(tiny_f32, tmp_path, libs, load_model) -> None:
    """The loss falls with chunking on. Necessary, not sufficient — but its absence is decisive."""
    losses, _, _ = _run(libs, load_model, tiny_f32, tmp_path / "c.gguf", 32, ckpt=1, steps=8)
    assert losses[-1] < losses[0], f"the loss did not fall with chunking on: {losses}"
