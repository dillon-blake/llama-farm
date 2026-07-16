"""S1-17: the same gradients, computed while keeping far fewer activations.

The backward pass reads the forward pass's activations, so in an ordinary graph every one of them
stays live from where it is produced to where its gradient is taken — for the first layer, that is
the whole graph. At long context it is activations, not weights, that exhaust a small machine.

Checkpointing keeps only the layer boundaries and **recomputes** everything between them, on demand,
immediately before the backward node that reads it. One extra forward pass of arithmetic, in
exchange for an activation footprint that stops growing with depth.

Two claims, and they need each other:

**The gradients do not change — bitwise.** Not "close", not "within tolerance": the recomputed
activations are the same ops on the same inputs, so they are the same bits, so the gradients are the
same bits. Anything less would mean the recompute is not reproducing the forward, and a tolerance
would hide exactly that. This is what the equality tests pin.

**The memory actually falls.** Bitwise equality would also pass for an implementation that quietly
did nothing at all — which would be *correct*, and would be the one bug that matters here. So peak
activation memory is measured directly, and asserted to fall.

The fixture is deliberately **deep** (12 layers). At 2 layers there is nothing between the
boundaries to recompute and checkpointing is pure overhead — a shallow fixture would have let a
do-nothing implementation pass the memory test too.
"""

import ctypes
import pathlib

import numpy as np
import pytest

from learning_llamas import _ffi
from learning_llamas.adapter import create_zero_adapter
from learning_llamas.train import TrainConfig, Trainer
from learning_llamas.train.loop import Batch

from .fixtures import gen_tiny_llama
from .fixtures.gen_tiny_llama import TinyLlamaHParams

# Deep enough that the layers between two boundaries are worth recomputing. This is the whole point
# of the fixture: at n_layer=2 an implementation that did nothing would pass every test below.
DEEP = TinyLlamaHParams(n_layer=12, n_ctx_train=1024)

N_CTX = 512
SEQ_LEN = 256
RANK = 4
N_STEPS = 3


@pytest.fixture(scope="session")
def deep_model(libs, fixture_cache_dir) -> pathlib.Path:
    path, _ = gen_tiny_llama.build("q4_k", fixture_cache_dir, hp=DEEP)
    return path


@pytest.fixture(scope="session")
def deep_adapter(deep_model, fixture_cache_dir) -> pathlib.Path:
    path = fixture_cache_dir / "grad-ckpt-adapter.gguf"
    if not path.exists():
        create_zero_adapter(deep_model, path, r=RANK, seed=7)
    return path


@pytest.fixture
def batch() -> Batch:
    rng = np.random.default_rng(1)
    tokens = [int(x) for x in rng.integers(1, 400, size=SEQ_LEN)]
    return Batch(
        tokens=tokens,
        targets=tokens[1:] + [tokens[0]],
        # A prompt/completion mask, as in any real batch: recompute must respect it too.
        weights=[0.0] * 16 + [1.0] * (SEQ_LEN - 16),
    )


class Run:
    """What one training run produced: its losses, its trained weights, and what it cost."""

    def __init__(self, losses: list[float], weights: np.ndarray, peak_bytes: int) -> None:
        self.losses = losses
        self.weights = weights
        self.peak_bytes = peak_bytes


def _train(libs, load_model, model_path, adapter_path, batch: Batch, segment_len: int) -> Run:
    """N steps at the given segment length, from a freshly zero-initialized adapter."""
    model = load_model(model_path, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True)
    model.attach_adapter(adapter_path, scale=1.0)

    with Trainer(libs, model, TrainConfig(lr=1e-2)) as trainer:
        if segment_len:
            _ffi.check(
                libs.farm.ll_set_grad_checkpointing(model.ctx, segment_len),
                "ll_set_grad_checkpointing",
            )

        losses = [trainer.step(batch).loss for _ in range(N_STEPS)]
        peak = libs.farm.ll_compute_buffer_bytes(model.ctx)

    # Read the trained A/B back out of the live adapter, after the optimizer is gone.
    weights = []
    for i in range(libs.farm.ll_adapter_n_tensors(model.adapter)):
        for is_b in (False, True):
            n = libs.farm.ll_adapter_get(model.adapter, i, is_b, None, 0)
            buf = (ctypes.c_float * n)()
            libs.farm.ll_adapter_get(model.adapter, i, is_b, buf, n)
            weights.append(np.frombuffer(buf, dtype=np.float32, count=n).copy())

    return Run(losses, np.concatenate(weights), peak)


# ---------------------------------------------------------------------------------------------
# THE test.
# ---------------------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("segment_len", [1, 2, 4])
def test_checkpointing_changes_the_memory_and_nothing_else(
    deep_model, deep_adapter, load_model, libs, batch, segment_len: int
) -> None:
    """Bitwise-identical losses and weights, at a fraction of the activation memory.

    Bitwise, not approximate. A recomputed activation is the same ops on the same inputs on the same
    backend, so it is the same bits — and if it were not, a tolerance here would hide precisely the
    bug worth finding. ADR-0002 makes the CPU backend deterministic per shape, which is what lets
    this be an equality rather than a hope.
    """
    off = _train(libs, load_model, deep_model, deep_adapter, batch, segment_len=0)
    on = _train(libs, load_model, deep_model, deep_adapter, batch, segment_len=segment_len)

    assert on.losses == off.losses, (
        f"segment {segment_len} changed the losses.\n"
        f"  off: {off.losses}\n"
        f"  on : {on.losses}\n"
        "The recomputed activations are not reproducing the forward pass."
    )

    assert np.array_equal(on.weights, off.weights), (
        f"segment {segment_len} trained to different weights, though the losses matched. The "
        f"gradients differ, which means the backward read something the forward did not write. "
        f"max |delta| = {np.abs(on.weights - off.weights).max()}"
    )

    assert on.peak_bytes < off.peak_bytes, (
        f"segment {segment_len} did not reduce activation memory: {on.peak_bytes} vs "
        f"{off.peak_bytes} bytes. Bitwise equality alone would also hold for an implementation "
        f"that did nothing at all, so this is the assertion that says it did something."
    )


@pytest.mark.slow
def test_a_shorter_segment_saves_more(deep_model, deep_adapter, load_model, libs, batch) -> None:
    """Segment length is a dial, not a switch: fewer layers per segment, less memory kept.

    Without this, `segment_len` could be ignored and every test above would still pass — one
    checkpoint anywhere reduces memory, and the parametrization would never notice.
    """
    peaks = {
        seg: _train(libs, load_model, deep_model, deep_adapter, batch, seg).peak_bytes
        for seg in (0, 4, 1)
    }

    assert peaks[0] > peaks[4] > peaks[1], (
        f"peak activation memory did not fall monotonically with the segment length: {peaks}"
    )


def test_checkpointing_is_correct_and_saves_memory_per_pr(
    deep_model, deep_adapter, load_model, libs, batch
) -> None:
    """A per-PR smoke that checkpointing is both correct AND not a silent no-op (S1-50).

    The two witnesses above are ``@slow`` (the full segment-length sweep is a nightly cost), so per-
    PR CI ran only the error-path contract -- nothing that would go red if a refactor made recompute
    wrong or turned the feature into a no-op. This is the minimal thing that catches both, cheaply:
    ONE on/off comparison on the deep fixture. The 12 layers are load-bearing -- they are what make
    the memory claim non-vacuous, since a no-op passes the bitwise equality but not the reduction.
    Measured ~2.7 s on the reference box (off 53 MiB, on 12 MiB), which is a per-PR price worth
    paying to keep a headline feature from silently breaking between nightly runs.
    """
    off = _train(libs, load_model, deep_model, deep_adapter, batch, segment_len=0)
    on = _train(libs, load_model, deep_model, deep_adapter, batch, segment_len=2)

    assert on.losses == off.losses, (
        f"checkpointing changed the losses -- recompute is not reproducing the forward.\n"
        f"  off: {off.losses}\n  on : {on.losses}"
    )
    assert np.array_equal(on.weights, off.weights), (
        "checkpointing trained to different weights though the losses matched: the backward read "
        "something the forward did not write"
    )
    assert on.peak_bytes < off.peak_bytes, (
        f"checkpointing did not reduce activation memory ({on.peak_bytes} vs {off.peak_bytes}); on "
        f"12 layers a real implementation must, so this is what catches a silent no-op per-PR"
    )


# ---------------------------------------------------------------------------------------------
# The contract.
# ---------------------------------------------------------------------------------------------


def test_off_is_the_default(tiny_q4_k, tmp_path, load_model, libs) -> None:
    """A context nobody configured trains the way it always has."""
    adapter = tmp_path / "a.gguf"
    create_zero_adapter(tiny_q4_k, adapter, r=RANK, seed=7)

    model = load_model(tiny_q4_k, n_ctx=64, n_ubatch=32, training=True)
    model.attach_adapter(adapter, scale=1.0)

    tokens = [7, 11, 13, 17] * 8
    b = Batch(tokens=tokens, targets=tokens[1:] + [tokens[0]], weights=[1.0] * 32)

    with Trainer(libs, model, TrainConfig(lr=1e-3)) as trainer:
        assert trainer.step(b).loss > 0.0  # it just trains; nothing had to be turned off


def test_changing_it_mid_run_is_refused(tiny_q4_k, tmp_path, load_model, libs) -> None:
    """Before the first step, or not at all.

    Checkpointing rewrites the step's memory and speed profile completely. A run whose steps are
    not alike is not a run, and the failure would show up as an inexplicable discontinuity in a
    memory graph rather than as an error.
    """
    adapter = tmp_path / "a.gguf"
    create_zero_adapter(tiny_q4_k, adapter, r=RANK, seed=7)

    model = load_model(tiny_q4_k, n_ctx=64, n_ubatch=32, training=True)
    model.attach_adapter(adapter, scale=1.0)

    tokens = [7, 11, 13, 17] * 8
    b = Batch(tokens=tokens, targets=tokens[1:] + [tokens[0]], weights=[1.0] * 32)

    with Trainer(libs, model, TrainConfig(lr=1e-3)) as trainer:
        # Before the first step: fine.
        _ffi.check(libs.farm.ll_set_grad_checkpointing(model.ctx, 1), "ll_set_grad_checkpointing")
        trainer.step(b)

        # After it: not.
        with pytest.raises(RuntimeError, match="ALREADY_INIT"):
            _ffi.check(
                libs.farm.ll_set_grad_checkpointing(model.ctx, 2), "ll_set_grad_checkpointing"
            )


def test_a_negative_segment_length_is_refused(tiny_q4_k, tmp_path, load_model, libs) -> None:
    adapter = tmp_path / "a.gguf"
    create_zero_adapter(tiny_q4_k, adapter, r=RANK, seed=7)

    model = load_model(tiny_q4_k, n_ctx=64, n_ubatch=32, training=True)
    model.attach_adapter(adapter, scale=1.0)

    with Trainer(libs, model, TrainConfig(lr=1e-3)):
        with pytest.raises(RuntimeError, match="INVALID_ARG"):
            _ffi.check(
                libs.farm.ll_set_grad_checkpointing(model.ctx, -1), "ll_set_grad_checkpointing"
            )


def test_it_needs_a_prepared_context(tiny_q4_k, load_model, libs) -> None:
    """There is nothing to checkpoint until there is something to train."""
    model = load_model(tiny_q4_k, n_ctx=64, n_ubatch=32, training=True)

    with pytest.raises(RuntimeError, match="NOT_INITIALIZED"):
        _ffi.check(libs.farm.ll_set_grad_checkpointing(model.ctx, 1), "ll_set_grad_checkpointing")
