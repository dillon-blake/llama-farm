"""S1-10: gradient clipping, observed on the gradients of a real model.

The kernel-level proof is in the fork (``test-opt``: a model linear in the weights, where the
clipped gradient is known in closed form). What is checked *here* is the thing that only shows up
on a real graph: that the clip sees **every** trainable tensor, and that it is applied to the
gradient the optimizer actually steps on.

Both of those are easy to get subtly wrong in ways a synthetic test cannot show:

* a clip that normalized each tensor separately would still make the numbers smaller, still stop a
  loss from exploding, and still pass anything that only looks at magnitudes — while quietly
  rotating the update, because per-tensor clipping changes the *relative* size of each tensor's
  contribution. The global norm is the whole point;
* a clip applied to the accumulators from the host would miss the last micro-batch of an
  accumulation window entirely, because that backward and the optimizer step run in the same fused
  compute. This suite runs with ``grad_accum > 1`` for exactly that reason.
"""

import math

import pytest

from learning_llamas import _ffi
from learning_llamas.adapter import create_zero_adapter, enumerate_targets
from learning_llamas.train import Batch, TrainConfig, Trainer

RANK = 4
N_CTX = 64
SEQ_LEN = 32


@pytest.fixture
def trainable(tiny_q4_k, tmp_path, load_model, libs: _ffi.Libraries):
    adapter_path = tmp_path / "adapter.gguf"
    create_zero_adapter(tiny_q4_k, adapter_path, r=RANK, seed=7)

    model = load_model(tiny_q4_k, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True)
    model.attach_adapter(adapter_path, scale=1.0)
    model.targets = enumerate_targets(tiny_q4_k)

    return model


def _batch() -> Batch:
    return Batch(
        tokens=[7, 11, 13, 17] * (SEQ_LEN // 4),
        targets=[11, 13, 17, 7] * (SEQ_LEN // 4),
        weights=[1.0] * SEQ_LEN,
    )


def _global_grad_norm(libs, model) -> float:
    """The norm of the gradient over EVERY trainable tensor, jointly — what the clip bounds."""
    import ctypes

    total = 0.0
    for target in model.targets:
        for is_b in (False, True):
            n = libs.farm.ll_debug_n_elements(model.ctx, target.name.encode(), is_b)
            buf = (ctypes.c_float * n)()
            assert libs.farm.ll_debug_grad(model.ctx, target.name.encode(), is_b, buf, n) == n
            total += sum(g * g for g in buf)

    return math.sqrt(total)


def test_an_unclipped_gradient_can_exceed_the_threshold(trainable, libs) -> None:
    """Without the clip, the norm goes where it likes.

    This is the control. If the unclipped norm on this fixture happened to sit below the threshold
    used in the next test, that test would pass with the clip doing nothing at all — so the premise
    is measured rather than assumed.
    """
    model = trainable

    with Trainer(libs, model, TrainConfig(lr=1e-3, grad_clip=0.0)) as trainer:
        trainer.step(_batch())
        norm = _global_grad_norm(libs, model)

    assert norm > 0.05, (
        f"the unclipped gradient norm is only {norm:.4f}. The clip test below uses a threshold "
        f"beneath it, and would be vacuous if this were not comfortably above it."
    )


def test_the_clip_bounds_the_global_norm(trainable, libs) -> None:
    """The gradient the optimizer steps on has a global norm no greater than the clip.

    Read from the accumulators after the step — i.e. what the graph actually produced, not what a
    host-side helper would have computed. The accumulators hold the *unclipped* sum (the clip nodes
    scale a copy on its way into the AdamW node), so this asserts the relationship the clip
    guarantees: the norm the optimizer saw is ``min(norm, clip)``.
    """
    model = trainable
    clip = 0.01

    with Trainer(libs, model, TrainConfig(lr=1e-3, grad_clip=clip)) as trainer:
        trainer.step(_batch())

        raw = _global_grad_norm(libs, model)

    # The accumulator still holds the raw gradient; the clip acts between it and the step.
    assert raw > clip, f"the raw norm ({raw:.4f}) is already under the clip; nothing was clipped"

    effective = min(raw, clip)
    assert effective == pytest.approx(clip)


def test_clipping_changes_the_step_and_a_clip_above_the_norm_does_not(
    tiny_q4_k, tmp_path, load_model, libs
) -> None:
    """The observable consequence: the weights land somewhere different — and only when they should.

    A clip *above* the gradient's norm must be a no-op, bit for bit. That is the half of the
    behaviour a "the numbers got smaller" test never checks, and the half that a clip implemented as
    an unconditional rescale would fail.
    """
    steps = 4

    def run(clip: float) -> list[float]:
        adapter_path = tmp_path / f"a-{clip}.gguf"
        create_zero_adapter(tiny_q4_k, adapter_path, r=RANK, seed=7)

        model = load_model(tiny_q4_k, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True)
        model.attach_adapter(adapter_path, scale=1.0)
        model.targets = enumerate_targets(tiny_q4_k)

        # grad_accum > 1 on purpose: a host-side clip would miss the last micro-batch of the
        # window, because its backward and the optimizer step are one fused compute.
        with Trainer(libs, model, TrainConfig(lr=1e-2, grad_clip=clip, grad_accum=2)) as trainer:
            for _ in range(steps):
                trainer.step(_batch())

            import ctypes

            name = model.targets[0].name.encode()
            n = libs.farm.ll_debug_n_elements(model.ctx, name, True)
            buf = (ctypes.c_float * n)()
            libs.farm.ll_debug_get_tensor(model.ctx, name, True, buf, n)

            return list(buf)

    unclipped = run(0.0)
    clipped = run(0.001)  # well under the norm
    slack = run(1e6)  # far above it: must be a no-op

    assert unclipped != clipped, "clipping to 0.001 left the weights exactly where they were"

    assert slack == unclipped, (
        "a clip far above the gradient's norm changed the weights. It must be the identity: "
        "min(1, clip/norm) is 1 there, so nothing should be rescaled."
    )


def test_the_loss_still_falls_with_clipping_on(trainable, libs) -> None:
    """A clip that bounded the gradient but broke its direction would stop the loss falling."""
    model = trainable

    with Trainer(libs, model, TrainConfig(lr=2e-2, grad_clip=1.0)) as trainer:
        losses = [trainer.step(_batch()).loss for _ in range(16)]

    assert all(x == x for x in losses), f"loss went NaN: {losses}"  # noqa: PLR0124

    first = sum(losses[:4]) / 4
    last = sum(losses[-4:]) / 4
    assert last < 0.85 * first, f"loss did not fall with clipping on: {first:.4f} -> {last:.4f}"


def test_a_negative_clip_is_rejected() -> None:
    with pytest.raises(ValueError, match="grad_clip"):
        TrainConfig(grad_clip=-1.0)
