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

import numpy as np
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


def test_the_clip_bounds_the_global_norm(tiny_q4_k, tmp_path, load_model, libs) -> None:
    """The clip's threshold is the global norm over every tensor, and it scales a copy.

    The accumulator the optimizer reads back is therefore left unclipped. AdamW's step-1 update is
    nearly invariant to a *uniform* scaling of the gradient — so a single-step assertion cannot read
    the clipped magnitude off the *weights* directly. (The clipped magnitude is read off the
    *norms* directly, by ``ll_grad_norms``, in the sibling test below; the closed-form magnitude
    proof lives in the fork's ``test-opt``.) What is observable here, and asserted, are the two
    properties that pin the clip to the *global* norm:

    * the accumulator still holds the unclipped global gradient after a clipped step — the clip
      nodes scale a copy on the way into AdamW, they do not rewrite the accumulator. A clip applied
      in place would instead collapse this norm toward the clip;
    * a clip set *above* the measured global norm is a no-op to within last-ulp arithmetic
      residue, across **every** trainable tensor — while a *binding* clip moves weights four
      orders of magnitude more. That boundary sits at the global norm, not at any per-tensor
      one, which is the whole point. (Why not bitwise: see the comment at the assertion — the
      answer is different on two hosts, and instructive.)

    The earlier version of this test asserted ``min(raw, clip) == clip`` after establishing
    ``raw > clip`` — a pure tautology that never observed the clip at all.
    """
    import ctypes

    def _run(clip: float):
        adapter_path = tmp_path / f"a-{clip}.gguf"
        create_zero_adapter(tiny_q4_k, adapter_path, r=RANK, seed=7)
        model = load_model(tiny_q4_k, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True)
        model.attach_adapter(adapter_path, scale=1.0)
        model.targets = enumerate_targets(tiny_q4_k)

        with Trainer(libs, model, TrainConfig(lr=1e-3, grad_clip=clip)) as trainer:
            trainer.step(_batch())
            norm = _global_grad_norm(libs, model)
            weights: list[float] = []
            for target in model.targets:
                for is_b in (False, True):
                    n = libs.farm.ll_debug_n_elements(model.ctx, target.name.encode(), is_b)
                    buf = (ctypes.c_float * n)()
                    libs.farm.ll_debug_get_tensor(model.ctx, target.name.encode(), is_b, buf, n)
                    weights.extend(buf)
        return norm, weights

    clip = 0.01
    norm_off, w_off = _run(0.0)
    assert norm_off > clip, (
        f"the unclipped global norm ({norm_off:.4f}) is under the clip; nothing would be clipped"
    )

    # A clipped step must leave the accumulator's global norm untouched: the clip scaled a copy
    # into AdamW, it did not rewrite what the accumulator holds.
    norm_on, _ = _run(clip)
    assert norm_on == pytest.approx(norm_off), (
        f"a clipped step changed the accumulator's global norm ({norm_on:.4f} vs {norm_off:.4f}); "
        f"the clip must scale a copy into AdamW, not rewrite the accumulator"
    )

    # A clip set above the measured GLOBAL norm must be a no-op — but not a BITWISE one, and the
    # two failed attempts at a stronger claim are worth recording. (1) Against the clip-DISABLED
    # run, bitwise fails because grad_clip=0 builds a graph without the clip nodes, and the
    # allocation shift moves SIMD reduction splits by an ulp on macos-14 arm64. (2) Even two
    # DIFFERENT above-norm clips differ bitwise there: the graph does not form a literal
    # factor == 1.0 and scale once — the clip constant participates in the arithmetic
    # (scale-then-divide / FMA), so each clip value leaves its own last-ulp residue. Linux merely
    # cancels by coincidence. ADR-0002's caveat, twice in one test.
    #
    # What IS host-invariant: a non-binding clip perturbs weights only at rounding scale (~1e-7
    # relative), while a BINDING clip moves them at the 1e-3 scale — four orders of magnitude of
    # separation, asserted here and in the binding-clip test below.
    _, w_slack = _run(norm_off + 1.0)
    _, w_slack2 = _run(norm_off + 2.0)
    assert np.allclose(w_slack, w_slack2, rtol=1e-6, atol=1e-9), (
        "two clips both above the measured global norm disagree beyond rounding residue; "
        "the clip is doing real work in a regime where factor must be 1"
    )
    assert np.allclose(w_slack, w_off, rtol=1e-6, atol=1e-9), (
        "a clip set above the measured global norm changed the weights beyond rounding residue; "
        "a non-binding clip must be a no-op up to last-ulp arithmetic jitter"
    )


def test_the_grad_norms_getter_reports_pre_and_post_clip(
    tiny_q4_k, tmp_path, load_model, libs
) -> None:
    """S1-10 item 3: ``ll_grad_norms`` exposes the global grad norm before and after the clip.

    ggml-opt computes both norms IN the backward graph — the pre-clip norm as the sqrt of the
    summed unclipped-gradient squares, the post-clip norm measured off the clipped gradients the
    optimizer actually steps on (not inferred as ``pre * factor``, and emphatically not a host-side
    ``min(pre, clip)``). This reads the two scalars back out.

    The load-bearing check against a tautology is the first assertion: the reported pre-clip norm
    is pinned to an INDEPENDENT host-side norm of the same accumulators (``_global_grad_norm``).
    A getter wired to the wrong tensor, or reporting a fabricated number, fails there. Given a true
    pre, the two clip relationships (``post <= pre``, ``post`` lands on the threshold when it binds)
    then say something real, because ``post`` is its own graph measurement — a broken clamp would
    move it off the threshold rather than have it agree by construction.
    """
    import ctypes

    def run(clip: float) -> tuple[float, float, float]:
        adapter_path = tmp_path / f"norms-{clip}.gguf"
        create_zero_adapter(tiny_q4_k, adapter_path, r=RANK, seed=7)
        model = load_model(tiny_q4_k, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True)
        model.attach_adapter(adapter_path, scale=1.0)
        model.targets = enumerate_targets(tiny_q4_k)

        with Trainer(libs, model, TrainConfig(lr=1e-3, grad_clip=clip)) as trainer:
            trainer.step(_batch())
            host = _global_grad_norm(libs, model)
            pre, post = ctypes.c_float(), ctypes.c_float()
            rc = libs.farm.ll_grad_norms(model.ctx, ctypes.byref(pre), ctypes.byref(post))
        assert rc == 0, f"ll_grad_norms returned {rc}"
        return host, pre.value, post.value

    # A BINDING clip, well under the natural norm, so it actually bites.
    clip = 0.01
    host_b, pre_b, post_b = run(clip)

    # The oracle: the reported pre-clip norm IS the real gradient's global norm.
    assert pre_b == pytest.approx(host_b, rel=1e-5), (
        f"the reported pre-clip norm {pre_b:.6f} disagrees with an independent host-side norm of "
        f"the accumulators {host_b:.6f} — the getter is not reading the gradient it claims to"
    )
    assert pre_b > 0.0 and post_b > 0.0, (
        f"a norm came back zero (pre={pre_b}, post={post_b}); the getter is reporting nothing"
    )
    assert pre_b > clip, f"premise failed: the natural norm {pre_b:.4f} is under the clip {clip}"
    # Clipping cannot RAISE the norm, and a binding clip lands it on the threshold.
    assert post_b <= pre_b + 1e-5, f"post-clip norm {post_b:.6f} exceeds pre-clip {pre_b:.6f}"
    assert post_b < pre_b, f"a binding clip left the norm unchanged ({post_b:.6f} == {pre_b:.6f})"
    assert post_b == pytest.approx(clip, rel=1e-4), (
        f"a binding clip should rescale the global norm to the threshold {clip}, got {post_b:.6f}"
    )

    # A SLACK clip, above the natural norm: nothing is clipped, so post == pre.
    host_s, pre_s, post_s = run(pre_b + 1.0)
    assert pre_s == pytest.approx(host_s, rel=1e-5)
    assert post_s == pytest.approx(pre_s, rel=1e-6), (
        f"a non-binding clip changed the norm ({post_s:.6f} vs {pre_s:.6f}); it must be a no-op"
    )
    assert post_s <= (pre_s + 1.0) + 1e-5


def test_the_grad_norms_are_nan_without_a_clip(tiny_q4_k, tmp_path, load_model, libs) -> None:
    """With ``grad_clip == 0`` ggml-opt builds no norm node, so the getter reports NaN.

    Not a stale or fabricated value.
    This is the contract that lets a caller distinguish "the clip was off" from "the norm was
    zero": an unclipped run's graph is unchanged by the feature, so there is genuinely nothing to
    report, and the getter says so rather than inventing a number.
    """
    import ctypes

    adapter_path = tmp_path / "noclip.gguf"
    create_zero_adapter(tiny_q4_k, adapter_path, r=RANK, seed=7)
    model = load_model(tiny_q4_k, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True)
    model.attach_adapter(adapter_path, scale=1.0)
    model.targets = enumerate_targets(tiny_q4_k)

    with Trainer(libs, model, TrainConfig(lr=1e-3, grad_clip=0.0)) as trainer:
        trainer.step(_batch())
        pre, post = ctypes.c_float(), ctypes.c_float()
        rc = libs.farm.ll_grad_norms(model.ctx, ctypes.byref(pre), ctypes.byref(post))

    assert rc == 0, f"ll_grad_norms returned {rc}"
    assert math.isnan(pre.value) and math.isnan(post.value), (
        f"an unclipped step should report NaN norms (no norm node is built), got "
        f"pre={pre.value}, post={post.value}"
    )


def test_clipping_changes_the_step_and_a_clip_above_the_norm_does_not(
    tiny_q4_k, tmp_path, load_model, libs
) -> None:
    """The observable consequence: the weights land somewhere different — and only when they should.

    A clip *above* the gradient's norm must be a no-op. That is the half of the behaviour a "the
    numbers got smaller" test never checks, and the half that a clip implemented as an
    unconditional rescale would fail.

    A no-op up to rounding residue, though, and **not** bit for bit — the same lesson the sibling
    test above records twice and this one was left out of by S1-49. ``grad_clip=0`` builds a graph
    with no clip nodes at all, so comparing an above-norm clip against it is a comparison across
    two DIFFERENT graphs, and the allocation shift moves SIMD reduction splits by an ulp on
    macos-14 arm64 (ADR-0002's caveat in miniature). Even the clip constant itself participates in
    the arithmetic rather than forming a literal ``factor == 1.0``, so each clip value leaves its
    own last-ulp residue; Linux merely cancels by coincidence.

    The tolerance is therefore the sibling's measured one (~1e-7 relative rounding residue), and it
    stays discriminating because the two claims are asserted against each other: a *binding* clip
    has to move these same weights by more than that same tolerance. Four steps at lr=1e-2 put that
    separation four orders of magnitude apart.
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

    # The two halves, at the SAME tolerance, which is what keeps either of them meaningful: a
    # binding clip must move the weights further than the rounding residue an above-norm one is
    # allowed, or the no-op claim below would be satisfied by a clip that does nothing ever.
    assert not np.allclose(clipped, unclipped, rtol=1e-6, atol=1e-9), (
        "clipping to 0.001 moved the weights no further than arithmetic rounding would; the clip "
        "is not binding, and the no-op assertion below would then be vacuous"
    )

    assert np.allclose(slack, unclipped, rtol=1e-6, atol=1e-9), (
        "a clip far above the gradient's norm changed the weights beyond rounding residue. It "
        "must be the identity: min(1, clip/norm) is 1 there, so nothing should be rescaled."
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
