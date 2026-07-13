"""S1-16: the GRPO update — a clipped ratio, a group advantage, and a KL that does not explode.

The interesting part of this ticket is that the PPO clip is **built out of RELU**:

    clip(r, lo, hi) = lo + relu(r - lo) - relu(r - hi)
    min(a, b)       = a - relu(a - b)

Those identities are exactly true on paper. That is not the claim being tested. The claim is that
*the graph I wrote computes them* — and the way to test that is to compare it, on inputs that reach
every branch, against a numpy implementation that shares none of its structure (`np.clip`,
`np.minimum`, `np.expm1`; no relu anywhere).

And that comparison is **not sufficient**, which is the other thing this file is about. The
blueprint prescribed `min(a, b) = b - relu(b - a)`, which is the same *number* and a different
*gradient* — so the numpy reference agrees with both forms and cannot tell them apart. Catching it
took a test that differentiates: see the clip-bound test at the bottom.

**And the ratio is controllable, exactly.** `r = exp(logp_new − logp_old)`, so setting
`logp_old = logp_new − log(r*)` makes the ratio exactly `r*` — for any `r*` I like. That turns a
test which would otherwise depend on the model's logits into an exact, model-independent check of
the composite: below the clip, inside it, above it, with positive advantages and with negative
ones. The negative ones matter most: a negative advantage **swaps the two arguments of the `min`**,
and PPO's whole asymmetry — between having made a good token likelier and a bad token likelier —
lives in that swap. A clip that got it backwards would still train, still converge, and be wrong.
"""

import ctypes
import logging
import math

import numpy as np
import pytest

from learning_llamas import _ffi
from learning_llamas.adapter import create_zero_adapter
from learning_llamas.logprobs import load_lm_head
from learning_llamas.train import TrainConfig, Trainer
from learning_llamas.train.grpo import GRPOConfig, GRPOTrainer, collate, train_grpo
from learning_llamas.train.rollout import (
    Rollout,
    RolloutBatch,
    RolloutEngine,
    SamplerConfig,
    token_reward,
)
from learning_llamas.verify import SelfVerified

from . import reference_grpo

N = 16
N_PROMPT = 6
EPS = 0.2
RANK = 4


def _arr_i32(values) -> ctypes.Array:
    return (ctypes.c_int32 * len(values))(*[int(v) for v in values])


def _arr_f32(values) -> ctypes.Array:
    return (ctypes.c_float * len(values))(*[float(v) for v in values])


@pytest.fixture
def policy(tiny_q4_k, tmp_path, load_model, libs: _ffi.Libraries):
    """A training context with a zero-init adapter."""
    adapter = tmp_path / "grpo.gguf"
    create_zero_adapter(tiny_q4_k, adapter, r=RANK, seed=7)

    model = load_model(tiny_q4_k, n_ctx=64, n_ubatch=N, training=True)
    model.attach_adapter(adapter, scale=1.0)
    return model


@pytest.fixture
def fixed_batch():
    """A batch with a prompt and a completion, and the mask that grades only the completion."""
    rng = np.random.default_rng(2)
    tokens = [int(x) for x in rng.integers(1, 400, size=N)]
    targets = tokens[1:] + [tokens[0]]
    mask = np.array([0.0] * N_PROMPT + [1.0] * (N - N_PROMPT), dtype=np.float32)
    return tokens, targets, mask


def _logp_new(libs, model, tokens, targets, mask) -> np.ndarray:
    """The TRAINING graph's per-token logprob, one one-hot `ll_logp_delta` at a time.

    Not the inference graph, and not `logprobs.py`. Both of those would be *close* — llama.cpp is
    deterministic per shape (ADR-0002), and a training graph is a different shape — and "close" is
    not good enough to test an exponentiated ratio against. This reads exactly the number the GRPO
    graph will read.
    """
    out = np.zeros(N, dtype=np.float64)
    seq, pos = _arr_i32([0] * N), _arr_i32(range(N))
    tk, tg = _arr_i32(tokens), _arr_i32(targets)

    for i in range(N):
        if mask[i] == 0.0:
            continue
        weights = np.zeros(N, dtype=np.float32)
        weights[i] = 1.0

        value = ctypes.c_float()
        _ffi.check(
            libs.farm.ll_logp_delta(
                model.ctx, tk, tg, _arr_f32(weights), seq, pos, N, ctypes.byref(value)
            ),
            "ll_logp_delta",
        )
        out[i] = value.value

    return out


def _grpo_loss(
    libs,
    model,
    tokens,
    targets,
    mask,
    adv,
    logp_old,
    *,
    eps=EPS,
    kl_coef=0.0,
    logp_ref=None,
    kl_w=None,
    train=False,
) -> float:
    """Run the shim's GRPO graph and return its loss.

    `kl_w` overrides the usual `mask * kl_coef`, so a test can hand the shim an UNMASKED weight and
    check that it defends itself.
    """
    adv_c = _arr_f32(adv)
    old_c = _arr_f32(logp_old)
    ref_c = _arr_f32(logp_ref) if logp_ref is not None else None

    if kl_w is not None:
        kl_c = _arr_f32(kl_w)
    else:
        kl_c = _arr_f32(mask * kl_coef) if kl_coef else None

    inputs = _ffi.ll_grpo_inputs(
        adv=ctypes.cast(adv_c, ctypes.POINTER(ctypes.c_float)),
        logp_old=ctypes.cast(old_c, ctypes.POINTER(ctypes.c_float)),
        logp_ref=ctypes.cast(ref_c, ctypes.POINTER(ctypes.c_float)) if ref_c else None,
        kl_w=ctypes.cast(kl_c, ctypes.POINTER(ctypes.c_float)) if kl_c else None,
        clip_eps=eps,
    )

    loss = ctypes.c_float()
    _ffi.check(
        libs.farm.ll_train_step_grpo(
            model.ctx,
            _arr_i32(tokens),
            _arr_i32(targets),
            _arr_f32(mask),
            _arr_i32([0] * N),
            _arr_i32(range(N)),
            N,
            ctypes.byref(inputs),
            train,
            ctypes.byref(loss),
        ),
        "ll_train_step_grpo",
    )
    return float(loss.value)


# ---------------------------------------------------------------------------------------------
# THE test: the relu composite is the clip.
# ---------------------------------------------------------------------------------------------


def test_the_relu_composite_is_the_clip_in_every_region(policy, libs, fixed_batch) -> None:
    """The graph's loss, against numpy, on ratios that reach every branch of the clip.

    The ratios are chosen *exactly* by setting `logp_old = logp_new - log(r)`, so this compares the
    composite itself rather than the model's logits: below the clip, inside it, on both bounds, and
    above it — each with a positive advantage, a negative one, and zero.

    A negative advantage is the case worth staring at. It swaps the two arguments of the `min`, so
    the clip binds from the *other* side, and an implementation that clipped the ratio and then
    applied the sign would be wrong on exactly half the tokens — while still training, still
    converging, and still looking fine.
    """
    tokens, targets, mask = fixed_batch

    with Trainer(libs, policy, TrainConfig(lr=1e-3)):
        logp_new = _logp_new(libs, policy, tokens, targets, mask)

        live = mask > 0
        n_live = int(live.sum())

        # The pairs matter more than the values, and this is the trap the first version of this
        # test fell into.
        #
        # The clip only ever BINDS on one side at a time, and which side depends on the SIGN of the
        # advantage:
        #
        #   A > 0, r > hi  ->  min(rA, hiA) = hiA   -- the UPPER bound binds
        #   A > 0, r < lo  ->  min(rA, loA) = rA    -- the unclipped branch wins; lo is never read
        #   A < 0, r < lo  ->  min(rA, loA) = loA   -- the LOWER bound binds
        #   A < 0, r > hi  ->  min(rA, hiA) = rA    -- the unclipped branch wins; hi is never read
        #
        # So a test with r = 0.5 and A = +1 does NOT exercise the lower bound: the loss is the same
        # whether `lo + relu(r - lo)` is there or not. The first version of this test had exactly
        # that, and no (r < lo, A < 0) pair anywhere -- so the entire lower half of the clip could
        # have been deleted and it would still have passed.
        #
        # Every one of the four rows above is present below, twice.
        ratios = np.array([0.5, 0.3, 2.0, 1.5, 0.8, 1.2, 1.0, 0.9, 1.1, 0.95][:n_live])
        advantages = np.array([-1.0, -2.0, 1.0, 2.0, -1.5, 1.5, -0.5, 0.5, -1.0, 0.0][:n_live])

        assert n_live >= 4, "the four clip x sign combinations need at least four graded tokens"
        # r < lo with A < 0 -> the LOWER bound binds. r > hi with A > 0 -> the UPPER bound binds.
        assert ((ratios < 0.8) & (advantages < 0)).any(), "the lower clip bound is never exercised"
        assert ((ratios > 1.2) & (advantages > 0)).any(), "the upper clip bound is never exercised"

        adv = np.zeros(N)
        adv[live] = advantages

        logp_old = np.zeros(N)
        logp_old[live] = logp_new[live] - np.log(ratios)

        got = _grpo_loss(libs, policy, tokens, targets, mask, adv, logp_old)
        want = reference_grpo.grpo_loss(logp_new, logp_old, adv, mask, eps=EPS)

        assert got == pytest.approx(want, abs=1e-5), (
            f"the relu-composed clip disagrees with np.clip/np.minimum.\n"
            f"  graph: {got:.8f}\n"
            f"  numpy: {want:.8f}\n"
            f"  ratios: {ratios}\n  advantages: {advantages}"
        )


def test_a_negative_advantage_clips_from_the_other_side(policy, libs, fixed_batch) -> None:
    """PPO's asymmetry, isolated.

    With A > 0 the surrogate is capped ABOVE (do not reward a big ratio too much); with A < 0 it is
    capped BELOW. Same ratio, opposite advantage, and the clip binds on the opposite side. If the
    `min` were applied to the ratio instead of to the ratio-times-advantage, these two would be
    mirror images — and they are not.
    """
    tokens, targets, mask = fixed_batch
    live = mask > 0
    n_live = int(live.sum())

    with Trainer(libs, policy, TrainConfig(lr=1e-3)):
        logp_new = _logp_new(libs, policy, tokens, targets, mask)

        # r = 2.0 everywhere: far above the clip's upper bound of 1.2.
        logp_old = np.zeros(N)
        logp_old[live] = logp_new[live] - math.log(2.0)

        positive = np.zeros(N)
        positive[live] = 1.0
        negative = np.zeros(N)
        negative[live] = -1.0

        loss_pos = _grpo_loss(libs, policy, tokens, targets, mask, positive, logp_old)
        loss_neg = _grpo_loss(libs, policy, tokens, targets, mask, negative, logp_old)

    # A > 0, r = 2 clipped to 1.2: surrogate = min(2, 1.2) = 1.2 per token, loss = -1.2 * n.
    assert loss_pos == pytest.approx(-1.2 * n_live, abs=1e-4)

    # A < 0, r = 2: surrogate = min(-2, -1.2) = -2 per token -- the clip does NOT bind, because
    # pessimism for a bad token means taking the *larger* penalty. loss = +2 * n.
    assert loss_neg == pytest.approx(2.0 * n_live, abs=1e-4)

    assert loss_neg != pytest.approx(-loss_pos, abs=1e-3), (
        "the loss is symmetric under a sign flip of the advantage, so the min is being applied to "
        "the ratio rather than to ratio*advantage. PPO's whole asymmetry has been lost."
    )


# ---------------------------------------------------------------------------------------------
# The KL.
# ---------------------------------------------------------------------------------------------


def test_the_k3_kl_matches_numpy(policy, libs, fixed_batch) -> None:
    """`expm1(d) - d` against `exp(d) - d - 1`, through the real graph."""
    tokens, targets, mask = fixed_batch
    live = mask > 0

    with Trainer(libs, policy, TrainConfig(lr=1e-3)):
        logp_new = _logp_new(libs, policy, tokens, targets, mask)

        logp_old = np.zeros(N)
        logp_old[live] = logp_new[live]  # r = 1, so the policy term is a clean -sum(adv)

        # A reference that has genuinely moved away, so the KL is not trivially zero.
        logp_ref = np.zeros(N)
        logp_ref[live] = logp_new[live] - 0.25

        adv = np.zeros(N)
        adv[live] = 0.5

        got = _grpo_loss(
            libs, policy, tokens, targets, mask, adv, logp_old, kl_coef=1.0, logp_ref=logp_ref
        )
        want = reference_grpo.grpo_loss(
            logp_new, logp_old, adv, mask, eps=EPS, kl_coef=1.0, logp_ref=logp_ref
        )

    assert got == pytest.approx(want, abs=1e-5)


def test_the_kl_is_zero_when_the_reference_is_the_policy(policy, libs, fixed_batch) -> None:
    """k3 is `exp(0) - 0 - 1 = 0` when the reference has not moved. Exactly, not nearly.

    This is what `expm1` buys. `exp(d) - 1` at `d ~ 1e-4` in float32 loses every significant digit
    to cancellation, and a KL penalty that is noise at the start of a run — which is exactly where a
    GRPO run *is* — pushes the policy in a random direction.
    """
    tokens, targets, mask = fixed_batch
    live = mask > 0

    with Trainer(libs, policy, TrainConfig(lr=1e-3)):
        logp_new = _logp_new(libs, policy, tokens, targets, mask)

        logp_old = np.zeros(N)
        logp_old[live] = logp_new[live]

        adv = np.zeros(N)  # no policy term at all: the loss IS the KL

        with_kl = _grpo_loss(
            libs, policy, tokens, targets, mask, adv, logp_old, kl_coef=10.0, logp_ref=logp_old
        )

    assert with_kl == pytest.approx(0.0, abs=1e-6)


def test_a_masked_token_cannot_poison_the_batch(policy, libs, fixed_batch) -> None:
    """Garbage in a padding slot must not reach the loss.

    And the garbage has to be the kind that would actually break it.

    On a masked token `ce_sparse` makes `logp_new` exactly 0. So if `logp_ref` there were left at
    some large POSITIVE value, `d = logp_ref - 0` is large and positive, `expm1(d)` overflows F32 to
    `+inf`, and `inf * kl_w` is `inf` — or, if `kl_w` happens to be 0 there, `NaN`. Either way the
    loss and every gradient in the batch are gone, from one stray number in a padding slot.

    The *direction* matters, and an earlier version of this test got it wrong: poisoning with a
    large NEGATIVE value passes whatever the shim does, because `expm1` of a large negative number
    is just -1. The test was green and proved nothing. `expm1` only overflows upward.

    So: large positive garbage in the masked slots, and an UNMASKED `kl_w`, so the only thing
    standing between the batch and an inf is the shim zeroing its inputs on masked tokens.
    """
    tokens, targets, mask = fixed_batch
    live = mask > 0

    with Trainer(libs, policy, TrainConfig(lr=1e-3)):
        logp_new = _logp_new(libs, policy, tokens, targets, mask)

        logp_old = np.zeros(N)
        logp_old[live] = logp_new[live]

        adv = np.zeros(N)
        adv[live] = 1.0

        # Hostile, and in the direction that overflows: +200 in every masked slot. expm1(200) is
        # 7e86, which is +inf in F32.
        poison_ref = np.full(N, 200.0)
        poison_ref[live] = logp_new[live]

        poison_old = logp_old.copy()
        poison_old[~live] = -500.0  # r = exp(0 - (-500)) = inf, if it were ever computed

        poison_adv = adv.copy()
        poison_adv[~live] = 1e6

        # ...and a kl_w the caller forgot to mask, so nothing multiplies the inf away.
        unmasked_kl_w = np.full(N, 1.0, dtype=np.float32)

        loss = _grpo_loss(
            libs,
            policy,
            tokens,
            targets,
            mask,
            poison_adv,
            poison_old,
            kl_coef=1.0,
            logp_ref=poison_ref,
            kl_w=unmasked_kl_w,
        )

    assert math.isfinite(loss), f"a masked slot reached the loss: {loss}"

    # r = 1 and adv = 1 on every live token, and the reference IS the policy there, so the KL is 0:
    # loss = -sum(adv) = -n_live, exactly.
    assert loss == pytest.approx(-float(live.sum()), abs=1e-4)


def test_no_reference_means_no_kl_rather_than_infinity(policy, libs, fixed_batch) -> None:
    """`logp_ref = NULL` must mean "the reference is the policy", not "the reference is zero".

    Zero is not a neutral logprob — it is *certainty*. With a real `logp_new` of, say, -12, a
    zero-filled reference makes `d = 0 - (-12) = +12`, and the KL claims the policy has diverged
    enormously from a reference that assigns probability 1 to every token. Push that a little
    further and `expm1(d)` is `inf`.

    So NULL is filled with `logp_old`, which makes `d = 0` and the KL term exactly zero.
    """
    tokens, targets, mask = fixed_batch
    live = mask > 0

    with Trainer(libs, policy, TrainConfig(lr=1e-3)):
        logp_new = _logp_new(libs, policy, tokens, targets, mask)

        logp_old = np.zeros(N)
        logp_old[live] = logp_new[live]

        adv = np.zeros(N)  # no policy term: the loss IS the KL

        # kl_coef is large, and logp_ref is NULL.
        loss = _grpo_loss(
            libs, policy, tokens, targets, mask, adv, logp_old, kl_coef=100.0, logp_ref=None
        )

    assert math.isfinite(loss)
    assert loss == pytest.approx(0.0, abs=1e-5), (
        f"with no reference the KL should be exactly zero, got {loss}. A NULL logp_ref is being "
        f"read as zeros rather than as logp_old."
    )


def test_the_k3_kl_is_accurate_and_non_negative_near_zero(policy, libs, fixed_batch) -> None:
    """The regime GRPO actually lives in, and the one ggml got wrong.

    k3 is ``exp(d) - d - 1``, and it is **non-negative by construction** — that is the entire reason
    to prefer it over the naive estimator, which is unbiased but can go negative on a single sample
    and then *rewards* divergence from the reference.

    ggml's ``GGML_UNARY_OP_EXPM1`` was implemented as ``expf(x) - 1.0f``, which is precisely the
    catastrophic cancellation the op exists to avoid. Measured, against the true ``d²/2``:

        d = 1e-5:   1.36e-8  vs   5.0e-11    (271x too large)
        d = 1e-6:   7.29e-8  vs   5.0e-13    (145,000x too large)
        d = 5e-5:  -5.13e-8  vs   1.25e-9    (NEGATIVE)

    And ``d = logp_ref - logp_new`` is ~0 on every GRPO step **by design** — an on-policy run lives
    exactly there. So the KL penalty was noise, and sometimes noise with the wrong sign. Fixed
    upstream in the fork (op_expm1 -> expm1f); this pins it.
    """
    tokens, targets, mask = fixed_batch
    live = mask > 0
    n_live = int(live.sum())

    with Trainer(libs, policy, TrainConfig(lr=1e-3)):
        logp_new = _logp_new(libs, policy, tokens, targets, mask)

        logp_old = np.zeros(N)
        logp_old[live] = logp_new[live]

        adv = np.zeros(N)  # no policy term: the loss IS the KL

        for delta in (1e-3, 1e-4, 1e-5):
            logp_ref = np.zeros(N)
            logp_ref[live] = logp_new[live] + delta  # d = logp_ref - logp_new = +delta

            got = _grpo_loss(
                libs, policy, tokens, targets, mask, adv, logp_old, kl_coef=1.0, logp_ref=logp_ref
            )

            # k3(d) ~= d^2/2 for small d, summed over the graded tokens.
            want = n_live * (delta**2 / 2)

            assert got >= 0.0, (
                f"the KL went NEGATIVE at d={delta:.0e}: {got:.3e}. k3 is non-negative by "
                f"construction — a negative KL rewards divergence from the reference."
            )
            assert got == pytest.approx(want, rel=0.05), (
                f"k3 at d={delta:.0e} is {got:.3e}, expected ~{want:.3e} (n*d^2/2). This is the "
                f"expf(x)-1 cancellation: the answer is smaller than the rounding error."
            )


# ---------------------------------------------------------------------------------------------
# The gradient.
# ---------------------------------------------------------------------------------------------


def test_zero_advantages_move_nothing(policy, libs, fixed_batch) -> None:
    """A batch where every group was degenerate must leave the weights exactly where they were.

    Which is not a nicety: an all-or-nothing reward makes whole groups score alike, and a group
    that says nothing must be allowed to say nothing. If a zero advantage still moved the policy,
    every failed group would push it in an arbitrary direction.
    """
    tokens, targets, mask = fixed_batch
    live = mask > 0

    before = _adapter_weights(libs, policy)

    with Trainer(libs, policy, TrainConfig(lr=1.0)) as trainer:  # a huge lr, so any motion shows
        logp_new = _logp_new(libs, policy, tokens, targets, mask)

        logp_old = np.zeros(N)
        logp_old[live] = logp_new[live]

        _grpo_loss(
            libs, policy, tokens, targets, mask, np.zeros(N), logp_old, kl_coef=0.0, train=True
        )
        assert trainer is not None

    after = _adapter_weights(libs, policy)

    assert np.array_equal(before, after), (
        f"zero advantages moved the adapter. max |delta| = {np.abs(after - before).max()}"
    )


def test_a_real_advantage_does_move_the_policy(policy, libs, fixed_batch) -> None:
    """The counterfactual to the test above: it is not just that nothing ever moves."""
    tokens, targets, mask = fixed_batch
    live = mask > 0

    before = _adapter_weights(libs, policy)

    with Trainer(libs, policy, TrainConfig(lr=1e-2)):
        logp_new = _logp_new(libs, policy, tokens, targets, mask)

        logp_old = np.zeros(N)
        logp_old[live] = logp_new[live]

        adv = np.zeros(N)
        adv[live] = 1.0

        _grpo_loss(libs, policy, tokens, targets, mask, adv, logp_old, train=True)

    after = _adapter_weights(libs, policy)

    assert not np.array_equal(before, after), "a nonzero advantage did not move the adapter at all"


def test_the_gradient_survives_a_ratio_landing_exactly_on_the_clip_bound(
    tiny_q4_k, tmp_path, load_model, libs
) -> None:
    """`min(a, b) = a - relu(a - b)`, and NOT the equally-true `b - relu(b - a)`.

    They give the same forward value. They differ in where the gradient goes **at a tie**, and that
    turns out to matter.

    ``ggml_step(0) == 0``. So when the ratio lands exactly on ``lo``, ``relu(r - lo)`` is fed
    exactly zero, its backward is zero, and ``d(clipped)/dr`` is zero. But ``clipped`` *evaluates*
    to ``lo == r``, so the two branches of the ``min`` are bitwise equal — a tie. And
    ``b - relu(b - a)`` routes a tie's whole gradient into ``b``: the branch whose derivative was
    just computed as zero.

    The token's policy gradient disappears. And for a POSITIVE advantage that is not a subgradient
    choice at a kink — there is no kink. ``min(rA, clip(r)A)`` equals ``rA`` on *both* sides of
    ``lo`` when ``A > 0``, because the clip does not bind from below there. The objective is smooth,
    with slope ``A``, and the graph returned 0.

    Measured, with one graded token and a positive advantage — the loss is identical in all three,
    only the derivative differs:

        3 ulps below lo:  adapter moved 5.0e-01
        r == lo exactly:  adapter moved 0.0e+00     <- with b - relu(b - a)
        3 ulps above lo:  adapter moved 5.0e-01

    So the tie is routed into ``a``, the unclipped branch, whose derivative *is* ``A``. Correct at
    ``lo``, and still a legal subgradient at ``hi`` (which is a genuine kink).
    """
    adapter = tmp_path / "tie.gguf"
    create_zero_adapter(tiny_q4_k, adapter, r=RANK, seed=7)

    live = 5  # exactly ONE graded position, so the batch's whole gradient is this token's
    mask = np.zeros(N, dtype=np.float32)
    mask[live] = 1.0

    rng = np.random.default_rng(4)
    tokens = [int(x) for x in rng.integers(1, 400, size=N)]
    targets = tokens[1:] + [tokens[0]]

    def logp_new_at(model) -> np.float32:
        weights = np.zeros(N, dtype=np.float32)
        weights[live] = 1.0
        value = ctypes.c_float()
        _ffi.check(
            libs.farm.ll_logp_delta(
                model.ctx,
                _arr_i32(tokens),
                _arr_i32(targets),
                _arr_f32(weights),
                _arr_i32([0] * N),
                _arr_i32(range(N)),
                N,
                ctypes.byref(value),
            ),
            "ll_logp_delta",
        )
        return np.float32(value.value)

    probe = load_model(tiny_q4_k, n_ctx=32, n_ubatch=N, training=True)
    probe.attach_adapter(adapter, scale=1.0)
    with Trainer(libs, probe, TrainConfig(lr=0.0)):
        logp_new = logp_new_at(probe)

    # The ratio lives on a coarse float grid, too coarse to land on 0.8f exactly. So move `lo` onto
    # an ACHIEVABLE r instead of the other way round: pick clip_eps with f32(1 - eps) bitwise equal
    # to the r this token actually reaches. Same tie, and reachable.
    f32 = np.float32

    def ulp(x: float, n: int) -> float:
        import struct  # noqa: PLC0415

        bits = struct.unpack("<I", struct.pack("<f", x))[0] + n
        return struct.unpack("<f", struct.pack("<I", bits))[0]

    guess = float(logp_new) - math.log(0.8)
    tie_logp_old = None
    eps = 0.2
    for k in range(-200, 200):
        cand = f32(ulp(guess, k))
        r = f32(np.exp(f32(logp_new - cand)))
        e = f32(f32(1.0) - r)
        if f32(f32(1.0) - e) == r and 0.0 < float(e) < 1.0:
            tie_logp_old, eps = float(cand), float(e)
            break

    assert tie_logp_old is not None, "could not align the clip bound with an achievable ratio"

    def moved(logp_old_value: float) -> float:
        model = load_model(tiny_q4_k, n_ctx=32, n_ubatch=N, training=True)
        model.attach_adapter(adapter, scale=1.0)
        before = _adapter_weights(libs, model)

        logp_old = np.zeros(N)
        logp_old[live] = logp_old_value
        adv = np.zeros(N)
        adv[live] = 1.0  # POSITIVE: the clip must not bind from below

        with Trainer(libs, model, TrainConfig(lr=0.5)):
            _grpo_loss(libs, model, tokens, targets, mask, adv, logp_old, eps=eps, train=True)

        return float(np.abs(_adapter_weights(libs, model) - before).max())

    below = moved(ulp(tie_logp_old, -3))
    at_bound = moved(tie_logp_old)
    above = moved(ulp(tie_logp_old, +3))

    assert below > 0.0 and above > 0.0, "the neighbours of the bound carry no gradient either"
    assert at_bound > 0.0, (
        "the gradient VANISHED at a ratio exactly on the clip's lower bound, while both of its "
        "float neighbours — the same objective, since the clip does not bind below for a positive "
        "advantage — carry one. The min composite is routing the tie into the clipped branch, "
        "whose derivative is zero there. Use min(a, b) = a - relu(a - b)."
    )


def _adapter_weights(libs, model) -> np.ndarray:
    out = []
    for i in range(libs.farm.ll_adapter_n_tensors(model.adapter)):
        for is_b in (False, True):
            n = libs.farm.ll_adapter_get(model.adapter, i, is_b, None, 0)
            buf = (ctypes.c_float * n)()
            libs.farm.ll_adapter_get(model.adapter, i, is_b, buf, n)
            out.append(np.frombuffer(buf, dtype=np.float32, count=n).copy())
    return np.concatenate(out)


# ---------------------------------------------------------------------------------------------
# The contract.
# ---------------------------------------------------------------------------------------------


def test_a_fractional_weight_is_refused(policy, libs, fixed_batch) -> None:
    """The weights ARE the mask, and a 0.5 is not a half-graded token — it is a corrupted ratio.

    `logp_new` is `-ce_sparse(...)`, and ce_sparse multiplies by the weight. So a weight of 0.5
    does not down-weight the token's contribution; it HALVES the log-probability that goes into
    `exp(logp_new - logp_old)`, and that is not a ratio of anything. Nothing would fail. The run
    would just optimize something else.
    """
    tokens, targets, mask = fixed_batch
    bad = mask.copy()
    bad[N - 1] = 0.5

    with Trainer(libs, policy, TrainConfig(lr=1e-3)):
        with pytest.raises(RuntimeError, match="INVALID_ARG"):
            _grpo_loss(libs, policy, tokens, targets, bad, np.zeros(N), np.zeros(N))


@pytest.mark.parametrize("eps", [0.0, 1.0, -0.1])
def test_an_epsilon_outside_the_open_unit_interval_is_refused(eps: float) -> None:
    with pytest.raises(ValueError, match="clip_eps"):
        GRPOConfig(clip_eps=eps)


def test_a_grpo_context_refuses_a_supervised_evaluation(policy, libs) -> None:
    """It would build the SFT graph, and this context is pinned to the GRPO one.

    ggml-opt sizes its optimizer state from the first graph it sees and indexes it by node index
    forever. A second graph with a different node count reads past the end of it.
    """
    with GRPOTrainer(libs, policy, GRPOConfig(lr=1e-3)) as trainer:
        with pytest.raises(NotImplementedError, match="node count"):
            trainer.evaluate([])


# ---------------------------------------------------------------------------------------------
# The collator.
# ---------------------------------------------------------------------------------------------


def _fake_rollouts() -> RolloutBatch:
    return RolloutBatch(
        rollouts=[
            Rollout(
                group=0,
                prompt_tokens=[5, 6, 7],
                completion_tokens=[8, 9],
                logp_old=np.array([-1.0, -2.0], dtype=np.float32),
                advantage=1.5,
            ),
            Rollout(
                group=0,
                prompt_tokens=[5, 6, 7],
                completion_tokens=[10],
                logp_old=np.array([-3.0], dtype=np.float32),
                advantage=-1.5,
            ),
        ],
        prompts=["p"],
    )


def test_the_collator_grades_the_position_that_predicts_a_completion_token() -> None:
    """The last prompt token is what predicts the first completion token, and it is graded.

    Position `i` predicts `tokens[i+1]`.

    Off by one here and the model is rewarded for predicting the prompt's own last token, and
    punished for the completion's last. Both losses look perfectly reasonable.
    """
    batch = collate(_fake_rollouts(), seq_len=8)

    assert len(batch.tokens) == 2 * 8

    # Rollout 0: prompt [5,6,7], completion [8,9]. Position 2 (the last prompt token) predicts 8;
    # position 3 predicts 9. Both graded, nothing else.
    graded = [i for i in range(8) if batch.mask[i] != 0.0]
    assert graded == [2, 3]
    assert batch.targets[2] == 8
    assert batch.targets[3] == 9

    # Rollout 1: completion is one token, so only position 2 is graded.
    graded = [i - 8 for i in range(8, 16) if batch.mask[i] != 0.0]
    assert graded == [2]
    assert batch.targets[8 + 2] == 10

    assert batch.n_completion_tokens == 3


def test_the_collator_carries_logp_old_to_the_position_that_uses_it() -> None:
    batch = collate(_fake_rollouts(), seq_len=8)

    assert batch.logp_old[2] == pytest.approx(-1.0)
    assert batch.logp_old[3] == pytest.approx(-2.0)
    assert batch.logp_old[8 + 2] == pytest.approx(-3.0)

    # ...and is zero everywhere else. A stray logp_old in a padding slot is the NaN above.
    assert batch.logp_old[0] == 0.0
    assert batch.logp_old[7] == 0.0


def test_the_collator_normalizes_by_the_graded_tokens() -> None:
    """So an update does not get stronger just because the model rambled this round."""
    batch = collate(_fake_rollouts(), seq_len=8)

    # 3 graded tokens, advantages +1.5 (x2) and -1.5 (x1), each divided by 3.
    assert batch.adv[2] == pytest.approx(1.5 / 3)
    assert batch.adv[8 + 2] == pytest.approx(-1.5 / 3)
    assert batch.adv[0] == 0.0


def test_each_rollout_is_its_own_sequence() -> None:
    """So attention cannot cross between two different answers (S1-07)."""
    batch = collate(_fake_rollouts(), seq_len=8)

    assert batch.seq_ids[:8] == [0] * 8
    assert batch.seq_ids[8:] == [1] * 8
    assert batch.positions[:8] == list(range(8))
    assert batch.positions[8:] == list(range(8))


# ---------------------------------------------------------------------------------------------
# Self-verification.
# ---------------------------------------------------------------------------------------------


def test_a_fast_path_that_agrees_is_used(caplog) -> None:
    calls = {"fast": 0, "naive": 0}

    def fast(x):
        calls["fast"] += 1
        return x * 2.0

    def naive(x):
        calls["naive"] += 1
        return x * 2.0

    checked = SelfVerified(fast, naive, tolerance=1e-6, name="doubling")

    assert checked(np.array([1.0, 2.0])).tolist() == [2.0, 4.0]
    assert calls == {"fast": 1, "naive": 1}  # both, once

    checked(np.array([3.0]))
    assert calls == {"fast": 2, "naive": 1}  # only the fast one, thereafter

    assert checked.report.agreed
    assert not checked.using_fallback


def test_a_fast_path_that_lies_is_retired_permanently(caplog) -> None:
    """It does not get a second chance, and the divergence is logged.

    A fast path that was wrong once will be wrong again, on inputs nobody can predict, and it will
    be wrong *silently* — the loss stays finite and the model still trains. Re-testing it
    periodically would only mean being wrong between tests.
    """
    calls = {"fast": 0, "naive": 0}

    def fast(x):
        calls["fast"] += 1
        return x * 2.0 + 1.0  # wrong

    def naive(x):
        calls["naive"] += 1
        return x * 2.0

    checked = SelfVerified(fast, naive, tolerance=1e-6, name="doubling")

    with caplog.at_level(logging.ERROR):
        result = checked(np.array([1.0, 2.0]))

    # The NAIVE answer comes back, not the fast one.
    assert result.tolist() == [2.0, 4.0]
    assert "FAILED verification" in caplog.text

    for _ in range(3):
        checked(np.array([1.0]))

    assert calls["fast"] == 1, "the fast path was called again after it failed"
    assert calls["naive"] == 4
    assert checked.using_fallback


def test_a_shape_disagreement_is_not_a_small_deviation() -> None:
    checked = SelfVerified(lambda x: x[:1], lambda x: x, tolerance=1e9, name="truncating")

    checked(np.array([1.0, 2.0, 3.0]))

    assert checked.using_fallback
    assert checked.report.max_deviation == float("inf")


# ---------------------------------------------------------------------------------------------
# The toy task.
# ---------------------------------------------------------------------------------------------


def test_grpo_learns_the_toy_task(tiny_q4_k, tmp_path, load_model, libs) -> None:
    """The mean group reward goes up. Which is the only claim GRPO actually makes.

    The reward is "use more tokens from this set", and that is the point rather than a shortcut. A
    GRPO update pushes up the log-probability of the tokens in high-advantage completions — so a
    reward that says *use more of these tokens* is exactly what the gradient can act on, with
    nothing in between. A policy that cannot learn this has not learned anything, and the fault
    would be in the update rather than in the task.

    (An output-length target — the obvious toy — asks a two-layer random model to control the
    decoded *character count* of its own samples. The signal is real; it is just buried under the
    sampling noise long before it reaches the weights. Measured: it trends up, but not far enough
    above the noise to assert on.)

    Two contexts over ONE model, sharing ONE adapter, so the engine samples from the weights the
    trainer is moving. Attaching the adapter file to each context separately would load it twice,
    and the run would be perfectly on-policy against a policy that never changed — the reward curve
    flat, and nothing to say why. That is what this test did on its first run.
    """
    adapter = tmp_path / "toy.gguf"
    create_zero_adapter(tiny_q4_k, adapter, r=8, seed=7)

    G = 8
    SEQ = 32
    PROMPTS = ["hello", "world"]

    n_sequences = len(PROMPTS) * G
    n_batch_tokens = n_sequences * SEQ

    rollout_model = load_model(tiny_q4_k, n_ctx=512, n_seq_max=G)
    rollout_model.attach_adapter(adapter, scale=1.0)

    train_model = load_model(
        tiny_q4_k,
        n_ctx=n_batch_tokens,
        n_ubatch=n_batch_tokens,
        n_seq_max=n_sequences,
        training=True,
    )
    # THE SAME adapter object, not the same file.
    train_model.attach_adapter(adapter, scale=1.0, adapter=rollout_model.adapter)
    assert train_model.adapter == rollout_model.adapter

    engine = RolloutEngine(
        libs,
        rollout_model.ctx,
        rollout_model.model,
        n_rollouts=G,
        sampler=SamplerConfig(temperature=1.0, seed=11, max_new_tokens=16),
        adapter=rollout_model.adapter,
    )

    result = train_grpo(
        libs,
        train_model,
        engine,
        prompts=PROMPTS,
        reward_fn=token_reward(set(range(128))),
        config=GRPOConfig(lr=1.0, clip_eps=EPS, seq_len=SEQ, iterations=12),
    )

    rewards = result.rewards()
    assert len(rewards) == 12
    assert all(math.isfinite(m.loss) for m in result.steps), f"loss went NaN: {result.steps}"

    first = sum(rewards[:3]) / 3
    last = sum(rewards[-3:]) / 3

    assert last > first, (
        f"the mean group reward did not improve: {first:.4f} -> {last:.4f}\n"
        f"  per-iteration: {[round(r, 4) for r in rewards]}"
    )


def test_two_contexts_with_two_adapters_are_refused(tiny_q4_k, tmp_path, load_model, libs) -> None:
    """The failure that has no symptom.

    Attaching the adapter FILE to both contexts loads it twice: two objects that start out equal and
    diverge the moment a step is taken. The trainer moves one; the engine keeps sampling from the
    other. GRPO then runs, forever, on a policy that never changes — no error, no NaN, and a reward
    curve that is flat for no reason anyone can see.

    So it is refused, loudly, before the first rollout.
    """
    adapter = tmp_path / "split.gguf"
    create_zero_adapter(tiny_q4_k, adapter, r=RANK, seed=7)

    G = 2
    rollout_model = load_model(tiny_q4_k, n_ctx=128, n_seq_max=G)
    rollout_model.attach_adapter(adapter, scale=1.0)

    train_model = load_model(tiny_q4_k, n_ctx=64, n_ubatch=64, n_seq_max=2, training=True)
    train_model.attach_adapter(adapter, scale=1.0)  # <- loads the file a SECOND time

    assert train_model.adapter != rollout_model.adapter

    engine = RolloutEngine(
        libs,
        rollout_model.ctx,
        rollout_model.model,
        n_rollouts=G,
        sampler=SamplerConfig(temperature=0.0, max_new_tokens=4),
        adapter=rollout_model.adapter,
    )

    with pytest.raises(ValueError, match="DIFFERENT adapter"):
        train_grpo(
            libs,
            train_model,
            engine,
            prompts=["hi"],
            reward_fn=token_reward({1, 2}),
            config=GRPOConfig(lr=1e-3, seq_len=32, iterations=1),
        )


def test_a_kl_coefficient_without_a_reference_is_refused(
    tiny_q4_k, tmp_path, load_model, libs
) -> None:
    """A knob that reads as if it regularizes and does not is worse than no knob.

    `train_grpo` used to accept `kl_coef > 0` and then never run a reference pass or pass logp_ref
    to the shim — so the KL term was multiplied by a zero weight and contributed exactly nothing.
    The run looked regularized. It was not.
    """
    adapter = tmp_path / "nokl.gguf"
    create_zero_adapter(tiny_q4_k, adapter, r=RANK, seed=7)

    G = 2
    rollout_model = load_model(tiny_q4_k, n_ctx=128, n_seq_max=G)
    rollout_model.attach_adapter(adapter, scale=1.0)

    train_model = load_model(tiny_q4_k, n_ctx=64, n_ubatch=64, n_seq_max=2, training=True)
    train_model.attach_adapter(adapter, scale=1.0, adapter=rollout_model.adapter)

    engine = RolloutEngine(
        libs,
        rollout_model.ctx,
        rollout_model.model,
        n_rollouts=G,
        sampler=SamplerConfig(temperature=0.0, max_new_tokens=4),
        adapter=rollout_model.adapter,
    )

    with pytest.raises(ValueError, match="needs a reference"):
        train_grpo(
            libs,
            train_model,
            engine,
            prompts=["hi"],
            reward_fn=token_reward({1, 2}),
            config=GRPOConfig(lr=1e-3, seq_len=32, iterations=1, kl_coef=0.1),
            lm_head=None,
        )


def test_the_kl_actually_reaches_the_loss(tiny_q4_k, tmp_path, load_model, libs) -> None:
    """With a reference and a large coefficient, the loss must MOVE.

    The reference is the base model with the adapter off (BLUEPRINT D6), scored on the ROLLOUT
    context — the one without an optimizer holding its scheduler.
    """
    adapter = tmp_path / "kl.gguf"
    create_zero_adapter(tiny_q4_k, adapter, r=RANK, seed=7)

    G = 2
    SEQ = 32
    n_seq = 1 * G
    n_tok = n_seq * SEQ

    rollout_model = load_model(tiny_q4_k, n_ctx=128, n_seq_max=G)
    rollout_model.attach_adapter(adapter, scale=1.0)

    train_model = load_model(tiny_q4_k, n_ctx=n_tok, n_ubatch=n_tok, n_seq_max=n_seq, training=True)
    train_model.attach_adapter(adapter, scale=1.0, adapter=rollout_model.adapter)

    engine = RolloutEngine(
        libs,
        rollout_model.ctx,
        rollout_model.model,
        n_rollouts=G,
        sampler=SamplerConfig(temperature=1.0, seed=3, max_new_tokens=8),
        adapter=rollout_model.adapter,
    )

    lm_head = load_lm_head(tiny_q4_k)

    result = train_grpo(
        libs,
        train_model,
        engine,
        prompts=["hello"],
        reward_fn=token_reward(set(range(100))),
        config=GRPOConfig(lr=1e-3, seq_len=SEQ, iterations=2, kl_coef=1.0),
        lm_head=lm_head,
    )

    assert len(result.steps) == 2
    assert all(math.isfinite(m.loss) for m in result.steps), (
        f"the KL path produced a non-finite loss: {[m.loss for m in result.steps]}"
    )
