"""Twenty DPO steps against a float64 oracle (S1-39).

The audit confirmed what ``test_dpo.py`` could not claim: its log-2-at-init identity proves the
*reference pipeline* is consistent at step zero, and "the loss falls / the model prefers chosen"
proves direction — but no multi-step DPO trajectory was ever compared against an independent
implementation. A β plumbed wrong by a factor, a sign error that only matters once the policy has
moved off the reference, or an optimizer interaction specific to the ±1-weighted gradient would
have passed everything.

Two comparisons, factoring the pipeline into independently-checked stages:

1. **The frozen reference log-ratios** (``ll_logp_delta`` with B zeroed) against the float64
   forward's log-ratios, per pair.
2. **The training trajectory**: per-step losses of ``train_dpo`` against a float64 run — same
   packed batches, same signed weights, ``ggml``'s own reference deltas (stage 1 already vouches
   for them, so a disagreement here is *training* numerics, not the reference pass) — with the
   loss written as ``np.logaddexp``, not the graph's softplus construction, and the gradient
   derived from the math (``tests/reference_dpo.py``).

And the oracle proves it can fail: the reference trajectory must move when β changes and when the
reference deltas are zeroed — the two knobs whose silent loss would otherwise be invisible.
"""

from __future__ import annotations

import numpy as np
import pytest

from learning_llamas import create_zero_adapter
from learning_llamas.data import MaskedSample
from learning_llamas.train import DPOConfig, Preference, train_dpo
from learning_llamas.train.dpo import to_batch

from . import reference_dpo as ref_dpo
from . import reference_llama as ref
from .convergence import config as conv_config

RANK = 4
ALPHA = 4.0
N_CTX = 128
SEQ_LEN = 32
BETA = 0.1
LR = 1e-3
EPOCHS = 5
N_PAIRS = 4

# ggml (F32) vs the float64 oracle, per step over 20 steps.
#
#   observed: 1.11e-05 worst (loss scale ~0.69).
#
# An order noisier than SFT's 5.3e-07, and the mechanism is worth naming: z = Δ_policy - Δ_ref is
# a *difference of two ~35-magnitude log-prob sums*, so f32 cancellation leaves ~ulp(35) ≈ 4e-6 of
# noise in z before the loss ever sees it. That is a property of the objective, not of the
# implementation. 2e-4 keeps ~18x margin over the floor and still sits 190x below the smallest
# thing it must catch (a dropped Δ_ref lands at 3.8e-02).
TRAJECTORY_TOL = 2e-4

#   observed: 2.21e-06 worst per pair.
REF_DELTA_TOL = 1e-4

# The knobs must matter: reference trajectories with β tripled / Δ_ref dropped sit this far away.
#   observed: 2.86e-01 (beta x3), 3.82e-02 (Δ_ref dropped). Floor is 20x the band.
SEPARATION_FLOOR = 20 * TRAJECTORY_TOL


def _sample(n_prompt: int, n_completion: int, start: int) -> MaskedSample:
    n = n_prompt + n_completion
    return MaskedSample(
        tokens=[start + (i % 9) for i in range(n)],
        weights=[0.0] * n_prompt + [1.0] * n_completion,
    )


def _pairs() -> list[Preference]:
    """Four pairs with distinct prompts and completions, long enough to carry real gradients."""
    return [
        Preference(
            chosen=_sample(3, 5, start=7 + 2 * i),
            rejected=_sample(3, 5, start=30 + 3 * i),
        )
        for i in range(N_PAIRS)
    ]


def _reference_trajectory(base_path, adapter_path, batches, ref_deltas, beta: float) -> list[float]:  # noqa: ANN001
    """The float64 run: forward the packed pair, DPO loss, backward, AdamW — per step."""
    from .convergence import harness

    base, hp = ref.load_model(base_path)
    loras = harness.load_loras(base_path, adapter_path)
    scale = ref.lora_scale(ALPHA, RANK, 1.0)

    params: dict[str, np.ndarray] = {}
    for name, lora in loras.items():
        params[f"{name}.lora_a"] = lora.a
        params[f"{name}.lora_b"] = lora.b

    opt = ref.AdamW(lr=LR)

    curve: list[float] = []
    for _epoch in range(EPOCHS):
        for batch, ref_delta in zip(batches, ref_deltas, strict=True):
            tokens = np.array(batch.tokens)
            targets = np.array(batch.targets)
            weights = np.array(batch.weights, dtype=np.float64)
            positions = np.array(batch.positions)
            seq_ids = np.array(batch.seq_ids)

            logits, cache = ref.forward(
                base, hp, loras, scale, tokens, positions=positions, seq_ids=seq_ids
            )
            loss, dlogits = ref_dpo.loss_and_dlogits(logits, targets, weights, beta, ref_delta)
            grads = ref.backward(base, hp, loras, scale, cache, dlogits)
            opt.step(params, grads)
            curve.append(loss)

    return curve


@pytest.fixture(scope="module")
def dpo_run(tiny_f32, tmp_path_factory, libs):  # noqa: ANN201
    """One real ``train_dpo`` run, shared by the tests below."""
    adapter = tmp_path_factory.mktemp("dpo") / "a.gguf"
    create_zero_adapter(tiny_f32, adapter, r=RANK, alpha=ALPHA, seed=conv_config.ADAPTER_SEED)

    from learning_llamas import Model

    model = Model(
        tiny_f32,
        libs=libs,
        n_ctx=N_CTX,
        n_ubatch=SEQ_LEN,
        training=True,
        n_seq_max=4,
        n_threads=2,
    )
    try:
        model.attach_adapter(adapter, scale=1.0)
        pairs = _pairs()
        result = train_dpo(
            libs,
            model,
            pairs,
            DPOConfig(lr=LR, beta=BETA, seq_len=SEQ_LEN, epochs=EPOCHS, shuffle=False),
        )
    finally:
        model.close()

    batches = [to_batch(p, SEQ_LEN) for p in pairs]
    return adapter, batches, result


def test_the_frozen_reference_logratios_match_float64(tiny_f32, dpo_run) -> None:
    """Stage 1: ``ll_logp_delta`` with B zeroed, against the float64 base-model forward.

    The reference model is the base model exactly (B = 0 makes the LoRA delta exactly zero), so
    the float64 forward with no adapter is the same function in higher precision.
    """
    _, batches, result = dpo_run
    base, hp = ref.load_model(tiny_f32)

    worst = 0.0
    for batch, got in zip(batches, result.reference, strict=True):
        logits, _ = ref.forward(
            base,
            hp,
            {},
            1.0,
            np.array(batch.tokens),
            positions=np.array(batch.positions),
            seq_ids=np.array(batch.seq_ids),
        )
        want = ref_dpo.logratio_from_logits(
            logits, np.array(batch.targets), np.array(batch.weights, dtype=np.float64)
        )
        worst = max(worst, abs(got - want))

    assert worst < REF_DELTA_TOL, (
        f"the frozen reference log-ratios disagree with the float64 base model by {worst:.2e} — "
        f"the DPO objective is being computed against the wrong reference"
    )


@pytest.fixture(scope="module")
def oracle_trajectory(tiny_f32, dpo_run) -> np.ndarray:
    """The float64 trajectory for the real run's config — shared with the can-fail test."""
    adapter, batches, result = dpo_run
    return np.array(_reference_trajectory(tiny_f32, adapter, batches, result.reference, BETA))


def test_the_dpo_trajectory_matches_the_float64_oracle(dpo_run, oracle_trajectory) -> None:
    """Stage 2: twenty optimizer steps of the ±1-weighted objective, per step."""
    _, _, result = dpo_run

    got = np.array([s.loss for s in result.steps])
    want = oracle_trajectory

    assert len(got) == len(want) == N_PAIRS * EPOCHS

    diff = np.abs(got - want)
    worst = int(diff.argmax())
    assert diff.max() < TRAJECTORY_TOL, (
        f"the DPO loss curve left the float64 oracle at step {worst}: "
        f"{got[worst]:.8f} vs {want[worst]:.8f} (worst |diff| {diff.max():.2e})"
    )


def test_the_oracle_can_fail(tiny_f32, dpo_run, oracle_trajectory) -> None:
    """β and Δ_ref must both move the trajectory, or the comparison above proves nothing.

    These are exactly the two knobs a plumbing bug would silently drop — β multiplied somewhere it
    cancels, or the precomputed reference ignored — and both would leave a perfectly healthy-
    looking falling loss.
    """
    adapter, batches, result = dpo_run
    base = oracle_trajectory

    tripled_beta = np.array(
        _reference_trajectory(tiny_f32, adapter, batches, result.reference, 3 * BETA)
    )
    sep_beta = np.abs(tripled_beta - base).max()
    assert sep_beta > SEPARATION_FLOOR, f"beta barely moves the trajectory ({sep_beta:.2e})"

    dropped_ref = np.array(
        _reference_trajectory(tiny_f32, adapter, batches, [0.0] * len(batches), BETA)
    )
    sep_ref = np.abs(dropped_ref - base).max()
    assert sep_ref > SEPARATION_FLOOR, (
        f"zeroing the reference deltas barely moves the trajectory ({sep_ref:.2e}) — "
        f"either the pairs are too symmetric or the oracle cannot see a dropped reference"
    )
