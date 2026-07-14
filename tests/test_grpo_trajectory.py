"""Eight GRPO updates against a float64 oracle (S1-40).

``test_grpo.py`` proves the composite loss/gradient are right *at a point* — exact ratio control,
one step, a gradient-differentiating clip test. What it cannot say is that eight optimizer steps of
the whole objective — ratio drift as the policy moves off the behaviour policy, the clip starting
to bind, the k3 anchor pulling, AdamW's state threading it all together — land where the math
says. The audit confirmed no multi-step GRPO trajectory was ever compared against anything.

The rollouts are hand-built and FIXED for the whole run — no sampling, no reward model — so the
run is deterministic and deliberately off-policy from step one: ``logp_old`` is seeded noise, so
the ratios start scattered across both clip branches, and the two groups carry advantages of both
signs. The run must *prove* that coverage (asserted below), because a trajectory whose clip never
binds is a trajectory that cannot catch a wrong clip.

The oracle: the float64 forward (packed, block-causal), ``reference_grpo.grpo_loss`` for the value
and ``reference_grpo.grpo_dlogp`` for the gradient — ``np.clip``/``np.minimum``/``np.where``, none
of the graph's relu identities — then the float64 backward and AdamW. Per-step comparison.
"""

from __future__ import annotations

import numpy as np
import pytest

from learning_llamas import Model, create_zero_adapter
from learning_llamas.train.grpo import GRPOBatch, GRPOConfig, GRPOTrainer, collate
from learning_llamas.train.rollout import Rollout, RolloutBatch, RolloutStats

from . import reference_grpo as ref_grpo
from . import reference_llama as ref
from .convergence import config as conv_config
from .convergence import harness

RANK = 4
ALPHA = 4.0
SEQ_LEN = 16
N_ROLLOUTS = 4  # two groups of two, advantages +1/-1 within each
N_PROMPT = 3
N_COMPLETION = 10
CLIP_EPS = 0.2
KL_COEF = 0.1
LR = 1e-3
ITERATIONS = 8

# ggml (F32) vs the float64 oracle, per step over 8 updates.
#   observed: 1.45e-06 worst (loss range 0.51 .. 3.73), with the clip binding on 161 of 192
#   live token-steps and both advantage signs on 160 each — the objective's every branch, live.
TRAJECTORY_TOL = 1e-4

# The knobs must matter (reference trajectories with the knob moved, worst |diff|):
#   observed: 8.69e-02 (clip_eps 0.2 -> 0.05), 1.36e+01 (kl_coef 0.1 -> 0.5).
SEPARATION_FLOOR = 20 * TRAJECTORY_TOL


def _rollouts() -> RolloutBatch:
    """Four fixed rollouts, two per group, with sampled-looking ``logp_old`` noise.

    ``logp_old`` is *not* the policy's true logprob — that is the point. The run starts off-policy
    with ratios spread across (0.3, 3), so both clip branches and both advantage signs are live
    from the first step rather than after some number of drift steps.
    """
    rng = np.random.default_rng(20260715)
    rollouts = []
    for r in range(N_ROLLOUTS):
        tokens = [3 + (7 * r + i) % 11 for i in range(N_PROMPT + N_COMPLETION)]
        rollouts.append(
            Rollout(
                group=r // 2,
                prompt_tokens=tokens[:N_PROMPT],
                completion_tokens=tokens[N_PROMPT:],
                logp_old=rng.uniform(-6.0, -1.0, size=N_COMPLETION).astype(np.float32),
                advantage=1.0 if r % 2 == 0 else -1.0,
            )
        )
    return RolloutBatch(rollouts=rollouts, prompts=["p0", "p1"], stats=RolloutStats())


def _logp_ref(batch: GRPOBatch) -> np.ndarray:
    """A fixed 'reference policy' logprob per slot: near logp_old, zero off-mask."""
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 0.3, size=len(batch.tokens))
    return ((batch.logp_old + noise) * batch.mask).astype(np.float32)


def _oracle_trajectory(
    base_path,  # noqa: ANN001
    adapter_path,  # noqa: ANN001
    batch: GRPOBatch,
    logp_ref: np.ndarray,
    clip_eps: float,
    kl_coef: float,
) -> tuple[list[float], dict[str, int]]:
    """The float64 run. Returns the loss curve and the branch-coverage counts it saw."""
    base, hp = ref.load_model(base_path)
    loras = harness.load_loras(base_path, adapter_path)
    scale = ref.lora_scale(ALPHA, RANK, 1.0)

    params: dict[str, np.ndarray] = {}
    for name, lora in loras.items():
        params[f"{name}.lora_a"] = lora.a
        params[f"{name}.lora_b"] = lora.b
    opt = ref.AdamW(lr=LR)

    tokens = np.array(batch.tokens)
    targets = np.array(batch.targets)
    positions = np.array(batch.positions)
    seq_ids = np.array(batch.seq_ids)
    mask = np.asarray(batch.mask, dtype=np.float64)
    adv = np.asarray(batch.adv, dtype=np.float64)
    logp_old = np.asarray(batch.logp_old, dtype=np.float64)
    ref_lp = np.asarray(logp_ref, dtype=np.float64)

    # collate() folded 1/n_completion into adv; fold it into the kl coefficient the same way.
    kl_scaled = kl_coef / batch.n_completion_tokens

    rows = np.arange(len(targets))
    live = mask != 0.0
    coverage = {"clip_bound": 0, "unclipped": 0, "adv_pos": 0, "adv_neg": 0}

    curve: list[float] = []
    for _it in range(ITERATIONS):
        logits, cache = ref.forward(
            base, hp, loras, scale, tokens, positions=positions, seq_ids=seq_ids
        )
        z = logits - logits.max(axis=-1, keepdims=True)
        logp_all = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
        logp_new = np.where(live, logp_all[rows, targets], 0.0)

        loss = ref_grpo.grpo_loss(
            logp_new, logp_old, adv, mask, eps=clip_eps, kl_coef=kl_scaled, logp_ref=ref_lp
        )
        dlogp = ref_grpo.grpo_dlogp(
            logp_new, logp_old, adv, mask, eps=clip_eps, kl_coef=kl_scaled, logp_ref=ref_lp
        )

        ratio = np.where(live, np.exp(logp_new - logp_old), 1.0)
        clipped = live & ((ratio <= 1.0 - clip_eps) | (ratio >= 1.0 + clip_eps))
        u, v = ratio * adv, ref_grpo.clip(ratio, clip_eps) * adv
        coverage["clip_bound"] += int(np.sum(live & clipped & (v < u)))
        coverage["unclipped"] += int(np.sum(live & ~clipped))
        coverage["adv_pos"] += int(np.sum(live & (adv > 0)))
        coverage["adv_neg"] += int(np.sum(live & (adv < 0)))

        # dL/dlogits: each live row's softmax pulls its dL/dlogp share.
        probs = np.exp(logp_all)
        dlogits = -probs * dlogp[:, None]
        dlogits[rows, targets] += dlogp

        grads = ref.backward(base, hp, loras, scale, cache, dlogits)
        opt.step(params, grads)
        curve.append(loss)

    return curve, coverage


@pytest.fixture(scope="module")
def grpo_run(tiny_f32, tmp_path_factory, libs):  # noqa: ANN201
    """One real 8-iteration GRPO run on fixed rollouts, shared by the tests below."""
    adapter = tmp_path_factory.mktemp("grpo") / "a.gguf"
    create_zero_adapter(tiny_f32, adapter, r=RANK, alpha=ALPHA, seed=conv_config.ADAPTER_SEED)

    batch = collate(_rollouts(), seq_len=SEQ_LEN)
    logp_ref = _logp_ref(batch)

    model = Model(
        tiny_f32,
        libs=libs,
        n_ctx=N_ROLLOUTS * SEQ_LEN,
        n_ubatch=N_ROLLOUTS * SEQ_LEN,
        training=True,
        n_seq_max=N_ROLLOUTS,
        n_threads=2,
    )
    losses: list[float] = []
    try:
        model.attach_adapter(adapter, scale=1.0)
        config = GRPOConfig(lr=LR, clip_eps=CLIP_EPS, kl_coef=KL_COEF, seq_len=SEQ_LEN)
        with GRPOTrainer(libs, model, config) as trainer:
            for _ in range(ITERATIONS):
                metrics = trainer.grpo_step(batch, mean_reward=0.0, logp_ref=logp_ref)
                losses.append(metrics.loss)
    finally:
        model.close()

    return adapter, batch, logp_ref, losses


def test_the_grpo_trajectory_matches_the_float64_oracle(tiny_f32, grpo_run) -> None:
    """Eight updates of the full objective — clip, k3 anchor, AdamW — per step."""
    adapter, batch, logp_ref, got = grpo_run

    want, coverage = _oracle_trajectory(tiny_f32, adapter, batch, logp_ref, CLIP_EPS, KL_COEF)

    # The trajectory must have EXERCISED the objective, or matching it proves nothing: tokens on
    # both sides of the clip, advantages of both signs. This is what the seeded logp_old buys.
    assert min(coverage.values()) > 0, (
        f"the run never exercised part of the objective: {coverage}. A trajectory whose clip "
        f"never binds cannot catch a wrong clip; re-seed the rollouts."
    )

    diff = np.abs(np.array(got) - np.array(want))
    worst = int(diff.argmax())
    assert diff.max() < TRAJECTORY_TOL, (
        f"the GRPO loss curve left the float64 oracle at step {worst}: "
        f"{got[worst]:.8f} vs {want[worst]:.8f} (worst |diff| {diff.max():.2e})"
    )


def test_the_oracle_can_fail(tiny_f32, grpo_run) -> None:
    """``clip_eps`` and ``kl_coef`` must both move the trajectory.

    These are the two knobs the graph bakes in most obscurely — eps inside the relu-composite's
    op_params, the KL folded into per-token weights — so a bug that dropped either would be
    invisible to everything except a trajectory that *depends* on them.
    """
    adapter, batch, logp_ref, _ = grpo_run

    base, _ = _oracle_trajectory(tiny_f32, adapter, batch, logp_ref, CLIP_EPS, KL_COEF)

    tighter_clip, _ = _oracle_trajectory(tiny_f32, adapter, batch, logp_ref, 0.05, KL_COEF)
    sep_eps = np.abs(np.array(tighter_clip) - np.array(base)).max()
    assert sep_eps > SEPARATION_FLOOR, f"clip_eps barely moves the trajectory ({sep_eps:.2e})"

    heavier_kl, _ = _oracle_trajectory(tiny_f32, adapter, batch, logp_ref, CLIP_EPS, 5 * KL_COEF)
    sep_kl = np.abs(np.array(heavier_kl) - np.array(base)).max()
    assert sep_kl > SEPARATION_FLOOR, f"kl_coef barely moves the trajectory ({sep_kl:.2e})"
