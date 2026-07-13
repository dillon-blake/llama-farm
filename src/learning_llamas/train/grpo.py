"""S1-16 — the GRPO update: reward what beat its siblings, and do not move too far doing it.

The objective, per completion token:

    r_i      = exp(logp_new_i - logp_old_i)                      the importance ratio
    L_i      = min( r_i * A_i,  clip(r_i, 1-eps, 1+eps) * A_i )  PPO's pessimistic surrogate
    loss     = -mean_i L_i  +  kl_coef * mean_i k3_i             optionally anchored to a reference

`A_i` is the token's **group** advantage — how its completion scored against the other G−1 answers
to the same prompt (S1-15). That is GRPO's whole trick: the group is the baseline, so there is no
value network to train.

`logp_old` is the behaviour policy's logprob, captured when the token was *sampled*. On a strictly
on-policy step it should be ≈ `logp_new` and the ratio ≈ 1; it stops being so the moment you take
more than one gradient step per batch of rollouts, and the clip is what stops that from going
anywhere dangerous.

Three things about the shape of a step, in decreasing order of how surprising they are:

**It runs on two contexts.** Generation happens on an inference context; the update happens on a
training one. They share one `llama_adapter_lora`, so a step mutates the very weights the next
rollout samples from. This is not an optimization — a `llama_decode` on a training context frees the
scheduler the optimizer is holding, and the second one segfaults.

**The batch shape is fixed for the whole run.** ggml-opt sizes its optimizer state from the first
graph it sees and indexes it by node index forever, so every step must present the same number of
tokens. Rollouts are ragged, so they are padded to a constant layout, and a padded slot carries
weight 0 — which makes its loss and its gradient bitwise zero (ADR-0003).

**The KL term is always in the graph, even when it is off.** Building it conditionally would change
the node count between runs, which is the same thing as corrupting the optimizer state. `kl_coef`
is folded into a per-token weight instead, and zero weights make it contribute exactly nothing.

Provenance (ROADMAP §13): the maths is standard PPO/GRPO and the chunked-CE kernel it rests on is
Apache-2.0 (already imported by S1-04). unsloth's GRPO orchestration is AGPL-marked and was not
read; the orchestration here — the collator, the normalization, the two-context structure — is
original.
"""

from __future__ import annotations

import ctypes
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .. import _ffi
from .loop import TrainConfig, Trainer
from .rollout import RewardFn, RolloutBatch, RolloutEngine

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GRPOConfig:
    """How to run GRPO.

    Attributes:
        lr: Learning rate.
        clip_eps: PPO's epsilon. **Fixed for the life of the run** — it is baked into the graph's
            op_params, so changing it mid-run would keep the node count identical while silently
            changing the objective.
        kl_coef: Weight on the k3 KL penalty to the reference policy. ``0.0`` turns it off, and
            then no reference pass is run at all.
        seq_len: Tokens per rollout in the training batch. Every rollout is padded to exactly this,
            because ggml-opt requires a fixed batch shape across the whole run.
        iterations: How many generate-then-update rounds.
        grad_clip: Global-norm gradient clip (S1-10). ``0.0`` disables it.

    Raises:
        ValueError: If ``clip_eps`` is not in ``(0, 1)`` or ``kl_coef`` is negative.
    """

    lr: float = 1e-3
    clip_eps: float = 0.2
    kl_coef: float = 0.0
    seq_len: int = 64
    iterations: int = 5
    grad_clip: float = 0.0

    def __post_init__(self) -> None:
        """Reject a clip epsilon or KL coefficient that cannot mean anything."""
        if not 0.0 < self.clip_eps < 1.0:
            raise ValueError(
                f"clip_eps must be in (0, 1), got {self.clip_eps}. At 0 the clipped branch is the "
                f"constant 1 and the ratio's gradient vanishes; at 1 the lower bound is 0 and the "
                f"clip never binds from below."
            )
        if self.kl_coef < 0.0:
            raise ValueError(f"kl_coef must not be negative, got {self.kl_coef}")


@dataclass
class GRPOMetrics:
    """What one update did.

    Attributes:
        loss: The surrogate, as the graph computed it.
        mean_reward: Averaged over every rollout in the batch. **This is the number that should go
            up.** The loss is only a proxy for it, and a falling loss with a flat reward means the
            policy is exploiting the surrogate rather than the task.
        mean_ratio: ``exp(logp_new - logp_old)`` averaged over completion tokens. Should sit near 1
            on a fresh batch of rollouts; drifting far from it means the policy has moved a long
            way from the one that generated them.
        clip_fraction: Fraction of tokens where the clip actually bound. A number near 0 means the
            clip is doing nothing; near 1 means every step is being held back by it.
        n_tokens: Completion tokens the update was averaged over.
    """

    loss: float
    mean_reward: float
    mean_ratio: float = 0.0
    clip_fraction: float = 0.0
    n_tokens: int = 0


@dataclass
class GRPOResult:
    """A whole run."""

    steps: list[GRPOMetrics] = field(default_factory=list)

    def rewards(self) -> list[float]:
        """The mean group reward at each iteration — the curve that should be going up."""
        return [m.mean_reward for m in self.steps]


@dataclass
class GRPOBatch:
    """A rollout batch flattened into the fixed layout the training graph wants.

    Every array is ``[n_rollouts * seq_len]``, and a masked slot is zero in all of them.
    """

    tokens: list[int]
    targets: list[int]
    mask: np.ndarray  # 1.0 on completion tokens, 0.0 on prompt and padding
    adv: np.ndarray  # advantage * mask, normalized by the completion-token count
    logp_old: np.ndarray
    seq_ids: list[int]
    positions: list[int]
    n_completion_tokens: int


def collate(rollouts: RolloutBatch, seq_len: int) -> GRPOBatch:
    """Pack rollouts into one fixed-shape batch.

    Each rollout becomes its own sequence, so attention cannot cross between them (S1-07), and each
    is padded to exactly ``seq_len`` — the shape is a property of the *run*, not of the batch, and
    ggml-opt will reject a change to it.

    The mask marks the positions whose **target** is a completion token. That is one position to the
    left of the token itself: position ``i`` predicts ``tokens[i+1]``, so the last prompt token is
    what predicts the first completion token, and it is graded.

    Args:
        rollouts: What the engine generated.
        seq_len: Tokens per rollout. A rollout longer than this is truncated, and the tokens that
            fall off carry no gradient, which is a silent loss of signal — so it warns.

    Returns:
        The flattened batch.

    Raises:
        ValueError: If there are no rollouts.
    """
    if not rollouts.rollouts:
        raise ValueError("no rollouts to collate")

    n = len(rollouts.rollouts)
    total = n * seq_len

    tokens = [0] * total
    targets = [0] * total
    mask = np.zeros(total, dtype=np.float32)
    adv = np.zeros(total, dtype=np.float32)
    logp_old = np.zeros(total, dtype=np.float32)
    seq_ids = [0] * total
    positions = [0] * total

    for r, rollout in enumerate(rollouts.rollouts):
        base = r * seq_len
        full = rollout.prompt_tokens + rollout.completion_tokens

        if len(full) > seq_len:
            log.warning(
                "rollout %d is %d tokens but seq_len is %d: the tail carries no gradient",
                r,
                len(full),
                seq_len,
            )
            full = full[:seq_len]

        n_prompt = len(rollout.prompt_tokens)

        for i in range(seq_len):
            seq_ids[base + i] = r
            positions[base + i] = i
            tokens[base + i] = full[i] if i < len(full) else 0
            targets[base + i] = full[i + 1] if i + 1 < len(full) else 0

        # Position `n_prompt - 1 + t` predicts completion token t. Grade exactly those.
        for t, logp in enumerate(rollout.logp_old):
            i = n_prompt - 1 + t
            if i >= seq_len or i + 1 >= len(full):
                break

            mask[base + i] = 1.0
            adv[base + i] = rollout.advantage
            logp_old[base + i] = logp

    n_completion = int(mask.sum())

    # Normalize by the completion tokens actually graded, so the update does not get quietly
    # stronger just because the model generated longer answers this round. Folded into the
    # advantages rather than the weights: the weights ARE the mask that ce_sparse multiplies
    # logp_new by, and scaling them would scale the log-probability inside the importance ratio.
    if n_completion > 0:
        adv = adv / n_completion

    return GRPOBatch(
        tokens=tokens,
        targets=targets,
        mask=mask,
        adv=adv,
        logp_old=logp_old,
        seq_ids=seq_ids,
        positions=positions,
        n_completion_tokens=n_completion,
    )


class GRPOTrainer(Trainer):
    """A :class:`~learning_llamas.train.loop.Trainer` that steps the GRPO objective.

    Deliberately does **not** expose ``evaluate()``. The base class's evaluation runs a *supervised*
    forward pass, which builds the SFT loss graph — and this context has been pinned to the GRPO
    objective. ggml-opt sizes its optimizer state from the first graph it sees; a second graph with
    a different node count reads past the end of it. The shim would refuse the step, but the honest
    thing is not to offer it.
    """

    def __init__(
        self,
        libs: _ffi.Libraries,
        model,  # noqa: ANN001 - a TrainableModel; typed structurally, as elsewhere in train/
        config: GRPOConfig,
    ) -> None:
        super().__init__(
            libs,
            model,
            TrainConfig(lr=config.lr, grad_clip=config.grad_clip),
        )
        self.grpo = config

        # Borrowed by the C struct on every step, so they have to outlive it. Rebinding these to a
        # temporary inside the call would hand the shim a freed pointer.
        self._live: dict[str, np.ndarray] = {}

    def evaluate(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201, ARG002
        """Not available: evaluation builds the SFT graph, and this context is pinned to GRPO."""
        raise NotImplementedError(
            "a GRPO context cannot run a supervised evaluation step: it would build a second loss "
            "graph with a different node count, and ggml-opt indexes its optimizer state by node "
            "index off the first graph it saw. Score the policy on the rollout context instead."
        )

    def grpo_step(
        self,
        batch: GRPOBatch,
        mean_reward: float,
        logp_ref: np.ndarray | None = None,
        train: bool = True,
    ) -> GRPOMetrics:
        """One GRPO update on a collated batch.

        Args:
            batch: From :func:`collate`.
            mean_reward: For the metrics — the number that should be going up.
            logp_ref: ``[n]`` the reference policy's logprob per token, for the KL term. ``None``
                means no reference, and the shim then makes the KL exactly zero rather than NaN.
            train: False runs the forward only, which is how the tests read the loss without moving.

        Returns:
            What the step did.
        """
        libs = self._libs
        n = len(batch.tokens)

        adv = np.ascontiguousarray(batch.adv, dtype=np.float32)
        logp_old = np.ascontiguousarray(batch.logp_old, dtype=np.float32)

        kl_w = None
        ref = None
        if self.grpo.kl_coef > 0.0 and logp_ref is not None:
            ref = np.ascontiguousarray(logp_ref, dtype=np.float32)
            scale = self.grpo.kl_coef / max(batch.n_completion_tokens, 1)
            kl_w = np.ascontiguousarray(batch.mask * scale, dtype=np.float32)

        # Held on the instance for the duration of the call: the C struct borrows these pointers.
        self._live = {"adv": adv, "logp_old": logp_old}
        if ref is not None:
            self._live["logp_ref"] = ref
        if kl_w is not None:
            self._live["kl_w"] = kl_w

        def as_ptr(array: np.ndarray | None):  # noqa: ANN202
            if array is None:
                return None
            return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        inputs = _ffi.ll_grpo_inputs(
            adv=as_ptr(adv),
            logp_old=as_ptr(logp_old),
            logp_ref=as_ptr(ref),
            kl_w=as_ptr(kl_w),
            clip_eps=self.grpo.clip_eps,
        )

        loss = ctypes.c_float()
        _ffi.check(
            libs.farm.ll_train_step_grpo(
                self._model.ctx,
                (ctypes.c_int32 * n)(*batch.tokens),
                (ctypes.c_int32 * n)(*batch.targets),
                (ctypes.c_float * n)(*batch.mask.tolist()),
                (ctypes.c_int32 * n)(*batch.seq_ids),
                (ctypes.c_int32 * n)(*batch.positions),
                n,
                ctypes.byref(inputs),
                train,
                ctypes.byref(loss),
            ),
            "ll_train_step_grpo",
        )

        return GRPOMetrics(
            loss=float(loss.value),
            mean_reward=mean_reward,
            n_tokens=batch.n_completion_tokens,
        )


def train_grpo(
    libs: _ffi.Libraries,
    policy,  # noqa: ANN001 - the TRAINING context, adapter attached
    engine: RolloutEngine,
    prompts: Sequence[str],
    reward_fn: RewardFn,
    config: GRPOConfig,
) -> GRPOResult:
    """Generate, score, update — ``config.iterations`` times.

    The engine and the policy are **two contexts over one model**, sharing one adapter. The engine
    samples from the weights the policy is training, so every round is on-policy without a copy.

    Args:
        libs: The loaded native libraries.
        policy: The **training** context, with the adapter attached.
        engine: The rollout engine, on an **inference** context with the same adapter.
        prompts: What to generate answers to.
        reward_fn: How good was an answer.
        config: The run.

    Returns:
        One :class:`GRPOMetrics` per iteration.
    """
    # The engine and the policy must be sampling and training THE SAME WEIGHTS.
    #
    # Two contexts, yes -- but one llama_adapter_lora between them. Loading the adapter file twice
    # gives two objects that start out equal and then diverge the moment a step is taken: the
    # trainer moves one, the engine keeps sampling from the other, and GRPO runs forever on a policy
    # that never changes. There is no error, no NaN, and no clue -- just a reward curve that is
    # perfectly flat. (This is not hypothetical. It is what the toy-task test did on its first run.)
    if engine.adapter is not None and getattr(policy, "adapter", None) is not None:
        if engine.adapter != policy.adapter:
            raise ValueError(
                "the rollout engine and the training context are using DIFFERENT adapter objects. "
                "They must share one: attach the same llama_adapter_lora to both contexts rather "
                "than loading the adapter file twice. Otherwise the trainer updates weights the "
                "engine never samples from, and the reward never moves."
            )

    # Every rollout is its own sequence in the training batch (S1-07), so the TRAINING context has
    # to have room for all of them at once -- not just for one group. The shim would refuse the step
    # with a bare INVALID_ARG; saying it here says which number to change.
    n_sequences = len(prompts) * engine.n_rollouts
    n_seq_max = int(libs.llama.llama_n_seq_max(policy.ctx))
    if n_sequences > n_seq_max:
        raise ValueError(
            f"{len(prompts)} prompts x {engine.n_rollouts} rollouts is {n_sequences} "
            f"sequences, but "
            f"the training context was created with n_seq_max={n_seq_max}. Each rollout is packed "
            f"as its own sequence, so the training context needs n_seq_max >= {n_sequences}."
        )

    # ...and they all have to fit in one ubatch: a training step is one forward pass (the shim will
    # not split it, because ggml-opt keys its optimizer state to the graph).
    n_tokens = n_sequences * config.seq_len
    n_ubatch = int(libs.llama.llama_n_ubatch(policy.ctx))
    if n_tokens > n_ubatch:
        raise ValueError(
            f"the batch is {n_sequences} rollouts x seq_len {config.seq_len} = {n_tokens} "
            f"tokens, but the training context's n_ubatch is {n_ubatch}. Create it with "
            f"n_ctx >= n_ubatch >= {n_tokens}."
        )

    result = GRPOResult()

    with GRPOTrainer(libs, policy, config) as trainer:
        for _ in range(config.iterations):
            rollouts = engine.generate(prompts, reward_fn)
            batch = collate(rollouts, config.seq_len)

            metrics = trainer.grpo_step(batch, mean_reward=rollouts.mean_reward())
            result.steps.append(metrics)

    return result
