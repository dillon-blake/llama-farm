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
from ..logprobs import LmHead, sequence_logprobs
from ..verify import SelfVerified, VerificationReport
from .loop import TrainConfig, Trainer
from .rollout import RewardFn, RolloutBatch, RolloutEngine, captured_logp, recompute_logp

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
        logp_old_tol: Tolerance for the sample-time ``logp_old`` cross-check (S1-16 §3). On the
            first real batch the captured ``logp_old`` is compared against an independent S1-13
            chunked recompute of the same tokens; within this tolerance the free capture is trusted
            and used for the rest of the run, and outside it the run falls back — loudly — to the
            recompute. The check needs ``lm_head`` (it is the same chunked pass the KL uses), so it
            is silently skipped when ``lm_head`` is not passed; ``0.0`` disables it outright.
            Observed capture-vs-recompute deviation on the CPU fixtures is ~1e-6.

    Raises:
        ValueError: If ``clip_eps`` is not in ``(0, 1)``, ``kl_coef`` is negative, or
            ``logp_old_tol`` is negative.
    """

    lr: float = 1e-3
    clip_eps: float = 0.2
    kl_coef: float = 0.0
    seq_len: int = 64
    iterations: int = 5
    grad_clip: float = 0.0
    logp_old_tol: float = 5e-3

    def __post_init__(self) -> None:
        """Reject a clip epsilon, KL coefficient, or tolerance that cannot mean anything."""
        if not 0.0 < self.clip_eps < 1.0:
            raise ValueError(
                f"clip_eps must be in (0, 1), got {self.clip_eps}. At 0 the clipped branch is the "
                f"constant 1 and the ratio's gradient vanishes; at 1 the lower bound is 0 and the "
                f"clip never binds from below."
            )
        if self.kl_coef < 0.0:
            raise ValueError(f"kl_coef must not be negative, got {self.kl_coef}")
        if self.logp_old_tol < 0.0:
            raise ValueError(f"logp_old_tol must not be negative, got {self.logp_old_tol}")


@dataclass
class GRPOMetrics:
    """What one update did.

    There is deliberately **no ``mean_ratio`` and no ``clip_fraction``** here, and their absence is
    the point. Both are the standard PPO drift diagnostics, and both are functions of ``logp_new``
    — which lives inside the graph and never crosses the FFI: ``ll_train_step_grpo`` hands back one
    scalar, the loss. They can only be produced by a second chunked scoring pass over the same batch
    (doubling the step's cost) or by new shim surface to read the ratio tensor back. Fields that
    read ``0.0`` forever would be worse than nothing: 0 is also what a *healthy* ``clip_fraction``
    looks like, so a caller watching for the clip to start binding would watch a constant and
    conclude it never does.

    Attributes:
        loss: The surrogate, as the graph computed it.
        mean_reward: Averaged over every rollout in the batch. **This is the number that should go
            up.** The loss is only a proxy for it, and a falling loss with a flat reward means the
            policy is exploiting the surrogate rather than the task.
        n_tokens: Completion tokens the update was averaged over.
    """

    loss: float
    mean_reward: float
    n_tokens: int = 0


@dataclass
class GRPOResult:
    """A whole run.

    Attributes:
        steps: One :class:`GRPOMetrics` per iteration.
        logp_verification: What the sample-time ``logp_old`` cross-check found (S1-16 §3), or
            ``None`` when it did not run (no ``lm_head``, or ``logp_old_tol == 0``). ``agreed`` is
            the headline: ``False`` means the capture diverged from the S1-13 recompute and the run
            fell back to the recompute — the update trained on correct numbers, but the fast path is
            not to be trusted on this model.
    """

    steps: list[GRPOMetrics] = field(default_factory=list)
    logp_verification: VerificationReport | None = None

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
        rollouts: What the engine generated. Each may be up to ``seq_len + 1`` tokens long: the
            layout holds ``seq_len`` *predictions*, and a rollout's final token is only ever a
            target.
        seq_len: Predictions per rollout. A rollout of more than ``seq_len + 1`` tokens is
            truncated, and the tokens that fall off carry no gradient, which is a silent loss of
            signal — so it warns.

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

        # A rollout with no prompt would make the first graded index `n_prompt - 1 + 0` = -1, and
        # numpy would happily write it -- into the LAST slot of the PREVIOUS rollout, corrupting a
        # different sequence's padding with this one's advantage. Nothing would fail.
        if not rollout.prompt_tokens:
            raise ValueError(
                f"rollout {r} has an empty prompt. Position i predicts token i+1, so there has to "
                f"be at least one prompt token for the first completion token to be predicted BY."
            )

        # The fixed layout holds `seq_len` PREDICTIONS, and a rollout's last token is only ever a
        # target -- `targets[i] = full[i+1]`, so `full[seq_len]` is graded from position seq_len - 1
        # and is never fed in. `seq_len + 1` tokens therefore fit exactly, which is the same
        # arithmetic sft.to_batch and packing._prepare do with `usable = len(tokens) - 1`. Cutting
        # at `seq_len` instead dropped the final completion token -- usually the EOS, the one token
        # that says the answer ended -- out of the grading, and warned about a tail that fitted.
        if len(full) > seq_len + 1:
            log.warning(
                "rollout %d is %d tokens but seq_len is %d (room for %d): the tail carries no "
                "gradient",
                r,
                len(full),
                seq_len,
                seq_len + 1,
            )
            full = full[: seq_len + 1]

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

    # Zero graded tokens is not an empty batch -- it is a batch that will run, cost a forward and a
    # backward, produce a loss of exactly 0, and report success. It happens when every prompt is
    # longer than seq_len, which is a configuration error rather than an unlucky round.
    if n_completion == 0:
        raise ValueError(
            f"not one token in this batch is graded: seq_len={seq_len} leaves no room for a "
            f"completion after the prompt. The step would run, cost a full forward and backward, "
            f"return a loss of 0, and train on nothing."
        )

    # Normalize by the completion tokens actually graded, so the update does not get quietly
    # stronger just because the model generated longer answers this round. Folded into the
    # advantages rather than the weights: the weights ARE the mask that ce_sparse multiplies
    # logp_new by, and scaling them would scale the log-probability inside the importance ratio.
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
            # The C struct borrows a bare pointer with no length. A short array is not an error over
            # there; it is an out-of-bounds read, and what it reads is whatever is next in the heap.
            if len(logp_ref) != n:
                raise ValueError(
                    f"logp_ref has {len(logp_ref)} entries but the batch has {n} tokens. The shim "
                    f"borrows this as a bare pointer and would read off the end of it."
                )
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


def reference_logprobs(
    libs: _ffi.Libraries,
    engine: RolloutEngine,
    lm_head: LmHead,
    batch: GRPOBatch,
) -> np.ndarray:
    """The reference policy's logprob of every token in the batch.

    The reference is **this model with the adapter off** — never a second model, never a second set
    of weights (BLUEPRINT D6). And it is scored on the ROLLOUT context, because that is the one
    without an optimizer holding its scheduler: a decode on the training context would free it.

    The adapter is detached with ``llama_set_adapters_lora(ctx, NULL, 0, NULL)`` and re-attached
    afterwards. Note what is *not* done here: DPO's ``_zero_b`` trick zeroes the adapter's B tensors
    in place, and that adapter is the one the optimizer is training. Doing it mid-run would zero the
    policy.

    **This scores the whole flattened batch in one decode**, so the engine's context has to be sized
    for the *batch* and not just for one group: ``n_seq_max >= n_prompts * n_rollouts`` and
    ``n_ctx >= n_prompts * n_rollouts * seq_len``. :func:`train_grpo` checks both up front when
    ``kl_coef > 0``; calling this directly on an engine sized only to generate gets a bare
    ``llama_decode`` failure from :func:`~learning_llamas.logprobs.hidden_states`.

    Args:
        libs: The loaded native libraries.
        engine: The rollout engine, for its inference context and its adapter handle.
        lm_head: The base model's output projection (S1-13).
        batch: The collated rollouts.

    Returns:
        ``[n_tokens]`` F32, zero wherever the batch is masked.
    """
    detached = False
    try:
        libs.llama.llama_set_adapters_lora(engine.ctx, None, 0, None)
        detached = True

        return sequence_logprobs(
            libs,
            engine.ctx,
            lm_head,
            batch.tokens,
            batch.targets,
            batch.mask.tolist(),
            batch.seq_ids,
            batch.positions,
        )
    finally:
        if detached and engine.adapter is not None:
            adapters = (ctypes.c_void_p * 1)(engine.adapter)
            scales = (ctypes.c_float * 1)(1.0)
            libs.llama.llama_set_adapters_lora(engine.ctx, adapters, 1, scales)


def _logp_old_verifier(
    engine: RolloutEngine, lm_head: LmHead, tolerance: float
) -> SelfVerified[np.ndarray]:
    """The self-verification harness wired to its first real customer (S1-16 §3).

    ``fast`` is the sample-time capture (:func:`~learning_llamas.train.rollout.captured_logp`) —
    already recorded as the tokens were drawn, so it costs nothing. ``naive`` is the S1-13 chunked
    recompute (:func:`~learning_llamas.train.rollout.recompute_logp`) — a genuine second forward
    that shares no code with the capture. On the first batch both run and are compared; thereafter
    only the capture, unless it diverged once, in which case only the recompute, for the rest of the
    process (:class:`~learning_llamas.verify.SelfVerified`).
    """
    return SelfVerified(
        captured_logp,
        lambda rollouts: recompute_logp(engine, rollouts, lm_head),
        tolerance,
        "grpo logp_old: sample-time capture vs S1-13 chunked recompute",
    )


def _apply_logp_old(rollouts: RolloutBatch, verified: np.ndarray) -> None:
    """Write the verified ``logp_old`` back onto the rollouts the collator will read.

    A no-op when the capture agreed — the verified values are the ones already on the rollouts. The
    correction when it did not, so the update trains on the recompute the harness fell back to
    rather than on the numbers a divergent fast path produced.
    """
    offset = 0
    for rollout in rollouts.rollouts:
        n = len(rollout.completion_tokens)
        rollout.logp_old = np.asarray(verified[offset : offset + n], dtype=np.float32)
        offset += n


def train_grpo(
    libs: _ffi.Libraries,
    policy,  # noqa: ANN001 - the TRAINING context, adapter attached
    engine: RolloutEngine,
    prompts: Sequence[str],
    reward_fn: RewardFn,
    config: GRPOConfig,
    lm_head: LmHead | None = None,
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
        lm_head: The base model's output projection
            (:func:`~learning_llamas.logprobs.load_lm_head`). **Required when
            ``config.kl_coef > 0``**: the KL is measured against a reference pass that needs it.
            A ``kl_coef`` that quietly did nothing would be the worst of both worlds — a knob
            that reads as if it regularizes and does not. Also enables the sample-time ``logp_old``
            cross-check (S1-16 §3, ``config.logp_old_tol``): given an ``lm_head``, the first batch's
            captured ``logp_old`` is verified against the S1-13 recompute. Passing it with
            ``kl_coef == 0`` is a valid way to ask for that check alone.

    Returns:
        One :class:`GRPOMetrics` per iteration.

    Raises:
        ValueError: If the engine and the policy are using different adapters, if the training
            context is too small, if ``kl_coef > 0`` and the *rollout* context is too small to
            score the whole batch in one pass, or if ``kl_coef > 0`` with no ``lm_head``.
    """
    if config.kl_coef > 0.0 and lm_head is None:
        raise ValueError(
            f"kl_coef={config.kl_coef} needs a reference to measure against, and the reference "
            f"pass needs the base model's lm_head. Pass lm_head=load_lm_head(base_gguf), or set "
            f"kl_coef=0.0 to train without a KL penalty. (Silently ignoring it would leave you "
            f"with a knob that reads as if it regularizes and does not.)"
        )

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

    # The same two numbers, for the ENGINE -- but only when the KL is on, because that is what makes
    # the engine score a batch instead of merely generating one.
    #
    # `reference_logprobs` decodes the WHOLE flattened batch on the inference context: every rollout
    # as its own sequence, seq_ids 0..n_sequences-1, all n_tokens of it in one llama_decode. But
    # generation only ever needs room for a single group at a time, so an engine sized exactly as
    # RolloutEngine documents (n_seq_max >= n_rollouts) is big enough to sample from and too small
    # to score with. llama.cpp rejects a seq_id >= n_seq_max in llama_batch_allocr::init
    # (llama-batch.cpp) by returning false, which surfaces here as a bare "llama_decode failed with
    # status -1" from the middle of iteration 1 -- after the rollouts, with nothing naming the knob.
    if config.kl_coef > 0.0:
        engine_seq_max = int(libs.llama.llama_n_seq_max(engine.ctx))
        if n_sequences > engine_seq_max:
            raise ValueError(
                f"{len(prompts)} prompts x {engine.n_rollouts} rollouts is {n_sequences} "
                f"sequences, but the ROLLOUT context was created with n_seq_max="
                f"{engine_seq_max}. Generation only needs one group at a time, but the KL's "
                f"reference pass scores every rollout at once, each as its own sequence, so the "
                f"rollout context needs n_seq_max >= {n_sequences} too. (Or set kl_coef=0.0, and "
                f"no reference pass runs at all.)"
            )

        # ...and the reference pass is ONE decode of the whole thing, which puts it under two
        # DIFFERENT limits. They are checked separately because they are different numbers, and the
        # obvious single number -- llama_n_ctx() -- is the wrong one for the harsher of the two.
        #
        # The BATCH limit is n_batch, and overrunning it is not a status code to be caught: llama
        # asserts `n_tokens_all <= cparams.n_batch` (llama-context.cpp) and GGML_ASSERT is
        # GGML_ABORT, so the process dies. n_batch is NOT llama_n_ctx(): cparams.n_batch is
        # min(cparams.n_ctx, params.n_batch) taken BEFORE cparams.n_ctx is padded up to a multiple
        # of 256, so on a context whose requested n_ctx is not already a multiple of 256 --
        # load_model(..., n_ctx=64) is the common one -- llama_n_ctx() reports up to 255 more than
        # a decode will accept, and a guard written against it waves through the abort.
        engine_n_batch = int(libs.llama.llama_n_batch(engine.ctx))
        if n_tokens > engine_n_batch:
            raise ValueError(
                f"the KL's reference pass scores {n_sequences} rollouts x seq_len "
                f"{config.seq_len} = {n_tokens} tokens in one decode on the rollout context, but "
                f"that context's n_batch is {engine_n_batch}. Create it with n_batch >= "
                f"{n_tokens} (learning_llamas.Model takes n_batch from n_ctx, so n_ctx >= "
                f"{n_tokens}). llama.cpp does not report this one -- it aborts the process."
            )

        # The CACHE limit is per sequence, because every rollout is scored as its own. On a unified
        # cache n_ctx_seq is the whole pool (and the n_batch check above already bounded the total
        # against it, since n_batch <= the unpadded n_ctx <= the padded one); with kv_unified=False
        # each stream instead gets n_ctx / n_seq_max of its own, and n_tokens <= n_ctx says nothing
        # about whether one rollout fits in one stream.
        engine_n_ctx_seq = int(libs.llama.llama_n_ctx_seq(engine.ctx))
        if config.seq_len > engine_n_ctx_seq:
            raise ValueError(
                f"each of the {n_sequences} rollouts is scored as its own sequence of "
                f"{config.seq_len} tokens, but the rollout context holds only {engine_n_ctx_seq} "
                f"KV cells per sequence (n_ctx_seq). Create it with a larger n_ctx -- or with "
                f"kv_unified=True, which gives every sequence the whole pool."
            )

    result = GRPOResult()

    # The self-verification harness's first customer (S1-16 §3): sample-time logp_old capture (fast)
    # vs the S1-13 chunked recompute (naive). It needs the lm_head to recompute against, so it runs
    # only when one was passed -- the same lm_head the KL already requires.
    verifier: SelfVerified[np.ndarray] | None = None
    if config.logp_old_tol > 0.0 and lm_head is not None:
        verifier = _logp_old_verifier(engine, lm_head, config.logp_old_tol)

    with GRPOTrainer(libs, policy, config) as trainer:
        for _ in range(config.iterations):
            rollouts = engine.generate(prompts, reward_fn)

            # Cross-check the captured logp_old against the recompute on the first batch, then trust
            # the capture (or, if it lied once, the recompute) for the rest of the run. Whichever
            # the harness returns is what the update trains on -- a divergent capture never reaches
            # the loss.
            if verifier is not None:
                _apply_logp_old(rollouts, verifier(rollouts))

            batch = collate(rollouts, config.seq_len)

            # The reference does not move, but the TOKENS do -- these are new rollouts every round,
            # so the reference has to score them afresh. It is a no-grad pass on the rollout context
            # with the adapter off.
            logp_ref = None
            if config.kl_coef > 0.0:
                logp_ref = reference_logprobs(libs, engine, lm_head, batch)

            metrics = trainer.grpo_step(
                batch, mean_reward=rollouts.mean_reward(), logp_ref=logp_ref
            )
            result.steps.append(metrics)

    if verifier is not None:
        result.logp_verification = verifier.report

    return result
