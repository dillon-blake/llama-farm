"""The step loop: the part of training that is the same for SFT, DPO and GRPO.

What differs between those three is how a batch is *built* and what the loss *means*. What does not
differ is: push a fixed-shape batch through ``ll_train_step``, mutate the learning rate, accumulate
gradients, report. That is what lives here, and it is why SFT (S1-05), DPO (S1-14) and GRPO (S1-16)
share one loop rather than three.

The interface between the data layer and this loop is deliberately narrow: **an iterable of
fixed-shape** :class:`Batch` **objects**. S1-07's packing collator produces the same thing from
several samples at once, and plugs in with no change here.
"""

from __future__ import annotations

import ctypes
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Protocol

from learning_llamas import _ffi
from learning_llamas.preflight import PreflightError, preflight_adapter

from .schedule import constant, warmup_cosine


class TrainableModel(Protocol):
    """What the loop needs from a model: three handles.

    Loading a model, attaching an adapter and freeing them are the caller's business — the trainer
    does not own that lifecycle and has no business guessing at it.

    Attributes:
        ctx: A ``llama_context *``, already in training mode with the adapter attached.
        model: The ``llama_model *``.
        adapter: The ``llama_adapter_lora *`` being trained.
    """

    ctx: int
    model: int
    adapter: int


@dataclass(frozen=True)
class Batch:
    """One fixed-shape training batch.

    What the model sees, what it should predict, and how much each prediction counts.

    The three lists are the same length, and that length must be **identical for every batch of a
    run** — the shim rejects a change with ``LL_ERR_SHAPE_MISMATCH``, because ggml-opt sizes its
    optimizer state from the first graph it sees and indexes it by node index thereafter. Pad short
    batches instead; a pad token carries weight 0 and so contributes exactly nothing.

    Attributes:
        tokens: The input token at each position.
        targets: The token position ``i`` is asked to predict — i.e. ``tokens[i + 1]``.
        weights: How much position ``i``'s prediction counts. ``0.0`` masks it out entirely: the
            loss is zero there and so is the gradient, bitwise (ADR-0003).
        seq_ids: Which sample each position belongs to, when several are **packed** into one batch
            (S1-07). llama.cpp masks attention across sequences, so packed samples cannot see one
            another. ``None`` means the whole batch is one sequence.
        positions: Each position's index *within its own sequence* — so a packed sample's positions
            restart at 0. ``None`` means ``0..n-1``.
    """

    tokens: list[int]
    targets: list[int]
    weights: list[float]
    seq_ids: list[int] | None = None
    positions: list[int] | None = None

    def __post_init__(self) -> None:
        """Reject a ragged batch at construction, not three layers down in ctypes."""
        n = len(self.tokens)
        lengths = {
            "targets": len(self.targets),
            "weights": len(self.weights),
        }
        if self.seq_ids is not None:
            lengths["seq_ids"] = len(self.seq_ids)
        if self.positions is not None:
            lengths["positions"] = len(self.positions)

        wrong = {k: v for k, v in lengths.items() if v != n}
        if wrong:
            raise ValueError(
                f"every field of a batch must be as long as its tokens ({n}); got {wrong}"
            )

    @property
    def n_valid(self) -> int:
        """How many positions actually carry loss."""
        return sum(1 for w in self.weights if w > 0.0)


@dataclass
class TrainConfig:
    """Everything that shapes a run but not the data.

    Attributes:
        lr: Peak learning rate.
        betas: AdamW's ``(beta1, beta2)``.
        eps: AdamW's epsilon.
        weight_decay: AdamW's decoupled weight decay. 0 disables it.
        grad_accum: How many batches make one optimizer step. This is ggml-opt's ``opt_period``:
            the gradients of ``grad_accum`` batches are summed and the optimizer steps on the last.
        grad_clip: Clip the gradients to this **global** norm — one norm over every trainable
            tensor jointly, so the update's length is bounded without rotating its direction.
            0 disables it, and a clip above the gradient's norm is a bit-exact no-op.

            Applied inside the graph, which is the only place it can be: the optimizer step is
            fused into the backward, so there is no moment on the host between the two.
        schedule: ``"constant"`` or ``"cosine"`` (linear warmup, then cosine decay).
        warmup_steps: Optimizer steps to ramp the LR over. Only used by ``"cosine"``.
        min_lr: The floor the cosine decays to.
        preflight: Run the trainability preflight (S1-11) before the first step, and refuse to train
            a model whose gradient path has an op ggml cannot differentiate. Leave it on: the
            alternative is a raw ``GGML_ABORT`` inside ``ggml_compute_backward`` on the first
            backward pass, naming an op enum and nothing else, after the data has tokenized and the
            user has waited. ``False`` is the escape hatch — for reaching that abort deliberately
            (debugging), or when you have reason to believe the op table itself is wrong.
    """

    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_accum: int = 1
    grad_clip: float = 0.0
    schedule: str = "constant"
    warmup_steps: int = 0
    min_lr: float = 0.0
    preflight: bool = True

    def __post_init__(self) -> None:
        """Reject a nonsensical config here, rather than mid-run."""
        if self.grad_accum < 1:
            raise ValueError(f"grad_accum must be at least 1, got {self.grad_accum}")
        if self.grad_clip < 0:
            raise ValueError(f"grad_clip cannot be negative, got {self.grad_clip}")
        if self.schedule not in ("constant", "cosine"):
            raise ValueError(f"unknown schedule {self.schedule!r}; use 'constant' or 'cosine'")


@dataclass(frozen=True)
class StepMetrics:
    """What one call to :meth:`Trainer.step` did.

    Attributes:
        micro_step: How many batches have been pushed through, this one included (0-based).
        opt_step: Which optimizer step this batch belongs to (0-based). With ``grad_accum > 1``
            several consecutive micro-steps share one.
        stepped: Whether the optimizer actually moved the weights on this call. True only on the
            last micro-step of an accumulation window.
        loss: The loss of this batch, **per valid token**. Masked positions are not in the average.
        lr: The learning rate in force.
        n_valid: How many positions carried loss.
        seconds: Wall-clock time of the step.
    """

    micro_step: int
    opt_step: int
    stepped: bool
    loss: float
    lr: float
    n_valid: int
    seconds: float

    @property
    def tokens_per_second(self) -> float:
        """Valid tokens per second — the number that actually sizes a run."""
        return self.n_valid / self.seconds if self.seconds > 0 else 0.0


@dataclass
class Hooks:
    """Named places for the tickets that come later to attach to, without touching the loop.

    Attributes:
        on_micro_step: Called after every batch, optimizer step or not.
        on_optimizer_step: Called only when the weights actually moved. Where S1-09's checkpointing
            and S1-10's gradient clipping belong.
    """

    on_micro_step: Callable[[StepMetrics], None] | None = None
    on_optimizer_step: Callable[[StepMetrics], None] | None = None


@dataclass
class _State:
    micro_step: int = 0
    metrics: list[StepMetrics] = field(default_factory=list)


class Trainer:
    """Owns the optimizer state for one model, and pushes batches through it.

    The learning-rate schedule is applied by assigning to the ``ll_opt_params`` struct the shim is
    reading — no callback, no reconfiguration. :mod:`learning_llamas._ffi.farm` keeps that struct
    alive; see the lifetime note there, which is load-bearing.
    """

    def __init__(
        self,
        libs: _ffi.Libraries,
        model: TrainableModel,
        config: TrainConfig,
        total_steps: int | None = None,
        hooks: Hooks | None = None,
    ) -> None:
        """Flag the adapter as trainable and set up the schedule.

        Args:
            libs: The loaded native libraries.
            model: A context in training mode with the adapter attached.
            config: The hyperparameters.
            total_steps: Total *optimizer* steps the run will take. Required by the cosine
                schedule, which has nowhere to decay to without it.
            hooks: Optional callbacks.

        Raises:
            ValueError: If a cosine schedule is asked for without ``total_steps``.
            RuntimeError: If the shim rejects the adapter (see :class:`_ffi.LLError`).
        """
        if config.schedule == "cosine" and total_steps is None:
            raise ValueError("a cosine schedule needs total_steps: it has nothing to decay towards")

        self._libs = libs
        self._model = model
        self._config = config
        self._hooks = hooks or Hooks()
        self._state = _State()
        self._closed = False

        # The gate, and it runs BEFORE the training optimizer state exists -- deliberately. The walk
        # needs flagged trainable tensors to seed from, so it stands up throwaway optimizer state of
        # its own and tears it down (preflight_adapter); it must not borrow the training context,
        # because opt_step_custom sizes an optimizer context from the first graph it sees and the
        # preflight's forward-only graph is not the training graph -- sharing it aborts the first
        # real backward. Gating first also leaves the training context below free to see the
        # training graph first, as ggml-opt requires. A blocker here is the whole reason S1-11 built
        # the preflight; S1-42 is what finally calls it. preflight=False is the escape hatch.
        if config.preflight:
            report = preflight_adapter(libs, model.ctx, model.model, [model.adapter])
            if not report.trainable:
                raise PreflightError(report)

        self._params = _ffi.ll_opt_params(
            alpha=config.lr,
            beta1=config.betas[0],
            beta2=config.betas[1],
            eps=config.eps,
            wd=config.weight_decay,
        )

        self.n_params = _ffi.opt_init_lora(
            libs,
            model.ctx,
            model.model,
            [model.adapter],
            self._params,
            opt_period=config.grad_accum,
            grad_clip=config.grad_clip,
        )

        if config.schedule == "cosine":
            self._schedule = warmup_cosine(
                config.lr,
                total_steps=total_steps,
                warmup_steps=config.warmup_steps,
                min_lr=config.min_lr,
            )
        else:
            self._schedule = constant(config.lr)

    @property
    def metrics(self) -> list[StepMetrics]:
        """Every step so far, in order."""
        return list(self._state.metrics)

    def step(self, batch: Batch) -> StepMetrics:
        """Train on one batch.

        With ``grad_accum > 1`` this accumulates a gradient and only *sometimes* moves the weights;
        :attr:`StepMetrics.stepped` says which happened.

        Args:
            batch: A fixed-shape batch. Every batch of a run must be the same length.

        Returns:
            What the step did.

        Raises:
            RuntimeError: If the shim rejects the step — most usefully ``SHAPE_MISMATCH``, which
                means this batch is a different length than the first.
        """
        lr = self.apply_schedule()

        started = time.perf_counter()
        loss = self._run(batch, train=True)
        elapsed = time.perf_counter() - started

        return self.record(loss=loss, lr=lr, n_valid=batch.n_valid, seconds=elapsed)

    def apply_schedule(self) -> float:
        """Set the learning rate for the step that is about to run, and return it.

        This is the whole of "applying a learning-rate schedule": the shim reads the params struct
        on
        its way into the optimizer, so assigning to it *is* the schedule. Nothing needs telling.

        Split out from :meth:`step` so that a trainer with a different objective — DPO, GRPO — can
        reuse the bookkeeping instead of reimplementing it slightly differently.

        Returns:
            The learning rate now in force.
        """
        lr = self._schedule(self._state.micro_step // self._config.grad_accum)
        self._params.alpha = lr

        return lr

    def record(self, loss: float, lr: float, n_valid: int, seconds: float = 0.0) -> StepMetrics:
        """Book a completed step: advance the counters, fire the hooks, keep the metrics.

        Args:
            loss: What the step's objective came out at.
            lr: The learning rate it used.
            n_valid: How many positions carried loss.
            seconds: Wall-clock time.

        Returns:
            What the step did.
        """
        opt_step = self._state.micro_step // self._config.grad_accum

        # The optimizer fires on the LAST micro-step of a window, so `stepped` is true when the
        # next micro-step would start a new one.
        stepped = (self._state.micro_step + 1) % self._config.grad_accum == 0

        metrics = StepMetrics(
            micro_step=self._state.micro_step,
            opt_step=opt_step,
            stepped=stepped,
            loss=loss,
            lr=lr,
            n_valid=n_valid,
            seconds=seconds,
        )

        self._state.micro_step += 1
        self._state.metrics.append(metrics)

        if self._hooks.on_micro_step:
            self._hooks.on_micro_step(metrics)
        if stepped and self._hooks.on_optimizer_step:
            self._hooks.on_optimizer_step(metrics)

        return metrics

    def evaluate(self, batches: Iterable[Batch]) -> float:
        """The masked loss over a held-out split, forward-only.

        Nothing is written: no gradient, no optimizer moment, no weight. The forward pass is the
        same graph the training step builds — ``ll_train_step(train=False)`` toggles the backward
        off, exactly as the stock epoch loop does — so this measures the model that is training,
        not a re-derivation of it.

        Args:
            batches: The held-out batches. Same fixed shape as the training ones.

        Returns:
            The mean loss per valid token, pooled over the split. Pooled, not averaged over batches:
            a batch with two valid tokens must not weigh as much as one with two hundred. Returns
            ``0.0`` for a split with no valid tokens at all.
        """
        total_loss = 0.0
        total_valid = 0

        for batch in batches:
            n_valid = batch.n_valid
            if n_valid == 0:
                continue

            loss = self._run(batch, train=False)

            total_loss += loss * n_valid
            total_valid += n_valid

        return total_loss / total_valid if total_valid else 0.0

    def close(self) -> None:
        """Release the shim's training state. Idempotent."""
        if not self._closed:
            _ffi.opt_free(self._libs, self._model.ctx)
            self._closed = True

    def __enter__(self) -> Trainer:
        """Enter a training scope; :meth:`close` runs on the way out."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Release the shim's training state, whatever happened inside."""
        self.close()

    def _run(self, batch: Batch, train: bool) -> float:
        n = len(batch.tokens)

        tokens = (ctypes.c_int32 * n)(*batch.tokens)
        targets = (ctypes.c_int32 * n)(*batch.targets)
        weights = (ctypes.c_float * n)(*batch.weights)

        # NULL is the unpacked default: one sequence, positions 0..n-1. The shim reads it that way.
        seq_ids = (ctypes.c_int32 * n)(*batch.seq_ids) if batch.seq_ids else None
        positions = (ctypes.c_int32 * n)(*batch.positions) if batch.positions else None

        loss = ctypes.c_float()

        _ffi.check(
            self._libs.farm.ll_train_step(
                self._model.ctx,
                tokens,
                targets,
                weights,
                seq_ids,
                positions,
                n,
                train,
                ctypes.byref(loss),
            ),
            "ll_train_step",
        )

        return loss.value


def run(
    trainer: Trainer,
    batches: Iterable[Batch],
    max_steps: int | None = None,
) -> Iterator[StepMetrics]:
    """Push batches through ``trainer`` until they run out, or ``max_steps`` optimizer steps pass.

    Args:
        trainer: The trainer.
        batches: The training batches.
        max_steps: Stop after this many *optimizer* steps. ``None`` means one pass over ``batches``.

    Yields:
        The metrics of each micro-step, as it happens — so a caller can log a long run without
        waiting for it to end.
    """
    for batch in batches:
        metrics = trainer.step(batch)
        yield metrics

        if max_steps is not None and metrics.stepped and metrics.opt_step + 1 >= max_steps:
            return
