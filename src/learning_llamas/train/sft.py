"""Supervised fine-tuning: turn masked chat samples into batches, and train on them.

Two things here are easy to get subtly wrong, so both are stated rather than assumed.

**The shift.** :class:`~learning_llamas.data.mask.MaskedSample` marks which *tokens* are the
assistant's completion. But a causal model does not take loss *on* a token — it takes loss on the
*prediction of* a token, which happens one position earlier. So position ``i`` is asked to produce
``tokens[i + 1]``, and it should be graded exactly when ``tokens[i + 1]`` is a completion token:

    tokens[i]  = sample.tokens[i]
    targets[i] = sample.tokens[i + 1]
    weights[i] = sample.weights[i + 1]      <-- the mask shifts with the target, not the input

Take the mask straight from ``sample.weights[i]`` instead and every position is graded on the token
it was *given* rather than the one it must *guess*: the model is trained one step out of phase, the
loss still falls, and it learns to predict the token it can already see.

**Padding is inert, and that is provable rather than hoped-for.** Pads go at the end, carry weight
0, and attention is causal — so a pad can neither be graded (weight 0) nor influence any position
that is (it comes after them all). The pad token's *identity* is therefore irrelevant, and the
per-valid-token loss of a batch does not depend on how much it was padded. There is a test for
that, because it is the assumption every fixed-shape collator rests on.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from learning_llamas import _ffi
from learning_llamas.data import MaskedSample

from .loop import Batch, Hooks, StepMetrics, TrainableModel, TrainConfig, Trainer


@dataclass
class SFTConfig(TrainConfig):
    """:class:`~learning_llamas.train.loop.TrainConfig`, plus how to shape the batches.

    Attributes:
        seq_len: The fixed length of every batch. Samples shorter than this are padded; a sample
            longer than this is an error, not a silent truncation.
        pad_id: The token to pad with. Its identity does not matter — see the module docstring —
            but the model's EOS or pad token is the conventional choice.
        epochs: Passes over the data.
        shuffle: Shuffle the samples each epoch.
        seed: Seeds the shuffle, so a run is reproducible.
        eval_every: Run the validation split every this many optimizer steps. 0 evaluates only at
            the end of the run.
    """

    seq_len: int = 512
    pad_id: int = 0
    epochs: int = 1
    shuffle: bool = True
    seed: int = 0
    eval_every: int = 0


@dataclass
class SFTResult:
    """What a run did.

    Attributes:
        steps: Every micro-step, in order.
        validation: ``(optimizer_step, loss_per_valid_token)`` at each evaluation.
    """

    steps: list[StepMetrics] = field(default_factory=list)
    validation: list[tuple[int, float]] = field(default_factory=list)

    @property
    def final_loss(self) -> float:
        """The training loss of the last step, per valid token."""
        return self.steps[-1].loss if self.steps else 0.0


def to_batch(sample: MaskedSample, seq_len: int, pad_id: int = 0) -> Batch:
    """Turn one masked sample into one fixed-shape batch.

    Args:
        sample: A tokenized sample with its per-token loss mask.
        seq_len: The fixed length to pad to.
        pad_id: The padding token. Inert — see the module docstring.

    Returns:
        The batch, with the loss mask shifted onto the *predictions*.

    Raises:
        ValueError: If the sample does not fit in ``seq_len``. Truncating it would silently drop
            the end of the completion — which is exactly the part being trained on — so it is
            refused instead. Raise ``seq_len``, or pack (S1-07).
    """
    # The last token has nothing after it to predict, so it can never be an input that carries loss.
    usable = len(sample.tokens) - 1

    if usable > seq_len:
        raise ValueError(
            f"a sample of {len(sample.tokens)} tokens does not fit in seq_len={seq_len} "
            f"(it needs {usable}). Truncating would drop the tail of the completion, which is the "
            f"part being trained on, so it is refused. Raise seq_len or pack the samples (S1-07)."
        )
    if usable <= 0:
        raise ValueError("a sample needs at least two tokens: one to read, one to predict")

    pad = seq_len - usable

    tokens = sample.tokens[:usable] + [pad_id] * pad
    targets = sample.tokens[1 : usable + 1] + [pad_id] * pad
    weights = sample.weights[1 : usable + 1] + [0.0] * pad

    # One sample per batch, padded to seq_len -- the throughput counters need both to report a pad
    # fraction and a samples-per-pack of 1 (S1-07).
    return Batch(tokens=tokens, targets=targets, weights=weights, pad_count=pad, n_samples=1)


def collate(samples: Sequence[MaskedSample], seq_len: int, pad_id: int = 0) -> list[Batch]:
    """One batch per sample, all the same shape.

    This is the simple collator. S1-07's packing collator produces the same thing — an iterable of
    fixed-shape :class:`~learning_llamas.train.loop.Batch` — from several samples per batch, and
    plugs into the same loop.

    Args:
        samples: The tokenized, masked samples.
        seq_len: The fixed length of every batch.
        pad_id: The padding token.

    Returns:
        One batch per sample.
    """
    return [to_batch(s, seq_len, pad_id) for s in samples]


def train_sft(
    libs: _ffi.Libraries,
    model: TrainableModel,
    samples: Sequence[MaskedSample],
    config: SFTConfig,
    validation: Sequence[MaskedSample] | None = None,
    hooks: Hooks | None = None,
) -> SFTResult:
    """Run supervised fine-tuning.

    Args:
        libs: The loaded native libraries.
        model: A context in training mode with the adapter attached. Its lifecycle is the caller's.
        samples: The training samples, already tokenized and masked by
            :func:`~learning_llamas.data.mask.build_masked_sample`.
        config: The hyperparameters and batch shape.
        validation: A held-out split. Evaluated forward-only; nothing about it is trained on.
        hooks: Optional per-step callbacks.

    Returns:
        The metrics of every step, and the validation loss at each evaluation.

    Raises:
        ValueError: If ``samples`` is empty, or a sample does not fit in ``config.seq_len``.
    """
    if not samples:
        raise ValueError("no training samples")

    train_batches = collate(samples, config.seq_len, config.pad_id)
    val_batches = collate(validation, config.seq_len, config.pad_id) if validation else []

    # Optimizer steps, not micro-steps: the cosine schedule decays over these, and a schedule that
    # thought there were grad_accum times as many would barely leave the warmup.
    total_micro = len(train_batches) * config.epochs
    total_steps = max(1, total_micro // config.grad_accum)

    result = SFTResult()
    rng = random.Random(config.seed)

    with Trainer(libs, model, config, total_steps=total_steps, hooks=hooks) as trainer:
        for _epoch in range(config.epochs):
            order = list(train_batches)
            if config.shuffle:
                rng.shuffle(order)

            for batch in order:
                metrics = trainer.step(batch)
                result.steps.append(metrics)

                due = (
                    val_batches
                    and metrics.stepped
                    and config.eval_every
                    and (metrics.opt_step + 1) % config.eval_every == 0
                )
                if due:
                    result.validation.append((metrics.opt_step, trainer.evaluate(val_batches)))

        if val_batches:
            last_step = result.steps[-1].opt_step
            if not result.validation or result.validation[-1][0] != last_step:
                result.validation.append((last_step, trainer.evaluate(val_batches)))

    return result
