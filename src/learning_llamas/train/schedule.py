"""Learning-rate schedules.

Each schedule is a plain function of the **optimizer step** (not the micro-step), returning a
learning rate. That is deliberate: it makes a schedule a closed form that a test can check against
its own arithmetic, rather than a stateful object whose correctness depends on being called exactly
once per step.

Applying one is a single attribute assignment — ``params.alpha = schedule(step)`` — because the
shim reads the caller's ``ll_opt_params`` struct on every step (S1-01). There is no callback and no
optimizer to reconfigure.
"""

from __future__ import annotations

import math
from collections.abc import Callable


def constant(lr: float) -> Callable[[int], float]:
    """A flat learning rate.

    Args:
        lr: The learning rate.

    Returns:
        A schedule that returns ``lr`` at every step.
    """
    return lambda step: lr


def warmup_cosine(
    peak_lr: float,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr: float = 0.0,
) -> Callable[[int], float]:
    """Linear warmup, then cosine decay from ``peak_lr`` to ``min_lr``.

    Warmup ramps over ``warmup_steps`` optimizer steps and reaches ``peak_lr`` on the *last* warmup
    step, not the one after it — so step ``warmup_steps - 1`` is the peak, and the cosine begins at
    ``warmup_steps``. Off-by-one here is the difference between ever seeing the peak LR and not.
    A consequence worth stating, because it looks like a bug and is not: ``cos(0) == 1``, so the
    cosine's own first point is ``peak_lr`` too, and steps ``warmup_steps - 1`` and ``warmup_steps``
    both run at exactly the peak. The peak occupies two optimizer steps of a warmed-up run.

    **The cosine's endpoint is step** ``total_steps`` **— which the run does not execute.**
    ``progress = (step - warmup_steps) / (total_steps - warmup_steps)`` reaches 1 — and the schedule
    reaches ``min_lr`` — only at ``step == total_steps``, while
    :func:`~learning_llamas.train.loop.run` calls this with steps ``0 .. total_steps - 1``. So the
    last step a run actually takes sits one cosine increment *above* ``min_lr``: with
    ``peak_lr=1e-4, min_lr=1e-5, total_steps=4, warmup_steps=0`` the final executed step is
    ``2.32e-5``, not ``1e-5``.

    That is the standard convention (it is what HF's ``get_cosine_schedule_with_warmup`` does) and
    it is deliberate here for a reason of its own: ``min_lr`` defaults to 0, and a schedule that hit
    0 on the last executed step would spend the run's final optimizer step multiplying the update by
    zero — a full forward and backward that moves nothing. Decaying *towards* the floor and stopping
    just short of it keeps every step useful. If you want the run to end at ``min_lr``, pass
    ``total_steps = n_steps - 1``.

    Args:
        peak_lr: The learning rate at the end of warmup.
        total_steps: Total optimizer steps in the run. The cosine bottoms out at ``min_lr`` at step
            ``total_steps`` — one past the last step a run of that length takes; see above.
        warmup_steps: How many steps to ramp over. 0 starts at ``peak_lr``.
        min_lr: The floor the cosine decays towards.

    Returns:
        A schedule mapping optimizer step to learning rate.

    Raises:
        ValueError: If ``warmup_steps`` exceeds ``total_steps``, or ``total_steps`` is not positive.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if warmup_steps > total_steps:
        raise ValueError(f"warmup_steps ({warmup_steps}) exceeds total_steps ({total_steps})")

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return peak_lr * (step + 1) / warmup_steps

        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / decay_steps)

        return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    return schedule
