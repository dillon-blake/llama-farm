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

    Args:
        peak_lr: The learning rate at the end of warmup.
        total_steps: Total optimizer steps in the run. The cosine reaches ``min_lr`` at the last.
        warmup_steps: How many steps to ramp over. 0 starts at ``peak_lr``.
        min_lr: The floor the cosine decays to.

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
