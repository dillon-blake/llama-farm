"""S1-16 — check the fast path against the slow one, once, and fall back forever if it lied.

Every optimization in this project claims to be *exactness-preserving*: the chunked logprob pass
(S1-13) claims to compute what a full-logits pass computes; sample-time `logp_old` capture (S1-15)
claims to compute what a recompute computes. Those claims are testable, and a test proves they held
on the fixture, on the day it was written.

They are not the same as the claim that matters, which is that they hold **on your model, on your
data, right now**. A quantization the tests never saw, an architecture with a logit softcap, a batch
shape that trips a different kernel — any of these can make a fast path diverge from the slow one
while every test in the suite stays green.

So: compare them, on the first real call, against the real inputs. If they agree, use the fast one
and never pay for the check again. If they do not, say so loudly and use the slow one **for the rest
of the process** — because a fast path that was wrong once has forfeited the benefit of the doubt,
and silently correct-most-of-the-time is the worst thing a numerical routine can be.

Provenance (ROADMAP §13): the *discipline* is unsloth's — compare against the naive path on first
use, fall back permanently on mismatch. The code is not. unsloth's GRPO orchestration is AGPL-marked
and was not read; this interface and its implementation are original.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np

log = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class VerificationReport:
    """What the first-use comparison found.

    Attributes:
        name: What was being checked.
        verified: True once the comparison has run.
        agreed: Whether the fast path matched the naive one.
        max_deviation: The largest absolute difference seen.
        tolerance: What was allowed.
    """

    name: str
    verified: bool = False
    agreed: bool = True
    max_deviation: float = 0.0
    tolerance: float = 0.0

    def summary(self) -> str:
        """One line, for a log or a test failure."""
        if not self.verified:
            return f"{self.name}: not yet verified"
        if self.agreed:
            return (
                f"{self.name}: verified, max deviation {self.max_deviation:.3e} "
                f"(tolerance {self.tolerance:.3e})"
            )
        return (
            f"{self.name}: FAILED verification — max deviation {self.max_deviation:.3e} exceeds "
            f"tolerance {self.tolerance:.3e}. Permanently using the naive path."
        )


class SelfVerified(Generic[T]):
    """A fast path that has to earn its place.

    On the first call, runs both paths on the real inputs and compares. Afterwards, runs only the
    fast one — unless the comparison failed, in which case it runs only the naive one, forever.

    "Forever" is deliberate. A fast path that diverged once will diverge again, on inputs you cannot
    predict, and the failure is silent by construction: the numbers stay finite, the loss still
    falls, and the model trains against something subtly other than what you asked for. Re-testing
    it periodically would just mean being wrong between tests.

    Args:
        fast: The optimized path.
        naive: The obvious one — the definition of correct.
        tolerance: Largest absolute deviation that still counts as agreement.
        name: For the log line and the report.

    Example:
        >>> capture = SelfVerified(sample_time_logp, chunked_recompute, 5e-3, "logp_old")
        >>> values = capture(rollouts)   # runs both, compares, then uses the fast one
    """

    def __init__(
        self,
        fast: Callable[..., T],
        naive: Callable[..., T],
        tolerance: float,
        name: str,
    ) -> None:
        self.fast = fast
        self.naive = naive
        self.report = VerificationReport(name=name, tolerance=tolerance)

    @property
    def using_fallback(self) -> bool:
        """True once the fast path has been retired for lying."""
        return self.report.verified and not self.report.agreed

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """Run the fast path — verifying it first, or refusing it if it already failed."""
        if self.using_fallback:
            return self.naive(*args, **kwargs)

        if self.report.verified:
            return self.fast(*args, **kwargs)

        fast_value = self.fast(*args, **kwargs)
        naive_value = self.naive(*args, **kwargs)

        deviation = _max_deviation(fast_value, naive_value)

        self.report.verified = True
        self.report.max_deviation = deviation
        self.report.agreed = deviation <= self.report.tolerance

        if self.report.agreed:
            log.info("%s", self.report.summary())
            return fast_value

        # Loudly. A silent fallback is a performance mystery six months from now, and the deviation
        # is the only evidence anyone will have about which of the two is wrong.
        log.error("%s", self.report.summary())
        return naive_value


def _max_deviation(fast: Any, naive: Any) -> float:
    """The largest absolute difference between two results, whatever shape they are."""
    a = np.asarray(fast, dtype=np.float64)
    b = np.asarray(naive, dtype=np.float64)

    if a.shape != b.shape:
        # A shape disagreement is not a small deviation; it is a different answer.
        return float("inf")

    if a.size == 0:
        return 0.0

    return float(np.max(np.abs(a - b)))
