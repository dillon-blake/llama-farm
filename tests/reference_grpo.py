"""A numpy GRPO loss, written to disagree with the graph.

The shim builds the PPO clip out of RELU identities, because that is what has a backward rule:

    clip(r, lo, hi) = lo + relu(r - lo) - relu(r - hi)
    min(a, b)       = b - relu(b - a)

Those identities are *exactly* true, which is the point — but "exactly true on paper" and "the graph
I wrote computes them" are different claims, and only the second one matters. So this module
computes the same loss the obvious way: `np.clip`, `np.minimum`, and `np.expm1`, in float64, with
no relu anywhere. If the graph agrees with this, the composite is right.

Deliberately *not* a translation of the graph. A reference that shares the graph's structure shares
its bugs.
"""

from __future__ import annotations

import numpy as np


def clip(r: np.ndarray, eps: float) -> np.ndarray:
    """``np.clip``, not a relu composite. That is the whole point of this file."""
    return np.clip(r, 1.0 - eps, 1.0 + eps)


def surrogate(ratio: np.ndarray, adv: np.ndarray, eps: float) -> np.ndarray:
    """PPO's clipped surrogate, per token: ``min(r·A, clip(r)·A)``.

    The `min` is taken **after** multiplying by the advantage, and it has to be: for a negative
    advantage the two arguments swap, and PPO's whole asymmetry — between having made a good token
    likelier and having made a bad token likelier — lives in that swap. Clipping the ratio and
    *then* deciding would get the sign of the pessimism backwards on exactly half the tokens.
    """
    return np.minimum(ratio * adv, clip(ratio, eps) * adv)


def k3_kl(logp_new: np.ndarray, logp_ref: np.ndarray) -> np.ndarray:
    """Schulman's k3 estimator: ``exp(d) - d - 1`` with ``d = logp_ref - logp_new``.

    Always non-negative, and unbiased for the KL — unlike the naive ``logp_new - logp_ref``, which
    is unbiased but can go negative on a single sample and then *rewards* divergence.
    """
    d = logp_ref - logp_new
    return np.exp(d) - d - 1.0


def grpo_loss(
    logp_new: np.ndarray,
    logp_old: np.ndarray,
    adv: np.ndarray,
    mask: np.ndarray,
    eps: float = 0.2,
    kl_coef: float = 0.0,
    logp_ref: np.ndarray | None = None,
) -> float:
    """The scalar the shim's graph should produce.

    Args:
        logp_new: ``[n]`` the current policy's logprob of each target token.
        logp_old: ``[n]`` the behaviour policy's, from sample time.
        adv: ``[n]`` each token's group advantage (already masked).
        mask: ``[n]`` 1.0 on completion tokens, 0.0 on prompt and padding.
        eps: PPO's epsilon.
        kl_coef: Weight on the KL penalty. 0.0 turns it off.
        logp_ref: ``[n]`` the reference policy's logprob. None means "no reference", which makes
            the KL term exactly zero.

    Returns:
        ``-sum(surrogate) + kl_coef * sum(k3)``, summed over completion tokens only.
    """
    logp_new = np.asarray(logp_new, dtype=np.float64)
    logp_old = np.asarray(logp_old, dtype=np.float64)
    adv = np.asarray(adv, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)

    # The shim zeroes every GRPO input on a masked token, and so does this: on a masked position
    # ce_sparse makes logp_new exactly 0, so a nonzero logp_ref there would blow the KL up to inf
    # and inf*0 is NaN. Mirroring it here keeps the two implementations comparing the same thing.
    live = mask != 0.0
    logp_new = np.where(live, logp_new, 0.0)
    logp_old = np.where(live, logp_old, 0.0)
    adv = np.where(live, adv, 0.0)

    ratio = np.exp(logp_new - logp_old)
    policy = surrogate(ratio, adv, eps)

    total = -float(policy.sum())

    if kl_coef != 0.0:
        reference = logp_old if logp_ref is None else np.asarray(logp_ref, dtype=np.float64)
        reference = np.where(live, reference, 0.0)
        total += kl_coef * float((k3_kl(logp_new, reference) * mask).sum())

    return total
