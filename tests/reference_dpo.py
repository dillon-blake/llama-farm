"""The DPO objective in float64 numpy, written to disagree with the graph.

The shim builds ``-log σ(βz)`` as ``softplus(-βz)`` because ggml's SIGMOID has no backward rule
(see ``train/dpo.py``). This file computes the same number the other way — ``np.logaddexp(0, -x)``,
no softplus construction, no relu identities — so agreement means the graph computes the *math*,
not that two transcriptions of the same graph agree with each other.

The gradient is derived from the math too. With ``z = Δ_policy - Δ_ref`` and
``Δ_policy = Σ_i w_i · logp_i(target_i)`` (weights **signed**: +1 on the chosen completion's
tokens, -1 on the rejected's, 0 elsewhere — exactly what ``to_batch`` builds):

    dL/dz          = -β · σ(-βz)
    dΔ/dlogits_i   = w_i · (onehot(target_i) - softmax(logits_i))

so a sign error on the rejected weights, a dropped β, or an ignored ``Δ_ref`` all land in the
gradient, not just the loss.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x: float) -> float:
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # tanh form: stable, and not the graph's construction


def logratio_from_logits(logits: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> float:
    """``Δ = Σ_i w_i · logp_i(target_i)`` with signed weights. Pads and prompts carry w = 0."""
    z = logits - logits.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    rows = np.arange(len(targets))
    return float(np.sum(weights * logp[rows, targets]))


def loss_and_dlogits(
    logits: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    beta: float,
    ref_delta: float,
) -> tuple[float, np.ndarray]:
    """The DPO loss ``-log σ(β(Δ_policy - Δ_ref))`` and its gradient w.r.t. the logits.

    Args:
        logits: ``(T, n_vocab)`` policy logits for the packed pair.
        targets: ``(T,)`` next-token targets.
        weights: ``(T,)`` signed weights (+1 chosen completion, -1 rejected, 0 elsewhere).
        beta: DPO's temperature.
        ref_delta: The frozen reference model's log-ratio for this pair.

    Returns:
        A ``(loss, dlogits)`` pair.
    """
    delta = logratio_from_logits(logits, targets, weights)
    z = beta * (delta - ref_delta)

    loss = float(np.logaddexp(0.0, -z))  # -log sigmoid(z), computed as neither of those
    dz = -beta * _sigmoid(-z)

    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)

    dlogits = -probs * weights[:, None]
    dlogits[np.arange(len(targets)), targets] += weights
    return loss, dz * dlogits
