"""The single source of truth for the convergence run: config, dataset, and identity.

Both sides of the gate import this — the learning-llamas run and the one-time PEFT recorder — so
there is exactly one place a hyperparameter is written down. The ticket's failure mode here is
obvious once named: type the learning rate into two files, change one, and the "reference" quietly
becomes a reference to a different experiment.

:func:`identity` hashes everything that defines the run. It is written into
``reference_curve.json`` and re-checked by the gate, so a curve recorded against a different model,
dataset or config cannot silently be compared against.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib

import numpy as np

# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

RANK = 4

# alpha == rank makes the effective scale exactly 1.0 on BOTH sides, which is the only reason the
# two are comparable:
#   llama.cpp: scale = alpha ? user_scale * alpha / rank : user_scale   (llama-adapter.h:52-57)
#   PEFT:      scaling = lora_alpha / r
# Note llama.cpp's trapdoor: alpha == 0 does not mean "scale by zero", it means "drop the
# alpha/rank factor". Setting alpha = rank sidesteps the branch entirely.
ALPHA = RANK

LR = 1e-3
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 0.0

# lora_dropout MUST be 0. PEFT's default is 0.0 already, but it is stated here because a nonzero
# dropout makes the reference non-deterministic and no amount of seeding fixes that -- the curve
# would simply not be reproducible, and the gate would be comparing against noise.
LORA_DROPOUT = 0.0

SEQ_LEN = 32
N_SAMPLES = 8
EPOCHS = 5  # -> 40 optimizer steps, at grad_accum = 1 and no shuffling

# Shuffling is OFF. The reference has to see the batches in the same order, and reproducing
# python's `random.Random(seed).shuffle` inside torch is a pointless thing to have to be right
# about.
SHUFFLE = False

DATA_SEED = 20260714
ADAPTER_SEED = 3
PAD_ID = 0

# The LoRA target set, in both naming conventions. The GGUF names are what `enumerate_targets`
# returns; the HF names are what PEFT's `target_modules` wants. They must denote the same tensors.
GGUF_TO_HF = {
    "attn_q": "q_proj",
    "attn_k": "k_proj",
    "attn_v": "v_proj",
    "attn_output": "o_proj",
    "ffn_gate": "gate_proj",
    "ffn_up": "up_proj",
    "ffn_down": "down_proj",
}
HF_TARGET_MODULES = sorted(GGUF_TO_HF.values())

CURVE_PATH = pathlib.Path(__file__).parent / "reference_curve.json"


@dataclasses.dataclass(frozen=True)
class RunSpec:
    """One convergence run's adapter and optimizer parameters.

    The module constants above describe *the recorded run* — the one ``reference_curve.json``
    holds, whose ``alpha == rank`` and ``weight_decay == 0`` were chosen to make the PEFT
    comparison clean. Those same choices make the recorded run numerically blind to a dropped
    ``alpha/rank`` factor and to the decay term, so ``tests/test_convergence_variants.py`` re-runs
    the gate with one parameter at a time moved off the recorded value.

    The dataset is deliberately **not** part of a spec: every variant trains on the same committed
    dataset, so the only thing that moved is the parameter under test.

    Defaults are exactly the recorded run.
    """

    rank: int = RANK
    alpha: float = float(ALPHA)
    user_scale: float = 1.0
    lr: float = LR
    betas: tuple[float, float] = BETAS
    eps: float = EPS
    weight_decay: float = WEIGHT_DECAY
    epochs: int = EPOCHS
    adapter_seed: int = ADAPTER_SEED
    grad_accum: int = 1


RECORDED = RunSpec()


# ---------------------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------------------


def dataset(n_vocab: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The committed token/mask dataset, generated deterministically from :data:`DATA_SEED`.

    A masked prompt followed by a completion — so the loss normalization (``1 / sum(w)``, not
    ``1 / n_tokens``) is actually exercised rather than degenerating into a plain mean.

    Args:
        n_vocab: The fixture's vocabulary size.

    Returns:
        ``(tokens, targets, weights)``, each ``(N_SAMPLES, SEQ_LEN)``. ``targets[i, j]`` is the
        token position ``j`` must predict, and ``weights[i, j]`` is 0 on the prompt and 1 on the
        completion — already shifted onto the *prediction*, exactly as
        :func:`learning_llamas.train.sft.to_batch` does it.
    """
    rng = np.random.default_rng(DATA_SEED)

    # Token 0/1/2 are unk/bos/eos in the fixture's sliced SPM vocab; keep clear of them so a stray
    # special token cannot change what is being measured.
    raw = rng.integers(3, n_vocab, size=(N_SAMPLES, SEQ_LEN + 1))

    tokens = raw[:, :SEQ_LEN]
    targets = raw[:, 1 : SEQ_LEN + 1]

    weights = np.ones((N_SAMPLES, SEQ_LEN), dtype=np.float64)
    weights[:, : SEQ_LEN // 2] = 0.0  # the "prompt" half carries no loss

    return tokens, targets, weights


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def identity(n_vocab: int) -> str:
    """A hash of everything that defines this run.

    Recorded into ``reference_curve.json`` and re-checked by the gate. If the fixture generator,
    the config or the dataset changes, the recorded curve describes a different experiment and must
    be regenerated rather than compared against — and this is what makes that failure loud.
    """
    from ..fixtures import gen_tiny_llama

    tokens, targets, weights = dataset(n_vocab)
    payload = json.dumps(
        {
            "fixture": gen_tiny_llama.cache_key(),
            "rank": RANK,
            "alpha": ALPHA,
            "lr": LR,
            "betas": list(BETAS),
            "eps": EPS,
            "weight_decay": WEIGHT_DECAY,
            "lora_dropout": LORA_DROPOUT,
            "seq_len": SEQ_LEN,
            "n_samples": N_SAMPLES,
            "epochs": EPOCHS,
            "shuffle": SHUFFLE,
            "adapter_seed": ADAPTER_SEED,
            "targets": HF_TARGET_MODULES,
            "data": hashlib.sha256(
                tokens.tobytes() + targets.tobytes() + weights.tobytes()
            ).hexdigest(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
