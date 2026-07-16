"""Direct Preference Optimization: train on pairs, not on labels.

SFT teaches a model to imitate one completion. DPO teaches it to *prefer* one over another, which is
a different and often more useful thing to know — and it needs no reward model, because the reward
is implicit in the ratio between the policy and a frozen reference.

    L = -log σ( β · [ Δ_policy - Δ_reference ] )

        where  Δ = logp(chosen) - logp(rejected)

Four things about that make the implementation, and three of them are pinned by facts rather than
preference.

**It is written as a softplus, not as `-log(sigmoid(x))`.** Not for elegance: ggml's `SIGMOID` has
no backward rule — it falls into the unary switch's default and **aborts** — so `-log σ(x)` cannot
be built that way at all. The identity ``-log σ(x) = softplus(-x)`` gives one node, with a VJP that
exists, and it is numerically stable at large ``|x|`` where a naive ``log(1 + exp(-x))`` is not.

**The reference model is the base model with the adapter off.** Never a second model, never a second
set of weights (BLUEPRINT D6). And its log-ratios are precomputed *before training starts*, not
recomputed each step — toggling the adapter set forces llama.cpp to rebuild its graph, which is not
something to do inside a training loop.

**Chosen and rejected are packed into one batch** (S1-07), as two sequences, so one forward pass
covers both. Attention cannot cross between them; the packer already guarantees that.

**And the weights are ±1.** ``ce_sparse`` produces ``-w_i · logp_i``, so weights of +1 on the chosen
completion's tokens and -1 on the rejected's make its sum exactly ``-(logp(chosen) -
logp(rejected))``
— the log-ratio, in one op, with no second pass and no gather.

Which leaves one invariant worth knowing, because it makes the whole reference pipeline testable in
a
single line: **at initialization the policy IS the reference** (a zero-init adapter is a bitwise
no-op), so the bracket is exactly zero and the loss is exactly ``-log σ(0) = log 2 = 0.6931...``.
Anything else means the reference pass and the training pass are not looking at the same model.
"""

from __future__ import annotations

import ctypes
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from learning_llamas import _ffi
from learning_llamas.data import MaskedSample

from .loop import Batch, Hooks, StepMetrics, TrainableModel, TrainConfig, Trainer
from .packing import pack

# -log sigmoid(0). The loss of a policy that has not moved from its reference.
LOG_2 = math.log(2.0)


@dataclass
class DPOConfig(TrainConfig):
    """:class:`~learning_llamas.train.loop.TrainConfig`, plus DPO's own knobs.

    Attributes:
        beta: How hard the implicit reward pushes. Small beta keeps the policy near the reference;
            large beta lets it move further and, eventually, forget.
        seq_len: The fixed length of every batch. A pair must fit in it together.
        pad_id: The padding token.
        epochs: Passes over the data.
        shuffle: Shuffle the pairs each epoch.
        seed: Seeds the shuffle.
    """

    beta: float = 0.1
    seq_len: int = 512
    pad_id: int = 0
    epochs: int = 1
    shuffle: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject a nonsensical config here, rather than mid-run."""
        super().__post_init__()
        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")


@dataclass(frozen=True)
class Preference:
    """One preference pair: the same prompt, two completions, one of them better.

    Attributes:
        chosen: The preferred completion, masked so only *it* carries weight.
        rejected: The dispreferred one, masked the same way.
    """

    chosen: MaskedSample
    rejected: MaskedSample


@dataclass
class DPOResult:
    """What a run did.

    Attributes:
        steps: Every step, in order.
        reference: The reference model's log-ratio for each pair, in the order they were given.
    """

    steps: list[StepMetrics] = field(default_factory=list)
    reference: list[float] = field(default_factory=list)


def to_batch(pair: Preference, seq_len: int, pad_id: int = 0) -> Batch:
    """Pack a preference pair into one batch, with ±1 weights.

    Chosen and rejected become two sequences, so attention cannot cross between them, and the
    weights
    are +1 on the chosen completion's tokens and -1 on the rejected's. The weighted sum
    ``ce_sparse``
    then produces is exactly the negative log-ratio.

    Args:
        pair: The preference pair.
        seq_len: The fixed batch length. Both completions must fit in it *together*.
        pad_id: The padding token.

    Returns:
        The batch.

    Raises:
        ValueError: If the pair does not fit in ``seq_len``.
    """
    batches = pack([pair.chosen, pair.rejected], seq_len=seq_len, pad_id=pad_id, max_per_pack=2)

    if len(batches) != 1:
        raise ValueError(
            f"a preference pair must fit in one batch of seq_len={seq_len}, but chosen "
            f"({len(pair.chosen.tokens)} tokens) and rejected ({len(pair.rejected.tokens)}) needed "
            f"{len(batches)}. DPO compares them in a single forward pass, so they cannot be split."
        )

    batch = batches[0]

    # The packer lays samples down longest-first, so which sequence id the chosen sample got is not
    # something to assume. Find it by its tokens.
    chosen_inputs = tuple(pair.chosen.tokens[:-1])

    runs: dict[int, list[int]] = {}
    for i, seq_id in enumerate(batch.seq_ids):
        runs.setdefault(seq_id, []).append(i)

    chosen_seq = next(
        (s for s, slots in runs.items() if tuple(batch.tokens[i] for i in slots) == chosen_inputs),
        None,
    )
    if chosen_seq is None:
        raise ValueError("could not find the chosen completion in the packed batch")

    # +1 on the chosen completion's graded tokens, -1 on the rejected's, 0 on prompts and pads.
    weights = [(w if batch.seq_ids[i] == chosen_seq else -w) for i, w in enumerate(batch.weights)]

    return Batch(
        tokens=batch.tokens,
        targets=batch.targets,
        weights=weights,
        seq_ids=batch.seq_ids,
        positions=batch.positions,
        pad_count=batch.pad_count,
        n_samples=batch.n_samples,
    )


def reference_logratios(
    libs: _ffi.Libraries,
    model: TrainableModel,
    batches: Sequence[Batch],
) -> list[float]:
    """The reference model's ``logp(chosen) - logp(rejected)`` for each pair.

    The reference model is this model with **the adapter off**. It is not a second model and it
    costs
    no extra weight memory (BLUEPRINT D6).

    But "adapter off" is **not** ``llama_set_adapters_lora`` with a scale of 0, and that is worth
    knowing because it is the obvious thing to reach for and it does not work here. That call does::

        for (size_t i = 0; i < n_adapters; i++) {
            if (scales[i] != 0.0f) {
                loras->insert({adapters[i], scales[i]});
            }
        }

    — a scale of zero **removes the adapter from the map entirely** (``llama-context.cpp``). So
    ``build_lora_mm`` never injects it, the adapter tensors are not in the graph at all, and the
    next
    graph build dies on::

        GGML_ASSERT(any_params && "no trainable parameters found, did you forget to call
                    ggml_set_param?")

    It is a perfectly good way to turn a LoRA off for *inference*. It is not compatible with a
    training context, whose graph must still contain the parameters.

    What is done instead is exact and does not touch the graph: **zero B**. The LoRA delta is
    ``scale · B(A·x)``, so with ``B = 0`` it is exactly zero — the base model, bit for bit, for any
    A
    — and the adapter tensors stay where the graph expects them. Afterwards B is put back.

    All of them are computed **once, before training starts**, and never again. The reference does
    not
    move, so there is nothing to recompute.

    And it must happen **before the optimizer exists** — before :class:`DPOTrainer` is constructed.

    That is not a nicety. ``ll_logp_delta`` runs a plain ``llama_decode``, and ``llama_context``
    does
    ``sched.reset(ggml_backend_sched_new(...))`` whenever it decides to re-reserve — which a decode
    does. ``ggml_opt_init`` captured the scheduler *pointer* and holds it for the life of the
    optimizer context, so a decode after training is set up leaves the optimizer holding a **freed
    scheduler**, and the next training step dereferences it. (The shim now detects that and returns
    ``SCHED_INVALIDATED`` rather than aborting somewhere inside ggml-backend, but the fix is to not
    do it.)

    Args:
        libs: The loaded native libraries.
        model: A context in training mode with the adapter attached. The optimizer must **not** have
            been initialized yet.
        batches: The packed pairs, as :func:`to_batch` produces them.

    Returns:
        One log-ratio per batch.
    """
    saved = _snapshot_b(libs, model)

    _zero_b(libs, model)  # <- the reference model, exactly
    try:
        return [_logp_delta(libs, model, b) for b in batches]
    finally:
        _restore_b(libs, model, saved)


def _adapter_names(libs: _ffi.Libraries, model: TrainableModel) -> list[str]:
    """Every adapted base tensor's name, from the adapter itself (S1-08)."""
    n = _ffi.check(libs.farm.ll_adapter_n_tensors(model.adapter), "ll_adapter_n_tensors")

    names = []
    for i in range(n):
        buf = ctypes.create_string_buffer(256)
        ne_a = (ctypes.c_int64 * 4)()
        ne_b = (ctypes.c_int64 * 4)()
        _ffi.check(
            libs.farm.ll_adapter_tensor_info(model.adapter, i, buf, 256, ne_a, ne_b),
            "ll_adapter_tensor_info",
        )
        names.append(buf.value.decode())

    return names


# The B tensors are read and written through the ADAPTER handle (S1-08), not through the ll_debug_*
# accessors, and that is load-bearing: the debug accessors need a training context, and the
# reference
# pass has to run BEFORE one exists.
def _snapshot_b(libs: _ffi.Libraries, model: TrainableModel) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}

    n = _ffi.check(libs.farm.ll_adapter_n_tensors(model.adapter), "ll_adapter_n_tensors")

    for i in range(n):
        k = _ffi.check(libs.farm.ll_adapter_get(model.adapter, i, True, None, 0), "ll_adapter_get")
        buf = (ctypes.c_float * k)()
        _ffi.check(libs.farm.ll_adapter_get(model.adapter, i, True, buf, k), "ll_adapter_get")
        out[i] = list(buf)

    return out


def _zero_b(libs: _ffi.Libraries, model: TrainableModel) -> None:
    """The reference model: delta = scale * B(A·x) is exactly zero when B is."""
    n = _ffi.check(libs.farm.ll_adapter_n_tensors(model.adapter), "ll_adapter_n_tensors")

    for i in range(n):
        k = _ffi.check(libs.farm.ll_adapter_get(model.adapter, i, True, None, 0), "ll_adapter_get")
        _ffi.check(
            libs.farm.ll_adapter_set(model.adapter, i, True, (ctypes.c_float * k)(), k),
            "ll_adapter_set",
        )


def _restore_b(libs: _ffi.Libraries, model: TrainableModel, saved: dict[int, list[float]]) -> None:
    for i, values in saved.items():
        buf = (ctypes.c_float * len(values))(*values)
        _ffi.check(
            libs.farm.ll_adapter_set(model.adapter, i, True, buf, len(buf)), "ll_adapter_set"
        )


def _logp_delta(libs: _ffi.Libraries, model: TrainableModel, batch: Batch) -> float:
    n = len(batch.tokens)
    out = ctypes.c_float()

    _ffi.check(
        libs.farm.ll_logp_delta(
            model.ctx,
            (ctypes.c_int32 * n)(*batch.tokens),
            (ctypes.c_int32 * n)(*batch.targets),
            (ctypes.c_float * n)(*batch.weights),
            (ctypes.c_int32 * n)(*batch.seq_ids),
            (ctypes.c_int32 * n)(*batch.positions),
            n,
            ctypes.byref(out),
        ),
        "ll_logp_delta",
    )

    return out.value


class DPOTrainer(Trainer):
    """A :class:`~learning_llamas.train.loop.Trainer` whose step is the DPO objective."""

    def __init__(self, libs, model, config: DPOConfig, total_steps=None, hooks=None) -> None:
        """Same as the base trainer, plus beta."""
        super().__init__(libs, model, config, total_steps=total_steps, hooks=hooks)
        self._beta = config.beta

    def dpo_step(self, batch: Batch, ref_delta: float, train: bool = True) -> float:
        """One DPO step against a precomputed reference log-ratio.

        Args:
            batch: The packed pair, with ±1 weights.
            ref_delta: The reference model's log-ratio for this pair.
            train: False runs the forward pass only.

        Returns:
            The loss.
        """
        n = len(batch.tokens)
        loss = ctypes.c_float()

        _ffi.check(
            self._libs.farm.ll_train_step_dpo(
                self._model.ctx,
                (ctypes.c_int32 * n)(*batch.tokens),
                (ctypes.c_int32 * n)(*batch.targets),
                (ctypes.c_float * n)(*batch.weights),
                (ctypes.c_int32 * n)(*batch.seq_ids),
                (ctypes.c_int32 * n)(*batch.positions),
                n,
                self._beta,
                ref_delta,
                train,
                ctypes.byref(loss),
            ),
            "ll_train_step_dpo",
        )

        return loss.value


def train_dpo(
    libs: _ffi.Libraries,
    model: TrainableModel,
    pairs: Sequence[Preference],
    config: DPOConfig,
    hooks: Hooks | None = None,
) -> DPOResult:
    """Run DPO.

    Args:
        libs: The loaded native libraries.
        model: A context in training mode with the adapter attached. Its ``n_seq_max`` must be at
            least 3 — one sequence per completion, plus one for the pads.
        pairs: The preference pairs.
        config: The hyperparameters.
        hooks: Optional per-step callbacks.

    Returns:
        Every step, and the reference log-ratios.

    Raises:
        ValueError: If ``pairs`` is empty, or a pair does not fit in ``config.seq_len``.
    """
    if not pairs:
        raise ValueError("no preference pairs")

    batches = [to_batch(p, config.seq_len, config.pad_id) for p in pairs]

    total_steps = max(1, len(batches) * config.epochs // config.grad_accum)

    result = DPOResult()
    rng = random.Random(config.seed)

    # BEFORE the trainer, and never again. The reference does not move -- and a decode after the
    # optimizer exists would free the scheduler it is holding. See reference_logratios.
    reference = reference_logratios(libs, model, batches)
    result.reference = reference

    with DPOTrainer(libs, model, config, total_steps=total_steps, hooks=hooks) as trainer:
        order = list(range(len(batches)))

        for _epoch in range(config.epochs):
            if config.shuffle:
                rng.shuffle(order)

            for i in order:
                lr = trainer.apply_schedule()
                loss = trainer.dpo_step(batches[i], reference[i])

                result.steps.append(
                    trainer.record(
                        loss=loss,
                        lr=lr,
                        n_valid=batches[i].n_valid,
                        n_tokens=len(batches[i].tokens),
                        pad_tokens=batches[i].pad_count,
                        n_samples=batches[i].n_samples,
                    )
                )

    return result
