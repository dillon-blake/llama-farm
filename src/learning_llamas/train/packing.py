"""Sample packing: several short samples in one fixed-shape batch, kept invisible to each other.

Short samples waste almost everything. A 40-token sample padded to 512 spends 92% of the forward
pass on padding — and a training step costs the same whether a slot holds a real token or a pad. So
pack: lay several samples end to end in one batch, give each its own ``seq_id``, and restart each
one's positions at 0.

**What stops them contaminating each other** is not a convention, it is llama.cpp's attention mask.
It masks across sequences unconditionally — ``if (s0 != s1) continue`` (``llama-graph.cpp``), on the
training path as much as the cached one — so a token in one packed sample cannot attend to a token
in another, no matter how adjacent they are in the buffer. The seq_id *is* the isolation.

**Boundary masking is structural here, not applied afterwards.** The usual way to pack is to
concatenate whole samples and then remember to zero the weight at each sample's last token, because
otherwise that token is trained to predict the first token of the *next* sample — a sample it has
never seen and cannot see. Forget it, and the model learns a small amount of pure noise.

This packer cannot make that mistake, because it never builds the cross-boundary pair in the first
place: each sample contributes its own ``(input, target)`` pairs, all drawn from within itself, and
its final token appears only as a *target*, never as an input. There is nothing at the boundary to
mask. The test suite still checks the invariant directly — no position may predict a token from
another sample — because the claim is worth more than the argument.

Packing changes throughput, not arithmetic: the per-valid-token loss of a packed batch is the same
number as the pooled loss of the same samples unpacked, and there is a test that says so on a real
model.

Note on placement: the ticket calls for ``data/packing.py``. It lives under ``train/`` because it
produces :class:`~learning_llamas.train.loop.Batch`, and ``data`` importing ``train`` would invert
the dependency between them. The interface is the one the ticket specifies — an iterable of
fixed-shape batches — which is all ``loop.py`` cares about.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from learning_llamas.data import MaskedSample

from .loop import Batch


@dataclass(frozen=True)
class _Placed:
    """One sample's contribution to a pack: its (input, target, weight) triples."""

    tokens: list[int]
    targets: list[int]
    weights: list[float]

    @property
    def n_slots(self) -> int:
        return len(self.tokens)


def _prepare(sample: MaskedSample) -> _Placed:
    """Shift a sample onto its predictions. Every pair is drawn from within the sample.

    The last token is used as a *target* and never as an input, which is what makes a cross-boundary
    prediction structurally impossible rather than merely masked out.
    """
    usable = len(sample.tokens) - 1
    if usable <= 0:
        raise ValueError("a sample needs at least two tokens: one to read, one to predict")

    return _Placed(
        tokens=sample.tokens[:usable],
        targets=sample.tokens[1 : usable + 1],
        weights=sample.weights[1 : usable + 1],
    )


def pack(
    samples: Sequence[MaskedSample],
    seq_len: int,
    pad_id: int = 0,
    max_per_pack: int | None = None,
) -> list[Batch]:
    """Pack samples into fixed-shape batches, first-fit-decreasing.

    Longest first, into the first pack with room. It is deterministic (no shuffling, no ties broken
    at random), which matters more here than optimality: a packer that rearranges itself between
    runs makes a training run unreproducible for no gain.

    Args:
        samples: The tokenized, masked samples.
        seq_len: The fixed length of every batch. Every emitted batch has exactly this length.
        pad_id: The token to pad leftover slots with. Inert: pads carry weight 0 and sit in their
            own sequence, so nothing attends to them and nothing is graded on them.
        max_per_pack: Cap on samples per batch. Leave ``None`` for no cap. A context can only hold
            ``n_seq_max`` sequences, and the shim rejects a seq_id beyond it — so this is how you
            stay inside that budget.

    Returns:
        The packed batches, all of length ``seq_len``.

    Raises:
        ValueError: If a sample cannot fit in ``seq_len`` even alone. Truncating would drop the tail
            of a completion, which is the part being trained on, so it is refused.
    """
    if max_per_pack is not None and max_per_pack < 1:
        raise ValueError(f"max_per_pack must be at least 1, got {max_per_pack}")

    placed = [_prepare(s) for s in samples]

    for original, p in zip(samples, placed, strict=True):
        if p.n_slots > seq_len:
            raise ValueError(
                f"a sample of {len(original.tokens)} tokens needs {p.n_slots} slots and does not "
                f"fit in seq_len={seq_len}, even alone. Truncating would drop the tail of the "
                f"completion, which is the part being trained on, so it is refused. Raise seq_len."
            )

    # Longest first: the classic first-fit-decreasing heuristic. Sorted by (-slots, index) so ties
    # break on the original order and the result does not depend on the sort's stability.
    order = sorted(range(len(placed)), key=lambda i: (-placed[i].n_slots, i))

    packs: list[list[_Placed]] = []
    used: list[int] = []

    for i in order:
        item = placed[i]

        for p, (bin_pack, bin_used) in enumerate(zip(packs, used, strict=True)):
            has_room = bin_used + item.n_slots <= seq_len
            under_cap = max_per_pack is None or len(bin_pack) < max_per_pack

            if has_room and under_cap:
                bin_pack.append(item)
                used[p] += item.n_slots
                break
        else:
            packs.append([item])
            used.append(item.n_slots)

    return [_emit(p, seq_len, pad_id) for p in packs]


def _emit(pack_items: list[_Placed], seq_len: int, pad_id: int) -> Batch:
    """Lay one pack's samples end to end and pad the rest."""
    tokens: list[int] = []
    targets: list[int] = []
    weights: list[float] = []
    seq_ids: list[int] = []
    positions: list[int] = []

    for seq_id, item in enumerate(pack_items):
        tokens.extend(item.tokens)
        targets.extend(item.targets)
        weights.extend(item.weights)
        seq_ids.extend([seq_id] * item.n_slots)

        # Positions restart at 0 for each sample. They are what the attention mask compares, so a
        # packed sample must not inherit its neighbour's offset -- it would then see itself as
        # starting halfway through a sequence, and RoPE would rotate it accordingly.
        positions.extend(range(item.n_slots))

    pad = seq_len - len(tokens)

    # Pads go in a sequence of their own, after every real sample. Nothing attends to them (they are
    # a different sequence) and nothing is graded on them (weight 0), so their contents do not
    # matter -- but they are given a coherent (seq_id, pos) anyway, because a batch with garbage in
    # it is a batch nobody can debug.
    pad_seq = len(pack_items)

    tokens.extend([pad_id] * pad)
    targets.extend([pad_id] * pad)
    weights.extend([0.0] * pad)
    seq_ids.extend([pad_seq] * pad)
    positions.extend(range(pad))

    return Batch(
        tokens=tokens,
        targets=targets,
        weights=weights,
        seq_ids=seq_ids,
        positions=positions,
        pad_count=pad,
        n_samples=len(pack_items),
    )


def n_sequences(batch: Batch) -> int:
    """How many sequences a batch uses — the ``n_seq_max`` a context must have to run it."""
    return max(batch.seq_ids) + 1 if batch.seq_ids else 1
