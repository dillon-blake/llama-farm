"""S1-07: packing several samples into one batch, without letting them see each other.

The whole value of packing is throughput; the whole risk of packing is contamination. So the test
that matters is the one at the bottom of this file: **the same samples, packed and unpacked, must
produce the same loss on a real model**. Everything above it is structure — necessary, cheap, and
not sufficient.

Two ways for packing to be silently wrong, both of which that test catches and neither of which a
shape assertion would:

1. **Attention leaks across the boundary.** If packed samples share a sequence id, a token in one
   attends to tokens in another. The loss changes, the model learns from context it will never have
   at inference, and nothing complains.

2. **A sample is trained to predict its neighbour.** Concatenate whole samples and the last token of
   each is asked to produce the first token of the next — a token from a different document. It is
   the classic packing bug, and the classic fix is to remember to zero that weight. This packer
   cannot make the mistake, because it never forms the pair (each sample's targets come only from
   within itself). The invariant is asserted anyway, below, because a structural argument is worth
   less than a check.
"""

import random

import pytest

from learning_llamas import _ffi
from learning_llamas.adapter import create_zero_adapter
from learning_llamas.data import MaskedSample
from learning_llamas.train import TrainConfig, Trainer, collate
from learning_llamas.train.packing import n_sequences, pack

SEQ_LEN = 64
N_CTX = 128
RANK = 4


def _sample(n_prompt: int, n_completion: int, start: int = 7) -> MaskedSample:
    n = n_prompt + n_completion
    tokens = [start + (i % 11) for i in range(n)]
    weights = [0.0] * n_prompt + [1.0] * n_completion
    return MaskedSample(tokens=tokens, weights=weights)


# ---------------------------------------------------------------------------
# Structure.
# ---------------------------------------------------------------------------


def test_every_packed_batch_has_the_same_shape() -> None:
    """The loop requires it and the shim enforces it — the packer must deliver it."""
    rng = random.Random(0)
    samples = [_sample(rng.randint(2, 8), rng.randint(2, 12)) for _ in range(20)]

    batches = pack(samples, seq_len=SEQ_LEN)

    assert batches
    for b in batches:
        assert len(b.tokens) == SEQ_LEN
        assert len(b.targets) == SEQ_LEN
        assert len(b.weights) == SEQ_LEN
        assert len(b.seq_ids) == SEQ_LEN
        assert len(b.positions) == SEQ_LEN


def test_packing_conserves_every_trained_token() -> None:
    """Packing is a throughput change. It may not add or drop a single graded position."""
    rng = random.Random(1)
    samples = [_sample(rng.randint(1, 6), rng.randint(1, 10)) for _ in range(25)]

    packed = pack(samples, seq_len=SEQ_LEN)
    unpacked = collate(samples, seq_len=SEQ_LEN)

    assert sum(b.n_valid for b in packed) == sum(b.n_valid for b in unpacked)
    assert len(packed) < len(unpacked), "packing did not actually pack anything"


def test_each_packed_sample_gets_its_own_sequence_and_restarts_its_positions() -> None:
    """The seq_id IS the isolation, and the positions are what RoPE and the mask compare.

    A packed sample that inherited its neighbour's position offset would be rotated as though it
    began halfway through a document — a subtle, entirely silent corruption.
    """
    samples = [_sample(2, 3), _sample(1, 4, start=30), _sample(3, 2, start=50)]

    batches = pack(samples, seq_len=SEQ_LEN)
    assert len(batches) == 1, "these should all fit in one pack"

    batch = batches[0]

    # Each sample's slots are contiguous, its seq_id unique, its positions 0, 1, 2, ...
    seen: dict[int, list[int]] = {}
    for seq_id, position in zip(batch.seq_ids, batch.positions, strict=True):
        seen.setdefault(seq_id, []).append(position)

    for seq_id, positions in seen.items():
        assert positions == list(range(len(positions))), f"seq {seq_id} positions: {positions}"

    # Three samples plus one sequence for the pads.
    assert n_sequences(batch) == 4


def test_no_position_is_ever_asked_to_predict_a_token_from_another_sample() -> None:
    """The packing bug, asserted out of existence.

    Every ``(input, target)`` pair must come from within one sample. This is the property the
    boundary mask exists to provide, and it holds here by construction rather than by remembering.

    Checked against the SOURCE samples, not against the batch's own internal consistency. A packer
    that pointed every target at its neighbour would be perfectly self-consistent; what it would not
    be is a faithful rendering of the data it was given. Each sample gets a distinct token range, so
    a run's tokens identify which sample produced it, and its targets and weights must then be that
    sample's — exactly, element for element.
    """  # noqa: D208
    rng = random.Random(2)
    samples = [_sample(rng.randint(1, 5), rng.randint(2, 9), start=7 + 13 * i) for i in range(12)]

    by_inputs = {tuple(s.tokens[:-1]): s for s in samples}
    assert len(by_inputs) == len(samples), "the samples must be distinguishable for this to work"

    matched = 0

    for batch in pack(samples, seq_len=SEQ_LEN):
        runs: dict[int, list[int]] = {}
        for i, seq_id in enumerate(batch.seq_ids):
            runs.setdefault(seq_id, []).append(i)

        for seq_id, slots in runs.items():
            # Every slot of a run is contiguous: a sample laid down in one piece.
            assert slots == list(range(slots[0], slots[0] + len(slots))), seq_id

            inputs = tuple(batch.tokens[i] for i in slots)
            if inputs not in by_inputs:
                continue  # the pad sequence

            source = by_inputs[inputs]
            matched += 1

            # THE assertion: this run's targets and weights are its own sample's, shifted. Not the
            # next sample's first token; not anything from outside the document.
            assert [batch.targets[i] for i in slots] == source.tokens[1:]
            assert [batch.weights[i] for i in slots] == source.weights[1:]

    assert matched == len(samples), f"only {matched} of {len(samples)} samples were placed"


def test_pads_are_inert_and_live_in_their_own_sequence() -> None:
    batches = pack([_sample(2, 3)], seq_len=SEQ_LEN, pad_id=99)
    batch = batches[0]

    real = 4  # a 5-token sample yields 4 (input, target) pairs

    assert all(w == 0.0 for w in batch.weights[real:])
    assert all(t == 99 for t in batch.tokens[real:])
    assert len(set(batch.seq_ids[real:])) == 1
    assert batch.seq_ids[real] not in batch.seq_ids[:real]


def test_max_per_pack_is_respected() -> None:
    """A context has only ``n_seq_max`` sequences. This is how a caller stays inside that budget."""
    samples = [_sample(1, 3) for _ in range(9)]

    batches = pack(samples, seq_len=SEQ_LEN, max_per_pack=2)

    assert len(batches) == 5  # ceil(9 / 2)
    for b in batches:
        assert n_sequences(b) <= 3  # at most 2 samples + 1 pad sequence


def test_a_sample_too_long_to_fit_alone_is_refused() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        pack([_sample(4, SEQ_LEN)], seq_len=SEQ_LEN)


def test_packing_is_deterministic() -> None:
    """A packer that rearranges itself between runs makes a training run unreproducible."""
    rng = random.Random(3)
    samples = [_sample(rng.randint(1, 6), rng.randint(2, 10)) for _ in range(15)]

    a = pack(samples, seq_len=SEQ_LEN)
    b = pack(samples, seq_len=SEQ_LEN)

    assert [x.tokens for x in a] == [x.tokens for x in b]
    assert [x.seq_ids for x in a] == [x.seq_ids for x in b]


# ---------------------------------------------------------------------------
# The one that matters.
# ---------------------------------------------------------------------------


@pytest.fixture
def trainable(tiny_q4_k, tmp_path, load_model, libs: _ffi.Libraries):
    """A model with room for several sequences — one per packed sample, plus one for the pads."""
    adapter_path = tmp_path / "adapter.gguf"
    create_zero_adapter(tiny_q4_k, adapter_path, r=RANK, seed=7)

    model = load_model(
        tiny_q4_k,
        n_ctx=N_CTX,
        n_ubatch=SEQ_LEN,
        training=True,
        n_seq_max=8,
    )
    model.attach_adapter(adapter_path, scale=1.0)

    return model


def test_packed_and_unpacked_give_the_same_loss(trainable, libs: _ffi.Libraries) -> None:
    """The test this whole module exists for.

    Packing changes throughput, not arithmetic. If a packed batch's per-valid-token loss differs
    from the pooled loss of the same samples run one at a time, then something crossed a boundary —
    attention leaked between samples, or a position was graded on a neighbour's token. Both are
    invisible to every structural check above, and both would train a real model on real noise.

    The comparison is pooled, not averaged over batches: a sample with three graded tokens must not
    weigh as much as one with twenty.
    """
    model = trainable

    samples = [
        _sample(3, 5, start=7),
        _sample(2, 7, start=20),
        _sample(4, 4, start=33),
        _sample(1, 6, start=46),
    ]

    packed = pack(samples, seq_len=SEQ_LEN, max_per_pack=6)
    unpacked = collate(samples, seq_len=SEQ_LEN)

    assert len(packed) == 1, "the point of the test is that they share a batch"
    assert sum(b.n_valid for b in packed) == sum(b.n_valid for b in unpacked)

    # Forward-only, on the zero-init adapter, so both runs measure exactly the same model.
    with Trainer(libs, model, TrainConfig(lr=1e-4)) as trainer:
        packed_loss = trainer.evaluate(packed)

    with Trainer(libs, model, TrainConfig(lr=1e-4)) as trainer:
        unpacked_loss = trainer.evaluate(unpacked)

    assert packed_loss > 1.0, "implausibly small; this test would prove nothing"

    # F32 summation over a different order of the same terms. 1e-4 is generous for that and far
    # tighter than any real contamination: a single leaked sample moves this by percent, not by
    # parts in ten thousand.
    assert packed_loss == pytest.approx(unpacked_loss, rel=1e-4), (
        f"packed loss {packed_loss:.6f} != unpacked loss {unpacked_loss:.6f}. Attention leaked "
        f"across a sample boundary, or a position was graded on a neighbour's token."
    )


def test_packing_needs_kv_unified(tiny_q4_k, tmp_path, load_model, libs: _ffi.Libraries) -> None:
    """Without it llama.cpp regroups the batch by sequence, and the loss would silently be wrong.

    The ubatch splitter is chosen by the KV cache's stream count: ``n_stream == 1 ? split_simple :
    split_equal``, and ``n_stream`` is ``unified ? 1 : n_seq_max``. ``split_simple`` hands out
    contiguous slices in the original order, which is what the loss builder assumes when it reads
    ``targets[pos + i]``. ``split_equal`` **regroups the tokens by sequence** — after which the
    targets and weights pair with the wrong tokens, and the loss still looks entirely plausible,
    because it is a perfectly valid loss of the wrong thing.

    So the shim refuses rather than computing it.
    """
    adapter_path = tmp_path / "adapter.gguf"
    create_zero_adapter(tiny_q4_k, adapter_path, r=RANK, seed=7)

    model = load_model(
        tiny_q4_k,
        n_ctx=N_CTX,
        n_ubatch=SEQ_LEN,
        training=True,
        n_seq_max=4,
        kv_unified=False,  # <-- the thing under test
    )
    model.attach_adapter(adapter_path, scale=1.0)

    packed = pack([_sample(2, 4), _sample(2, 4, start=30)], seq_len=SEQ_LEN)

    with Trainer(libs, model, TrainConfig(lr=1e-4)) as trainer:
        with pytest.raises(RuntimeError, match="INVALID_ARG"):
            trainer.step(packed[0])


def test_a_seq_id_beyond_n_seq_max_is_a_clear_error(
    tiny_q4_k, tmp_path, load_model, libs: _ffi.Libraries
) -> None:
    """Otherwise the failure surfaces as a batch-allocator complaint that never mentions packing."""
    adapter_path = tmp_path / "adapter.gguf"
    create_zero_adapter(tiny_q4_k, adapter_path, r=RANK, seed=7)

    model = load_model(tiny_q4_k, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True, n_seq_max=2)
    model.attach_adapter(adapter_path, scale=1.0)

    # Three samples plus a pad sequence needs n_seq_max >= 4; the context has 2.
    packed = pack(
        [_sample(1, 3), _sample(1, 3, start=20), _sample(1, 3, start=40)],
        seq_len=SEQ_LEN,
    )
    assert n_sequences(packed[0]) > 2

    with Trainer(libs, model, TrainConfig(lr=1e-4)) as trainer:
        with pytest.raises(RuntimeError, match="INVALID_ARG"):
            trainer.step(packed[0])
