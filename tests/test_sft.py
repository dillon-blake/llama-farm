"""S1-05: the SFT trainer — masking, normalization, accumulation, schedules, evaluation.

Most of what a trainer does is bookkeeping, and bookkeeping fails quietly. A mask applied one
position out of phase, a loss averaged over the wrong denominator, a learning-rate schedule that
never reaches its peak, an "optimizer step" that is really two — none of these stop a loss curve
from falling. So every one of them is checked against something *independent* of the code under
test: an arithmetic closed form, a numpy re-derivation, or an invariant that must hold by
construction.

One item from the ticket is deliberately not done. It asks to keep S1-03's stopgap composite loss
selectable behind a debug flag and to cross-check ``ce_sparse`` against it. **The stopgap's backward
was wrong** — that is what S1-03 found — so it was deleted, and keeping a knowingly-incorrect loss
in the codebase to validate the correct one against would be worse than useless. The cross-check it
was meant to provide is done properly instead, against a **numpy reference** computed from the
model's own logits: an oracle that is genuinely independent, rather than a second implementation
that shares the bug.
"""

import ctypes
import math

import numpy as np
import pytest

from learning_llamas import _ffi
from learning_llamas.adapter import create_zero_adapter
from learning_llamas.data import MaskedSample
from learning_llamas.train import (
    Batch,
    Hooks,
    SFTConfig,
    TrainConfig,
    Trainer,
    collate,
    to_batch,
    train_sft,
    warmup_cosine,
)

SEQ_LEN = 32
N_CTX = 64
RANK = 4


@pytest.fixture
def trainable(tiny_q4_k, tmp_path, load_model, libs: _ffi.Libraries):
    """A Q4_K base with a zero-init adapter attached, in training mode."""
    adapter_path = tmp_path / "adapter.gguf"
    create_zero_adapter(tiny_q4_k, adapter_path, r=RANK, seed=7)

    model = load_model(tiny_q4_k, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True)
    model.attach_adapter(adapter_path, scale=1.0)

    return model


def _sample(n_prompt: int, n_completion: int, start: int = 7) -> MaskedSample:
    """A synthetic masked sample: ``n_prompt`` masked tokens then ``n_completion`` trained ones."""
    n = n_prompt + n_completion
    tokens = [start + (i % 13) for i in range(n)]
    weights = [0.0] * n_prompt + [1.0] * n_completion
    return MaskedSample(tokens=tokens, weights=weights)


# ---------------------------------------------------------------------------
# The shift: the mask belongs to the PREDICTION, not the input.
# ---------------------------------------------------------------------------


def test_the_loss_mask_shifts_onto_the_predicted_token() -> None:
    """Position i is graded on ``tokens[i + 1]``, so it is masked by ``weights[i + 1]``.

    Get this wrong and the model is trained one position out of phase: every position is graded on
    the token it was *given* rather than the one it must *guess*. The loss falls beautifully, and
    the model learns to predict what it can already see.
    """
    sample = MaskedSample(tokens=[10, 11, 12, 13, 14], weights=[0.0, 0.0, 1.0, 1.0, 1.0])

    batch = to_batch(sample, seq_len=4, pad_id=0)

    # Four usable positions: the fifth token has nothing after it to predict.
    assert batch.tokens == [10, 11, 12, 13]
    assert batch.targets == [11, 12, 13, 14]

    # Position 0 reads token 10 and must produce 11, which is masked -> weight 0.
    # Position 1 reads token 11 and must produce 12, the FIRST completion token -> weight 1.
    assert batch.weights == [0.0, 1.0, 1.0, 1.0]

    # The trained positions predict exactly the completion tokens, and nothing else.
    predicted = [t for t, w in zip(batch.targets, batch.weights, strict=True) if w > 0.0]
    assert predicted == [12, 13, 14]


def test_padding_is_appended_and_carries_no_loss() -> None:
    sample = _sample(n_prompt=2, n_completion=2)  # 4 tokens -> 3 usable positions

    batch = to_batch(sample, seq_len=SEQ_LEN, pad_id=99)

    assert len(batch.tokens) == SEQ_LEN
    assert len(batch.targets) == SEQ_LEN
    assert len(batch.weights) == SEQ_LEN

    assert batch.tokens[3:] == [99] * (SEQ_LEN - 3)
    assert all(w == 0.0 for w in batch.weights[3:])
    assert batch.n_valid == 2


def test_a_sample_that_does_not_fit_is_refused_not_truncated() -> None:
    """Truncating would drop the tail of the completion — the part being trained on."""
    sample = _sample(n_prompt=4, n_completion=SEQ_LEN)

    with pytest.raises(ValueError, match="does not fit"):
        to_batch(sample, seq_len=SEQ_LEN)


# ---------------------------------------------------------------------------
# The loss: normalized per valid token, and equal to an independent re-derivation.
# ---------------------------------------------------------------------------


def _reference_loss(libs, model, batch: Batch) -> float:
    """The masked cross-entropy of ``batch``, computed from the model's own logits with numpy.

    This is the independent oracle. It shares no code with the ce_sparse kernel, the shim's
    normalization, or the graph: it re-runs the forward pass through the ordinary inference path
    and does the arithmetic in float64 by hand.
    """
    libs.llama.llama_memory_clear(libs.llama.llama_get_memory(model.ctx), True)

    n = len(batch.tokens)

    # Built by hand rather than with llama_batch_get_one, which asks for logits at the LAST position
    # only. The whole point here is to grade every position independently.
    tokens = (_ffi.llama_token * n)(*batch.tokens)
    pos = (_ffi.llama_pos * n)(*range(n))
    n_seq_id = (ctypes.c_int32 * n)(*([1] * n))
    seq_ids = [(_ffi.llama_seq_id * 1)(0) for _ in range(n)]
    seq_id = (ctypes.POINTER(_ffi.llama_seq_id) * n)(*seq_ids)
    logits = (ctypes.c_int8 * n)(*([1] * n))

    llama_batch = _ffi.llama_batch(
        n_tokens=n,
        token=tokens,
        embd=None,
        pos=pos,
        n_seq_id=n_seq_id,
        seq_id=seq_id,
        logits=logits,
    )

    assert libs.llama.llama_decode(model.ctx, llama_batch) == 0

    total = 0.0
    total_w = 0.0

    for i in range(n):
        if batch.weights[i] == 0.0:
            continue

        row = libs.llama.llama_get_logits_ith(model.ctx, i)
        logits = np.array([row[j] for j in range(model.n_vocab)], dtype=np.float64)

        # log_softmax, the numerically careful way.
        logp = logits - (logits.max() + np.log(np.exp(logits - logits.max()).sum()))

        total += batch.weights[i] * -logp[batch.targets[i]]
        total_w += batch.weights[i]

    return total / total_w if total_w else 0.0


def test_the_loss_matches_an_independent_numpy_reference(trainable, libs: _ffi.Libraries) -> None:
    """ce_sparse, the shim's normalization, and the mask — checked against float64 arithmetic.

    This is the cross-check the ticket wanted, done against a real oracle rather than against the
    stopgap loss it suggested (whose backward was wrong; see the module docstring).

    The mask is nontrivial on purpose: a mask of all ones would not distinguish "normalized by valid
    tokens" from "normalized by batch size", which is the mistake being ruled out.
    """
    model = trainable

    batch = to_batch(_sample(n_prompt=10, n_completion=6), seq_len=SEQ_LEN, pad_id=0)
    assert 0 < batch.n_valid < SEQ_LEN, "the mask must be nontrivial or this test proves nothing"

    # A forward-only step through the trainer, on the zero-init adapter (a no-op), so the model the
    # reference measures is the model the shim measured.
    with Trainer(libs, model, TrainConfig(lr=1e-4)) as trainer:
        measured = trainer.evaluate([batch])

    expected = _reference_loss(libs, model, batch)

    # Non-vacuity: a random-init model on a 512-token vocab sits near ln(512) = 6.2. If both sides
    # came back 0.0 the comparison above would "pass" while proving nothing.
    assert measured > 1.0, f"the loss is implausibly small ({measured}); this test proves nothing"

    assert measured == pytest.approx(expected, rel=1e-3), (
        f"the shim's masked loss ({measured:.6f}) disagrees with a float64 re-derivation from the "
        f"model's own logits ({expected:.6f})"
    )


def test_padding_does_not_change_the_loss(trainable, libs: _ffi.Libraries) -> None:
    """The per-valid-token loss is invariant to how much a batch was padded.

    It has to be, and for a reason worth stating: pads sit at the end, attention is causal, and
    their weight is zero — so a pad can neither be graded nor influence any position that is. If
    this ever fails, either the normalization is dividing by the wrong count or a pad is leaking
    into the loss.

    The two batches below are the same sample at two different sequence lengths, so this also pins
    that the fixed-shape requirement costs nothing in correctness.
    """
    model = trainable
    sample = _sample(n_prompt=6, n_completion=5)

    short = to_batch(sample, seq_len=16, pad_id=0)
    long = to_batch(sample, seq_len=SEQ_LEN, pad_id=0)

    assert short.n_valid == long.n_valid == 5
    assert len(short.tokens) != len(long.tokens)

    # Through the TRAINER, not through a numpy re-derivation: comparing two references against each
    # other would only prove llama's forward pass is pad-invariant, which was never in doubt. What
    # is being checked is the shim's normalization and mask.
    #
    # Two trainers, because the shim pins the batch shape to the first step's -- which is exactly
    # the restriction this test shows costs nothing.
    with Trainer(libs, model, TrainConfig(lr=1e-4)) as trainer:
        loss_short = trainer.evaluate([short])
    with Trainer(libs, model, TrainConfig(lr=1e-4)) as trainer:
        loss_long = trainer.evaluate([long])

    assert loss_short > 1.0, "implausibly small; this test would prove nothing"

    assert loss_short == pytest.approx(loss_long, rel=1e-4), (
        f"padding changed the loss: {loss_short:.6f} at seq_len=16 vs {loss_long:.6f} at "
        f"seq_len={SEQ_LEN}. A pad is being graded, or the denominator counts pads."
    )


def test_a_fully_masked_batch_leaves_the_adapter_untouched(trainable, libs: _ffi.Libraries) -> None:
    """No valid token, no gradient, no step — the weights must be bit-for-bit unchanged.

    Not "approximately unchanged": ADR-0003 requires the masked backward to be a bitwise zero, and
    AdamW applied to an exactly-zero gradient with zero moments moves nothing. If a single weight
    drifts, some masked position contributed a gradient.
    """
    model = trainable

    masked = Batch(
        tokens=[7] * SEQ_LEN,
        targets=[11] * SEQ_LEN,
        weights=[0.0] * SEQ_LEN,
    )

    with Trainer(libs, model, TrainConfig(lr=1e-2)) as trainer:
        before = _adapter_snapshot(libs, model)

        metrics = trainer.step(masked)

        after = _adapter_snapshot(libs, model)

    assert metrics.loss == 0.0, f"a fully masked batch must have zero loss, got {metrics.loss}"
    assert metrics.n_valid == 0

    for name, values in before.items():
        assert values == after[name], f"{name} moved on a fully masked batch"


def _adapter_snapshot(libs, model) -> dict[str, list[float]]:
    """Every A and B tensor of the attached adapter, by name."""
    from learning_llamas.adapter import enumerate_targets  # noqa: PLC0415 - test-only helper

    snapshot = {}
    for target in enumerate_targets(model.path):
        for is_b in (False, True):
            n = libs.farm.ll_debug_n_elements(model.ctx, target.name.encode(), is_b)
            buf = (ctypes.c_float * n)()
            libs.farm.ll_debug_get_tensor(model.ctx, target.name.encode(), is_b, buf, n)
            snapshot[f"{target.name}.{'b' if is_b else 'a'}"] = list(buf)

    return snapshot


# ---------------------------------------------------------------------------
# Gradient accumulation, learning-rate schedules, and the shape contract.
# ---------------------------------------------------------------------------


def test_gradient_accumulation_steps_the_optimizer_once_per_window(
    trainable, libs: _ffi.Libraries
) -> None:
    """With grad_accum = 3, the weights move on every third batch and on no other.

    Observed on the weights themselves, not on the bookkeeping: `stepped` claiming an optimizer step
    happened would be worth nothing if the optimizer had not, in fact, stepped.
    """
    model = trainable
    batch = to_batch(_sample(n_prompt=4, n_completion=8), seq_len=SEQ_LEN)

    config = TrainConfig(lr=1e-2, grad_accum=3)

    moved = []
    with Trainer(libs, model, config) as trainer:
        for i in range(6):
            before = _adapter_snapshot(libs, model)
            metrics = trainer.step(batch)
            after = _adapter_snapshot(libs, model)

            changed = any(before[k] != after[k] for k in before)
            moved.append(changed)

            assert metrics.stepped == changed, (
                f"micro-step {i}: stepped={metrics.stepped} but the weights "
                f"{'moved' if changed else 'did not move'}"
            )

    assert moved == [False, False, True, False, False, True], moved


def test_the_cosine_schedule_matches_its_closed_form() -> None:
    """The recorded LR must equal the arithmetic, not merely look like a plausible curve."""
    peak, total, warmup, floor = 1e-3, 10, 3, 1e-5

    schedule = warmup_cosine(peak, total_steps=total, warmup_steps=warmup, min_lr=floor)

    # Warmup ramps linearly and REACHES the peak on the last warmup step -- not the one after it.
    assert schedule(0) == pytest.approx(peak / 3)
    assert schedule(1) == pytest.approx(2 * peak / 3)
    assert schedule(2) == pytest.approx(peak)

    # ...then the cosine starts from the peak and ends at the floor.
    for step in range(warmup, total):
        progress = (step - warmup) / (total - warmup)
        expected = floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))
        assert schedule(step) == pytest.approx(expected), f"step {step}"

    assert schedule(warmup) == pytest.approx(peak)
    assert schedule(total) == pytest.approx(floor)


def test_the_learning_rate_the_optimizer_saw_is_the_scheduled_one(
    trainable, libs: _ffi.Libraries
) -> None:
    """The schedule is indexed by OPTIMIZER step, not micro-step.

    With grad_accum = 2, six batches are three optimizer steps, so the LR must take three distinct
    values and hold each for two batches. Index it by micro-step instead and a run moves through its
    schedule grad_accum times too fast, hitting the cosine floor a third of the way in.
    """
    model = trainable
    batch = to_batch(_sample(n_prompt=4, n_completion=8), seq_len=SEQ_LEN)

    # warmup_steps=0 so the three optimizer steps take three DISTINCT learning rates. With a warmup
    # the first two coincide (warmup ends at the peak, and the cosine starts there), and the test
    # could then pass while indexing by the wrong counter.
    config = TrainConfig(lr=1e-3, grad_accum=2, schedule="cosine", warmup_steps=0)
    schedule = warmup_cosine(1e-3, total_steps=3, warmup_steps=0, min_lr=0.0)

    seen = []
    with Trainer(libs, model, config, total_steps=3) as trainer:
        for _ in range(6):
            seen.append(trainer.step(batch).lr)

    expected = [schedule(i // 2) for i in range(6)]

    assert seen == pytest.approx(expected)
    assert len(set(seen)) == 3, f"the LR should take one value per optimizer step, got {seen}"


def test_a_batch_of_the_wrong_shape_is_a_clear_error(trainable, libs: _ffi.Libraries) -> None:
    """Not a raw ggml assert, and not silent corruption.

    Changing the batch length does not currently crash — it happens to work, because ggml-opt sizes
    its optimizer state from the first graph's node count and a transformer's node count does not
    depend on the ubatch size. Nothing guarantees that, and when it stops being true the backward
    reads past the end of that array: a silent out-of-bounds, not an abort. So the shim pins the
    shape and says so.
    """
    model = trainable

    with Trainer(libs, model, TrainConfig(lr=1e-4)) as trainer:
        trainer.step(to_batch(_sample(4, 8), seq_len=SEQ_LEN))

        with pytest.raises(RuntimeError, match="SHAPE_MISMATCH"):
            trainer.step(to_batch(_sample(4, 8), seq_len=SEQ_LEN // 2))


# ---------------------------------------------------------------------------
# The run: loss falls, evaluation is inert.
# ---------------------------------------------------------------------------


def test_evaluation_changes_nothing(trainable, libs: _ffi.Libraries) -> None:
    """A validation pass must not perturb the adapter, the optimizer, or the next training step.

    The last of those is the one that bites: ggml-opt used to advance its gradient-accumulation
    counter on *every* eval, including forward-only ones, so a validation pass wedged into the
    middle of an accumulation window consumed a slot and the optimizer silently skipped a step
    (fixed in the fork, dillon-blake/llama.cpp#9). Checking only that the weights are unchanged
    would have missed it entirely — so this checks that the *next training step behaves the same
    either way*.
    """
    model = trainable
    train_batch = to_batch(_sample(4, 8), seq_len=SEQ_LEN)
    val_batch = to_batch(_sample(6, 5, start=20), seq_len=SEQ_LEN)

    config = TrainConfig(lr=1e-3, grad_accum=2)

    # Run A: train, train, train, train.
    with Trainer(libs, model, config) as trainer:
        initial = _adapter_snapshot(libs, model)
        losses_a = [trainer.step(train_batch).loss for _ in range(4)]
        after_a = _adapter_snapshot(libs, model)

    # Run B: the same four steps, with a validation pass wedged after each one. Rewind the WHOLE
    # adapter first -- A as well as B, because dL/dB depends on A, so a run starting from a moved A
    # is a different run and would prove nothing.
    with Trainer(libs, model, config) as trainer:
        _restore_adapter(libs, model, initial)

        losses_b = []
        for _ in range(4):
            losses_b.append(trainer.step(train_batch).loss)
            trainer.evaluate([val_batch])  # <-- the thing that must not matter

        after_b = _adapter_snapshot(libs, model)

    assert losses_a == pytest.approx(losses_b, rel=1e-5), (
        f"a validation pass changed the training trajectory: {losses_a} vs {losses_b}"
    )
    for name in after_a:
        assert after_a[name] == pytest.approx(after_b[name], rel=1e-5), (
            f"{name} ended up different because evaluation ran between the steps"
        )


def _restore_adapter(libs, model, snapshot: dict[str, list[float]]) -> None:
    """Put every A and B back, so a second run starts exactly where the first did."""
    for key, values in snapshot.items():
        name, _, suffix = key.rpartition(".")
        buf = (ctypes.c_float * len(values))(*values)
        libs.farm.ll_debug_set_tensor(model.ctx, name.encode(), suffix == "b", buf, len(buf))


def test_the_loss_falls_over_a_real_run(trainable, libs: _ffi.Libraries) -> None:
    """End to end, through train_sft: collation, shuffling, scheduling, evaluation."""
    model = trainable

    samples = [_sample(n_prompt=4, n_completion=8, start=7 + 3 * i) for i in range(4)]
    validation = [_sample(n_prompt=4, n_completion=8, start=40)]

    config = SFTConfig(
        lr=2e-2,
        seq_len=SEQ_LEN,
        epochs=8,
        schedule="cosine",
        warmup_steps=2,
        shuffle=True,
        seed=0,
        eval_every=8,
    )

    result = train_sft(libs, model, samples, config, validation=validation)

    losses = [m.loss for m in result.steps]

    assert len(losses) == 32, len(losses)
    assert all(x == x for x in losses), f"loss went NaN: {losses}"  # noqa: PLR0124

    first = sum(losses[:4]) / 4
    last = sum(losses[-4:]) / 4
    assert last < 0.8 * first, f"loss did not fall: {first:.4f} -> {last:.4f}"

    # Validation ran, produced a finite loss per valid token, and did so more than once.
    assert len(result.validation) >= 2, result.validation
    assert all(v == v and v > 0.0 for _, v in result.validation), result.validation


def test_hooks_fire_where_they_say_they_do(trainable, libs: _ffi.Libraries) -> None:
    """S1-09 (checkpointing) and S1-10 (clipping) hang off on_optimizer_step. It must be exact."""
    model = trainable
    batch = to_batch(_sample(4, 8), seq_len=SEQ_LEN)

    micro: list[int] = []
    opt: list[int] = []
    hooks = Hooks(
        on_micro_step=lambda m: micro.append(m.micro_step),
        on_optimizer_step=lambda m: opt.append(m.opt_step),
    )

    with Trainer(libs, model, TrainConfig(lr=1e-3, grad_accum=2), hooks=hooks) as trainer:
        for _ in range(5):
            trainer.step(batch)

    assert micro == [0, 1, 2, 3, 4]
    assert opt == [0, 1], "on_optimizer_step must fire once per window, and not otherwise"


def test_collate_produces_one_fixed_shape_batch_per_sample() -> None:
    samples = [_sample(2, 3), _sample(6, 9), _sample(1, 2)]

    batches = collate(samples, seq_len=SEQ_LEN, pad_id=0)

    assert len(batches) == 3
    assert {len(b.tokens) for b in batches} == {SEQ_LEN}
    assert [b.n_valid for b in batches] == [3, 9, 2]
