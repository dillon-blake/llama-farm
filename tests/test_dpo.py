"""S1-14: DPO — train on preferences, against a reference that does not move.

There is one test here that is worth more than the rest put together, and it is the cheapest:

**At initialization, the policy IS the reference.** A zero-init adapter is a bitwise no-op (S0-05),
so ``logp_π == logp_ref`` for every token, the bracket in the DPO objective is *exactly zero*, and

    L = -log σ(β · 0) = -log(1/2) = log 2 = 0.693147...

That single number checks the entire reference pipeline at once. If the reference pass and the
training pass are looking at different models — a different adapter scale, a different mask, a
different packing, a sign error in the log-ratio, a `ce_sparse` whose sign is the other way round —
the loss at step 0 is *not* log 2, and it is not close to it either. Nothing else in this file could
find those, and no amount of "the loss goes down" would.

The rest is: the loss falls, the model actually learns to prefer the chosen completion (which is not
the same claim), and the reference genuinely does not move.
"""

import ctypes
import math

import numpy as np
import pytest

from learning_llamas import _ffi
from learning_llamas.adapter import create_zero_adapter, enumerate_targets
from learning_llamas.data import MaskedSample
from learning_llamas.train import DPOConfig, Preference, reference_logratios, train_dpo
from learning_llamas.train.dpo import LOG_2, DPOTrainer, to_batch

RANK = 4
N_CTX = 128
SEQ_LEN = 32
BETA = 0.1


def _sample(n_prompt: int, n_completion: int, start: int) -> MaskedSample:
    n = n_prompt + n_completion
    tokens = [start + (i % 9) for i in range(n)]
    weights = [0.0] * n_prompt + [1.0] * n_completion
    return MaskedSample(tokens=tokens, weights=weights)


def _pair(seed: int = 0) -> Preference:
    """A prompt with two different completions — the second one is the one we do not want."""
    return Preference(
        chosen=_sample(3, 5, start=7 + seed),
        rejected=_sample(3, 5, start=30 + seed),
    )


@pytest.fixture
def trainable(tiny_q4_k, tmp_path, load_model, libs: _ffi.Libraries):
    """Room for three sequences: chosen, rejected, and the pads."""
    adapter_path = tmp_path / "adapter.gguf"
    create_zero_adapter(tiny_q4_k, adapter_path, r=RANK, seed=7)

    model = load_model(tiny_q4_k, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True, n_seq_max=4)
    model.attach_adapter(adapter_path, scale=1.0)

    return model


# ---------------------------------------------------------------------------
# The batch.
# ---------------------------------------------------------------------------


def test_the_pair_is_packed_as_two_sequences_with_plus_and_minus_one_weights() -> None:
    """+1 on the chosen completion's tokens, -1 on the rejected's, 0 on prompts and pads.

    That is what makes ``ce_sparse``'s weighted sum the *log-ratio* in one op — no second pass, no
    gather. It is also the only place a sign error could hide, so it is checked directly.
    """
    batch = to_batch(_pair(), seq_len=SEQ_LEN)

    assert len(batch.tokens) == SEQ_LEN
    assert batch.seq_ids is not None

    positive = [i for i, w in enumerate(batch.weights) if w > 0]
    negative = [i for i, w in enumerate(batch.weights) if w < 0]

    assert positive and negative, "both completions must carry weight"
    assert all(w in (-1.0, 0.0, 1.0) for w in batch.weights)

    # Each sign lives entirely in one sequence, and they are different sequences.
    chosen_seqs = {batch.seq_ids[i] for i in positive}
    rejected_seqs = {batch.seq_ids[i] for i in negative}

    assert len(chosen_seqs) == 1
    assert len(rejected_seqs) == 1
    assert chosen_seqs != rejected_seqs, "the two completions must not share a sequence"

    # Five graded tokens each.
    assert len(positive) == 5
    assert len(negative) == 5


def test_a_pair_too_big_for_one_batch_is_refused() -> None:
    """DPO compares them in a single forward pass, so they cannot be split."""
    big = Preference(chosen=_sample(4, 20, 7), rejected=_sample(4, 20, 30))

    with pytest.raises(ValueError, match="single forward pass|does not fit"):
        to_batch(big, seq_len=SEQ_LEN)


# ---------------------------------------------------------------------------
# THE test.
# ---------------------------------------------------------------------------


def test_the_loss_at_initialization_is_exactly_log_two(trainable, libs: _ffi.Libraries) -> None:
    """The one that checks the whole reference pipeline in a single number.

    A zero-init adapter is a bitwise no-op, so at step 0 the policy **is** the reference: the
    bracket
    in the DPO objective is exactly zero, and the loss is exactly ``-log σ(0) = log 2``.

    A different adapter scale on the reference pass, a mask that does not line up, a packing that
    puts the two completions in the wrong sequences, a sign error in the log-ratio, a ``ce_sparse``
    whose sign is the other way round — every one of those moves this number, and none of them would
    stop the loss falling afterwards.
    """
    model = trainable
    batch = to_batch(_pair(), seq_len=SEQ_LEN)

    reference = reference_logratios(libs, model, [batch])
    assert len(reference) == 1

    with DPOTrainer(libs, model, DPOConfig(lr=1e-3, beta=BETA, seq_len=SEQ_LEN)) as trainer:
        loss = trainer.dpo_step(batch, reference[0], train=False)

    assert loss == pytest.approx(LOG_2, abs=1e-5), (
        f"the loss at initialization is {loss:.8f}, not log 2 = {LOG_2:.8f}. The policy and the "
        f"reference are not looking at the same model."
    )


def test_the_reference_is_the_model_with_the_adapter_off(trainable, libs: _ffi.Libraries) -> None:
    """And once the policy has moved, its log-ratio must differ from the reference's.

    Otherwise the test above passes for the wrong reason: it would also pass if the "reference" pass
    were secretly the policy pass, since at step 0 they coincide.
    """
    model = trainable
    batch = to_batch(_pair(), seq_len=SEQ_LEN)

    reference = reference_logratios(libs, model, [batch])[0]

    # At init the two coincide -- that is the previous test.
    policy_before = _policy_logratio(libs, model, batch)
    assert policy_before == pytest.approx(reference, abs=1e-5)

    with DPOTrainer(libs, model, DPOConfig(lr=5e-2, beta=BETA, seq_len=SEQ_LEN)) as trainer:
        for _ in range(8):
            trainer.dpo_step(batch, reference)

    # Every measurement below is inference, and it happens once the optimizer is gone -- a decode
    # while it is alive would free the scheduler it is holding (the shim now says so, loudly).
    policy_after = _policy_logratio(libs, model, batch)

    assert abs(policy_after - policy_before) > 1e-3, (
        "training did not move the policy's log-ratio at all"
    )

    reference_again = reference_logratios(libs, model, [batch])[0]
    assert reference_again == pytest.approx(reference, abs=1e-5), (
        "the reference model moved. It is supposed to be the frozen base with the adapter off."
    )


def _lora_grads(libs, model, tiny_q4_k) -> dict[tuple[str, bool], np.ndarray]:
    """Every LoRA A/B gradient accumulator, keyed by (base name, is_b)."""
    out: dict[tuple[str, bool], np.ndarray] = {}
    for target in enumerate_targets(tiny_q4_k):
        for is_b in (False, True):
            n = libs.farm.ll_debug_n_elements(model.ctx, target.name.encode(), is_b)
            assert n > 0
            buf = (ctypes.c_float * n)()
            got = libs.farm.ll_debug_grad(model.ctx, target.name.encode(), is_b, buf, n)
            assert got == n
            out[(target.name, is_b)] = np.frombuffer(buf, dtype=np.float32, count=n).copy()
    return out


def test_a_fully_masked_pair_contributes_zero_gradient(trainable, tiny_q4_k, libs) -> None:
    """S1-14 §6d: a preference pair with no graded token moves nothing.

    With every weight 0, ``ce_sparse`` is 0 at every position, so both log-probs are 0, the DPO
    bracket is ``β·(0 - 0) == 0``, the loss is exactly ``log 2``, and -- the claim under test --
    every
    gradient is bitwise zero. A stray nonzero anywhere (a masked position that still leaked a
    gradient) would show up here and nowhere else.
    """
    model = trainable
    pair = Preference(chosen=_sample(8, 0, start=7), rejected=_sample(9, 0, start=30))
    batch = to_batch(pair, seq_len=SEQ_LEN)

    with DPOTrainer(libs, model, DPOConfig(lr=1e-2, beta=BETA, seq_len=SEQ_LEN)) as trainer:
        loss = trainer.dpo_step(batch, ref_delta=0.0)
        grads = _lora_grads(libs, model, tiny_q4_k)

    assert loss == pytest.approx(LOG_2, abs=1e-6), (
        f"a fully-masked pair should sit at exactly log 2, got {loss}"
    )
    for (name, is_b), g in grads.items():
        assert np.array_equal(g, np.zeros_like(g)), (
            f"{name}.{'b' if is_b else 'a'} took a nonzero gradient from a fully-masked pair: "
            f"max |g| = {np.abs(g).max():.3e}"
        )


def test_a_dpo_step_moves_only_the_adapter(trainable, tiny_q4_k, libs) -> None:
    """S1-14 §6c: the step trains the adapter and nothing else.

    Two halves. (1) The base weights and the named-input buffers are *not* parameters in LoRA mode:
    ggml-opt never flags them, so they have no gradient accumulator -- ``ll_debug_base_grad`` on a
    base tensor errors rather than returning zeros, which is the machine-checkable form of "the
    optimizer cannot touch them". (2) The adapter's own A/B *do* take a gradient. Together with the
    sibling ``test_the_reference_is_the_model_with_the_adapter_off`` -- which shows the adapter-off
    forward is byte-stable across a whole training run, i.e. the base weights did not move -- this
    pins "only A/B move".
    """
    model = trainable
    batch = to_batch(_pair(), seq_len=SEQ_LEN)
    reference = reference_logratios(libs, model, [batch])[0]

    with DPOTrainer(libs, model, DPOConfig(lr=5e-2, beta=BETA, seq_len=SEQ_LEN)) as trainer:
        trainer.dpo_step(batch, reference)
        grads = _lora_grads(libs, model, tiny_q4_k)

        # A base weight has no gradient accumulator in LoRA mode: it is a constant input, not a
        # parameter, so there is nothing for the optimizer to step.
        for base in ("blk.0.attn_q.weight", "blk.0.ffn_down.weight", "output.weight"):
            assert (
                libs.farm.ll_debug_base_grad(model.ctx, base.encode(), (ctypes.c_float * 1)(), 1)
                < 0
            ), f"{base} has a gradient accumulator -- a named input is being trained as a parameter"

    # The adapter DID take a gradient: at least one A/B tensor is nonzero, so the step trained the
    # thing it is supposed to and the zero-gradient test above is not vacuous.
    assert any(np.abs(g).max() > 0 for g in grads.values()), (
        "no adapter tensor took a gradient; the DPO step trained nothing"
    )


def _policy_logratio(libs, model, batch) -> float:
    """The *policy's* log-ratio — the adapter left on."""
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


# ---------------------------------------------------------------------------
# The run.
# ---------------------------------------------------------------------------


def test_dpo_learns_to_prefer_the_chosen_completion(trainable, libs: _ffi.Libraries) -> None:
    """The claim DPO actually makes — and it is not "the loss goes down".

    A loss that falls tells you the objective is being minimized. What DPO is *for* is that the
    policy comes to assign the chosen completion more probability than the rejected one, relative to
    where the reference had them. That is the log-ratio going up, and it is what is checked.
    """
    model = trainable
    pairs = [_pair(seed=i) for i in range(3)]
    batches = [to_batch(p, seq_len=SEQ_LEN) for p in pairs]

    reference = reference_logratios(libs, model, batches)
    before = [_policy_logratio(libs, model, b) for b in batches]

    with DPOTrainer(libs, model, DPOConfig(lr=5e-2, beta=BETA, seq_len=SEQ_LEN)) as trainer:
        losses = []
        for _ in range(10):
            for i, b in enumerate(batches):
                losses.append(trainer.dpo_step(b, reference[i]))

    after = [_policy_logratio(libs, model, b) for b in batches]

    assert all(x == x for x in losses), f"loss went NaN: {losses}"  # noqa: PLR0124

    first = sum(losses[:3]) / 3
    last = sum(losses[-3:]) / 3
    assert last < first, f"the DPO loss did not fall: {first:.4f} -> {last:.4f}"

    # The thing the loss is a proxy for.
    improved = sum(1 for b, a in zip(before, after, strict=True) if a > b)
    assert improved == len(pairs), (
        f"only {improved} of {len(pairs)} pairs had their log-ratio increase. The loss fell, "
        f"but the model did not learn to prefer the chosen completion."
    )


def test_a_bigger_beta_moves_the_policy_further(trainable, libs: _ffi.Libraries) -> None:
    """Beta is the only knob DPO adds, and it must do what it says.

    Small beta keeps the policy near the reference; large beta lets it move further. If beta did
    nothing — if it were dropped on the way into the graph, say — the loss would still fall and
    every
    other test here would still pass.
    """
    model = trainable
    batch = to_batch(_pair(), seq_len=SEQ_LEN)

    def travel(beta: float) -> float:
        _zero_b(libs, model)  # back to a no-op adapter, so both runs start from the same place
        reference = reference_logratios(libs, model, [batch])[0]

        with DPOTrainer(libs, model, DPOConfig(lr=2e-2, beta=beta, seq_len=SEQ_LEN)) as trainer:
            for _ in range(6):
                trainer.dpo_step(batch, reference)

        return _policy_logratio(libs, model, batch) - reference

    small = travel(0.05)
    large = travel(0.5)

    assert large > small > 0, f"beta had no effect: 0.05 -> {small:.5f}, 0.5 -> {large:.5f}"


def _zero_b(libs, model) -> None:
    """Put every B back to zero — i.e. the adapter back to being a no-op."""
    from learning_llamas.train.dpo import _zero_b as zero  # noqa: PLC0415 - test-only helper

    zero(libs, model)


def test_train_dpo_end_to_end(trainable, libs: _ffi.Libraries) -> None:
    model = trainable
    pairs = [_pair(seed=i) for i in range(4)]

    config = DPOConfig(lr=2e-2, beta=BETA, seq_len=SEQ_LEN, epochs=4, shuffle=True, seed=0)

    result = train_dpo(libs, model, pairs, config)

    assert len(result.steps) == 16
    assert len(result.reference) == 4

    losses = [m.loss for m in result.steps]
    assert all(x == x for x in losses), f"loss went NaN: {losses}"  # noqa: PLR0124

    # The run starts at log 2 -- every pair does, because the policy starts as the reference.
    assert losses[0] == pytest.approx(LOG_2, abs=1e-4)

    assert sum(losses[-4:]) / 4 < sum(losses[:4]) / 4


def test_a_nonpositive_beta_is_rejected() -> None:
    with pytest.raises(ValueError, match="beta"):
        DPOConfig(beta=0.0)

    assert math.isclose(LOG_2, 0.6931471805599453)
