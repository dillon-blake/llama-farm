"""S1-15: GRPO rollouts — generate G answers per prompt and let the group be its own baseline.

Two claims here are load-bearing, and both are the kind that would produce a *plausible* wrong
answer rather than a crash:

**The KV-cache prefix share must not change what is generated.** The prompt is decoded once and its
KV copied to the other G−1 group members, instead of being decoded G times. That is the whole reason
a group of 8 is affordable. If `llama_memory_seq_cp` copied the wrong range, or the positions were
off by one, generation would still *work* — it would just be conditioning on a subtly different
prompt. So the test is not "does it generate": it is that greedy rollouts through the share are
**token-for-token identical** to greedy rollouts decoded from scratch, one sequence at a time.

**`logp_old` must be the policy's logprob, not the sampler's.** The importance ratio GRPO trains
on is `exp(logp_new − logp_old)`. Temperature and top-p change what gets *drawn*; they do not change
the distribution the ratio is defined against. Capturing after the sampler had warped the row would
give a ratio against the wrong policy — and every loss would still be finite, every gradient would
still flow, and the model would learn the wrong thing. So `logp_old` is cross-checked against an
independent full-logits recompute (S1-13) of the same rollouts.
"""

import numpy as np
import pytest

from learning_llamas import _ffi
from learning_llamas.data import Tokenizer
from learning_llamas.logprobs import load_lm_head, sequence_logprobs
from learning_llamas.train.rollout import (
    Rollout,
    RolloutEngine,
    SamplerConfig,
    captured_logp,
    group_advantages,
    length_reward,
    recompute_logp,
    substring_reward,
    token_reward,
)

G = 4
N_CTX = 256
MAX_NEW = 10

PROMPTS = ["hello", "world"]


@pytest.fixture
def engine(tiny_q4_k, load_model, libs: _ffi.Libraries):
    """An INFERENCE context — never a training one. Generation on a training context segfaults."""
    model = load_model(tiny_q4_k, n_ctx=N_CTX, n_seq_max=G)
    return RolloutEngine(
        libs,
        model.ctx,
        model.model,
        n_rollouts=G,
        sampler=SamplerConfig(temperature=1.0, seed=7, max_new_tokens=MAX_NEW),
    ), model


def _greedy_engine(libs, model, seed: int = 0) -> RolloutEngine:
    return RolloutEngine(
        libs,
        model.ctx,
        model.model,
        n_rollouts=G,
        sampler=SamplerConfig(temperature=0.0, seed=seed, max_new_tokens=MAX_NEW),
    )


# ---------------------------------------------------------------------------------------------
# THE test: the prefix share must be invisible.
# ---------------------------------------------------------------------------------------------


def test_the_kv_prefix_share_does_not_change_what_is_generated(tiny_q4_k, load_model, libs) -> None:
    """Greedy rollouts through `llama_memory_seq_cp` must equal rollouts decoded from scratch.

    The prompt is decoded ONCE and its KV copied to the other G−1 sequences — that is what makes a
    group of G cost one prompt forward instead of G. But a wrong copy range, or an off-by-one in the
    positions, does not fail: it conditions the group on a subtly different prompt and generates
    fluent, plausible, wrong text.

    Greedy, because greedy is a function of the KV alone: if the share is faithful, the tokens are
    identical, and if it is not, they diverge on the first one that matters.
    """
    model = load_model(tiny_q4_k, n_ctx=N_CTX, n_seq_max=G)
    with_share = _greedy_engine(libs, model).generate(PROMPTS, length_reward(10))

    # The reference: **sequence 0 only**, which decoded the prompt itself and never received a copy.
    #
    # That distinction is the entire test. Comparing the group's first member against this would
    # prove nothing at all — sequence 0 is the one that never uses the share, so it agrees whatever
    # the copy does. Every member from 1 upward is generating from KV it did not compute, and those
    # are the ones that have to match.
    solo_model = load_model(tiny_q4_k, n_ctx=N_CTX, n_seq_max=2)
    solo = RolloutEngine(
        libs,
        solo_model.ctx,
        solo_model.model,
        n_rollouts=2,
        sampler=SamplerConfig(temperature=0.0, seed=0, max_new_tokens=MAX_NEW),
    )
    reference = solo.generate(PROMPTS, length_reward(10))

    for group in range(len(PROMPTS)):
        want = next(r for r in reference.rollouts if r.group == group).completion_tokens
        members = [r for r in with_share.rollouts if r.group == group]

        for seq, rollout in enumerate(members):
            assert rollout.completion_tokens == want, (
                f"group {group}, sequence {seq}: the KV prefix share changed what was generated.\n"
                f"  from the copied KV: {rollout.completion_tokens}\n"
                f"  decoded directly:   {want}\n"
                "The copy is conditioning this sequence on a different prompt."
            )

    assert with_share.stats.kv_reuse_hits == len(PROMPTS) * (G - 1)


def test_greedy_members_of_a_group_agree(tiny_q4_k, load_model, libs) -> None:
    """At temperature 0 the G members are G copies of the same argmax walk.

    Which is worth asserting because it is what makes the test above a comparison of *generation*
    rather than of *seeding* — and because if the sequences were somehow sharing a KV slot rather
    than each having their own, they would drift apart here.
    """
    model = load_model(tiny_q4_k, n_ctx=N_CTX, n_seq_max=G)
    batch = _greedy_engine(libs, model).generate(["hello"], length_reward(10))

    completions = [r.completion_tokens for r in batch.rollouts]
    assert all(c == completions[0] for c in completions), (
        f"greedy rollouts of one prompt diverged: {completions}"
    )


# ---------------------------------------------------------------------------------------------
# logp_old is the POLICY's logprob.
# ---------------------------------------------------------------------------------------------


def test_logp_old_matches_an_independent_recompute(tiny_q4_k, load_model, libs) -> None:
    """Sample-time capture vs a full-logits recompute of the same tokens.

    `logp_old` is the denominator of an exponentiated ratio, so an error in it is *multiplied*, not
    added. And nothing about a wrong one looks wrong: the loss stays finite, the gradients still
    flow, and the model trains against the wrong policy.

    The recompute (S1-13) shares nothing with the capture: a different decode, all positions at
    once rather than one at a time, and the log-softmax done by a different code path. They are not
    bitwise equal and should not be — llama.cpp is deterministic per *shape* (ADR-0002), and a
    one-token incremental decode is a different shape from a full forward. They must agree
    numerically.
    """
    model = load_model(tiny_q4_k, n_ctx=N_CTX, n_seq_max=G)
    engine = _greedy_engine(libs, model)

    batch = engine.generate(["hello"], length_reward(10))
    rollout = batch.rollouts[0]

    lm_head = load_lm_head(tiny_q4_k)

    # position i predicts token i+1, so the completion's logprobs live at
    # [n_prompt-1 .. n_prompt+len(completion)-2] of the concatenated sequence.
    full = rollout.prompt_tokens + rollout.completion_tokens
    n_prompt = len(rollout.prompt_tokens)
    n_comp = len(rollout.completion_tokens)

    tokens = full[:-1]
    targets = full[1:]
    weights = [0.0] * len(tokens)
    for t in range(n_comp - 1 + 1):
        idx = n_prompt - 1 + t
        if idx < len(tokens):
            weights[idx] = 1.0

    recomputed = sequence_logprobs(libs, model.ctx, lm_head, tokens, targets, weights)

    captured = rollout.logp_old[: n_comp - 1] if n_comp > 1 else rollout.logp_old
    against = np.array(
        [recomputed[n_prompt - 1 + t] for t in range(len(captured))], dtype=np.float32
    )

    np.testing.assert_allclose(captured, against, atol=5e-3, rtol=0)

    # ...and they are real logprobs, not zeros that would trivially agree.
    assert (captured < 0.0).all()
    assert np.isfinite(captured).all()


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_recompute_logp_agrees_with_the_capture_over_a_whole_batch(
    tiny_q4_k, load_model, libs, temperature: float
) -> None:
    """``recompute_logp`` vs ``captured_logp`` on a full batch (S1-15 §4) — S1-16's cross-check.

    The manual check above proves it for one greedy rollout with the last token dropped. This is the
    helper GRPO actually uses: every rollout in the batch, every completion token graded, greedy and
    sampled. It is the ``naive`` path of :class:`~learning_llamas.verify.SelfVerified`'s first
    customer, exercised here on its own.

    The two paths share no code — the capture reads one raw logits row per drawn token off a chain
    of one-token incremental decodes; the recompute scores each finished rollout in a single forward
    and log-softmaxes through ``ce_sparse``. Deterministic per shape (ADR-0002) makes them
    numerically-close, not bitwise. Observed here: max ``|capture − recompute|`` ~1e-6, band 1e-4.
    """
    model = load_model(tiny_q4_k, n_ctx=N_CTX, n_seq_max=G)
    engine = RolloutEngine(
        libs,
        model.ctx,
        model.model,
        n_rollouts=G,
        sampler=SamplerConfig(temperature=temperature, seed=13, max_new_tokens=MAX_NEW),
    )
    batch = engine.generate(PROMPTS, length_reward(10))
    lm_head = load_lm_head(tiny_q4_k)

    captured = captured_logp(batch)
    recomputed = recompute_logp(engine, batch, lm_head)

    # Aligned slot-for-slot: one entry per completion token, concatenated in rollout order.
    n_completion = sum(len(r.completion_tokens) for r in batch.rollouts)
    assert captured.shape == recomputed.shape == (n_completion,)
    assert n_completion > 0

    deviation = float(np.max(np.abs(captured - recomputed)))
    assert deviation < 1e-4, (
        f"the sample-time capture and the S1-13 recompute diverged by {deviation:.2e} over the "
        f"batch (temperature={temperature}). One of them is scoring the wrong policy."
    )

    # ...and they are real logprobs, not zeros that would trivially agree.
    assert (captured < 0.0).all()
    assert np.isfinite(recomputed).all()
    assert (recomputed < 0.0).all()


def test_logp_old_is_one_per_generated_token(engine) -> None:
    eng, _ = engine
    batch = eng.generate(PROMPTS, length_reward(10))

    for rollout in batch.rollouts:
        assert len(rollout.logp_old) == len(rollout.completion_tokens)
        assert rollout.logp_old.dtype == np.float32
        assert (rollout.logp_old <= 0.0).all(), "a logprob is never positive"


# ---------------------------------------------------------------------------------------------
# Determinism.
# ---------------------------------------------------------------------------------------------


def test_the_same_seed_reproduces_the_run_bitwise(tiny_q4_k, load_model, libs) -> None:
    """Same seed, same tokens, and `logp_old` equal to the bit.

    Reproducibility is not a nicety in RL: `logp_old` is recorded at sample time and *reused* by
    the training step. A run that cannot be replayed cannot be debugged, and a `logp_old` that
    drifts from the policy that produced it silently biases every ratio.
    """
    first = None
    for _ in range(2):
        model = load_model(tiny_q4_k, n_ctx=N_CTX, n_seq_max=G)
        engine = RolloutEngine(
            libs,
            model.ctx,
            model.model,
            n_rollouts=G,
            sampler=SamplerConfig(temperature=1.0, seed=99, max_new_tokens=MAX_NEW),
        )
        batch = engine.generate(PROMPTS, length_reward(10))
        run = [(r.completion_tokens, r.logp_old) for r in batch.rollouts]

        if first is None:
            first = run
            continue

        for (tokens_a, logp_a), (tokens_b, logp_b) in zip(first, run, strict=True):
            assert tokens_a == tokens_b, "same seed produced different tokens"
            assert np.array_equal(logp_a, logp_b), "same seed produced different logp_old"


def test_different_seeds_explore_differently(tiny_q4_k, load_model, libs) -> None:
    """The G members of a group must not be G copies of each other when sampling.

    A group whose members all drew the same completion has zero variance, so zero advantage, so
    zero gradient — GRPO would run happily and learn nothing at all. That is the failure mode of
    seeding every chain identically, and it is silent.
    """
    model = load_model(tiny_q4_k, n_ctx=N_CTX, n_seq_max=G)
    engine = RolloutEngine(
        libs,
        model.ctx,
        model.model,
        n_rollouts=G,
        sampler=SamplerConfig(temperature=1.0, seed=3, max_new_tokens=MAX_NEW),
    )

    batch = engine.generate(["hello"], length_reward(10))
    completions = [tuple(r.completion_tokens) for r in batch.rollouts]

    assert len(set(completions)) > 1, (
        "every rollout in the group is identical — the sampler chains share a seed, so the group "
        "has no variance, so no advantage, so no gradient."
    )


# ---------------------------------------------------------------------------------------------
# Advantages.
# ---------------------------------------------------------------------------------------------


def test_advantages_are_group_normalized() -> None:
    rng = np.random.default_rng(0)
    rewards = rng.normal(5.0, 2.0, size=16).astype(np.float32)

    advantages = group_advantages(rewards)

    assert abs(float(advantages.mean())) < 1e-5
    assert abs(float(advantages.std()) - 1.0) < 1e-2


def test_a_degenerate_group_gets_zero_advantage() -> None:
    """Every rollout earned the same reward, so the group says nothing about which is better.

    Not an edge case — it is *common*. An all-or-nothing reward ("did it contain the answer") makes
    whole groups succeed or fail together, and a group that all failed must not be pushed anywhere.
    Dividing by a near-zero std would push it very hard indeed, in an arbitrary direction.
    """
    assert np.array_equal(
        group_advantages(np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32)),
        np.zeros(4, dtype=np.float32),
    )
    assert np.array_equal(
        group_advantages(np.zeros(4, dtype=np.float32)), np.zeros(4, dtype=np.float32)
    )


def test_advantages_are_computed_per_group_not_across_the_batch(engine) -> None:
    """Two prompts, and each is normalized against its OWN siblings.

    Normalizing across the whole batch would compare a completion to answers for a *different*
    question — which is exactly the baseline GRPO is designed not to need.
    """
    eng, _ = engine
    batch = eng.generate(PROMPTS, length_reward(10))

    advantages = batch.advantages()
    for group in range(len(PROMPTS)):
        members = advantages[group * G : (group + 1) * G]
        assert abs(float(members.sum())) < 1e-4, (
            f"group {group}'s advantages do not sum to zero: {members}. They are being normalized "
            f"across the batch rather than within the group."
        )


# ---------------------------------------------------------------------------------------------
# Throughput, and the contract.
# ---------------------------------------------------------------------------------------------


def test_the_counters_are_populated(engine) -> None:
    """Rollout throughput is GRPO's bottleneck, so it is counted rather than guessed at."""
    eng, _ = engine
    batch = eng.generate(PROMPTS, length_reward(10))
    stats = batch.stats

    assert stats.decode_calls > 0
    assert stats.prompt_tokens > 0
    assert stats.generated_tokens > 0
    assert stats.tokens_per_second > 0.0

    # The prefix share: G-1 prompt forward passes skipped, per prompt.
    assert stats.kv_reuse_hits == len(PROMPTS) * (G - 1)

    # One decode for the prompt, then one per token step -- for the WHOLE group, not per sequence.
    # If this were per-sequence the count would be G times larger, and the batching would be a lie.
    assert stats.decode_calls <= len(PROMPTS) * MAX_NEW


def test_the_batch_counters_are_this_iterations_alone(engine) -> None:
    """A GRPO run reads these per iteration, so they must mean "this iteration".

    ``generate`` used to hand the batch ``self.stats`` — the engine-lifetime accumulator — by
    reference. Two things went wrong and neither was visible in a single-iteration test: iteration 5
    reported the sum of iterations 1 through 5 (a tokens/second that averages the whole run and
    hides exactly the slowdown you would want to see), and every batch already returned mutated
    underneath the caller the moment the next one was generated.

    ``kv_reuse_hits`` is the crisp probe: it is exactly ``n_prompts * (G - 1)`` per call, with no
    dependence on sampling, so a cumulative counter reads double on the second call.
    """
    eng, _ = engine
    per_call = len(PROMPTS) * (G - 1)

    first = eng.generate(PROMPTS, length_reward(10))
    assert first.stats.kv_reuse_hits == per_call

    second = eng.generate(PROMPTS, length_reward(10))
    assert second.stats.kv_reuse_hits == per_call, (
        "the second batch reported the run's total, not its own iteration"
    )

    # The first batch is a snapshot: generating again must not have rewritten it.
    assert first.stats.kv_reuse_hits == per_call
    assert first.stats is not eng.stats
    assert second.stats is not first.stats

    # ...and nothing is lost: the engine still accumulates, and the parts sum to the whole.
    assert eng.stats.kv_reuse_hits == 2 * per_call
    assert eng.stats.generated_tokens == (
        first.stats.generated_tokens + second.stats.generated_tokens
    )
    assert eng.stats.decode_calls == first.stats.decode_calls + second.stats.decode_calls
    assert eng.stats.wall_seconds == pytest.approx(
        first.stats.wall_seconds + second.stats.wall_seconds
    )


def test_the_engine_tokenizes_special_tokens_like_the_rest_of_the_project(
    tiny_q4_k, load_model, libs
) -> None:
    """One tokenizer convention, or the prompt GRPO trains on is not the prompt anyone serves.

    The engine used to pass ``parse_special=False`` while the data layer
    (:meth:`learning_llamas.data.Tokenizer.encode`) defaults to ``True``. GRPO is internally
    consistent either way — ``logp_old``, ``logp_new`` and the reference all read whatever ids came
    out of here — so nothing in a GRPO run can notice. What notices is everything else: a rendered
    chat template's ``</s>`` or ``<|im_start|>`` becomes a handful of ordinary character tokens
    during training and one control token at serving time, and the only symptom is a model that is
    quietly worse.
    """
    model = load_model(tiny_q4_k, n_ctx=N_CTX, n_seq_max=G)
    eng = RolloutEngine(libs, model.ctx, model.model, n_rollouts=G)
    tokenizer = Tokenizer(libs, model.model)

    # The vocab's own spelling of EOS, so this cannot degenerate into byte-fallback on a token the
    # 512-entry fixture vocab does not have (the failure mode the data-layer audit found).
    text = f"{tokenizer.piece(tokenizer.eos)} and on we go"

    assert tokenizer.eos >= 0
    assert not tokenizer.add_eos, (
        "this model appends EOS itself, so finding the EOS id below would prove nothing"
    )

    # The discriminating half: unparsed, that prefix is several ordinary character tokens and the
    # EOS id never appears at all.
    assert tokenizer.eos in eng.tokenize(text), (
        f"the engine did not recognize special-token text: {eng.tokenize(text)}"
    )

    # The engine adds BOS itself (a raw prompt, not a rendered template), which is the one
    # deliberate difference from the data layer's default.
    assert eng.tokenize(text) == tokenizer.encode(text, add_special=True, parse_special=True)


def test_a_group_of_one_is_refused(tiny_q4_k, load_model, libs) -> None:
    """It has no baseline: every advantage would be exactly zero, and nothing would ever train."""
    model = load_model(tiny_q4_k, n_ctx=N_CTX, n_seq_max=G)

    with pytest.raises(ValueError, match="at least 2"):
        RolloutEngine(libs, model.ctx, model.model, n_rollouts=1)


def test_a_group_bigger_than_the_context_allows_is_refused(tiny_q4_k, load_model, libs) -> None:
    model = load_model(tiny_q4_k, n_ctx=N_CTX, n_seq_max=2)

    with pytest.raises(ValueError, match="n_seq_max"):
        RolloutEngine(libs, model.ctx, model.model, n_rollouts=8)


def _rollout(text: str = "", tokens: list[int] | None = None) -> Rollout:
    return Rollout(
        group=0,
        prompt_tokens=[1],
        completion_tokens=tokens or [1],
        logp_old=np.zeros(1, dtype=np.float32),
        text=text,
    )


def test_the_builtin_rewards() -> None:
    length = length_reward(target=10)
    assert length("p", _rollout("x" * 10)) == 0.0
    assert length("p", _rollout("x" * 5)) < 0.0
    assert length("p", _rollout("x" * 10)) > length("p", _rollout("x" * 3))

    contains = substring_reward("yes")
    assert contains("p", _rollout("oh yes indeed")) == 1.0
    assert contains("p", _rollout("no")) == 0.0

    # The one the toy task uses: how much of the completion came from a wanted set of tokens. It is
    # the most direct thing a policy gradient can act on -- the update literally raises the logprob
    # of the tokens in a high-advantage completion.
    wanted = token_reward({7, 8})
    assert wanted("p", _rollout(tokens=[7, 8, 7, 8])) == 1.0
    assert wanted("p", _rollout(tokens=[7, 9, 7, 9])) == 0.5
    assert wanted("p", _rollout(tokens=[1, 2, 3])) == 0.0
