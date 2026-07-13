"""S1-15 — GRPO rollouts: generate G answers per prompt, and score them against each other.

GRPO's idea is that you do not need a value network. Sample G completions for the same prompt, and
let the group be its own baseline: a completion that beat its siblings gets a positive advantage,
one that lost gets a negative one. That is the whole of it, and it means the expensive part is not
the maths — it is the *generation*.

Which is why this module contains **no new native code at all**. A rollout is ordinary llama.cpp
inference: KV cache, sampler chains, parallel sequences, adapter attached. Everything here is public
C API.

Three things are worth knowing before reading the code.

**The prompt is decoded once, not G times.** Decode it on sequence 0, then `llama_memory_seq_cp` its
KV to the other G−1 group members, and only then let them diverge. llama.cpp's KV cache does prefix
sharing natively, so a group of 8 costs one prompt forward, not eight. (This is also the sanctioned
replacement for unsloth's prefix-grouper, which is AGPL — ROADMAP §13.)

**`logp_old` is captured at sample time, from the RAW logits.** The importance ratio GRPO trains
on is `exp(logp_new − logp_old)`, and `logp_old` must be the *policy's* logprob of the token that
was actually drawn — not the sampler's. Temperature and top-p reshape what gets sampled; they do
not change the distribution the ratio is defined against. So the logits row is snapshotted
**before** the sampler touches it, and the log-softmax is done here, in float64. It costs no extra
forward pass: the row is already sitting there from the decode that produced it.

**Generation never happens on a training context.** A second `llama_decode` on a context with
`cparams.training` set segfaults, and it frees the scheduler the optimizer is holding. So GRPO runs
**two contexts over one model**: this one for rollouts, and a training one for the update. They
share the *same* `llama_adapter_lora` object — adapters are model-level and their A/B tensors live
in the adapter, not the context — so a training step mutates the weights this engine samples from,
the rollouts are on-policy for free, with no copying.
"""

from __future__ import annotations

import ctypes
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .. import _ffi

#: A reward function: given the prompt and the whole rollout, how good was it?
#:
#: The rollout, and not just the text, because a reward often needs to see the *tokens* — and
#: because the tokens are what the policy gradient can actually act on.
RewardFn = Callable[[str, "Rollout"], float]


@dataclass(frozen=True)
class SamplerConfig:
    """How to draw the G completions.

    Attributes:
        temperature: Softmax temperature. ``0.0`` means greedy, which makes a run exactly
            reproducible and is what the tests use.
        top_p: Nucleus threshold. ``1.0`` disables it.
        seed: Base seed. Each sequence gets a derived seed, so the G members of a group are
            genuinely different draws rather than G copies of the same one.
        max_new_tokens: Hard cap on completion length.
    """

    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 0
    max_new_tokens: int = 32

    @property
    def greedy(self) -> bool:
        """True when temperature is 0: the argmax walk, and therefore exactly reproducible."""
        return self.temperature <= 0.0


@dataclass
class RolloutStats:
    """What the generation actually cost.

    Rollout throughput is GRPO's bottleneck, so it is counted from day one rather than guessed at
    later.

    Attributes:
        decode_calls: Calls into ``llama_decode``. One per prompt, then one per generated token
            *per group* — not per sequence, which is the point of batching them.
        prompt_tokens: Prompt tokens actually forwarded.
        generated_tokens: Completion tokens sampled.
        kv_reuse_hits: Times a prompt's KV was copied to a sibling instead of being recomputed.
            With G rollouts per prompt this is ``G - 1`` per group, and each one is a prompt
            forward pass that did not happen.
        wall_seconds: Wall time inside :meth:`RolloutEngine.generate`.
    """

    decode_calls: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    kv_reuse_hits: int = 0
    wall_seconds: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        """Completion tokens generated per second of wall time."""
        if self.wall_seconds <= 0.0:
            return 0.0
        return self.generated_tokens / self.wall_seconds


@dataclass
class Rollout:
    """One sampled completion.

    Attributes:
        group: Which prompt this answers. The advantage is computed *within* a group.
        prompt_tokens: The prompt, tokenized.
        completion_tokens: What the model produced. May be shorter than ``max_new_tokens`` if it
            emitted an end-of-generation token.
        logp_old: ``[len(completion_tokens)]`` F32 — the behaviour policy's logprob of each token
            it drew, captured at sample time from the raw logits.
        text: The completion, detokenized. What the reward function sees.
        reward: What the reward function said.
        advantage: The reward, normalized within its group. Filled in after all G are generated.
    """

    group: int
    prompt_tokens: list[int]
    completion_tokens: list[int]
    logp_old: np.ndarray
    text: str = ""
    reward: float = 0.0
    advantage: float = 0.0


@dataclass
class RolloutBatch:
    """Everything one GRPO iteration generated.

    Attributes:
        rollouts: Every completion, grouped by prompt (``rollouts[i].group`` is the prompt index).
        prompts: The prompt texts, indexed by group.
        stats: What it cost.
    """

    rollouts: list[Rollout]
    prompts: list[str]
    stats: RolloutStats = field(default_factory=RolloutStats)

    def rewards(self) -> np.ndarray:
        """Every rollout's reward, in rollout order."""
        return np.array([r.reward for r in self.rollouts], dtype=np.float32)

    def advantages(self) -> np.ndarray:
        """Every rollout's group-normalized advantage, in rollout order."""
        return np.array([r.advantage for r in self.rollouts], dtype=np.float32)

    def mean_reward(self) -> float:
        """The reward averaged over every rollout — the number a GRPO run should move."""
        return float(np.mean(self.rewards())) if self.rollouts else 0.0


def group_advantages(rewards: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Normalize rewards within a group: ``(r - mean) / (std + eps)``.

    This is the whole reason GRPO needs no value network — the group is its own baseline.

    **A degenerate group gets zero advantages, not a division by a tiny number.** If every rollout
    in a group earned the same reward, the group says nothing about which is better, and the honest
    answer is "no signal": std is 0, and `(r - mean)` is 0 too, so the ratio is 0/eps = 0 rather
    than 0/0. That is not an edge case to be tolerated — it is *common*, because a reward like
    "did it contain the answer" is often all-or-nothing across a group, and a group that all failed
    must not be pushed anywhere.

    Args:
        rewards: ``[G]`` rewards for one group.
        eps: Floor on the standard deviation.

    Returns:
        ``[G]`` advantages, mean 0 and (for a non-degenerate group) std ≈ 1.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    if rewards.size == 0:
        return rewards

    centred = rewards - rewards.mean()
    return (centred / (rewards.std() + eps)).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Built-in rewards, for tests and for a first run.
# ---------------------------------------------------------------------------------------------


def length_reward(target: int) -> RewardFn:
    """Reward completions for being about ``target`` characters long.

    A toy, and deliberately so: it is *learnable from the policy alone* (no world knowledge, no
    tokenizer tricks), dense, and bounded — which makes it the right thing to assert convergence
    against. A real reward would not be.
    """

    def reward(_prompt: str, rollout: Rollout) -> float:
        return -abs(len(rollout.text) - target) / max(target, 1)

    return reward


def substring_reward(needle: str) -> RewardFn:
    """Return 1.0 if the completion contains ``needle``, else 0.0.

    All-or-nothing, so it produces degenerate groups often — which is exactly why
    :func:`group_advantages` has to handle them.
    """

    def reward(_prompt: str, rollout: Rollout) -> float:
        return 1.0 if needle in rollout.text else 0.0

    return reward


def token_reward(wanted: set[int]) -> RewardFn:
    """Reward a completion for how much of it is drawn from ``wanted``.

    The most direct test of a GRPO update there is. The update pushes up the log-probability of the
    tokens in high-advantage completions — so a reward that says *use more of these tokens* is
    precisely the thing the gradient can act on, with nothing in between. A policy that cannot learn
    this has not learned anything, and the failure is in the update rather than in the task.

    Which is why it is the toy task: a reward like "produce text of about this length" is a fine
    thing to want, but it asks a two-layer random model to control the *decoded character count* of
    its own samples, and the signal is buried under the sampling noise long before it reaches the
    weights.
    """

    def reward(_prompt: str, rollout: Rollout) -> float:
        if not rollout.completion_tokens:
            return 0.0
        hits = sum(1 for t in rollout.completion_tokens if t in wanted)
        return hits / len(rollout.completion_tokens)

    return reward


# ---------------------------------------------------------------------------------------------
# The engine.
# ---------------------------------------------------------------------------------------------


class RolloutEngine:
    """Generates G completions per prompt on an **inference** context.

    Args:
        libs: The loaded native libraries.
        ctx: A ``llama_context`` that is **not** in training mode, with ``n_seq_max >= n_rollouts``
            and enough ``n_ctx`` for ``n_rollouts * (prompt + max_new_tokens)``. The adapter must
            be attached to it — that is what makes the rollouts on-policy.
        model: The ``llama_model``, for its vocabulary.
        n_rollouts: G, the group size. Must be at least 2: a group of one has no baseline, so
            every advantage would be exactly zero and nothing would ever be learned.
        sampler: How to draw.

    Raises:
        ValueError: If ``n_rollouts`` is less than 2 or exceeds the context's ``n_seq_max``.
    """

    def __init__(
        self,
        libs: _ffi.Libraries,
        ctx: int,
        model: int,
        n_rollouts: int = 4,
        sampler: SamplerConfig | None = None,
        adapter: int | None = None,
    ) -> None:
        if n_rollouts < 2:
            raise ValueError(
                f"n_rollouts must be at least 2, got {n_rollouts}: a group of one is its own "
                f"baseline, so every advantage is zero and no gradient ever flows."
            )

        n_seq_max = int(libs.llama.llama_n_seq_max(ctx))
        if n_rollouts > n_seq_max:
            raise ValueError(
                f"n_rollouts={n_rollouts} exceeds the context's n_seq_max={n_seq_max}. Create the "
                f"rollout context with n_seq_max >= n_rollouts."
            )

        self.libs = libs
        self.ctx = ctx
        self.model = model
        self.n_rollouts = n_rollouts
        self.sampler = sampler or SamplerConfig()

        # The `llama_adapter_lora *` this context is sampling through, if the caller said. GRPO
        # checks it against the training context's: they must be the SAME OBJECT, not two adapters
        # loaded from the same file, or the trainer updates one set of weights and the engine keeps
        # sampling from the other -- and nothing fails, the reward simply never moves.
        self.adapter = adapter

        self.vocab = libs.llama.llama_model_get_vocab(model)
        self.n_vocab = int(libs.llama.llama_vocab_n_tokens(self.vocab))
        self.memory = libs.llama.llama_get_memory(ctx)

        self.stats = RolloutStats()

    # -- the public surface -------------------------------------------------------------------

    def generate(self, prompts: Sequence[str], reward_fn: RewardFn) -> RolloutBatch:
        """Generate G completions for each prompt, score them, and normalize within each group."""
        started = time.perf_counter()

        rollouts: list[Rollout] = []
        for group, prompt in enumerate(prompts):
            rollouts.extend(self._generate_group(group, prompt))

        for rollout in rollouts:
            rollout.text = self._detokenize(rollout.completion_tokens)
            rollout.reward = float(reward_fn(prompts[rollout.group], rollout))

        # Advantages are per-group: the group IS the baseline.
        for group in range(len(prompts)):
            members = [r for r in rollouts if r.group == group]
            advantages = group_advantages(np.array([r.reward for r in members], dtype=np.float32))
            for rollout, advantage in zip(members, advantages, strict=True):
                rollout.advantage = float(advantage)

        self.stats.wall_seconds += time.perf_counter() - started

        return RolloutBatch(rollouts=rollouts, prompts=list(prompts), stats=self.stats)

    def tokenize(self, text: str) -> list[int]:
        """Tokenize, with the model's BOS convention applied."""
        buf = (_ffi.llama_token * 512)()
        n = self.libs.llama.llama_tokenize(
            self.vocab, text.encode(), len(text.encode()), buf, 512, True, False
        )
        if n < 0:
            raise RuntimeError(f"llama_tokenize needs {-n} tokens, more than the 512 buffer")
        return [int(buf[i]) for i in range(n)]

    # -- one group ----------------------------------------------------------------------------

    def _generate_group(self, group: int, prompt: str) -> list[Rollout]:
        """Decode the prompt once, fan its KV out to G sequences, then step them together."""
        libs = self.libs
        g = self.n_rollouts

        prompt_tokens = self.tokenize(prompt)
        n_prompt = len(prompt_tokens)

        # A fresh cache per group: sequence ids are reused across groups, and a leftover KV entry
        # would silently prepend the previous prompt to this one.
        libs.llama.llama_memory_clear(self.memory, True)

        batch = libs.llama.llama_batch_init(max(n_prompt, g), 0, 1)
        chains = [self._make_chain(group, i) for i in range(g)]

        try:
            # 1. The prompt, once, on sequence 0. Only the last position needs a logits row -- that
            #    is the one every group member samples its first token from.
            self._fill(
                batch,
                prompt_tokens,
                positions=range(n_prompt),
                seqs=[0] * n_prompt,
                want_logits=[i == n_prompt - 1 for i in range(n_prompt)],
            )
            self._decode(batch)
            self.stats.prompt_tokens += n_prompt

            # 2. Fan the prompt's KV out to the rest of the group. This is the prefix share: G-1
            #    prompt forward passes that do not happen.
            for seq in range(1, g):
                libs.llama.llama_memory_seq_cp(self.memory, 0, seq, -1, -1)
                self.stats.kv_reuse_hits += 1

            # 3. Every member samples its first token from the SAME logits row -- one row, G draws.
            row = self._logits_row(n_prompt - 1)

            live = list(range(g))
            tokens: list[list[int]] = [[] for _ in range(g)]
            logps: list[list[float]] = [[] for _ in range(g)]

            for seq in range(g):
                token = int(libs.llama.llama_sampler_sample(chains[seq], self.ctx, n_prompt - 1))
                tokens[seq].append(token)
                logps[seq].append(_logprob(row, token))
                self.stats.generated_tokens += 1

            live = [s for s in live if not self._is_eog(tokens[s][-1])]

            # 4. ...and then step every live sequence together, one decode call per token.
            for step in range(1, self.sampler.max_new_tokens):
                if not live:
                    break

                self._fill(
                    batch,
                    [tokens[s][-1] for s in live],
                    positions=[n_prompt + step - 1] * len(live),
                    seqs=live,
                    want_logits=[True] * len(live),
                )
                self._decode(batch)

                still_live = []
                for i, seq in enumerate(live):
                    # `i` is the index within THIS batch, which is what llama_get_logits_ith wants.
                    row = self._logits_row(i)
                    token = int(libs.llama.llama_sampler_sample(chains[seq], self.ctx, i))

                    tokens[seq].append(token)
                    logps[seq].append(_logprob(row, token))
                    self.stats.generated_tokens += 1

                    if not self._is_eog(token):
                        still_live.append(seq)

                live = still_live

            return [
                Rollout(
                    group=group,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=tokens[seq],
                    logp_old=np.array(logps[seq], dtype=np.float32),
                )
                for seq in range(g)
            ]
        finally:
            # The chain OWNS what was added to it (llama.h:1308). Free the chain, never its members.
            for chain in chains:
                libs.llama.llama_sampler_free(chain)
            libs.llama.llama_batch_free(batch)

    # -- the plumbing -------------------------------------------------------------------------

    def _make_chain(self, group: int, member: int) -> int:
        """One sampler chain per sequence, seeded so the G members are different draws."""
        libs = self.libs
        params = libs.llama.llama_sampler_chain_default_params()
        chain = libs.llama.llama_sampler_chain_init(params)

        if self.sampler.greedy:
            libs.llama.llama_sampler_chain_add(chain, libs.llama.llama_sampler_init_greedy())
            return chain

        if self.sampler.top_p < 1.0:
            libs.llama.llama_sampler_chain_add(
                chain, libs.llama.llama_sampler_init_top_p(self.sampler.top_p, 1)
            )
        libs.llama.llama_sampler_chain_add(
            chain, libs.llama.llama_sampler_init_temp(self.sampler.temperature)
        )

        # A distinct, DERIVED seed per (group, member): same config -> same rollouts, run to run,
        # while the G members of a group still explore differently.
        seed = (self.sampler.seed * 1_000_003 + group * 1_009 + member) & 0xFFFFFFFF
        libs.llama.llama_sampler_chain_add(chain, libs.llama.llama_sampler_init_dist(seed))

        return chain

    def _fill(self, batch, tokens, positions, seqs, want_logits) -> None:  # noqa: ANN001
        """Write one step into the reusable llama_batch.

        `n_tokens` is set every time on purpose: llama_batch_init leaves it uninitialized, and a
        stale value from the previous step would decode the wrong number of rows.
        """
        for i, (token, pos, seq, wants) in enumerate(
            zip(tokens, positions, seqs, want_logits, strict=True)
        ):
            batch.token[i] = token
            batch.pos[i] = pos
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = seq
            batch.logits[i] = 1 if wants else 0

        batch.n_tokens = len(tokens)

    def _decode(self, batch) -> None:  # noqa: ANN001
        status = self.libs.llama.llama_decode(self.ctx, batch)
        if status != 0:
            raise RuntimeError(f"llama_decode failed with status {status} during a rollout")
        self.stats.decode_calls += 1

    def _logits_row(self, index: int) -> np.ndarray:
        """The RAW logits for one batch position, snapshotted.

        Copied out before the sampler runs, deliberately. `logp_old` must be the *policy's* logprob
        of the drawn token — temperature and top-p change what gets drawn, not the distribution the
        importance ratio is defined against. Taking the snapshot first makes that true by
        construction, rather than by trusting that no sampler ever writes in place.

        Raises:
            RuntimeError: If this position had no logits row. llama.cpp returns NULL for that, and
                ctypes would happily dereference it.
        """
        ptr = self.libs.llama.llama_get_logits_ith(self.ctx, index)
        if not ptr:
            raise RuntimeError(
                f"no logits row at batch index {index}: the batch did not request one there"
            )
        return np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)).copy()

    def _is_eog(self, token: int) -> bool:
        return bool(self.libs.llama.llama_vocab_is_eog(self.vocab, token))

    def _detokenize(self, tokens: Sequence[int]) -> str:
        if not tokens:
            return ""
        n = len(tokens)
        buf = ctypes.create_string_buffer(n * 32 + 64)
        written = self.libs.llama.llama_detokenize(
            self.vocab, (_ffi.llama_token * n)(*tokens), n, buf, len(buf), False, False
        )
        if written < 0:
            raise RuntimeError(f"llama_detokenize needs {-written} bytes")
        return buf.raw[:written].decode("utf-8", errors="replace")


def _logprob(row: np.ndarray, token: int) -> float:
    """``logits[token] - logsumexp(logits)``, in float64.

    The stable form, and in double precision, because this number is the denominator of an
    exponentiated ratio: an error here is multiplied, not added.
    """
    logits = row.astype(np.float64)
    peak = logits.max()
    lse = peak + math.log(np.exp(logits - peak).sum())
    return float(logits[token] - lse)
