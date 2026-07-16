r"""Build the per-token loss mask for a chat sample.

This is the piece of the data layer where the subtle bug lives, so it is worth being explicit
about what the bug *is*.

The naive way to mask a prompt is to render the prompt, tokenize it, count the tokens, and mark
that many positions as weight 0. **That is wrong**, and it is wrong in a way that no assertion on
lengths will catch: a BPE tokenizer can **merge across the boundary**. Tokenizing
``"...assistant\\n"`` and ``"...assistant\\nHello"`` need not give a common prefix — the last
prompt token and the first completion character can fuse into a single token that belongs to
neither. Mask by count and you either train on a token that is half prompt, or you skip the first
real token of the completion. Either way the model learns a slightly wrong thing, forever, and
the loss curve looks fine.

So this module never counts. It renders the *incremental prefixes*, tokenizes each, and **checks
the token-prefix property explicitly**. If tokenization merged across a boundary, the check fails
and says so, naming the boundary and the offending tokens — rather than silently shifting the
mask by one.
"""

from __future__ import annotations

from dataclasses import dataclass

from .template import ChatTemplate, Message
from .tokenize import Tokenizer


@dataclass(frozen=True)
class MaskedSample:
    """A tokenized chat sample with its per-token loss weights.

    Attributes:
        tokens: The token ids of the whole rendered conversation.
        weights: One weight per token. ``0.0`` on prompt and template tokens, ``1.0`` on the
            assistant content this sample trains on.
    """

    tokens: list[int]
    weights: list[float]

    @property
    def n_trained(self) -> int:
        """How many tokens actually carry loss."""
        return sum(1 for w in self.weights if w > 0.0)


class BoundaryMergeError(ValueError):
    """Tokenization merged across a prompt/completion boundary.

    Raised rather than papered over. A mask that is off by one token is not a rounding error —
    it means the model is trained to predict a token it will never be asked to predict, or not
    trained on one it will be.
    """


def _assert_token_prefix(
    prefix: list[int], full: list[int], boundary: str, tokenizer: Tokenizer
) -> None:
    if full[: len(prefix)] == prefix:
        return

    # Find where they diverge, so the message names the actual culprit.
    i = 0
    while i < len(prefix) and i < len(full) and prefix[i] == full[i]:
        i += 1

    got = full[i] if i < len(full) else None
    expected = prefix[i] if i < len(prefix) else None

    raise BoundaryMergeError(
        f"tokenization merged across the {boundary} boundary: at position {i}, the prefix has "
        f"{expected!r} ({tokenizer.piece(expected)!r}) but the full text has {got!r} "
        f"({tokenizer.piece(got) if got is not None else ''!r}). Masking by token count here "
        "would silently shift the loss mask by a token. Adjust the template so the boundary "
        "falls on a token break."
    )


def build_masked_sample(
    messages: list[Message],
    template: ChatTemplate,
    tokenizer: Tokenizer,
) -> MaskedSample:
    """Tokenize a chat sample and mark only the final assistant turn for loss.

    Args:
        messages: The conversation. The last message must be the assistant turn to train on.
        template: The model's chat template.
        tokenizer: The model's tokenizer.

    Returns:
        The tokens and their per-token loss weights.

    Raises:
        ValueError: If the last message is not an assistant turn.
        BoundaryMergeError: If tokenization merged across the prompt/completion boundary.
    """
    if not messages or messages[-1].role != "assistant":
        raise ValueError("the last message must be the assistant turn to train on")

    # The prompt is everything up to the point where the assistant's content begins — including
    # the template's assistant-turn opener, which the model is *given*, not asked to produce.
    prompt_text = template.render(messages[:-1], add_generation_prompt=True)
    full_text = template.render(messages, add_generation_prompt=False)

    if not full_text.startswith(prompt_text):
        raise BoundaryMergeError(
            "the rendered prompt is not a text prefix of the rendered conversation. The chat "
            "template renders the assistant turn differently depending on whether it is the "
            "generation target, so there is no well-defined completion span to train on."
        )

    # The token-prefix property is checked at EVERY message boundary, not only the one that moves
    # the loss mask (S1-06 item 4). The chain of rendered prefixes, in order:
    #   render(m[:1]), render(m[:2]), ..., render(m[:-1])   -- each complete turn, no opener,
    #   prompt_text (= render(m[:-1]) + the assistant opener) -- the pre-completion prefix,
    #   full_text                                             -- the whole conversation.
    # Each must be a TOKEN-prefix of the next. A BPE/SPM merge at any boundary is exactly the bug
    # this module exists to catch; naming which boundary merged is far more useful than only
    # checking the one boundary the mask happens to fall on. add_special=False throughout: the
    # template already emits BOS/EOS as text, and letting the tokenizer add its own would double
    # them on the full text but not the prefixes, breaking the property for an unrelated reason.
    stages: list[tuple[str, str]] = [
        (f"message {k}", template.render(messages[:k], add_generation_prompt=False))
        for k in range(1, len(messages))
    ]
    stages.append(("prompt/completion", prompt_text))
    stages.append(("full conversation", full_text))

    tokenized: dict[str, list[int]] = {}
    prev_tokens: list[int] = []
    prev_label = "start"
    for label, text in stages:
        toks = tokenizer.encode(text, add_special=False, parse_special=True)
        _assert_token_prefix(prev_tokens, toks, f"{prev_label} -> {label}", tokenizer)
        tokenized[label] = toks
        prev_tokens, prev_label = toks, label

    prompt_tokens = tokenized["prompt/completion"]
    full_tokens = tokenized["full conversation"]

    weights = [0.0] * len(prompt_tokens) + [1.0] * (len(full_tokens) - len(prompt_tokens))

    return MaskedSample(tokens=full_tokens, weights=weights)


def completion_text(sample: MaskedSample, tokenizer: Tokenizer) -> str:
    """Detokenize exactly the span that carries loss.

    Useful for inspecting a mask, and used by the tests to prove it lines up. But mind one
    artifact, because it looks like a bug and is not:

    **A SentencePiece detokenizer strips the leading space of a span's first token.** SPM encodes
    " leading" as a single ``▁leading`` token, and ``decode`` drops that space when the token
    starts the sequence it is given. So a completion of ``" foo"`` comes back as ``"foo"`` — the
    space is still *in the trained token*, and the mask is still correct; it is the detokenizer
    that is not a faithful inverse of a mid-text span.

    Do not "fix" a mask because of this. The property that matters is the token-prefix check in
    :func:`build_masked_sample`, which is exact.
    """
    trained = [t for t, w in zip(sample.tokens, sample.weights, strict=True) if w > 0.0]
    return tokenizer.decode(trained)
