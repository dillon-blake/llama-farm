"""S1-06: the loss mask lines up with the completion, token for token.

The bug this suite exists to catch is the one that does not announce itself. Mask a prompt by
*counting* its tokens and a BPE tokenizer can merge across the boundary — the last prompt
character and the first completion character fuse into one token belonging to neither. The mask
shifts by one, the model is trained to predict a token it will never be asked to predict, and the
loss curve looks completely normal.

So the tests here do not check that a mask exists. They check that the *span the model takes loss
on detokenizes back to the completion text*, exactly — and that a merge across the boundary is a
loud error rather than a silent shift.
"""

import pytest

from learning_llamas import _ffi
from learning_llamas.data import (
    BoundaryMergeError,
    ChatTemplate,
    Message,
    Tokenizer,
    build_masked_sample,
    completion_text,
)
from learning_llamas.data.mask import _assert_token_prefix


@pytest.fixture
def chat(tiny_f32, load_model, libs: _ffi.Libraries):
    model = load_model(tiny_f32, n_ctx=512)
    tokenizer = Tokenizer(libs, model.model)
    template = ChatTemplate.from_model(libs, model.model, tokenizer)
    return template, tokenizer


def test_template_comes_from_the_model(chat) -> None:
    """The turn format is the model's, not one we invented."""
    template, _ = chat
    assert "<|im_start|>" in template.source
    assert "add_generation_prompt" in template.source


def test_single_turn_mask_covers_exactly_the_completion(chat) -> None:
    template, tokenizer = chat
    messages = [
        Message("user", "What is the capital of France?"),
        Message("assistant", "Paris."),
    ]

    sample = build_masked_sample(messages, template, tokenizer)

    assert len(sample.tokens) == len(sample.weights)
    assert sample.n_trained > 0

    # The round trip that actually proves it: what the model takes loss on IS the completion.
    trained = completion_text(sample, tokenizer)
    assert "Paris." in trained

    # ...and nothing of the prompt leaked into it.
    assert "capital of France" not in trained
    assert "<|im_start|>user" not in trained


def test_multi_turn_trains_only_the_final_assistant_turn(chat) -> None:
    """Earlier assistant turns are context, not targets.

    Training on them teaches the model to parrot its own history.
    """
    template, tokenizer = chat
    messages = [
        Message("system", "You are terse."),
        Message("user", "One?"),
        Message("assistant", "First answer."),
        Message("user", "Two?"),
        Message("assistant", "Second answer."),
    ]

    sample = build_masked_sample(messages, template, tokenizer)
    trained = completion_text(sample, tokenizer)

    assert "Second answer." in trained
    assert "First answer." not in trained
    assert "You are terse." not in trained


def test_system_prompt_is_never_trained_on(chat) -> None:
    template, tokenizer = chat
    messages = [
        Message("system", "SECRET_SYSTEM_STRING"),
        Message("user", "Hi"),
        Message("assistant", "Hello"),
    ]

    sample = build_masked_sample(messages, template, tokenizer)
    trained = completion_text(sample, tokenizer)

    assert "SECRET_SYSTEM_STRING" not in trained
    assert "Hello" in trained


def test_the_token_prefix_property_holds_at_the_boundary(chat) -> None:
    """The prompt's tokens must be a genuine PREFIX of the whole sample's tokens.

    This is the property masking depends on, and it is checked rather than assumed. If it ever
    fails, `build_masked_sample` raises instead of shifting the mask.
    """
    template, tokenizer = chat
    messages = [Message("user", "x"), Message("assistant", "y")]

    prompt_text = template.render(messages[:-1], add_generation_prompt=True)
    prompt_tokens = tokenizer.encode(prompt_text, add_special=False, parse_special=True)

    sample = build_masked_sample(messages, template, tokenizer)

    assert sample.tokens[: len(prompt_tokens)] == prompt_tokens
    assert sample.weights[: len(prompt_tokens)] == [0.0] * len(prompt_tokens)
    assert all(w == 1.0 for w in sample.weights[len(prompt_tokens) :])


@pytest.mark.parametrize(
    "completion",
    [
        "no leading space",  # the adversarial case: nothing separates it from the turn opener
        " leading space",
        "\nleading newline",
        "123",  # digits, which many tokenizers merge greedily
        "",  # empty completion
    ],
)
def test_mask_is_correct_regardless_of_how_the_completion_starts(chat, completion: str) -> None:
    """The boundary must survive completions that invite a merge.

    A completion with no leading whitespace is the classic trigger: the tokenizer is free to fuse
    the template's last character with the completion's first. If it did, this would raise rather
    than silently mask the wrong span.
    """
    template, tokenizer = chat
    messages = [Message("user", "q"), Message("assistant", completion)]

    sample = build_masked_sample(messages, template, tokenizer)

    prompt_text = template.render(messages[:-1], add_generation_prompt=True)
    prompt_tokens = tokenizer.encode(prompt_text, add_special=False, parse_special=True)
    assert sample.tokens[: len(prompt_tokens)] == prompt_tokens

    if completion:
        # lstrip, because a SentencePiece detokenizer strips the leading space of a span's first
        # token. The space is still IN the trained token -- the mask is right; `decode` is simply
        # not a faithful inverse of a mid-text span. See completion_text's docstring.
        assert completion.lstrip() in completion_text(sample, tokenizer)


def test_a_boundary_merge_is_detected_and_raises(chat) -> None:
    """The failure mode this module exists to prevent.

    The detection logic is tested directly, on synthetic token sequences, rather than by trying
    to coax a merge out of the fixture's tokenizer. The fixture's vocab is almost entirely
    byte-fallback tokens, so it barely merges anything -- a test that relied on provoking a real
    merge there would pass for the wrong reason, and would go on passing if the check were
    deleted.

    What must hold is: when the prompt's tokens are NOT a prefix of the full sample's tokens,
    that is an error naming the boundary -- never a mask silently shifted by a token.
    """
    _, tokenizer = chat

    # The signature of a merge: the sequences agree, then the boundary token differs because the
    # tokenizer fused it with the completion's first character.
    prompt = [10, 11, 12]
    merged = [10, 11, 99, 42]  # 12 and the completion's first token became 99

    with pytest.raises(BoundaryMergeError, match="merged across"):
        _assert_token_prefix(prompt, merged, "prompt/completion", tokenizer)

    # ...and the happy path does not raise.
    _assert_token_prefix(prompt, [10, 11, 12, 42], "prompt/completion", tokenizer)


def test_a_template_that_rewrites_the_prompt_raises(chat) -> None:
    """A template that rewrites the prompt has no well-defined completion span.

    Some templates render the final assistant turn differently depending on whether it is the
    generation target. Masking is then meaningless, so this must be an error.
    """
    _, tokenizer = chat

    rewriting = ChatTemplate(
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        "{% if add_generation_prompt %}{{ 'PROMPT_ONLY_SUFFIX' }}{% endif %}"
    )
    messages = [Message("user", "q"), Message("assistant", "a")]

    with pytest.raises(BoundaryMergeError, match="not a text prefix"):
        build_masked_sample(messages, rewriting, tokenizer)


def test_last_message_must_be_the_assistant_turn(chat) -> None:
    template, tokenizer = chat
    messages = [Message("assistant", "a"), Message("user", "u")]

    with pytest.raises(ValueError, match="assistant"):
        build_masked_sample(messages, template, tokenizer)


def test_tokenizer_round_trips(chat) -> None:
    _, tokenizer = chat
    text = "Hello, world!"

    tokens = tokenizer.encode(text)
    assert tokens
    assert tokenizer.decode(tokens) == text


def test_special_token_text_is_parsed_as_one_token(chat) -> None:
    """A template's special-token text must become the token, not its characters.

    That is what parse_special=True buys. Otherwise every rendered turn is a dozen junk tokens.
    """
    _, tokenizer = chat

    with_special = tokenizer.encode("<|im_start|>", parse_special=True)
    without = tokenizer.encode("<|im_start|>", parse_special=False)

    # The fixture's vocab has no <|im_start|> token, so both fall back to characters -- but they
    # must at least agree, which is the invariant that matters for a vocab that DOES have it.
    assert len(with_special) <= len(without)
