"""Render chat messages with the model's own embedded chat template.

The template comes out of the GGUF (``tokenizer.chat_template``), not out of a config file we
invent. A model trained with one turn format and fine-tuned with another learns the mismatch, and
nothing in the loss curve says so.

A caveat worth knowing: llama.cpp renders templates with **minja**, its own minimal Jinja engine,
while this renders with **jinja2**. For the templates the supported model set actually ships,
these agree — and ``tests/test_data_roundtrip.py`` is what holds that claim to account, by
checking the rendered text tokenizes to a token sequence with the prefix property the mask
depends on. If a template ever renders differently under the two engines, the round-trip test is
where it surfaces.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import jinja2
import jinja2.sandbox

from learning_llamas import _ffi

from .tokenize import Tokenizer


class UnknownSpecialTokenError(ValueError):
    """A chat template names a special token this model's vocabulary does not contain.

    BLUEPRINT §8's one v1 hard rule: **no vocab resize and no new special tokens.** The base
    token embeddings are frozen and quantized, so a control token the base vocab lacks has no
    embedding to train — ``parse_special`` cannot match it, the tokenizer byte-falls-back into a
    dozen meaningless per-character tokens, and the model is trained to reproduce *that* instead
    of the single control token the template meant. That is silent: the loss still falls, the
    mask still lines up on a byte boundary, and the model quietly learns to spell out
    ``< | i m _ s t a r t | >`` where it should emit one token. This error is the data layer
    refusing, loudly and by name, rather than shipping that garbage into the training set.
    """


def guard_special_tokens(tokenizer: Tokenizer, special_tokens: Iterable[str]) -> None:
    """Verify every declared special-token string exists in the model's vocabulary.

    The check is exact and cheap: a genuine special token tokenizes, under ``parse_special``, to
    **exactly one** id (its own). A token the vocab lacks byte-falls-back to many — the same
    sequence ``parse_special=False`` would produce — so ``len != 1`` is precisely the
    no-new-special-tokens failure (BLUEPRINT §8).

    What this does *not* do is scan rendered output for ``<|...|>``-shaped substrings and flag any
    that byte-fall-back. That would be unsound: a tokenizer's byte-fallback of an ordinary control
    string is indistinguishable from a legitimately literal one, so such a scan would reject a
    perfectly valid template (the tiny fixtures' ChatML markers are byte-fallback and correct —
    their mask lands on the trailing newline). Instead the guard validates a *declared* set: the
    special-token strings the model exposes to the template (its BOS/EOS text) plus any the caller
    names via ``require_special``. See ``tickets/stage-1-cpu/S1-45-vocab-guard.md`` for the choice.

    Args:
        tokenizer: The model's tokenizer.
        special_tokens: The special-token strings to validate. Empty strings are skipped: a model
            with no BOS reports its BOS text as ``""``, which means "no such token", not "an
            unknown one".

    Raises:
        UnknownSpecialTokenError: If any string does not tokenize to exactly one existing vocab id,
            naming the offending token and the byte-fallback it produced instead.
    """
    for text in special_tokens:
        if not text:
            continue
        ids = tokenizer.encode(text, add_special=False, parse_special=True)
        if len(ids) != 1:
            raise UnknownSpecialTokenError(
                f"the special token {text!r} is not in this model's vocabulary: with "
                f"parse_special it tokenizes to {len(ids)} tokens ({ids}), not one, so it would "
                f"be trained as byte-fallback characters rather than a control token. BLUEPRINT "
                f"§8 forbids adding special tokens in v1 — the base embeddings are frozen and "
                f"quantized, so a token the base vocab lacks cannot be learned. Use a template "
                f"built from this model's own special tokens, or a base whose vocab already "
                f"contains {text!r}."
            )


@dataclass(frozen=True)
class Message:
    """One chat turn.

    Attributes:
        role: ``system``, ``user``, or ``assistant``.
        content: The turn's text.
    """

    role: str
    content: str


class ChatTemplate:
    """The model's chat template, rendered.

    Attributes:
        source: The raw Jinja source, as embedded in the GGUF.
    """

    def __init__(self, source: str, bos_token: str = "", eos_token: str = "") -> None:
        """Compile a template.

        Args:
            source: The Jinja source.
            bos_token: The BOS token's text, exposed to the template as ``bos_token``.
            eos_token: Likewise for ``eos_token``.
        """
        self.source = source
        self._bos = bos_token
        self._eos = eos_token

        # A sandboxed environment: a chat template is data that arrives inside a model file, and
        # a model file is something a user downloads from the internet. Rendering it with an
        # unsandboxed Jinja environment would let a crafted GGUF execute arbitrary Python.
        env = jinja2.sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, undefined=jinja2.StrictUndefined
        )
        env.filters["tojson"] = jinja2.filters.do_tojson
        self._template = env.from_string(source)

    @classmethod
    def from_model(
        cls,
        libs: _ffi.Libraries,
        model: int,
        tokenizer: Tokenizer,
        name: str | None = None,
        require_special: Iterable[str] = (),
    ) -> ChatTemplate:
        """Extract the template embedded in a loaded model.

        Args:
            libs: The loaded native libraries.
            model: A ``llama_model *``.
            tokenizer: A :class:`~learning_llamas.data.tokenize.Tokenizer` for the same model,
                used to resolve the BOS/EOS token *text*.
            name: A named template variant, or None for the default.
            require_special: Extra special-token strings the template (or a caller who brought a
                ``--chat-template`` override) depends on, e.g. ``("<|im_start|>", "<|im_end|>")``.
                Each is run through the vocab guard alongside the model's own BOS/EOS text, so a
                template that names a control token the base vocab lacks fails here rather than
                byte-falling-back into garbage training data (BLUEPRINT §8).

        Returns:
            The compiled template.

        Raises:
            ValueError: If the GGUF embeds no chat template. Pass an explicit template source to
                the constructor instead — the error says so, because "no chat template" is a
                thing that happens to base models and is not a bug.
            UnknownSpecialTokenError: If any required special token — the model's own BOS/EOS or a
                ``require_special`` entry — is not a single id in this model's vocabulary.
        """
        raw = libs.llama.llama_model_chat_template(model, name.encode() if name else None)
        if not raw:
            raise ValueError(
                "this model embeds no chat template (tokenizer.chat_template is absent), which "
                "is normal for a base model. Construct ChatTemplate(source=...) with the "
                "template you intend to train against."
            )

        template = cls(
            raw.decode(),
            bos_token=tokenizer.piece(tokenizer.bos) if tokenizer.bos >= 0 else "",
            eos_token=tokenizer.piece(tokenizer.eos) if tokenizer.eos >= 0 else "",
        )

        # The vocab guard, wired into the normal load path so it is not decorative: validate the
        # special-token strings the model hands the template (BOS/EOS) plus any the caller declares.
        guard_special_tokens(tokenizer, (template._bos, template._eos, *require_special))

        return template

    def render(self, messages: list[Message], add_generation_prompt: bool = False) -> str:
        """Render messages to the text the model expects.

        Args:
            messages: The conversation.
            add_generation_prompt: Append the assistant-turn opener, so the model's next token is
                the start of its reply. Used when building the *prefix* of a training sample.

        Returns:
            The rendered text.
        """
        return self._template.render(
            messages=[{"role": m.role, "content": m.content} for m in messages],
            add_generation_prompt=add_generation_prompt,
            bos_token=self._bos,
            eos_token=self._eos,
        )
