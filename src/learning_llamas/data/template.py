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

from dataclasses import dataclass

import jinja2
import jinja2.sandbox

from learning_llamas import _ffi


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
        cls, libs: _ffi.Libraries, model: int, tokenizer, name: str | None = None
    ) -> ChatTemplate:
        """Extract the template embedded in a loaded model.

        Args:
            libs: The loaded native libraries.
            model: A ``llama_model *``.
            tokenizer: A :class:`~learning_llamas.data.tokenize.Tokenizer` for the same model,
                used to resolve the BOS/EOS token *text*.
            name: A named template variant, or None for the default.

        Returns:
            The compiled template.

        Raises:
            ValueError: If the GGUF embeds no chat template. Pass an explicit template source to
                the constructor instead — the error says so, because "no chat template" is a
                thing that happens to base models and is not a bug.
        """
        raw = libs.llama.llama_model_chat_template(model, name.encode() if name else None)
        if not raw:
            raise ValueError(
                "this model embeds no chat template (tokenizer.chat_template is absent), which "
                "is normal for a base model. Construct ChatTemplate(source=...) with the "
                "template you intend to train against."
            )

        return cls(
            raw.decode(),
            bos_token=tokenizer.piece(tokenizer.bos) if tokenizer.bos >= 0 else "",
            eos_token=tokenizer.piece(tokenizer.eos) if tokenizer.eos >= 0 else "",
        )

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
