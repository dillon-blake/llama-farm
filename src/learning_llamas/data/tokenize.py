"""Tokenize and detokenize through llama.cpp's own vocab.

Not through a Python tokenizer. The model was trained by llama.cpp's tokenizer and will be
served by it, so anything else is a second implementation that can disagree — and a tokenizer
disagreement between training and serving is invisible until the model is quietly worse.
"""

from __future__ import annotations

import ctypes

from learning_llamas import _ffi


class Tokenizer:
    """The model's vocabulary, as llama.cpp sees it.

    Attributes:
        n_tokens: Vocabulary size.
        bos: Beginning-of-sequence token id, or -1 if the model has none.
        eos: End-of-sequence token id, or -1.
        eot: End-of-turn token id, or -1. Chat templates use this to close an assistant turn.
        add_bos: Whether the model's tokenizer prepends BOS by default.
        add_eos: Whether it appends EOS by default.
    """

    def __init__(self, libs: _ffi.Libraries, model: int) -> None:
        """Wrap the vocabulary of an already-loaded model.

        Args:
            libs: The loaded native libraries.
            model: A ``llama_model *``.
        """
        self._libs = libs
        self._vocab = libs.llama.llama_model_get_vocab(model)

        self.n_tokens: int = libs.llama.llama_vocab_n_tokens(self._vocab)
        self.bos: int = libs.llama.llama_vocab_bos(self._vocab)
        self.eos: int = libs.llama.llama_vocab_eos(self._vocab)
        self.eot: int = libs.llama.llama_vocab_eot(self._vocab)
        self.add_bos: bool = libs.llama.llama_vocab_get_add_bos(self._vocab)
        self.add_eos: bool = libs.llama.llama_vocab_get_add_eos(self._vocab)

    def encode(self, text: str, add_special: bool = False, parse_special: bool = True) -> list[int]:
        """Tokenize ``text``.

        Args:
            text: The text to tokenize.
            add_special: Let the tokenizer add BOS/EOS itself. **Defaults to False**, because
                templated text already carries them — see :mod:`learning_llamas.data.mask`.
            parse_special: Recognize special-token *text* (``<|im_start|>``) as the token rather
                than tokenizing it character by character. Defaults to True, because a rendered
                chat template is full of them.

        Returns:
            The token ids.

        Raises:
            RuntimeError: If llama.cpp reports a tokenization failure.
        """
        raw = text.encode("utf-8")

        # The two-call convention: a first call with no output buffer returns the required size
        # as a NEGATIVE number. Guessing a buffer size instead is how you get silent truncation.
        needed = self._libs.llama.llama_tokenize(
            self._vocab, raw, len(raw), None, 0, add_special, parse_special
        )
        n = -needed
        if n < 0:
            raise RuntimeError(f"llama_tokenize failed with {needed}")
        if n == 0:
            return []

        out = (_ffi.llama_token * n)()
        written = self._libs.llama.llama_tokenize(
            self._vocab, raw, len(raw), out, n, add_special, parse_special
        )
        if written < 0:
            raise RuntimeError(f"llama_tokenize failed with {written}")

        return list(out[:written])

    def decode(self, tokens: list[int], special: bool = True) -> str:
        """Detokenize ``tokens`` back to text.

        Args:
            tokens: The token ids.
            special: Render special tokens as their text rather than dropping them.

        Returns:
            The text.

        Raises:
            RuntimeError: If llama.cpp reports a detokenization failure.
        """
        if not tokens:
            return ""

        buf_in = (_ffi.llama_token * len(tokens))(*tokens)

        needed = self._libs.llama.llama_detokenize(
            self._vocab, buf_in, len(tokens), None, 0, False, special
        )
        n = -needed
        if n < 0:
            raise RuntimeError(f"llama_detokenize failed with {needed}")

        out = ctypes.create_string_buffer(n)
        written = self._libs.llama.llama_detokenize(
            self._vocab, buf_in, len(tokens), out, n, False, special
        )
        if written < 0:
            raise RuntimeError(f"llama_detokenize failed with {written}")

        return out.raw[:written].decode("utf-8", errors="replace")

    def piece(self, token: int, special: bool = True) -> str:
        """The text of a single token."""
        buf = ctypes.create_string_buffer(64)
        n = self._libs.llama.llama_token_to_piece(self._vocab, token, buf, 64, 0, special)
        if n < 0:
            buf = ctypes.create_string_buffer(-n)
            n = self._libs.llama.llama_token_to_piece(self._vocab, token, buf, -n, 0, special)
        return buf.raw[:n].decode("utf-8", errors="replace")
