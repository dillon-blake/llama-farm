"""S1-06: a template-less GGUF produces the documented error, not a byte-fallback surprise.

``ChatTemplate.from_model`` raises ``ValueError`` when the model embeds no chat template -- normal
for a base model, and the acceptance criterion says it is *tested*. Every committed fixture embeds a
ChatML template, so this test builds a template-less one: it copies the F32 fixture minus the single
``tokenizer.chat_template`` metadata key (reusing ``export``'s proven byte-exact tensor copy, so the
result is a fully loadable model that differs from the fixture in exactly that one key), and drives
the NULL branch.

Editing ``gen_tiny_llama.py`` to emit a template-less variant is off the table: its ``cache_key``
hashes the whole generator source, and that key is baked into the convergence reference-curve
identity, so any edit would force a re-record. Post-processing a copy sidesteps that entirely.
"""

from __future__ import annotations

import pathlib

import gguf
import numpy as np
import pytest

from learning_llamas import _ffi
from learning_llamas.data import ChatTemplate, Tokenizer
from learning_llamas.export import _SKIP_KEYS, _sub_type, _write


def _strip_chat_template(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Copy a GGUF, dropping only the ``tokenizer.chat_template`` key."""
    reader = gguf.GGUFReader(str(src), "r")
    writer = gguf.GGUFWriter(str(dst), arch="llama")

    skip = _SKIP_KEYS | {gguf.Keys.Tokenizer.CHAT_TEMPLATE}
    for key, field in reader.fields.items():
        if key in skip or not field.types:
            continue
        writer.add_key_value(key, field.contents(), field.types[0], sub_type=_sub_type(field))

    for tensor in reader.tensors:
        _write(writer, tensor.name, np.asarray(tensor.data), tensor.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.fixture
def template_less_gguf(tiny_f32, tmp_path) -> pathlib.Path:
    out = tmp_path / "no-template.gguf"
    _strip_chat_template(tiny_f32, out)
    return out


def test_a_template_less_model_raises_the_documented_error(
    template_less_gguf, load_model, libs: _ffi.Libraries
) -> None:
    """The NULL-template branch of ``from_model`` is a clear ValueError, not a crash."""
    model = load_model(template_less_gguf, n_ctx=512)
    tokenizer = Tokenizer(libs, model.model)

    # Sanity: the strip really removed the template (else the test proves nothing).
    assert not libs.llama.llama_model_chat_template(model.model, None)

    with pytest.raises(ValueError, match="no chat template"):
        ChatTemplate.from_model(libs, model.model, tokenizer)


def test_the_stripped_copy_is_otherwise_the_same_model(
    template_less_gguf, tiny_f32, load_model, libs: _ffi.Libraries
) -> None:
    """The copy differs in exactly one key: the ORIGINAL fixture still yields a template.

    Without this, the test above could pass because the copy is broken in some unrelated way rather
    than because the template is absent -- the mutation guard the repo asks for.
    """
    model = load_model(tiny_f32, n_ctx=512)
    tokenizer = Tokenizer(libs, model.model)
    template = ChatTemplate.from_model(libs, model.model, tokenizer)
    assert "<|im_start|>" in template.source
