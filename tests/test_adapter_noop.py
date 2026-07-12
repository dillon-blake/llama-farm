"""S0-06: the zero-B adapter is a provable no-op — the project's first end-to-end gate.

A freshly initialized LoRA adapter has ``A ~ N(0, sigma)`` and ``B = 0``, so the delta it adds
is ``scale * B(A @ x)`` — **exactly** zero, for any A, at any scale. Attaching it must therefore
leave the model's logits *bit-identical*, not merely close (BLUEPRINT D3).

That exactness is what makes this test worth having. It validates, end to end and against the
stock loader, everything S0-05 writes: the four KVs, the tensor names, both shape conventions,
the F32 dtypes, and the alpha caveat. Approximate equality would pass even if the adapter were
being applied with a tiny but non-zero scale; exact equality cannot.
"""

import pathlib

import pytest

from learning_llamas.adapter import create_zero_adapter


@pytest.fixture
def zero_adapter(tiny_model: pathlib.Path, tmp_path: pathlib.Path) -> pathlib.Path:
    out = tmp_path / "zero-adapter.gguf"
    create_zero_adapter(tiny_model, out, r=8)
    return out


def test_zero_adapter_at_scale_1_changes_nothing(
    tiny_model: pathlib.Path, zero_adapter: pathlib.Path, load_model
) -> None:
    """Logits with a zero-B adapter attached at scale 1.0 == logits without it. Exactly."""
    tokens = [1, 5, 9, 42, 7, 3]

    baseline = load_model(tiny_model).logits(tokens)

    adapted_model = load_model(tiny_model)
    adapted_model.attach_adapter(zero_adapter, scale=1.0)
    adapted = adapted_model.logits(tokens)

    assert adapted == baseline, "the zero-B adapter perturbed the logits"


def test_zero_adapter_at_a_large_scale_still_changes_nothing(
    tiny_model: pathlib.Path, zero_adapter: pathlib.Path, load_model
) -> None:
    """B == 0 makes the delta zero *independently of scale* — so a huge scale is still a no-op.

    This separates "the adapter is zero" from "the adapter is not being applied at all": a
    scale of 1000 would blow up any non-zero delta into something impossible to miss.
    """
    tokens = [1, 5, 9, 42, 7, 3]

    baseline = load_model(tiny_model).logits(tokens)

    adapted_model = load_model(tiny_model)
    adapted_model.attach_adapter(zero_adapter, scale=1000.0)
    adapted = adapted_model.logits(tokens)

    assert adapted == baseline


def test_no_op_holds_with_the_embedding_and_head_adapted(
    tiny_model: pathlib.Path, tmp_path: pathlib.Path, load_model
) -> None:
    """Same property with the opt-in targets on — which is what exercises the flipped
    token_embd convention against the *stock loader*, not just against our own reader.

    If the flipped shapes at src/llama-adapter.cpp:356-360 were written wrong, the adapter
    would not load at all and this test would error rather than fail.
    """
    tokens = [1, 5, 9, 42, 7, 3]

    out = tmp_path / "full-adapter.gguf"
    create_zero_adapter(tiny_model, out, r=8, include_output=True, include_token_embd=True)

    baseline = load_model(tiny_model).logits(tokens)

    adapted_model = load_model(tiny_model)
    adapted_model.attach_adapter(out, scale=1.0)
    adapted = adapted_model.logits(tokens)

    assert adapted == baseline
