"""S0-06: the generated fixture models are real, loadable llama.cpp models."""

import math
import pathlib
import subprocess

import gguf
import pytest

from .fixtures import gen_tiny_llama

TOKENS = [1, 5, 9, 42, 7, 3]


def test_every_variant_loads_and_decodes(tiny_model: pathlib.Path, load_model) -> None:
    """F32, Q8_0 and Q4_K all load through _ffi and produce finite logits."""
    model = load_model(tiny_model)
    logits = model.logits(TOKENS)

    assert len(logits) == gen_tiny_llama.HPARAMS.n_vocab
    assert all(math.isfinite(x) for x in logits)
    assert any(x != 0.0 for x in logits)


def test_quantized_variants_track_the_f32_model(
    tiny_f32: pathlib.Path, tiny_q8_0: pathlib.Path, tiny_q4_k: pathlib.Path, load_model
) -> None:
    """Quantization is lossy, but not arbitrary: the logits must track F32.

    This distinguishes a genuinely quantized fixture from one whose bytes were written wrong and
    happen to decode to *something*. These are garbage detectors, not numerics tolerances — the
    real cross-backend bounds live in ADR-0002. The sharp signal is the *ordering*: Q8_0 must be
    strictly closer to F32 than Q4_K, which no byte-level mistake would reproduce by accident.
    """
    reference = load_model(tiny_f32).logits(TOKENS)
    q8_0 = load_model(tiny_q8_0).logits(TOKENS)
    q4_k = load_model(tiny_q4_k).logits(TOKENS)

    def max_abs_diff(a: list[float], b: list[float]) -> float:
        return max(abs(x - y) for x, y in zip(a, b, strict=True))

    err_q8_0 = max_abs_diff(reference, q8_0)
    err_q4_k = max_abs_diff(reference, q4_k)

    # The fixture's weights are random N(0, 0.02), so there is no learned structure for the
    # quantizer to preserve; 4-bit drift of this size on untrained noise is expected.
    assert err_q8_0 < 0.05, f"Q8_0 logits drifted from F32 by {err_q8_0}"
    assert err_q4_k < 0.50, f"Q4_K logits drifted from F32 by {err_q4_k}"
    assert err_q8_0 < err_q4_k, "Q8_0 should be closer to F32 than Q4_K"


def test_quantized_variants_really_are_quantized(
    tiny_q8_0: pathlib.Path, tiny_q4_k: pathlib.Path
) -> None:
    """The 2-D tensors carry the quantized type; 1-D norms stay F32."""
    expected = {
        tiny_q8_0: gguf.GGMLQuantizationType.Q8_0,
        tiny_q4_k: gguf.GGMLQuantizationType.Q4_K,
    }

    for path, qtype in expected.items():
        reader = gguf.GGUFReader(str(path), "r")
        types = {t.name: t.tensor_type for t in reader.tensors}

        assert types["token_embd.weight"] == qtype
        assert types["blk.0.attn_q.weight"] == qtype
        assert types["blk.0.ffn_down.weight"] == qtype
        assert types["output.weight"] == qtype
        # Norms are 1-D; quantization tooling leaves them alone.
        assert types["blk.0.attn_norm.weight"] == gguf.GGMLQuantizationType.F32
        assert types["output_norm.weight"] == gguf.GGMLQuantizationType.F32


def test_quantized_variants_are_smaller(
    tiny_f32: pathlib.Path, tiny_q8_0: pathlib.Path, tiny_q4_k: pathlib.Path
) -> None:
    assert tiny_q8_0.stat().st_size < tiny_f32.stat().st_size
    assert tiny_q4_k.stat().st_size < tiny_q8_0.stat().st_size


def test_second_build_is_a_cache_hit(fixture_cache_dir: pathlib.Path) -> None:
    """Fixtures are generated once per content hash, then reused."""
    first_path, _ = gen_tiny_llama.build("f32", fixture_cache_dir)
    mtime = first_path.stat().st_mtime_ns

    second_path, generated = gen_tiny_llama.build("f32", fixture_cache_dir)

    assert generated is False
    assert second_path == first_path
    assert second_path.stat().st_mtime_ns == mtime


def test_cache_key_tracks_the_generator(fixture_cache_dir: pathlib.Path) -> None:
    """Changing the hyperparameters must invalidate the cache, not reuse a stale model."""
    default = gen_tiny_llama.cache_key()
    other = gen_tiny_llama.cache_key(gen_tiny_llama.TinyLlamaHParams(n_layer=3))
    assert default != other


def test_dimensions_are_k_quantizable() -> None:
    """Q4_K's superblock is 256 elements, so every quantized row must be a multiple of 256.

    The ticket's suggested n_embd=64 could not be Q4_K-quantized at all — hence n_embd=256 and
    n_ff=512. validate() is what stops a future edit from silently reintroducing the problem.
    """
    gen_tiny_llama.HPARAMS.validate()

    with pytest.raises(ValueError, match="multiples of 256"):
        gen_tiny_llama.TinyLlamaHParams(n_embd=64, n_head=4).validate()


def test_no_model_binaries_are_committed() -> None:
    """Fixtures are generated, never committed."""
    tracked = subprocess.run(
        ["git", "ls-files", "*.gguf"],
        capture_output=True,
        text=True,
        check=True,
        cwd=pathlib.Path(__file__).resolve().parents[1],
    )
    assert tracked.stdout.strip() == "", f"model binaries are tracked by git:\n{tracked.stdout}"
