"""S0-06: the generated fixture models are real, loadable llama.cpp models."""

import math
import pathlib
import subprocess

import gguf
import pytest

from .fixtures import gen_tiny_llama, gen_tiny_mamba, gen_tiny_mamba2, gen_tiny_moe

# The three generators that are NOT self-contained: they build their GGUFs out of gen_tiny_llama's
# CHAT_TEMPLATE and _load_vocab (and, for the MoE, its quantizer and type tables).
SHARED_SOURCE_GENERATORS = (gen_tiny_moe, gen_tiny_mamba, gen_tiny_mamba2)

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


def test_an_interrupted_build_leaves_no_false_cache_hit(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl-C partway through the first generation must not poison the cache permanently.

    ``build`` decides on a bare ``path.exists()`` and never rewrites a file that is already there,
    so a truncated GGUF sitting at the final path is forever: every later run returns it, and the
    damage surfaces as an unreadable model a long way from the generator. Writing to
    ``.gguf.partial`` and renaming makes the cache entry appear only once it is complete.

    The MoE generator is the one that wrote straight to the final path and the reason this test
    exists, but all four are checked: the write-then-rename discipline is a property of the family,
    and a generator added later would otherwise be free to drop it unnoticed.
    """
    for module, stem in (
        (gen_tiny_moe, "tiny-moe"),
        (gen_tiny_mamba, "tiny-mamba"),
        (gen_tiny_mamba2, "tiny-mamba2"),
        (gen_tiny_llama, "tiny-llama"),
    ):

        def crash_partway(path: pathlib.Path, *args: object, **kwargs: object) -> None:
            path.write_bytes(b"GGUF\x00truncated")
            raise KeyboardInterrupt

        monkeypatch.setattr(module, "_write", crash_partway)

        cache_dir = tmp_path / stem
        with pytest.raises(KeyboardInterrupt):
            module.build("f32", cache_dir)

        final = cache_dir / module.cache_key() / f"{stem}-f32.gguf"
        assert not final.exists(), (
            f"{module.__name__}: an interrupted build left a truncated file at the final cache "
            "path, and every later run will treat it as a valid fixture"
        )


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


def _round_tripped_source(module) -> bytes:  # noqa: ANN001 - a module, of four different ones
    """A generator's source as a CRLF checkout hands it over, normalized back the way keys do."""
    source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
    crlf = source.replace("\n", "\r\n")
    assert crlf != source, f"{module.__name__} has no newlines?"
    return crlf.replace("\r\n", "\n").encode()


def test_the_cache_key_does_not_depend_on_line_endings() -> None:
    """Git hands Windows a CRLF working tree, and the key must not notice.

    ``cache_key`` hashes the generator's source. Hashing the raw *bytes* makes the key depend on the
    checkout's line endings — so the same generator hashes differently on Windows. S1-12's
    ``reference_curve.json`` embeds that key as its identity, and the convergence gate duly failed
    on ci-windows with "recorded against a different fixture", against a fixture that was
    byte-for-byte the same model.

    Simulate the CRLF checkout directly: hash the source, then hash a CRLF copy of it, and require
    the same answer. The payload formula is mirrored here rather than called, because calling
    ``cache_key`` is exactly what this must not trust.
    """
    import hashlib
    from importlib.metadata import version

    llama_payload = b"".join(
        [
            _round_tripped_source(gen_tiny_llama),
            repr(gen_tiny_llama.HPARAMS).encode(),
            version("gguf").encode(),
        ]
    )
    assert hashlib.sha256(llama_payload).hexdigest()[:16] == gen_tiny_llama.cache_key(), (
        "gen_tiny_llama.cache_key() changes under a CRLF checkout"
    )

    # The siblings fold in the llama source and their seed as well, so both have to survive the
    # CRLF round trip too — the shared source is the bigger half of their payload.
    for module in SHARED_SOURCE_GENERATORS:
        payload = b"".join(
            [
                _round_tripped_source(module),
                _round_tripped_source(gen_tiny_llama),
                repr(module.HPARAMS).encode(),
                repr(module.SEED).encode(),
                version("gguf").encode(),
            ]
        )
        assert hashlib.sha256(payload).hexdigest()[:16] == module.cache_key(), (
            f"{module.__name__}.cache_key() changes under a CRLF checkout"
        )


@pytest.mark.parametrize(
    "module", SHARED_SOURCE_GENERATORS, ids=lambda m: m.__name__.rsplit(".", 1)[-1]
)
def test_the_sibling_cache_keys_track_the_shared_llama_generator(
    module, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Editing gen_tiny_llama.py must invalidate the MoE and Mamba fixtures, not only llama's.

    Those three do not build their GGUFs out of their own source alone — ``CHAT_TEMPLATE`` and
    ``_load_vocab`` come from ``gen_tiny_llama``, and the MoE takes its quantizer and type tables
    from there too. A key that hashed only the sibling's own file would not move when the chat
    template changed, so the llama fixtures would regenerate while a warm ``tests/.fixtures`` went
    on serving MoE and Mamba models carrying the OLD template. conftest's header promises that
    cannot happen; until the shared source was folded into these keys, it could.

    The edit is simulated rather than performed: ``gen_tiny_llama.py`` is frozen, because its key is
    the convergence reference-curve identity and touching it forces a re-record. Pointing the
    module's ``__file__`` at an edited copy exercises the same path — ``llama_source()`` resolves it
    per call — without writing to the real file.
    """
    before = module.cache_key()

    source = pathlib.Path(gen_tiny_llama.__file__).read_text(encoding="utf-8")
    assert "<|im_start|>" in source, "the chat template moved; this test is now editing nothing"
    edited = tmp_path / "gen_tiny_llama.py"
    edited.write_text(source.replace("<|im_start|>", "<|turn_start|>"), encoding="utf-8")
    monkeypatch.setattr(gen_tiny_llama, "__file__", str(edited))

    assert module.cache_key() != before, (
        f"{module.__name__}.cache_key() ignores gen_tiny_llama.py, so a warm fixture cache would "
        "keep serving a model built from the old chat template"
    )


@pytest.mark.parametrize(
    "module", SHARED_SOURCE_GENERATORS, ids=lambda m: m.__name__.rsplit(".", 1)[-1]
)
def test_the_seed_names_the_fixture(module) -> None:
    """A seed that is not in the key is a seed that does not select a model.

    ``build(seed=...)`` is public surface. The key names the directory the GGUF is cached under, so
    if the seed is absent from it, the second caller asking for a *different* seed gets the first
    caller's model straight back off the cache — silently, with no file on disk that disagrees.
    No caller does this today, which is exactly why it would have gone unnoticed.
    """
    assert module.cache_key(seed=module.SEED + 1) != module.cache_key()
