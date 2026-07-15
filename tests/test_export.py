"""S1-08: save a trained adapter, and merge it into a standalone model.

Two artefacts, two different ways to be wrong.

**The adapter save** is exact or it is broken — there is no tolerance to hide in. What comes out of
a live adapter must be what goes into the file, and what comes out of the file when *stock*
llama.cpp reloads it must be the same numbers again. So the round-trip is asserted bit-for-bit.

**The merge** cannot be exact, and that is the interesting part. Folding the delta into a Q4_K
tensor means dequantizing it, adding, and quantizing it back — so the merged model differs from
"base + adapter at runtime" by one round-trip of quantization noise, unavoidably. The test that
matters is therefore not "are they equal" (they are not, and a test that claimed so would only be
measuring its own tolerance) but **"is the merge the only thing that changed"**:

* at scale 0 the merge is the identity, so a merged model must equal a *re-quantize round-trip of
  the base* — bit for bit. That isolates the quantization path completely: if this fails, the
  arithmetic is not what is wrong.
* at scale 1 the merged model's logits must track base+adapter far more closely than they track the
  bare base. That is what says the delta went in, went in the right way round, and went in at the
  right size — a transposed or mis-scaled delta fails it badly.
"""

import ctypes

import numpy as np
import pytest

from learning_llamas import _ffi
from learning_llamas.adapter import create_zero_adapter, read_adapter, save_adapter
from learning_llamas.export import merge
from learning_llamas.quant import ImatrixRequired, dequantize, quantize
from learning_llamas.train import Batch, TrainConfig, Trainer

RANK = 4
N_CTX = 64
SEQ_LEN = 32
ALPHA = float(RANK)  # so the effective scale is exactly the user scale

PROMPT = [7, 11, 13, 17, 19]


# ---------------------------------------------------------------------------
# quant.py: the thing gguf-py cannot do.
# ---------------------------------------------------------------------------


def test_gguf_py_cannot_quantize_k_quants_but_we_can(libs: _ffi.Libraries) -> None:
    """The reason quant.py exists, asserted rather than asserted-in-a-comment.

    gguf-py dequantizes everything and quantizes almost nothing. If it ever grows a Q4_K quantizer
    this test starts failing, which is the correct moment to find out — the fallback to libggml
    could then be dropped.
    """
    import gguf

    values = np.random.default_rng(0).standard_normal((4, 256)).astype(np.float32)

    with pytest.raises(NotImplementedError):
        gguf.quants.quantize(values, gguf.GGMLQuantizationType.Q4_K)

    # ...whereas libggml's own quantizer, which is what llama-quantize uses, does it fine.
    raw = quantize(values, gguf.GGMLQuantizationType.Q4_K, libs)
    back = dequantize(
        np.frombuffer(raw, dtype=np.uint8), gguf.GGMLQuantizationType.Q4_K, values.shape
    )

    assert back.shape == values.shape
    assert np.abs(back - values).max() < 0.2, "the round trip is not even approximately the input"


def test_an_imatrix_type_is_refused_not_approximated(libs: _ffi.Libraries) -> None:
    """A user who asked for IQ2_XXS wanted IQ2_XXS, not a guess at it."""
    import gguf

    values = np.random.default_rng(0).standard_normal((4, 256)).astype(np.float32)

    with pytest.raises(ImatrixRequired, match="importance matrix"):
        quantize(values, gguf.GGMLQuantizationType.IQ2_XXS, libs)


# ---------------------------------------------------------------------------
# The adapter save: exact, or broken.
# ---------------------------------------------------------------------------


@pytest.fixture
def trained(tiny_q4_k, tmp_path, load_model, libs: _ffi.Libraries):
    """A model whose adapter has actually been trained — B is no longer zero."""
    adapter_path = tmp_path / "init.gguf"
    create_zero_adapter(tiny_q4_k, adapter_path, r=RANK, alpha=ALPHA, seed=7)

    model = load_model(tiny_q4_k, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True)
    model.attach_adapter(adapter_path, scale=1.0)

    batch = Batch(
        tokens=[7, 11, 13, 17] * (SEQ_LEN // 4),
        targets=([11, 13, 17, 7] * (SEQ_LEN // 4)),
        weights=[1.0] * SEQ_LEN,
    )

    with Trainer(libs, model, TrainConfig(lr=2e-2)) as trainer:
        for _ in range(12):
            trainer.step(batch)

    return model


def test_saving_a_trained_adapter_round_trips_bit_for_bit(
    trained, tmp_path, libs: _ffi.Libraries
) -> None:
    """Out of the live adapter, into a file, back out through STOCK llama.cpp — unchanged.

    The reload deliberately goes through ``llama_adapter_lora_init``, the same loader
    ``llama-cli --lora`` uses. A file only we can read is not a format, it is a pickle.
    """
    model = trained
    out = tmp_path / "trained.gguf"

    n = save_adapter(libs, model.adapter, out, architecture="llama", alpha=ALPHA)
    assert n > 0

    before = _adapter_values(libs, model.adapter)
    assert any(np.abs(b).max() > 0 for _, b in before.values()), "nothing trained; nothing to prove"

    # Reload through stock llama.cpp, into a fresh handle.
    reloaded = libs.llama.llama_adapter_lora_init(model.model, str(out).encode())
    assert reloaded, "stock llama.cpp refused the adapter we just wrote"

    after = _adapter_values(libs, reloaded)

    assert set(before) == set(after)
    for name, (a, b) in before.items():
        a2, b2 = after[name]
        # Bit for bit. F32 in, F32 out, no arithmetic in between: anything less is a bug.
        assert np.array_equal(a, a2), f"{name}.lora_a changed in the round trip"
        assert np.array_equal(b, b2), f"{name}.lora_b changed in the round trip"


def test_the_saved_adapter_records_its_alpha_and_rank(trained, tmp_path, libs) -> None:
    out = tmp_path / "trained.gguf"
    save_adapter(libs, trained.adapter, out, architecture="llama", alpha=ALPHA)

    info = read_adapter(out)

    assert info.architecture == "llama"
    assert info.alpha == pytest.approx(ALPHA)
    assert set(info.ranks.values()) == {RANK}


def test_saving_with_alpha_zero_is_refused(trained, tmp_path, libs) -> None:
    """Alpha == 0 does not mean "scale 0" to llama.cpp — it means "drop the alpha/rank factor"."""
    with pytest.raises(ValueError, match="non-zero"):
        save_adapter(libs, trained.adapter, tmp_path / "x.gguf", architecture="llama", alpha=0.0)


def _adapter_values(libs, adapter) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Every (A, B) of an adapter handle, by base tensor name, straight from the shim."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    n = libs.farm.ll_adapter_n_tensors(adapter)
    for i in range(n):
        name_buf = ctypes.create_string_buffer(256)
        ne_a = (ctypes.c_int64 * 4)()
        ne_b = (ctypes.c_int64 * 4)()
        assert libs.farm.ll_adapter_tensor_info(adapter, i, name_buf, 256, ne_a, ne_b) == 0

        def read(is_b: bool, i=i) -> np.ndarray:
            k = libs.farm.ll_adapter_get(adapter, i, is_b, None, 0)
            buf = (ctypes.c_float * k)()
            assert libs.farm.ll_adapter_get(adapter, i, is_b, buf, k) == k
            return np.frombuffer(buf, dtype=np.float32, count=k).copy()

        out[name_buf.value.decode()] = (read(False), read(True))

    return out


# ---------------------------------------------------------------------------
# The merge: not exact, and the tests know it.
# ---------------------------------------------------------------------------


def test_merging_at_scale_zero_is_a_pure_requantize_round_trip(
    tiny_q4_k, tmp_path, libs: _ffi.Libraries
) -> None:
    """At scale 0 the delta vanishes, so the merge must be the identity — modulo one round trip.

    This isolates the quantization path from the arithmetic completely. The merged file must be
    **bit-identical** to what you get by dequantizing every adapted tensor and quantizing it
    straight back. If it is not, the bug is in the quant plumbing and nowhere near the LoRA maths —
    which is exactly the thing you want to know before trusting any of the numbers below.
    """
    import gguf

    adapter_path = tmp_path / "a.gguf"
    create_zero_adapter(tiny_q4_k, adapter_path, r=RANK, alpha=ALPHA, seed=1)

    out = tmp_path / "merged0.gguf"
    touched = merge(tiny_q4_k, adapter_path, out, libs, scale=0.0)

    assert touched, "the adapter matched nothing"

    base = gguf.GGUFReader(str(tiny_q4_k), "r")
    got = gguf.GGUFReader(str(out), "r")

    by_name = {t.name: t for t in got.tensors}

    for tensor in base.tensors:
        merged_tensor = by_name[tensor.name]
        assert merged_tensor.tensor_type == tensor.tensor_type, tensor.name

        if tensor.name not in touched:
            # Never dequantized, so it must be byte-identical, not merely close.
            assert np.array_equal(
                np.asarray(merged_tensor.data).reshape(-1), np.asarray(tensor.data).reshape(-1)
            ), f"{tensor.name} was not adapted but its bytes changed"
            continue

        shape = tuple(int(x) for x in tensor.shape)[::-1]
        values = dequantize(tensor.data, tensor.tensor_type, shape)
        expected = quantize(values, tensor.tensor_type, libs)

        # Flattened: GGUFReader hands quantized data back as (n_rows, row_bytes) while quantize()
        # returns it flat. The bytes are what is being compared, not their shape.
        assert np.array_equal(
            np.asarray(merged_tensor.data).reshape(-1), np.frombuffer(expected, np.uint8)
        ), f"{tensor.name}: merging at scale 0 did not reproduce a plain requantize round trip"


def test_the_merged_model_tracks_base_plus_adapter(
    trained, tiny_q4_k, tmp_path, load_model, libs: _ffi.Libraries
) -> None:
    """The fidelity gate: a merged model behaves like the model it was merged from.

    It cannot be *identical*: folding the delta into a Q4_K tensor costs one round trip of
    quantization noise, and no amount of care removes it. So the claim is comparative, and it is the
    one that actually distinguishes a correct merge from a broken one:

        the merged model's logits are much closer to (base + adapter) than to the bare base.

    A transposed delta, a delta scaled by ``alpha`` instead of ``alpha/rank``, or a delta added to
    the wrong tensor all leave the merged model sitting nearer the bare base — or nowhere near
    either. Only a correct merge lands on top of base+adapter.
    """
    model = trained

    adapter_out = tmp_path / "trained.gguf"
    save_adapter(libs, model.adapter, adapter_out, architecture="llama", alpha=ALPHA)

    merged_path = tmp_path / "merged.gguf"
    merge(tiny_q4_k, adapter_out, merged_path, libs, scale=1.0)

    # Three models, one prompt.
    bare = load_model(tiny_q4_k, n_ctx=N_CTX)
    logits_base = np.array(bare.logits(PROMPT))

    with_adapter = load_model(tiny_q4_k, n_ctx=N_CTX)
    with_adapter.attach_adapter(adapter_out, scale=1.0)
    logits_adapter = np.array(with_adapter.logits(PROMPT))

    merged_model = load_model(merged_path, n_ctx=N_CTX)
    logits_merged = np.array(merged_model.logits(PROMPT))

    # The adapter does something at all, or none of the below means anything.
    moved = np.abs(logits_adapter - logits_base).max()
    assert moved > 1e-3, f"the trained adapter barely changes the logits ({moved:.3g})"

    to_adapter = np.abs(logits_merged - logits_adapter).max()
    to_base = np.abs(logits_merged - logits_base).max()

    assert to_adapter < to_base, (
        f"the merged model is closer to the BARE base ({to_base:.4f}) than to base+adapter "
        f"({to_adapter:.4f}). The delta went in transposed, mis-scaled, or not at all."
    )

    # ...and not merely closer: most of the adapter's effect must have survived the merge. The
    # residual is quantization noise, which is a fraction of the delta, not comparable to it.
    assert to_adapter < 0.35 * moved, (
        f"the merged model kept only part of the adapter: it sits {to_adapter:.4f} from "
        f"base+adapter, while the adapter itself moves the logits by {moved:.4f}"
    )

    # The greedy argmax -- what a user actually sees -- must agree.
    assert int(logits_merged.argmax()) == int(logits_adapter.argmax()), (
        "the merged model and base+adapter disagree on the very next token"
    )


def test_a_merged_model_keeps_its_tokenizer(tiny_q4_k, tmp_path, libs: _ffi.Libraries) -> None:
    """A merged model that lost its tokenizer or chat template is not a model."""
    import gguf

    adapter_path = tmp_path / "a.gguf"
    create_zero_adapter(tiny_q4_k, adapter_path, r=RANK, alpha=ALPHA, seed=1)

    out = tmp_path / "merged.gguf"
    merge(tiny_q4_k, adapter_path, out, libs, scale=1.0)

    base_keys = set(gguf.GGUFReader(str(tiny_q4_k), "r").fields)
    merged_keys = set(gguf.GGUFReader(str(out), "r").fields)

    missing = base_keys - merged_keys
    assert not missing, f"the merge dropped {sorted(missing)}"


def test_an_adapter_for_another_model_is_refused(
    tiny_q4_k, tiny_f32, tmp_path, libs: _ffi.Libraries
) -> None:
    """Better a clear error than a merged model with a delta from somewhere else in it."""
    import gguf

    adapter_path = tmp_path / "wrong.gguf"
    create_zero_adapter(tiny_q4_k, adapter_path, r=RANK, alpha=ALPHA, seed=1)

    # Rewrite its architecture so it no longer matches.
    reader = gguf.GGUFReader(str(adapter_path), "r")
    bad = tmp_path / "bad-arch.gguf"
    writer = gguf.GGUFWriter(str(bad), arch="mistral")
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, ALPHA)
    for tensor in reader.tensors:
        writer.add_tensor(
            tensor.name, np.array(tensor.data, dtype=np.float32).reshape(tuple(tensor.shape)[::-1])
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    with pytest.raises(ValueError, match="architecture"):
        merge(tiny_q4_k, bad, tmp_path / "out.gguf", libs)


# ---------------------------------------------------------------------------
# The token_embd flipped-convention merge (S1-44).
#
# Every default-preset target folds `delta = B @ A`. `token_embd.weight` does NOT: the loader
# applies it A-transposed (llama-adapter.cpp:356-368), and no existing test touches it, because the
# default preset excludes token_embd. A wrong transpose here yields a merged model whose embeddings
# are garbage -- and nothing would have caught it. These tests pin the convention numerically.
# ---------------------------------------------------------------------------

# alpha/rank == 2: a non-trivial token_embd scale, so a merge that (wrongly) dropped the alpha/rank
# factor -- effective 1.0 -- would MISS the oracle rather than coincide with it at 1.0.
TE_ALPHA = 2.0 * RANK


def _write_adapter(
    path, arch: str, alpha: float, pairs: list[tuple[str, np.ndarray, np.ndarray]]
) -> None:
    """Write a LoRA adapter GGUF from explicit ``(name, A, B)`` numpy pairs.

    Mirrors :func:`learning_llamas.adapter._write_adapter_gguf`, spelled out here so a test can put
    a chosen B into the file (``create_zero_adapter`` only ever writes ``B == 0``). A and B are in
    numpy layout -- the GGUF ``ne`` reversed -- and ``add_tensor`` reverses them back, so the ne
    that lands in the file is exactly the one the loader validates.
    """
    import gguf

    writer = gguf.GGUFWriter(str(path), arch=arch)
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, alpha)
    for name, a, b in pairs:
        writer.add_tensor(f"{name}.lora_a", a)
        writer.add_tensor(f"{name}.lora_b", b)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _read_ab(path) -> tuple[str, float, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Read an adapter GGUF back into ``arch, alpha, {base_name: (A, B)}`` numpy arrays.

    numpy shapes are the GGUF ``ne`` reversed, so an A written from ``(n_vocab, r)`` comes back as
    ``(n_vocab, r)`` -- the shape the oracle below reasons about.
    """
    import gguf

    reader = gguf.GGUFReader(str(path), "r")
    arch = str(reader.get_field(gguf.Keys.General.ARCHITECTURE).contents())
    alpha = float(reader.get_field(gguf.Keys.Adapter.LORA_ALPHA).contents())

    parts: dict[str, dict[str, np.ndarray]] = {}
    for t in reader.tensors:
        for suffix, role in ((".lora_a", "a"), (".lora_b", "b")):
            if t.name.endswith(suffix):
                base = t.name[: -len(suffix)]
                arr = np.array(t.data, dtype=np.float32).reshape(tuple(t.shape)[::-1])
                parts.setdefault(base, {})[role] = arr
    return arch, alpha, {name: (ab["a"], ab["b"]) for name, ab in parts.items()}


def _tensor(path, name: str):
    """One tensor of a GGUF, by name."""
    import gguf

    return {t.name: t for t in gguf.GGUFReader(str(path), "r").tensors}[name]


def _npshape(tensor) -> tuple[int, ...]:
    """A GGUF tensor's numpy shape -- its ``ne`` reversed."""
    return tuple(int(x) for x in tensor.shape)[::-1]


def _nonzero_token_embd_adapter(
    base_path, out_path, seed: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """A ``token_embd``-only adapter with nonzero A *and* B, plus the exact (A, B) it holds.

    ``create_zero_adapter`` writes ``A ~ N(0, sigma)`` and ``B == 0`` in the loader's flipped
    token_embd convention. We keep that A, fill B with real values, rewrite, and read both back --
    so the caller's oracle uses the bytes actually in the file, not the ones it meant to write.
    """
    zero = out_path.parent / (out_path.stem + "-zero.gguf")
    create_zero_adapter(
        base_path, zero, r=RANK, alpha=TE_ALPHA, preset=(), include_token_embd=True, seed=seed
    )
    arch, alpha, ab = _read_ab(zero)
    a, b_zero = ab["token_embd.weight"]
    assert not b_zero.any(), "a fresh adapter must have B == 0"

    b = (np.random.default_rng(seed + 1).standard_normal(b_zero.shape) * 0.02).astype(np.float32)
    _write_adapter(out_path, arch, alpha, [("token_embd.weight", a, b)])

    _, _, ab2 = _read_ab(out_path)
    return ab2["token_embd.weight"]


def test_token_embd_merge_uses_the_flipped_convention(tiny_f32, tmp_path, libs: _ffi.Libraries):
    """token_embd merges A-transposed -- checked against a numpy oracle, not against export._delta.

    The loader applies the token_embd LoRA in ``llm_build_inp_embd``
    (vendor/llama.cpp/src/llama-graph.cpp:2268-2273) as::

        delta_row(t) = scale * mul_mat(B, get_rows(A, tokens))

    so for token ``t`` and embedding component ``i`` the added value is ``sum_k B[i,k] * A[t,k]`` --
    the whole delta is ``D = A @ B.T`` of shape ``(n_vocab, n_embd)``, with A stored
    ``(n_vocab, r)`` and B stored ``(n_embd, r)`` (the flipped shape the loader validates at
    llama-adapter.cpp:356-368). This oracle is that math written out from first principles. Because
    the base is F32 the merge is exact (dequantize/quantize are identities), so the check is a
    strict equality, not a tolerance to hide in.
    """
    adapter_path = tmp_path / "te.gguf"
    a, b = _nonzero_token_embd_adapter(tiny_f32, adapter_path)

    n_vocab, r = a.shape
    n_embd, r_b = b.shape
    assert (r, r_b) == (RANK, RANK)
    assert n_vocab != n_embd, "the fixture must be non-square, so orientation errors cannot hide"

    scale = 1.0
    merged_path = tmp_path / "merged.gguf"
    touched = merge(tiny_f32, adapter_path, merged_path, libs, scale=scale)
    assert set(touched) == {"token_embd.weight"}

    # ---- the oracle, transcribed from the graph above (NOT by calling export._delta) ----
    rank = r  # llama.cpp reads the rank from B.ne[0] == r (llama-adapter.h:53)
    effective = scale * TE_ALPHA / rank  # alpha != 0, so the alpha/rank factor applies
    delta = a @ b.T
    assert delta.shape == (n_vocab, n_embd)

    base_te = _tensor(tiny_f32, "token_embd.weight")
    base_vals = dequantize(base_te.data, base_te.tensor_type, _npshape(base_te))
    expected = base_vals + effective * delta

    merged_te = _tensor(merged_path, "token_embd.weight")
    merged_vals = dequantize(merged_te.data, merged_te.tensor_type, _npshape(merged_te))

    tol = 1e-6
    off = float(np.abs(merged_vals - expected).max())
    # Observed: 0.0 (bit-exact) -- F32 in, F32 out, identical float ops on both sides.
    assert off < tol, f"merged token_embd is {off:.3g} from the flipped-convention oracle"

    # ---- can-fail: a transposed delta could not survive this test ----
    # (1) shape: n_vocab != n_embd, so the transpose D.T is (n_embd, n_vocab) and does not even fit
    #     the base tensor -- a `b @ a.T` / `(a @ b.T).T` mix-up RAISES in merge, it does not pass.
    assert delta.T.shape != base_vals.shape
    # (2) value: on the overlapping square block, A @ B.T is strongly non-symmetric, so had the fold
    #     used the transpose the merged tensor would differ from this oracle by ~1e5x `tol`.
    k = min(delta.shape)
    block = delta[:k, :k]
    asymmetry = float(np.abs(block - block.T).max())  # observed ~0.15
    assert asymmetry > 1e4 * tol, (
        f"the delta is nearly symmetric ({asymmetry:.3g}); this oracle could not tell A @ B.T from "
        "its transpose, so it would not catch a flipped-convention bug"
    )


def test_the_token_embd_merged_model_matches_base_plus_adapter(
    tiny_f32, tmp_path, load_model, libs: _ffi.Libraries
):
    """The token_embd merge is not just arithmetic on paper: the merged GGUF runs and agrees.

    A merged model folds the embedding delta into ``token_embd.weight``; base+adapter applies it at
    ``get_rows`` time. On an F32 base those are the same computation reordered, so the logits must
    agree to float32 round-off -- while both sit far from the bare base, proving the delta is really
    present. (tiny_f32 is not tied, so ``output.weight`` is untouched and the comparison is clean.)
    """
    adapter_path = tmp_path / "te.gguf"
    _nonzero_token_embd_adapter(tiny_f32, adapter_path)

    merged_path = tmp_path / "merged.gguf"
    merge(tiny_f32, adapter_path, merged_path, libs, scale=1.0)

    bare = load_model(tiny_f32, n_ctx=N_CTX)
    logits_base = np.array(bare.logits(PROMPT))

    with_adapter = load_model(tiny_f32, n_ctx=N_CTX)
    with_adapter.attach_adapter(adapter_path, scale=1.0)
    logits_adapter = np.array(with_adapter.logits(PROMPT))

    merged_model = load_model(merged_path, n_ctx=N_CTX)
    logits_merged = np.array(merged_model.logits(PROMPT))

    moved = float(np.abs(logits_adapter - logits_base).max())
    assert moved > 1e-2, f"the token_embd adapter barely moves the logits ({moved:.3g})"

    to_adapter = float(np.abs(logits_merged - logits_adapter).max())
    # Observed: 8.2e-05, against a bare-base distance of ~0.89 -- float32 associativity, not a
    # modelling difference. A transposed or mis-scaled embedding delta lands nowhere near here.
    assert to_adapter < 1e-3, (
        f"the merged model diverges from base+adapter by {to_adapter:.3g}; on an F32 base the "
        "flipped-convention fold should reproduce it to round-off"
    )
    assert int(logits_merged.argmax()) == int(logits_adapter.argmax()), (
        "the merged model and base+adapter disagree on the very next token"
    )


def test_a_q8_0_merge_preserves_every_tensors_quant_type(tiny_q8_0, tmp_path, libs: _ffi.Libraries):
    """S1-08 AC, the Q8_0 half: a merged Q8_0 base stays Q8_0 -- no silent F16 fallback.

    ``test_merging_at_scale_zero...`` pins type preservation for Q4_K only; Q8_0 is a *named* S1-08
    acceptance criterion that had no test. A real (nonzero) delta is folded into every
    default-preset target so the re-quantize path runs on merged data, not a pass-through.
    """
    import gguf

    zero = tmp_path / "zero.gguf"
    create_zero_adapter(tiny_q8_0, zero, r=RANK, alpha=ALPHA, seed=5)

    arch, alpha, ab = _read_ab(zero)
    rng = np.random.default_rng(7)
    pairs = [
        (name, a, (rng.standard_normal(b.shape) * 0.01).astype(np.float32))
        for name, (a, b) in ab.items()
    ]
    nonzero = tmp_path / "nz.gguf"
    _write_adapter(nonzero, arch, alpha, pairs)

    merged_path = tmp_path / "merged.gguf"
    touched = merge(tiny_q8_0, nonzero, merged_path, libs, scale=1.0)
    assert touched, "the adapter matched nothing"

    base_types = {t.name: t.tensor_type for t in gguf.GGUFReader(str(tiny_q8_0), "r").tensors}
    merged_types = {t.name: t.tensor_type for t in gguf.GGUFReader(str(merged_path), "r").tensors}

    assert set(base_types) == set(merged_types)
    # Every tensor keeps its exact type: Q8_0 weights re-quantize to Q8_0, the F32 norms pass
    # through as F32. Nothing here is an F32 *target*, so there is no legitimate type change to
    # exempt -- and FALLBACK_TYPE (F16) must appear nowhere, since that is what an unimplemented
    # re-quantizer would emit, quadrupling the file.
    for name, base_type in base_types.items():
        assert merged_types[name] == base_type, (
            f"{name}: base is {base_type.name}, merged is {merged_types[name].name}"
        )
    assert gguf.GGMLQuantizationType.F16 not in merged_types.values()

    # ...and the Q8_0 path was genuinely exercised -- not vacuously true over an all-F32 base.
    q8 = gguf.GGMLQuantizationType.Q8_0
    q8_touched = [n for n in touched if base_types[n] == q8]
    assert q8_touched, "no Q8_0 tensor was folded; this test would prove nothing about Q8_0"
    assert all(merged_types[n] == q8 for n in q8_touched)
