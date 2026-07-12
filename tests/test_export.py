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
