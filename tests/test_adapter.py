"""S0-05: enumerate targets, write a zero-init adapter GGUF, read it back.

Pure Python — no native build required. The base model is synthesized in-test with GGUFWriter,
so these tests pin the *file format* contract against llama.cpp's loader rules without ever
loading a model.
"""

import pathlib

import gguf
import numpy as np
import pytest

from learning_llamas.adapter import (
    DEFAULT_PRESET,
    create_zero_adapter,
    enumerate_targets,
    read_adapter,
)

N_EMBD = 32
N_FF = 64
N_VOCAB = 48
N_LAYER = 2


@pytest.fixture(scope="module")
def base_gguf(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """A minimal llama-arch base model: just enough tensors to enumerate targets from."""
    path = tmp_path_factory.mktemp("adapter") / "base.gguf"
    writer = gguf.GGUFWriter(str(path), arch="llama")
    writer.add_block_count(N_LAYER)
    writer.add_embedding_length(N_EMBD)
    writer.add_feed_forward_length(N_FF)

    def f32(*shape: int) -> np.ndarray:
        return np.zeros(shape, dtype=np.float32)

    # numpy shape is the reverse of GGUF ne: (n_out, n_in) is written as ne = [n_in, n_out].
    writer.add_tensor("token_embd.weight", f32(N_VOCAB, N_EMBD))  # ne = [n_embd, n_vocab]
    writer.add_tensor("output.weight", f32(N_VOCAB, N_EMBD))  # ne = [n_embd, n_vocab]
    writer.add_tensor("output_norm.weight", f32(N_EMBD))

    for il in range(N_LAYER):
        writer.add_tensor(f"blk.{il}.attn_q.weight", f32(N_EMBD, N_EMBD))
        writer.add_tensor(f"blk.{il}.attn_k.weight", f32(N_EMBD, N_EMBD))
        writer.add_tensor(f"blk.{il}.attn_v.weight", f32(N_EMBD, N_EMBD))
        writer.add_tensor(f"blk.{il}.attn_output.weight", f32(N_EMBD, N_EMBD))
        writer.add_tensor(f"blk.{il}.ffn_up.weight", f32(N_FF, N_EMBD))  # ne = [n_embd, n_ff]
        writer.add_tensor(f"blk.{il}.ffn_gate.weight", f32(N_FF, N_EMBD))
        writer.add_tensor(f"blk.{il}.ffn_down.weight", f32(N_EMBD, N_FF))  # ne = [n_ff, n_embd]
        # Norms are not LoRA targets (the loader ignores norm vectors), and the MoE expert
        # tensor must NOT be swept up by a naive `endswith("ffn_down.weight")` match.
        writer.add_tensor(f"blk.{il}.attn_norm.weight", f32(N_EMBD))
        writer.add_tensor(f"blk.{il}.ffn_down_exps.weight", f32(N_EMBD, N_FF))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def test_enumerate_returns_exactly_the_default_preset(base_gguf: pathlib.Path) -> None:
    targets = enumerate_targets(base_gguf)
    names = {t.name for t in targets}

    expected = {
        f"blk.{il}.{module}.weight"
        for il in range(N_LAYER)
        for module in (
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_up",
            "ffn_gate",
            "ffn_down",
        )
    }
    assert names == expected

    # Norms, the LM head, the embedding table, and the MoE expert tensor are all excluded.
    assert not any("norm" in n for n in names)
    assert not any("_exps" in n for n in names)
    assert "output.weight" not in names
    assert "token_embd.weight" not in names


def test_enumerate_opt_ins_extend_the_preset(base_gguf: pathlib.Path) -> None:
    targets = enumerate_targets(base_gguf, include_output=True, include_token_embd=True)
    names = {t.name for t in targets}

    assert "output.weight" in names
    assert "token_embd.weight" in names
    assert len(names) == len(enumerate_targets(base_gguf)) + 2

    token_embd = next(t for t in targets if t.name == "token_embd.weight")
    assert token_embd.is_token_embd
    assert (token_embd.n_in, token_embd.n_out) == (N_EMBD, N_VOCAB)


def test_enumerate_reads_ne_order(base_gguf: pathlib.Path) -> None:
    """n_in is ne[0] and n_out is ne[1] — the asymmetric ffn tensors prove it isn't reversed."""
    targets = {t.name: t for t in enumerate_targets(base_gguf)}

    up = targets["blk.0.ffn_up.weight"]
    assert (up.n_in, up.n_out) == (N_EMBD, N_FF)

    down = targets["blk.0.ffn_down.weight"]
    assert (down.n_in, down.n_out) == (N_FF, N_EMBD)


def test_adapter_has_the_four_kvs(base_gguf: pathlib.Path, tmp_path: pathlib.Path) -> None:
    out = tmp_path / "adapter.gguf"
    create_zero_adapter(base_gguf, out, r=8)

    reader = gguf.GGUFReader(str(out), "r")
    assert str(reader.get_field(gguf.Keys.General.TYPE).contents()) == "adapter"
    assert str(reader.get_field(gguf.Keys.Adapter.TYPE).contents()) == "lora"
    assert str(reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()) == "llama"

    alpha_field = reader.get_field(gguf.Keys.Adapter.LORA_ALPHA)
    assert alpha_field.types[0] == gguf.GGUFValueType.FLOAT32
    assert float(alpha_field.contents()) == 8.0  # alpha defaults to r


def test_normal_target_shapes_satisfy_the_loader(
    base_gguf: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """The checks at src/llama-adapter.cpp:362-367, applied to every non-embedding target.

    model.ne[0] == a.ne[0]  and  model.ne[1] == b.ne[1]  and  a.ne[1] == b.ne[0]
    """
    r = 8
    out = tmp_path / "adapter.gguf"
    targets = create_zero_adapter(base_gguf, out, r=r)
    info = read_adapter(out)

    for target in targets:
        a_ne, b_ne = info.shapes[target.name]
        assert a_ne == (target.n_in, r), target.name
        assert b_ne == (r, target.n_out), target.name
        assert a_ne[1] == b_ne[0], target.name
        assert info.ranks[target.name] == r


def test_token_embd_uses_the_flipped_convention(
    base_gguf: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """The checks at src/llama-adapter.cpp:356-360 — A and B are flipped for token_embd.

        model.ne[0] == b.ne[1]  and  model.ne[1] == a.ne[1]

    Getting this wrong is not a subtle numerical error: the loader throws "incorrect shape".
    """
    r = 8
    out = tmp_path / "adapter.gguf"
    create_zero_adapter(base_gguf, out, r=r, include_token_embd=True)
    info = read_adapter(out)

    a_ne, b_ne = info.shapes["token_embd.weight"]

    # token_embd.weight has ne = [n_embd, n_vocab].
    assert b_ne[1] == N_EMBD  # model.ne[0] == b.ne[1]
    assert a_ne[1] == N_VOCAB  # model.ne[1] == a.ne[1]
    assert b_ne[0] == r  # the loader reads the rank from b.ne[0]

    # ...and it is genuinely flipped relative to a normal target, whose A is (n_in, r).
    normal_a, _ = info.shapes["blk.0.attn_q.weight"]
    assert normal_a == (N_EMBD, r)
    assert a_ne == (r, N_VOCAB)


def test_b_is_exactly_zero_and_a_is_seeded(base_gguf: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """B == 0 exactly is the whole no-op property: scale * B(A @ x) == 0 for any A."""
    out = tmp_path / "adapter.gguf"
    create_zero_adapter(base_gguf, out, r=8, seed=1234)

    reader = gguf.GGUFReader(str(out), "r")
    saw_a = saw_b = False
    for tensor in reader.tensors:
        assert tensor.tensor_type == gguf.GGMLQuantizationType.F32
        if tensor.name.endswith(".lora_b"):
            assert np.count_nonzero(tensor.data) == 0, f"{tensor.name} is not all-zero"
            saw_b = True
        elif tensor.name.endswith(".lora_a"):
            assert np.count_nonzero(tensor.data) > 0, f"{tensor.name} is all-zero"
            saw_a = True

    assert saw_a and saw_b


def test_tensor_names_keep_the_weight_suffix(
    base_gguf: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """The loader strips only '.lora_a' and then calls model.get_tensor() on the remainder.

    So the name must be `blk.0.attn_q.weight.lora_a`. Dropping `.weight` yields a file that
    fails to load with "unexpected suffix" or a missing base tensor.
    """
    out = tmp_path / "adapter.gguf"
    create_zero_adapter(base_gguf, out, r=4)

    reader = gguf.GGUFReader(str(out), "r")
    names = [t.name for t in reader.tensors]

    assert "blk.0.attn_q.weight.lora_a" in names
    assert "blk.0.attn_q.weight.lora_b" in names
    for name in names:
        stripped = name.removesuffix(".lora_a").removesuffix(".lora_b")
        assert stripped.endswith(".weight"), name


def test_alpha_zero_is_rejected(base_gguf: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Alpha == 0 silently DROPS the alpha/rank factor rather than zeroing the adapter."""
    with pytest.raises(ValueError, match="alpha/rank"):
        create_zero_adapter(base_gguf, tmp_path / "bad.gguf", r=8, alpha=0)


def test_same_seed_is_byte_identical(base_gguf: pathlib.Path, tmp_path: pathlib.Path) -> None:
    first = tmp_path / "a.gguf"
    second = tmp_path / "b.gguf"
    third = tmp_path / "c.gguf"

    create_zero_adapter(base_gguf, first, r=8, seed=7)
    create_zero_adapter(base_gguf, second, r=8, seed=7)
    create_zero_adapter(base_gguf, third, r=8, seed=8)

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() != third.read_bytes()


def test_read_adapter_reports_metadata(base_gguf: pathlib.Path, tmp_path: pathlib.Path) -> None:
    out = tmp_path / "adapter.gguf"
    targets = create_zero_adapter(base_gguf, out, r=16, alpha=32.0)

    info = read_adapter(out)
    assert info.architecture == "llama"
    assert info.alpha == 32.0
    assert set(info.ranks) == {t.name for t in targets}
    assert set(info.ranks.values()) == {16}


def test_rank_must_be_positive(base_gguf: pathlib.Path, tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="rank must be positive"):
        create_zero_adapter(base_gguf, tmp_path / "bad.gguf", r=0)


def test_default_preset_is_the_blueprint_set() -> None:
    assert DEFAULT_PRESET == (
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_qkv",
        "attn_output",
        "ffn_up",
        "ffn_gate",
        "ffn_down",
    )
