"""Does a MoE model actually train? (S1-25 / S1-26 / S1-27 / S1-28).

The MoE kernels are checked op by op — against finite differences, against a double-precision
reference, across thread counts, across strides. All of that says the arithmetic is right. None of
it says a MoE model *trains*: that needs the router, the top-k, the expert gather, the
LoRA-on-a-3D-expert-stack pattern and the optimizer all working together, in a real graph, on a
real GGUF, with a loss that goes down.

Every one of those pieces was a separate assumption, and one of them was wrong (`enumerate_targets`
read only `ne[0]` and `ne[1]`, so a 3D expert stack came out as a 2D adapter — which llama.cpp's
loader *accepts*, because it validates only those two dimensions, and then reads a 4-expert model
as a 1-expert one).

So: a 4-expert, top-2 Mixtral-shaped fixture, LoRA on the expert stacks, and a loss that falls.
"""

from __future__ import annotations

import pathlib

import pytest

from learning_llamas import _ffi, create_zero_adapter, enumerate_targets, preflight
from learning_llamas.data import MaskedSample
from learning_llamas.train import SFTConfig, TrainConfig, Trainer, train_sft

from .fixtures import gen_tiny_moe

SEQ_LEN = 32
N_CTX = 64
RANK = 4

# One sequence per packed sample, plus one for the pads (S1-07).
N_SEQ_MAX = 4


@pytest.fixture(scope="session")
def tiny_moe_f32(libs: _ffi.Libraries, fixture_cache_dir: pathlib.Path) -> pathlib.Path:
    path, _ = gen_tiny_moe.build("f32", fixture_cache_dir)
    return path


@pytest.fixture(scope="session")
def tiny_moe_q4_k(libs: _ffi.Libraries, fixture_cache_dir: pathlib.Path) -> pathlib.Path:
    """Quantized experts — the case that makes OUT_PROD_ID's dequantize path load-bearing.

    The base expert stacks are frozen Q4_K. They take no gradient themselves, but the activations
    flowing *into* them do, because an earlier layer's LoRA is upstream. That is the only path in
    the project where d(b) has to propagate back through a quantized weight.
    """
    path, _ = gen_tiny_moe.build("q4_k", fixture_cache_dir)
    return path


def _samples(n: int = 6) -> list[MaskedSample]:
    """Synthetic samples: a masked prompt, then tokens to actually predict."""
    out = []
    for s in range(n):
        tokens = [7 + ((s + i) % 13) for i in range(SEQ_LEN // 2)]
        weights = [0.0] * 4 + [1.0] * (len(tokens) - 4)
        out.append(MaskedSample(tokens=tokens, weights=weights))
    return out


# ---------------------------------------------------------------------------
# The fixture is a real MoE, and the adapter is really 3D.
# ---------------------------------------------------------------------------


def test_the_fixture_has_expert_stacks_and_a_router(tiny_moe_f32) -> None:
    """If this is secretly a dense model, everything below passes and proves nothing."""
    import gguf

    reader = gguf.GGUFReader(str(tiny_moe_f32), "r")
    names = {t.name: t for t in reader.tensors}

    assert "blk.0.ffn_gate_inp.weight" in names, "no router — this is not a MoE"
    for stack in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
        t = names[f"blk.0.{stack}.weight"]
        assert len(t.shape) == 3, f"{stack} is not a 3D expert stack: {t.shape}"
        assert t.shape[2] == gen_tiny_moe.HPARAMS.n_expert

    # And the dense FFN must be ABSENT, or llama.cpp would build the dense path instead.
    assert "blk.0.ffn_gate.weight" not in names


def test_the_adapter_for_an_expert_stack_is_3d(tiny_moe_f32, tmp_path) -> None:
    """A 2D adapter on a 3D base LOADS FINE and is then read as a 1-expert stack.

    llama.cpp's loader validates only ne[0] and ne[1] (llama-adapter.cpp:362-367), so nothing
    downstream complains. The shape has to be right here or it is wrong silently.
    """
    import gguf

    targets = enumerate_targets(tiny_moe_f32)
    experts = [t for t in targets if t.name.endswith("_exps.weight")]
    assert experts, "the preset did not pick up the expert stacks"
    assert all(t.n_expert == gen_tiny_moe.HPARAMS.n_expert for t in experts)

    out = tmp_path / "moe-adapter.gguf"
    create_zero_adapter(tiny_moe_f32, out, r=RANK, seed=3)

    reader = gguf.GGUFReader(str(out), "r")
    shapes = {t.name: tuple(int(x) for x in t.shape) for t in reader.tensors}

    a = shapes["blk.0.ffn_gate_exps.weight.lora_a"]
    b = shapes["blk.0.ffn_gate_exps.weight.lora_b"]
    hp = gen_tiny_moe.HPARAMS

    # a.ne = [n_in, r, n_expert];  b.ne = [r, n_out, n_expert]
    assert a == (hp.n_embd, RANK, hp.n_expert), a
    assert b == (RANK, hp.n_ff, hp.n_expert), b


# ---------------------------------------------------------------------------
# The preflight was the thing that used to lie about this.
# ---------------------------------------------------------------------------


def test_preflight_says_a_moe_model_trains(tiny_moe_f32, tmp_path, load_model, libs) -> None:
    """Before S1-25, MUL_MAT_ID had no backward and this reported the model as untrainable.

    Now it should report clean — and if it does not, it names the op, which is the point of it.
    """
    adapter = tmp_path / "a.gguf"
    create_zero_adapter(tiny_moe_f32, adapter, r=RANK, seed=3)

    model = load_model(
        tiny_moe_f32, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True, n_seq_max=N_SEQ_MAX
    )
    model.attach_adapter(adapter, scale=1.0)

    # preflight walks the TRAINING graph, so the optimizer has to exist first: it needs the graph
    # the backward would actually be built over, not an inference one.
    with Trainer(libs, model, TrainConfig(lr=1e-4)):
        report = preflight(libs, model.ctx, [7, 11, 13, 17] * 4)

    blockers = [f for f in report.findings if f.status != 0]
    assert not blockers, "preflight says the MoE model cannot train: " + "; ".join(
        f"{f.op}: {f.detail}" for f in blockers
    )


# ---------------------------------------------------------------------------
# The one that matters.
# ---------------------------------------------------------------------------


def test_a_moe_model_trains_end_to_end(tiny_moe_f32, tmp_path, load_model, libs) -> None:
    """Router, top-k, expert gather, LoRA inside the expert operand, optimizer. Loss must fall.

    This is the first thing in the repo that runs OUT_PROD_ID and OUT_PROD_ID_GRP inside a real
    graph rather than a test harness. Every kernel check said the arithmetic was right; none of
    them could say the pieces fit together.
    """
    adapter = tmp_path / "a.gguf"
    create_zero_adapter(tiny_moe_f32, adapter, r=RANK, seed=3)

    model = load_model(
        tiny_moe_f32, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True, n_seq_max=N_SEQ_MAX
    )
    model.attach_adapter(adapter, scale=1.0)

    result = train_sft(
        libs, model, _samples(), SFTConfig(lr=1e-2, seq_len=SEQ_LEN, pad_id=0, epochs=6)
    )

    losses = [step.loss for step in result.steps]
    assert len(losses) >= 6
    assert losses[-1] < losses[0], f"the MoE loss did not fall: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_a_quantized_moe_trains_through_the_dequantize_path(
    tiny_moe_q4_k, tmp_path, load_model, libs
) -> None:
    """Q4_K experts: d(b) has to propagate back THROUGH a quantized weight.

    That is OUT_PROD_ID's dequantize-a-row-at-a-time path, and it is the one MODE_GRAD structurally
    cannot check (ggml's quantized matmul quantizes the activations on the fly, so the forward is a
    staircase in b and a finite difference of it measures quantization edges). It was verified by
    direct equivalence against an F32 reference — and this is the only place it runs for real.
    """
    adapter = tmp_path / "a.gguf"
    create_zero_adapter(tiny_moe_q4_k, adapter, r=RANK, seed=3)

    model = load_model(
        tiny_moe_q4_k, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True, n_seq_max=N_SEQ_MAX
    )
    model.attach_adapter(adapter, scale=1.0)

    result = train_sft(
        libs, model, _samples(), SFTConfig(lr=1e-2, seq_len=SEQ_LEN, pad_id=0, epochs=6)
    )

    losses = [step.loss for step in result.steps]
    assert losses[-1] < losses[0], f"the quantized MoE loss did not fall: {losses}"


def test_training_a_moe_moves_the_expert_adapters(tiny_moe_f32, tmp_path, load_model, libs) -> None:
    """A falling loss is necessary, not sufficient: the ATTENTION LoRA alone could produce it.

    If OUT_PROD_ID_GRP were broken -- or simply never scheduled -- the expert stacks' A/B tensors
    would stay exactly at their zero-init B and the model would still learn, through attention,
    and the loss would still fall. So check the expert adapters specifically.
    """
    from learning_llamas.adapter import _read

    adapter = tmp_path / "a.gguf"
    create_zero_adapter(tiny_moe_f32, adapter, r=RANK, seed=3)

    model = load_model(
        tiny_moe_f32, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True, n_seq_max=N_SEQ_MAX
    )
    handle = model.attach_adapter(adapter, scale=1.0)

    # B starts at exactly zero, by construction, which makes "did it move" an exact question.
    targets = enumerate_targets(tiny_moe_f32)
    expert_idx = [i for i, t in enumerate(targets) if t.name.endswith("_exps.weight")]
    assert expert_idx

    train_sft(libs, model, _samples(), SFTConfig(lr=1e-2, seq_len=SEQ_LEN, pad_id=0, epochs=6))

    moved = 0
    for i in expert_idx:
        b = _read(libs, handle, i, is_b=True)
        if abs(b).max() > 0.0:
            moved += 1

    assert moved == len(expert_idx), (
        f"only {moved}/{len(expert_idx)} expert LoRA B tensors moved — "
        "the weight gradient is not reaching the expert stacks"
    )
