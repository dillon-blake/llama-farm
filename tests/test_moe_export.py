"""S1-50: a MoE adapter can be SAVED, and merges at the scale llama.cpp would have used.

``tests/test_moe.py`` proves a MoE model trains. It stops there — at enumerate, create, and a
falling loss. Everything downstream of "the loss went down" was untested on MoE, and both halves of
it were broken, in the two ways a 3D expert stack breaks things:

* :func:`~learning_llamas.adapter.save_adapter` reshaped every tensor to two dimensions, so an
  ``A`` of ``ne = [n_in, r, n_expert]`` raised ``ValueError``. A MoE run could not save its adapter
  **after** spending the whole training budget producing one.
* :func:`~learning_llamas.export.merge` read the rank off ``A``'s leading numpy axis, which on a 3D
  stack is ``n_expert``, not ``r``. Every expert delta was folded at ``alpha/n_expert`` instead of
  the ``alpha/rank`` llama.cpp applies to the same file at runtime — with the usual ``alpha == r``,
  a silent constant factor of ``r/n_expert``.

Neither has a symptom before the end of a training run, so both are checked here against things
that are not the code under test: the loader's own shape validation (a saved adapter is reloaded
through stock ``llama_adapter_lora_init``), and a per-expert numpy oracle written out from
``llama-adapter.h:53-55`` rather than from :func:`~learning_llamas.export._delta`.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from learning_llamas import _ffi, create_zero_adapter
from learning_llamas.adapter import save_adapter
from learning_llamas.export import merge
from learning_llamas.quant import dequantize

from .fixtures import gen_tiny_moe
from .test_export import _npshape, _read_ab, _tensor, _write_adapter

N_CTX = 64

PROMPT = [7, 11, 13, 17, 19]

# The rank must NOT equal the fixture's expert count (4), or `alpha/rank` and the buggy
# `alpha/n_expert` are the same number and every scale assertion below passes for free.
RANK = 2

# alpha/rank == 2, so a merge that dropped the alpha/rank factor entirely (effective 1.0) misses
# the oracle too, rather than coinciding with it.
ALPHA = 2.0 * RANK


@pytest.fixture(scope="session")
def tiny_moe_f32(libs: _ffi.Libraries, fixture_cache_dir: pathlib.Path) -> pathlib.Path:
    path, _ = gen_tiny_moe.build("f32", fixture_cache_dir)
    return path


# ---------------------------------------------------------------------------
# Saving: the expert axis has to survive the round trip through the shim.
# ---------------------------------------------------------------------------


def test_a_moe_adapter_can_be_saved_at_all(tiny_moe_f32, tmp_path, load_model, libs) -> None:
    """Before S1-50 this raised ValueError, and only ever at the end of a training run.

    ``ll_adapter_tensor_info`` reports the full GGML_MAX_DIMS ``ne``; ``save_adapter`` reshaped to
    ``(ne[1], ne[0])`` regardless, so an expert stack's ``n_in * r * n_expert`` elements would not
    fit the ``(r, n_in)`` it asked for. numpy raised, the adapter was never written, and the
    compute was gone.
    """
    init = tmp_path / "init.gguf"
    targets = create_zero_adapter(tiny_moe_f32, init, r=RANK, alpha=ALPHA, seed=3)
    assert any(t.n_expert for t in targets), "the fixture produced no 3D targets; nothing to prove"

    model = load_model(tiny_moe_f32, n_ctx=N_CTX)
    model.attach_adapter(init, scale=1.0)

    out = tmp_path / "saved.gguf"
    n = save_adapter(libs, model.adapter, out, architecture="llama", alpha=ALPHA)

    assert n == len(targets)


def test_the_saved_moe_adapter_has_the_shape_the_loader_validates(
    tiny_moe_f32, tmp_path, load_model, libs
) -> None:
    """A wrong-but-writable layout is worse than the crash, so the ne is pinned exactly.

    ``a.ne = [n_in, r, n_expert]`` and ``b.ne = [r, n_out, n_expert]`` — the same convention
    :func:`~learning_llamas.adapter._zero_init_pair` writes and the only one for which
    ``build_lora_mm_id``'s ``mul_mat_id(B, mul_mat_id(A, cur, ids), ids)`` gathers the adapter slice
    belonging to the expert the base gathered. Note that llama.cpp's loader checks ``ne[0]`` and
    ``ne[1]`` only (llama-adapter.cpp:362-367): a 2D A would be *accepted* and then read as a
    one-expert stack, so nothing downstream would ever complain.
    """
    import gguf

    hp = gen_tiny_moe.HPARAMS

    init = tmp_path / "init.gguf"
    create_zero_adapter(tiny_moe_f32, init, r=RANK, alpha=ALPHA, seed=3)

    model = load_model(tiny_moe_f32, n_ctx=N_CTX)
    model.attach_adapter(init, scale=1.0)

    out = tmp_path / "saved.gguf"
    save_adapter(libs, model.adapter, out, architecture="llama", alpha=ALPHA)

    written = gguf.GGUFReader(str(out), "r").tensors
    shapes = {t.name: tuple(int(x) for x in t.shape) for t in written}

    # ffn_gate_exps/ffn_up_exps map n_embd -> n_ff; ffn_down_exps maps n_ff -> n_embd. Both
    # directions are checked so a transposed reshape cannot hide behind the fixture's square FFN.
    assert shapes["blk.0.ffn_gate_exps.weight.lora_a"] == (hp.n_embd, RANK, hp.n_expert)
    assert shapes["blk.0.ffn_gate_exps.weight.lora_b"] == (RANK, hp.n_ff, hp.n_expert)
    assert shapes["blk.0.ffn_down_exps.weight.lora_a"] == (hp.n_ff, RANK, hp.n_expert)
    assert shapes["blk.0.ffn_down_exps.weight.lora_b"] == (RANK, hp.n_embd, hp.n_expert)

    # The attention targets are still 2D — a fix that made everything 3D would be just as wrong.
    assert shapes["blk.0.attn_q.weight.lora_a"] == (hp.n_embd, RANK)


def test_stock_llama_cpp_reloads_the_saved_moe_adapter_unchanged(
    tiny_moe_f32, tmp_path, load_model, libs
) -> None:
    """The layout is judged by the loader, not by us — and the values must be identical.

    ``create_zero_adapter`` gives ``A ~ N(0, sigma)`` and ``B == 0``, so A carries
    ``n_expert * r * n_in`` distinct values: a reshape that permuted the expert axis (or folded it
    into the rank axis) would either be refused by the loader's ``a.ne[0] == n_in`` check or come
    back as the same numbers in the wrong order. Both fail here.
    """
    init = tmp_path / "init.gguf"
    create_zero_adapter(tiny_moe_f32, init, r=RANK, alpha=ALPHA, seed=3)

    model = load_model(tiny_moe_f32, n_ctx=N_CTX)
    model.attach_adapter(init, scale=1.0)

    out = tmp_path / "saved.gguf"
    save_adapter(libs, model.adapter, out, architecture="llama", alpha=ALPHA)

    reloaded = libs.llama.llama_adapter_lora_init(model.model, str(out).encode())
    assert reloaded, "stock llama.cpp refused the MoE adapter we just wrote"

    _, _, written = _read_ab(out)
    _, _, original = _read_ab(init)

    assert set(written) == set(original)
    experts = [n for n in written if n.endswith("_exps.weight")]
    assert experts, "no expert stack in the adapter; this test would prove nothing"

    for name in written:
        a_in, b_in = original[name]
        a_out, b_out = written[name]
        assert a_out.shape == a_in.shape, name
        assert b_out.shape == b_in.shape, name
        # F32 out of the shim, F32 into the file, no arithmetic in between.
        assert np.array_equal(a_out, a_in), f"{name}.lora_a changed on the way out"
        assert np.array_equal(b_out, b_in), f"{name}.lora_b changed on the way out"

    # ...and A really is per-expert distinct, so the equality above is not comparing four copies
    # of the same slice.
    a_expert = written[experts[0]][0]
    assert a_expert.shape[0] == gen_tiny_moe.HPARAMS.n_expert
    assert not np.array_equal(a_expert[0], a_expert[1])


# ---------------------------------------------------------------------------
# Merging: the scale, against an oracle that is not export.py.
# ---------------------------------------------------------------------------


def _nonzero_expert_adapter(base_path, out_path, seed: int = 11):
    """An ``ffn_gate_exps``-only adapter with nonzero A *and* B, plus the exact pairs it holds.

    ``create_zero_adapter`` only ever writes ``B == 0``, which makes every delta zero and every
    scale question unanswerable. So its A is kept, B is filled in, and both are read back out of
    the file — the oracle then reasons about the bytes that are actually there.
    """
    zero = out_path.parent / (out_path.stem + "-zero.gguf")
    create_zero_adapter(base_path, zero, r=RANK, alpha=ALPHA, preset=("ffn_gate_exps",), seed=seed)

    arch, alpha, ab = _read_ab(zero)
    rng = np.random.default_rng(seed + 1)
    pairs = [
        (name, a, (rng.standard_normal(b.shape) * 0.05).astype(np.float32))
        for name, (a, b) in ab.items()
    ]
    _write_adapter(out_path, arch, alpha, pairs)

    _, _, back = _read_ab(out_path)
    return back


def test_a_moe_merge_uses_alpha_over_rank_not_alpha_over_n_expert(
    tiny_moe_f32, tmp_path, libs
) -> None:
    """The scale, folded per expert, checked against numpy — not against ``export._delta``.

    llama.cpp's rule is ``rank = b->ne[0]`` and ``scale = alpha ? user * alpha / rank : user``
    (llama-adapter.h:53-55). ``b->ne[0]`` is B's LAST numpy axis, ``r``, whatever the convention —
    while ``a.shape[0]`` is ``r`` for a dense A and ``n_expert`` for a 3D one. Reading the rank off
    A therefore mis-scales every expert delta by ``r/n_expert`` and nothing says so: the file
    loads, the model runs, it is just a different model from the one that was trained.

    The base is F32, so dequantize/quantize are identities and the comparison is an equality rather
    than a tolerance with somewhere to hide.
    """
    hp = gen_tiny_moe.HPARAMS
    assert RANK != hp.n_expert, "with r == n_expert the two scale rules coincide; this is vacuous"

    adapter_path = tmp_path / "exps.gguf"
    pairs = _nonzero_expert_adapter(tiny_moe_f32, adapter_path)

    merged_path = tmp_path / "merged.gguf"
    touched = merge(tiny_moe_f32, adapter_path, merged_path, libs, scale=1.0)
    assert set(touched) == {f"blk.{il}.ffn_gate_exps.weight" for il in range(hp.n_layer)}

    name = "blk.0.ffn_gate_exps.weight"
    a, b = pairs[name]
    assert a.shape == (hp.n_expert, RANK, hp.n_embd)
    assert b.shape == (hp.n_expert, hp.n_ff, RANK)

    # ---- the oracle: one expert at a time, from llama-adapter.h ----
    effective = 1.0 * ALPHA / RANK  # rank == b.ne[0] == b's last numpy axis
    delta = np.stack([b[e] @ a[e] for e in range(hp.n_expert)])
    assert delta.shape == (hp.n_expert, hp.n_ff, hp.n_embd)

    base_t = _tensor(tiny_moe_f32, name)
    base_vals = dequantize(base_t.data, base_t.tensor_type, _npshape(base_t))
    expected = base_vals + effective * delta

    merged_t = _tensor(merged_path, name)
    assert tuple(int(x) for x in merged_t.shape) == tuple(int(x) for x in base_t.shape), (
        "the merge flattened the expert axis"
    )
    merged_vals = dequantize(merged_t.data, merged_t.tensor_type, _npshape(merged_t))

    tol = 1e-6
    off = float(np.abs(merged_vals - expected).max())
    assert off < tol, f"the merged expert stack is {off:.3g} from the alpha/rank oracle"

    # ---- can-fail: three ways this could have been a test of nothing ----
    # (1) the actual bug. rank = a.shape[0] = n_expert scales every delta by r/n_expert.
    wrong_scale = base_vals + (1.0 * ALPHA / hp.n_expert) * delta
    assert float(np.abs(wrong_scale - expected).max()) > 1e4 * tol, (
        "the two scale rules are indistinguishable on this fixture"
    )
    # (2) a transposed fold. The fixture's expert stacks are square (n_ff == n_embd), so a
    #     transpose FITS and would not raise -- it has to be excluded numerically.
    assert float(np.abs(delta[0] - delta[0].T).max()) > 1e4 * tol, (
        "the delta is nearly symmetric; a transposed fold would pass this oracle"
    )
    # (3) a mixed-up expert axis. If the experts' deltas were interchangeable, folding expert 1's
    #     into expert 0's slice would be invisible.
    assert float(np.abs(delta[0] - delta[1]).max()) > 1e4 * tol, (
        "the experts' deltas are nearly equal; an expert-axis permutation would pass this oracle"
    )


def test_the_merged_moe_model_matches_base_plus_adapter(
    tiny_moe_f32, tmp_path, load_model, libs
) -> None:
    """The arithmetic is not the artefact: the merged GGUF has to load and route and agree.

    base+adapter computes ``mul_mat_id(W, x) + s * mul_mat_id(B, mul_mat_id(A, x, ids), ids)``;
    the merged model computes ``mul_mat_id(W + s * B@A, x, ids)``. Same computation reordered, on
    an F32 base, with the router untouched (``ffn_gate_inp`` is not a LoRA target, so top-k picks
    the same experts) — so the logits must agree to float32 associativity while sitting far from
    the bare base.
    """
    adapter_path = tmp_path / "exps.gguf"
    _nonzero_expert_adapter(tiny_moe_f32, adapter_path)

    merged_path = tmp_path / "merged.gguf"
    merge(tiny_moe_f32, adapter_path, merged_path, libs, scale=1.0)

    bare = load_model(tiny_moe_f32, n_ctx=N_CTX)
    logits_base = np.array(bare.logits(PROMPT))

    with_adapter = load_model(tiny_moe_f32, n_ctx=N_CTX)
    with_adapter.attach_adapter(adapter_path, scale=1.0)
    logits_adapter = np.array(with_adapter.logits(PROMPT))

    merged_model = load_model(merged_path, n_ctx=N_CTX)
    logits_merged = np.array(merged_model.logits(PROMPT))

    moved = float(np.abs(logits_adapter - logits_base).max())
    assert moved > 1e-3, f"the expert adapter barely moves the logits ({moved:.3g})"

    to_adapter = float(np.abs(logits_merged - logits_adapter).max())

    # A ratio, not an absolute bound: what is being claimed is that the residual is round-off and
    # not modelling. The bug this replaces scaled the delta by r/n_expert == 0.5, which lands the
    # merged model roughly `moved` away from base+adapter -- twenty times this bound.
    assert to_adapter < 0.05 * moved, (
        f"the merged MoE model sits {to_adapter:.4g} from base+adapter while the adapter itself "
        f"moves the logits by {moved:.4g}; the delta went in mis-scaled, transposed, or into the "
        f"wrong expert"
    )
    assert int(logits_merged.argmax()) == int(logits_adapter.argmax()), (
        "the merged MoE model and base+adapter disagree on the very next token"
    )
