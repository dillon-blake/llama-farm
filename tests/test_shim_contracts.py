"""S1-50: the shim's stated contracts, on the paths where breaking one is silent.

Every case here comes from the 2026-07-22 audit of ``csrc/farm_train.cpp``. They share a shape:
the shim promises something in ``farm_api.h``, the promise is load-bearing, and the failure when it
is broken is a wrong number or a segfault rather than an error — which is exactly the class of bug
a contract test is for.

* the masked-mean denominator counts only the tokens that count, *including* after the shim itself
  force-masks a position whose target is out of vocabulary range;
* ``ll_logp_delta`` refuses packing without ``kv_unified``, as ``ll_train_step`` already did — the
  two must agree, because DPO uses them on the same batch;
* a null argument is ``LL_ERR_INVALID_ARG``, not a crash, on every entry point that reads one;
* ``ll_opt_free`` un-flags what it flagged, so a later, smaller parameter set really is smaller;
* two adapters whose parameter tensors would share a name are refused, because every
  name-addressed part of the ABI would alias them without saying so.
"""

import ctypes

import pytest

from learning_llamas import _ffi
from learning_llamas.adapter import create_zero_adapter

from .fixtures import gen_tiny_llama

N_CTX = 64
N_UBATCH = 32
RANK = 4

# The fixture's vocabulary. Any target outside [0, N_VOCAB) is what the shim force-masks.
N_VOCAB = gen_tiny_llama.HPARAMS.n_vocab


def _arr_i32(values) -> ctypes.Array:
    return (ctypes.c_int32 * len(values))(*[int(v) for v in values])


def _arr_f32(values) -> ctypes.Array:
    return (ctypes.c_float * len(values))(*[float(v) for v in values])


def _step(libs, model, tokens, targets, weights, *, train=False) -> float:
    loss = ctypes.c_float()
    _ffi.check(
        libs.farm.ll_train_step(
            model.ctx,
            _arr_i32(tokens),
            _arr_i32(targets),
            _arr_f32(weights),
            None,
            None,
            len(tokens),
            train,
            ctypes.byref(loss),
        ),
        "ll_train_step",
    )
    return float(loss.value)


@pytest.fixture
def prepared(tiny_f32, tmp_path, load_model, libs: _ffi.Libraries):
    """A LoRA-initialized context ready to take steps, torn down on the way out."""
    adapter_path = tmp_path / "adapter.gguf"
    create_zero_adapter(tiny_f32, adapter_path, r=RANK, seed=7)

    model = load_model(tiny_f32, n_ctx=N_CTX, n_ubatch=N_UBATCH, training=True)
    model.attach_adapter(adapter_path, scale=1.0)

    params = _ffi.ll_opt_params(alpha=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0)
    _ffi.opt_init_lora(libs, model.ctx, model.model, [model.adapter], params)

    yield model

    _ffi.opt_free(libs, model.ctx)


# ---------------------------------------------------------------------------
# A5: the denominator counts the tokens that count.
# ---------------------------------------------------------------------------


def test_a_force_masked_target_leaves_the_denominator(prepared, libs: _ffi.Libraries) -> None:
    """An out-of-range target is masked out — so it must also leave ``sum(w)``.

    ``upload_masked_ce`` drives the weight of any position whose target is outside
    ``[0, n_vocab)`` to zero rather than let it index off the end of a logits row. That position
    then contributes nothing to the numerator. If it is still counted in the denominator, *every
    other* token's loss and gradient is scaled by ``n_valid / n_total`` — a silent, data-dependent
    shrink of the learning rate that moves whenever the batch's mix of bad targets does.

    The oracle is the same batch with those positions honestly masked instead: identical valid
    terms, identical denominator, so identical loss. The two differ by 32/29 without the fix, which
    is four orders of magnitude outside this tolerance.
    """
    n = 32
    bad = (3, 9, 20)

    tokens = [7, 11, 13, 17] * (n // 4)
    targets = tokens[1:] + [tokens[0]]

    # Out of range, and told to count: the shim must mask it AND stop counting it.
    forced = list(targets)
    for j in bad:
        forced[j] = N_VOCAB + 100
    loss_forced = _step(libs, prepared, tokens, forced, [1.0] * n)

    # The same positions, masked the honest way.
    masked = [1.0] * n
    for j in bad:
        masked[j] = 0.0
    loss_masked = _step(libs, prepared, tokens, targets, masked)

    assert loss_masked > 0.0, "the control batch has no loss; nothing is being compared"
    assert loss_forced == pytest.approx(loss_masked, rel=1e-6), (
        f"a force-masked target is still in the denominator: {loss_forced:.6f} vs the same batch "
        f"masked explicitly, {loss_masked:.6f} (ratio {loss_forced / loss_masked:.4f}; "
        f"{n - len(bad)}/{n} = {(n - len(bad)) / n:.4f} is what the bug produces)"
    )


def test_an_all_out_of_range_batch_is_zero_loss_not_a_nan(prepared, libs) -> None:
    """Every target invalid zeroes every weight, so ``sum(w)`` is 0 — and 0/0 is not the answer.

    The guard is ``sum_w > 0``, and moving the sum after the force-masking is what puts this batch
    into that branch for the first time: before, the denominator was the caller's 32.0 and the
    numerator was 0, which happened to give 0 as well. It is asserted so that the new branch is
    covered rather than assumed.
    """
    n = 32
    tokens = [7, 11, 13, 17] * (n // 4)

    loss = _step(libs, prepared, tokens, [N_VOCAB + 1] * n, [1.0] * n)

    assert loss == 0.0, f"an entirely invalid batch should contribute nothing, got {loss}"


# ---------------------------------------------------------------------------
# A3: ll_logp_delta and ll_train_step must agree about packing.
# ---------------------------------------------------------------------------


def _packed_logp_delta(libs, model, n: int) -> int:
    """Call ``ll_logp_delta`` with seq_ids, returning its raw status code."""
    half = n // 2
    out = ctypes.c_float()
    return libs.farm.ll_logp_delta(
        model.ctx,
        _arr_i32([7] * n),
        _arr_i32([11] * n),
        _arr_f32([1.0] * half + [-1.0] * half),
        _arr_i32([0] * half + [1] * half),
        _arr_i32(list(range(half)) * 2),
        n,
        ctypes.byref(out),
    )


def test_logp_delta_refuses_packing_without_kv_unified(
    tiny_f32, tmp_path, load_model, libs: _ffi.Libraries
) -> None:
    """The guard ``ll_train_step`` has always had, on the call DPO actually packs with.

    ``ll_logp_delta`` exists to precompute DPO's reference log-ratio, and that batch is *always*
    packed: chosen and rejected are two sequences sharing one forward pass. With ``kv_unified``
    off, llama.cpp's ubatch splitter regroups the batch by sequence while ``upload_masked_ce``
    keeps indexing targets in the caller's order — so the answer is a well-formed log-probability
    of the wrong pairing, baked in as a constant that every later DPO step optimizes against.

    ``ll_train_step`` refuses exactly this configuration. The pair has to agree, or the refusal is
    only half a refusal.
    """
    adapter_path = tmp_path / "adapter.gguf"
    create_zero_adapter(tiny_f32, adapter_path, r=RANK, seed=7)

    split = load_model(
        tiny_f32, n_ctx=N_CTX, n_ubatch=N_UBATCH, training=True, n_seq_max=2, kv_unified=False
    )
    split.attach_adapter(adapter_path, scale=1.0)

    assert _packed_logp_delta(libs, split, 16) == _ffi.LLError.INVALID_ARG

    # ...and the refusal is about the packing, not about this batch or this shim being unable to
    # run at all: the identical call on a kv_unified context succeeds.
    unified = load_model(tiny_f32, n_ctx=N_CTX, n_ubatch=N_UBATCH, training=True, n_seq_max=2)
    unified.attach_adapter(adapter_path, scale=1.0)

    assert _packed_logp_delta(libs, unified, 16) == _ffi.LLError.OK


# ---------------------------------------------------------------------------
# A4: a null argument is an error code, not a segfault.
# ---------------------------------------------------------------------------


def test_grpo_rejects_a_null_weight_array(prepared, libs: _ffi.Libraries) -> None:
    """``weights`` is read by GRPO's own 0-or-1 mask check, before anything else validates it.

    ``train_step_impl`` does reject a null ``weights`` — but only after ``ll_train_step_grpo`` has
    already walked it looking for a fractional entry. farm_api.h promises ``LL_ERR_INVALID_ARG``
    for a null argument; without the guard this dereferences NULL and takes the process with it,
    so a failure here is a crash rather than an assertion.
    """
    n = 32
    adv = _arr_f32([0.0] * n)
    logp_old = _arr_f32([0.0] * n)

    grpo = _ffi.ll_grpo_inputs(
        adv=ctypes.cast(adv, ctypes.POINTER(ctypes.c_float)),
        logp_old=ctypes.cast(logp_old, ctypes.POINTER(ctypes.c_float)),
        logp_ref=None,
        kl_w=None,
        clip_eps=0.2,
    )

    status = libs.farm.ll_train_step_grpo(
        prepared.ctx,
        _arr_i32([7] * n),
        _arr_i32([11] * n),
        None,  # the null under test
        None,
        None,
        n,
        ctypes.byref(grpo),
        True,
        None,
    )

    assert status == _ffi.LLError.INVALID_ARG


# ---------------------------------------------------------------------------
# A7: two adapters must not collide on a parameter name.
# ---------------------------------------------------------------------------


def test_two_adapters_over_the_same_base_tensors_are_refused(
    tiny_f32, tmp_path, load_model, libs: _ffi.Libraries
) -> None:
    """Colliding names would alias every name-addressed structure in the shim, in silence.

    llama.cpp derives an adapter tensor's ggml name from the BASE tensor, not from the adapter —
    so two adapters over the same projection both produce ``blk.0.attn_q.weight.lora_a``. The
    gradient accumulators and the AdamW moments are ``unordered_map<string, ...>``, so the second
    capture overwrites the first; ``ll_debug_grad`` then hands back the other adapter's gradient,
    and ``ll_opt_state_count`` reports (and a checkpoint saves) one adapter's worth of moments
    while claiming to cover the run. Nothing errors and nothing disagrees out loud.

    Refusing is the honest answer rather than re-keying internally, because the ambiguity is in the
    ABI: ``ll_opt_state_get`` and the debug accessors take a parameter NAME, so with a duplicate
    there is no argument that means "the other one".
    """
    adapter_path = tmp_path / "adapter.gguf"
    create_zero_adapter(tiny_f32, adapter_path, r=RANK, seed=7)

    model = load_model(tiny_f32, n_ctx=N_CTX, n_ubatch=N_UBATCH, training=True)

    # The same file twice: two independent adapters whose tensors carry identical names.
    first = model.attach_adapter(adapter_path, scale=1.0)
    second = libs.llama.llama_adapter_lora_init(model.model, str(adapter_path).encode())
    assert second, "the second adapter failed to load"

    try:
        params = _ffi.ll_opt_params(alpha=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0)
        handles = (ctypes.c_void_p * 2)(first, second)

        status = libs.farm.ll_opt_init_lora(
            model.ctx, model.model, handles, 2, ctypes.byref(params), 1, 0.0
        )
        assert status == _ffi.LLError.INVALID_ARG, (
            f"colliding adapters were accepted (returned {status}); their gradients, moments and "
            f"checkpoint entries would silently alias"
        )

        # And the refusal left nothing half-built: the single-adapter init still works.
        _ffi.opt_init_lora(libs, model.ctx, model.model, [first], params)
        _ffi.opt_free(libs, model.ctx)
    finally:
        libs.llama.llama_adapter_lora_free(second)


# ---------------------------------------------------------------------------
# A6: ll_opt_free un-flags what it flagged.
# ---------------------------------------------------------------------------


def _adapter_b_values(libs, adapter: int) -> list[float]:
    """Every B tensor of an adapter, concatenated. Needs no training context."""
    n_pairs = libs.farm.ll_adapter_n_tensors(adapter)
    assert n_pairs > 0, f"ll_adapter_n_tensors returned {n_pairs}"

    values: list[float] = []
    for i in range(n_pairs):
        n = libs.farm.ll_adapter_get(adapter, i, True, None, 0)
        assert n > 0, f"ll_adapter_get sizing call returned {n}"
        buf = (ctypes.c_float * n)()
        assert libs.farm.ll_adapter_get(adapter, i, True, buf, n) == n
        values.extend(buf)

    return values


def test_opt_free_stops_the_optimizer_stepping_a_dropped_adapter(
    tiny_f32, tmp_path, load_model, libs: _ffi.Libraries
) -> None:
    """``ggml_set_param`` only ORs the flag in; nothing in ggml takes it back off.

    So without ``ll_opt_free`` clearing it, a flag outlives the training state that set it. Run
    ``init_lora([q, v]) -> opt_free -> init_lora([q])`` and v's tensors are still PARAM-flagged:
    ggml still promotes them to graph nodes, still allocates them a gradient and AdamW moments, and
    the optimizer still STEPS them — while ``param_tensors``, the ``ll_opt_state_*`` checkpoint and
    every ``ll_debug_*`` accessor enumerate q alone. The run trains a tensor the caller believes it
    detached, no checkpoint carries it, and a resume therefore starts from a different model than
    the one that was saved.

    The two adapters target DISJOINT projections on purpose — overlapping ones are refused outright
    (see the collision test above), and disjointness is also what makes v's movement attributable
    to the stale flag rather than to anything q did.
    """
    q_path = tmp_path / "adapter-q.gguf"
    v_path = tmp_path / "adapter-v.gguf"
    create_zero_adapter(tiny_f32, q_path, r=RANK, seed=7, preset=("attn_q",))
    create_zero_adapter(tiny_f32, v_path, r=RANK, seed=8, preset=("attn_v",))

    model = load_model(tiny_f32, n_ctx=N_CTX, n_ubatch=N_UBATCH, training=True)
    kept = model.attach_adapter(q_path, scale=1.0)
    dropped = libs.llama.llama_adapter_lora_init(model.model, str(v_path).encode())
    assert dropped, "the second adapter failed to load"

    try:
        # Both attached: the dropped one stays in the forward graph throughout, which is what makes
        # this a question about the FLAG rather than about graph membership.
        handles = (ctypes.c_void_p * 2)(kept, dropped)
        scales = (ctypes.c_float * 2)(1.0, 1.0)
        assert libs.llama.llama_set_adapters_lora(model.ctx, handles, 2, scales) == 0

        # Flag both, then let go of both.
        params = _ffi.ll_opt_params(alpha=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0)
        _ffi.opt_init_lora(libs, model.ctx, model.model, [kept, dropped], params)
        _ffi.opt_free(libs, model.ctx)

        before = _adapter_b_values(libs, dropped)
        assert before == [0.0] * len(before), "create_zero_adapter should have written B = 0"

        # Now train only the first one.
        _ffi.opt_init_lora(libs, model.ctx, model.model, [kept], params)
        try:
            n = N_UBATCH
            tokens = [7, 11, 13, 17] * (n // 4)
            _step(libs, model, tokens, tokens[1:] + [tokens[0]], [1.0] * n, train=True)
        finally:
            _ffi.opt_free(libs, model.ctx)

        after = _adapter_b_values(libs, dropped)
        assert after == before, (
            "an adapter dropped from the trainable set was still stepped by the optimizer: its B "
            "tensors moved. GGML_TENSOR_FLAG_PARAM survived ll_opt_free."
        )

        # Non-vacuity: the adapter that IS in the trainable set did move, so a step really happened
        # and "unchanged" is a claim about the flag rather than about a step that never ran.
        trained = _adapter_b_values(libs, kept)
        assert any(x != 0.0 for x in trained), (
            "the trained adapter's B is still zero; no optimizer step took effect and the "
            "assertion above is vacuous"
        )
    finally:
        libs.llama.llama_set_adapters_lora(model.ctx, None, 0, None)
        libs.llama.llama_adapter_lora_free(dropped)
