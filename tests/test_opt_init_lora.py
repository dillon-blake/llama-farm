"""S1-01: only the adapter's A/B tensors are trainable — the base model stays frozen.

The headline test here is ``test_lora_trains_a_memory_mapped_quantized_base``, and it is worth
saying why it proves what it proves.

Full fine-tuning **cannot** memory-map the base model: the AdamW step writes updated weights back
in place, and mmap maps them read-only, so the process dies with SIGSEGV inside
``ggml_compute_forward_opt_step_adamw`` (llama.cpp's own finetune example disables mmap for
exactly this reason, and S1-00's test hit it). That failure is not graceful and it is not
optional — the kernel refuses the write.

Which makes mmap a **free oracle for "did anything touch a base weight?"** If LoRA training runs
to completion on a mmap'd base, then nothing wrote to a base weight, because the operating system
would not have allowed it. No instrumentation, no golden values, no tolerance: the memory
protection bits are the assertion.

That is the BLUEPRINT D2 claim — base stays quantized and mmap'd — and this test is the proof.
"""

import ctypes

import pytest

from learning_llamas import _ffi
from learning_llamas.adapter import create_zero_adapter, enumerate_targets

N_CTX = 64
N_UBATCH = 32
RANK = 8


def _default_opt_params() -> _ffi.ll_opt_params:
    return _ffi.ll_opt_params(alpha=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0)


def _init_lora(libs: _ffi.Libraries, model, params: _ffi.ll_opt_params) -> int:
    """Call ll_opt_init_lora with the model's single attached adapter."""
    adapters = (ctypes.c_void_p * 1)(model.adapter)
    return libs.farm.ll_opt_init_lora(model.ctx, model.model, adapters, 1, ctypes.byref(params))


@pytest.fixture
def adapted(tiny_model, tmp_path, load_model, libs: _ffi.Libraries):
    """A model with a zero-init adapter attached — mmap left ON, deliberately."""
    adapter_path = tmp_path / "adapter.gguf"
    targets = create_zero_adapter(tiny_model, adapter_path, r=RANK)

    # training=True disables extra buffer types (the backward needs OUT_PROD, which repacked
    # bufts do not implement) but leaves use_mmap ON. That is the point: see the module docstring.
    model = load_model(tiny_model, n_ctx=N_CTX, n_ubatch=N_UBATCH, training=True)
    model.attach_adapter(adapter_path, scale=1.0)

    yield model, targets

    libs.farm.ll_opt_free(model.ctx)


def test_flags_exactly_two_tensors_per_adapted_base_tensor(
    adapted, libs: _ffi.Libraries, tiny_model
) -> None:
    """Every adapted base tensor contributes exactly its A and its B — nothing else."""
    model, targets = adapted
    params = _default_opt_params()

    n_flagged = _ffi.check(_init_lora(libs, model, params), "ll_opt_init_lora")

    assert n_flagged == 2 * len(targets)
    assert libs.farm.ll_opt_n_params(model.ctx) == n_flagged

    # And it is the default preset, not everything in the model.
    assert len(targets) == len(enumerate_targets(tiny_model))


def test_lora_trains_a_memory_mapped_quantized_base(
    adapted, libs: _ffi.Libraries, tmp_path
) -> None:
    """The BLUEPRINT D2 claim, proved by the memory-protection bits.

    A base weight write would SIGSEGV against the read-only mapping. Completing a training run
    therefore proves no base weight was written — and it runs on Q4_K and Q8_0 too, which is the
    other half of the claim: the base stays *quantized*.
    """
    model, _ = adapted
    params = _default_opt_params()
    _ffi.check(_init_lora(libs, model, params), "ll_opt_init_lora")

    n_ctx = libs.llama.llama_n_ctx(model.ctx)
    dataset, ndata = _make_dataset(libs, _corpus(n_ctx * 8), n_ctx)

    result = libs.ggml_base.ggml_opt_result_init()
    try:
        losses = []
        for _ in range(3):
            libs.ggml_base.ggml_opt_result_reset(result)
            # If anything here wrote a base weight, the process would already be dead.
            libs.llama.llama_opt_epoch(model.ctx, dataset, result, None, ndata, None, None)
            loss = ctypes.c_double()
            libs.ggml_base.ggml_opt_result_loss(result, ctypes.byref(loss), None)
            losses.append(loss.value)
    finally:
        libs.ggml_base.ggml_opt_result_free(result)
        libs.ggml_base.ggml_opt_dataset_free(dataset)

    assert all(x == x for x in losses), f"loss went NaN: {losses}"  # noqa: PLR0124
    assert losses[-1] < losses[0], f"the adapter did not learn: {losses}"


def test_learning_rate_is_a_plain_attribute_assignment(adapted, libs: _ffi.Libraries) -> None:
    """The shim holds a pointer to the caller's struct, so a schedule needs no callback."""
    model, _ = adapted
    params = _default_opt_params()
    _ffi.check(_init_lora(libs, model, params), "ll_opt_init_lora")

    # Nothing to call, nothing to re-register: the next step reads the new value.
    params.alpha = 5e-4
    assert params.alpha == pytest.approx(5e-4)


def test_double_init_is_rejected(adapted, libs: _ffi.Libraries) -> None:
    """Flagging twice would double-count parameters and silently corrupt the optimizer state."""
    model, _ = adapted
    params = _default_opt_params()

    _ffi.check(_init_lora(libs, model, params), "ll_opt_init_lora")

    assert _init_lora(libs, model, params) == _ffi.LLError.ALREADY_INIT
    with pytest.raises(RuntimeError, match="ALREADY_INIT"):
        _ffi.check(_init_lora(libs, model, params), "ll_opt_init_lora")


def test_zero_adapters_is_rejected(adapted, libs: _ffi.Libraries) -> None:
    """Training with no adapter would train nothing at all, silently."""
    model, _ = adapted
    params = _default_opt_params()
    adapters = (ctypes.c_void_p * 1)(model.adapter)

    result = libs.farm.ll_opt_init_lora(model.ctx, model.model, adapters, 0, ctypes.byref(params))
    assert result == _ffi.LLError.NO_ADAPTERS


def test_null_arguments_are_rejected(adapted, libs: _ffi.Libraries) -> None:
    model, _ = adapted
    params = _default_opt_params()
    adapters = (ctypes.c_void_p * 1)(model.adapter)

    assert (
        libs.farm.ll_opt_init_lora(None, model.model, adapters, 1, ctypes.byref(params))
        == _ffi.LLError.INVALID_ARG
    )
    assert (
        libs.farm.ll_opt_init_lora(model.ctx, model.model, adapters, 1, None)
        == _ffi.LLError.INVALID_ARG
    )


def test_n_params_before_init_reports_not_initialized(
    tiny_f32, load_model, libs: _ffi.Libraries
) -> None:
    model = load_model(tiny_f32, n_ctx=N_CTX, n_ubatch=N_UBATCH, training=True)
    assert libs.farm.ll_opt_n_params(model.ctx) == _ffi.LLError.NOT_INITIALIZED


def test_repacked_base_is_rejected_at_init_not_aborted_mid_training(
    tiny_q4_k, tmp_path, load_model, libs: _ffi.Libraries
) -> None:
    """A repacked base tensor cannot be differentiated — say so, rather than abort later.

    Loaded with extra buffer types ON (the default), Q4_K weights get repacked to `q4_K_8x8`.
    `supports_op` then returns early for any op whose src is in that buffer, delegating to a
    handler that implements MUL_MAT and **not OUT_PROD** — which is exactly what the backward
    pass needs (`ggml.c:6594-6630`). The gradient node becomes unschedulable and
    `ggml_backend_sched` aborts with:

        ggml-backend.cpp:1242: GGML_ASSERT(*cur_backend_id != -1) failed

    naming neither the op nor the tensor. Worse, it bites Q4_K and **not** Q8_0, purely because a
    q4_K repack variant exists — so it looks like "Q4_K training is broken" for no visible reason.

    The shim asks the question up front — it builds the exact OUT_PROD node the backward would
    build and asks the owning device whether it can run it — and returns an error that names the
    tensor, the buffer type, and the fix.
    """
    adapter_path = tmp_path / "adapter.gguf"
    create_zero_adapter(tiny_q4_k, adapter_path, r=RANK)

    # NOT training=True: extra buffer types stay on, so Q4_K repacks.
    model = load_model(tiny_q4_k, n_ctx=N_CTX, n_ubatch=N_UBATCH)
    model.attach_adapter(adapter_path, scale=1.0)

    params = _default_opt_params()
    result = _init_lora(libs, model, params)

    assert result == _ffi.LLError.BASE_BUFT_NO_BACKWARD
    with pytest.raises(RuntimeError, match="BASE_BUFT_NO_BACKWARD"):
        _ffi.check(result, "ll_opt_init_lora")


def test_opt_free_is_idempotent(adapted, libs: _ffi.Libraries) -> None:
    model, _ = adapted
    params = _default_opt_params()
    _ffi.check(_init_lora(libs, model, params), "ll_opt_init_lora")

    assert libs.farm.ll_opt_free(model.ctx) == _ffi.LLError.OK
    assert libs.farm.ll_opt_free(model.ctx) == _ffi.LLError.OK
    assert libs.farm.ll_opt_n_params(model.ctx) == _ffi.LLError.NOT_INITIALIZED


# --- dataset helpers (shared shape with test_training_graph) -------------------------------


def _corpus(n: int) -> list[int]:
    pattern = [7, 11, 13, 17, 19, 23, 29, 31]
    return [pattern[i % len(pattern)] for i in range(n)]


def _make_dataset(libs: _ffi.Libraries, tokens: list[int], n_ctx: int):
    stride = n_ctx // 2
    ndata = (len(tokens) - n_ctx - 1) // stride
    assert ndata > 0

    dataset = libs.ggml_base.ggml_opt_dataset_init(
        int(_ffi.GGMLType.I32), int(_ffi.GGMLType.I32), n_ctx, n_ctx, ndata, 1
    )
    data_ptr = libs.ggml_base.ggml_get_data(libs.ggml_base.ggml_opt_dataset_data(dataset))
    label_ptr = libs.ggml_base.ggml_get_data(libs.ggml_base.ggml_opt_dataset_labels(dataset))

    buf_type = ctypes.c_int32 * (ndata * n_ctx)
    data = buf_type.from_address(data_ptr)
    labels = buf_type.from_address(label_ptr)

    for idata in range(ndata):
        start = idata * stride
        for i in range(n_ctx):
            data[idata * n_ctx + i] = tokens[start + i]
            labels[idata * n_ctx + i] = tokens[start + i + 1]

    return dataset, ndata
