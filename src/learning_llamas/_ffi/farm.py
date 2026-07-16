"""ctypes mirror of ``csrc/farm_api.h`` — the learning-llamas shim's own flat C ABI."""

from __future__ import annotations

import ctypes
from enum import IntEnum

from .registry import Library, Symbol


class LLError(IntEnum):
    """Negative return codes from the shim (``farm_api.h``)."""

    OK = 0
    INVALID_ARG = -1
    NO_ADAPTERS = -2
    ALREADY_INIT = -3
    TENSOR_NOT_F32 = -4
    TENSOR_NOT_LEAF = -5
    NOT_INITIALIZED = -6
    BASE_BUFT_NO_BACKWARD = -7
    STEP_FAILED = -8
    SHAPE_MISMATCH = -9
    SCHED_INVALIDATED = -10


class ll_grpo_inputs(ctypes.Structure):  # noqa: N801 — mirrors the C name
    """``struct ll_grpo_inputs`` (farm_api.h) — everything GRPO needs beyond a batch.

    Every array is ``[n_tokens]`` and is indexed exactly like ``weights``. All of them are
    **constants**: ggml only propagates a gradient from tensors flagged PARAM or LOSS, so a named
    input never gets an accumulator. GRPO's gradient is the policy's alone, which is what makes the
    importance ratio an off-policy correction rather than a second model to train.

    The buffers are borrowed, not copied. Keep the numpy arrays they point at alive across the
    call — ``arr.ctypes.data_as(...)`` on a temporary is a use-after-free.
    """

    _fields_ = [
        ("adv", ctypes.POINTER(ctypes.c_float)),
        ("logp_old", ctypes.POINTER(ctypes.c_float)),
        ("logp_ref", ctypes.POINTER(ctypes.c_float)),
        ("kl_w", ctypes.POINTER(ctypes.c_float)),
        ("clip_eps", ctypes.c_float),
    ]


class ll_opt_params(ctypes.Structure):  # noqa: N801 — mirrors the C name
    """``struct ll_opt_params`` (farm_api.h) — AdamW hyperparameters.

    **Owned by Python and read afresh on every optimizer step.** Mutating a field between steps
    is how a learning-rate schedule works here: no callback, no GIL, no struct-return marshalling
    (see :class:`~learning_llamas._ffi.ggml_opt.OptimizerParams` for why that matters).

    The caller must keep it alive for the whole training run — the shim stores a pointer to it.
    """

    _fields_ = [
        ("alpha", ctypes.c_float),  # learning rate
        ("beta1", ctypes.c_float),
        ("beta2", ctypes.c_float),
        ("eps", ctypes.c_float),
        ("wd", ctypes.c_float),  # weight decay; 0 to disable
    ]


# Every ll_opt_params struct the shim is holding a pointer to, keyed by its llama_context.
#
# The shim stores a RAW POINTER to the caller's struct and re-reads alpha/beta1/beta2/eps/wd off it
# on every single step. That is the whole design: a learning-rate schedule is `params.alpha = ...`
# between steps, with no callback into Python and no GIL round-trip on the hot path.
#
# The cost is a lifetime the C side cannot see. If Python frees the struct, the shim goes on reading
# it, and the hyperparameters become whatever now occupies that heap: training silently produces
# NaN, or an alpha of zero, or it aborts on `GGML_ASSERT(alpha > 0)`, or it segfaults -- and which
# one you get depends on the heap layout, so it is nondeterministic across runs.
#
# And it is *far* too easy to do by accident. This is enough:
#
#     h, _ = make_trainer(...)    # `_` is the params struct
#     for _ in range(16):         # ...and now it is not: the loop rebound `_`, Python freed it
#         h.step()
#
# So the binding layer keeps the reference itself and the lifetime stops being the caller's problem.
# Nothing outside this module should call ll_opt_init_lora / ll_opt_free directly.
_LIVE_PARAMS: dict[int, ll_opt_params] = {}


def opt_init_lora(
    libs,
    ctx: int,
    model: int,
    adapters: list[int],
    params: ll_opt_params,
    opt_period: int = 1,
    grad_clip: float = 0.0,
) -> int:
    """Flag an adapter's A/B tensors as the only trainable parameters.

    Args:
        libs: The loaded native libraries.
        ctx: A ``llama_context *``.
        model: The ``llama_model *`` the adapters were loaded against.
        adapters: One or more ``llama_adapter_lora *``.
        params: The AdamW hyperparameters. **Kept alive by this module** for as long as ``ctx`` is
            initialized, because the shim holds a pointer to it and reads it on every step. Mutate
            its fields between steps to schedule the learning rate.
        opt_period: How many ``ll_train_step`` calls make one optimizer step. 1 steps every call;
            anything greater accumulates gradients across that many calls and steps on the last.
        grad_clip: Clip the gradients to this global norm before every optimizer step. 0 disables
            it. Unlike the learning rate this is fixed at init, because it is structural: it is
            nodes in the graph, not a number the optimizer reads.

    Returns:
        The number of tensors flagged — two per adapted base tensor.

    Raises:
        RuntimeError: If the shim rejects the adapters (see :class:`LLError`).
    """
    arr = (ctypes.c_void_p * len(adapters))(*adapters)

    n_flagged = check(
        libs.farm.ll_opt_init_lora(
            ctx, model, arr, len(adapters), ctypes.byref(params), opt_period, grad_clip
        ),
        "ll_opt_init_lora",
    )

    _LIVE_PARAMS[int(ctx)] = params

    return n_flagged


def opt_init_full(
    libs,
    ctx: int,
    model: int,
    params: ll_opt_params,
    opt_period: int = 1,
    grad_clip: float = 0.0,
) -> int:
    """Flag the base model's own weights as the trainable parameters (S1-48, full fine-tuning).

    The mirror image of :func:`opt_init_lora`: no adapter, the base weights themselves train, and
    the model must have been loaded with ``full_finetune=True`` (mmap off) so the AdamW step can
    write them back in place.

    Args:
        libs: The loaded native libraries.
        ctx: A ``llama_context *``.
        model: The ``llama_model *`` whose weights will be trained.
        params: The AdamW hyperparameters. **Kept alive by this module** for as long as ``ctx`` is
            initialized, exactly as for :func:`opt_init_lora` — the shim holds a pointer and reads
            it every step.
        opt_period: How many ``ll_train_step`` calls make one optimizer step.
        grad_clip: Clip the gradients to this global norm before every optimizer step. 0 disables.

    Returns:
        The number of base tensors flagged.

    Raises:
        RuntimeError: If the shim rejects the call (see :class:`LLError`).
    """
    n_flagged = check(
        libs.farm.ll_opt_init_full(ctx, model, ctypes.byref(params), opt_period, grad_clip),
        "ll_opt_init_full",
    )

    _LIVE_PARAMS[int(ctx)] = params

    return n_flagged


def opt_free(libs, ctx: int) -> None:
    """Release the shim's training state for ``ctx``, and the params struct it was reading.

    Args:
        libs: The loaded native libraries.
        ctx: The ``llama_context *`` passed to :func:`opt_init_lora` or :func:`opt_init_full`.
    """
    libs.farm.ll_opt_free(ctx)
    _LIVE_PARAMS.pop(int(ctx), None)


def check(result: int, what: str) -> int:
    """Raise if a shim call returned a negative error code.

    Args:
        result: The shim's return value.
        what: The call being made, for the message.

    Returns:
        ``result``, unchanged, when it is not an error.

    Raises:
        RuntimeError: If ``result`` is negative.
    """
    if result >= 0:
        return result

    try:
        name = LLError(result).name
    except ValueError:
        name = "UNKNOWN"

    raise RuntimeError(f"{what} failed: {name} ({result})")


SYMBOLS = [
    Symbol(Library.FARM, "ll_version", [], ctypes.c_char_p),
    Symbol(Library.FARM, "ll_probe", [], ctypes.c_char_p),
    # S1-01: flag the adapter's A/B tensors as the only trainable parameters.
    Symbol(
        Library.FARM,
        "ll_opt_init_lora",
        [
            ctypes.c_void_p,  # llama_context *
            ctypes.c_void_p,  # llama_model *
            ctypes.POINTER(ctypes.c_void_p),  # llama_adapter_lora **
            ctypes.c_size_t,  # n_adapters
            ctypes.POINTER(ll_opt_params),
            ctypes.c_int32,  # opt_period
            ctypes.c_float,  # grad_clip (0 disables)
        ],
        ctypes.c_int32,
    ),
    # S1-48: flag the base model's own weights as the trainable parameters (full fine-tuning).
    Symbol(
        Library.FARM,
        "ll_opt_init_full",
        [
            ctypes.c_void_p,  # llama_context *
            ctypes.c_void_p,  # llama_model *
            ctypes.POINTER(ll_opt_params),
            ctypes.c_int32,  # opt_period
            ctypes.c_float,  # grad_clip (0 disables)
        ],
        ctypes.c_int32,
    ),
    Symbol(Library.FARM, "ll_opt_free", [ctypes.c_void_p], ctypes.c_int32),
    # S1-03 debug accessors. Outside any API-stability promise; superseded by S1-08.
    Symbol(
        Library.FARM,
        "ll_debug_n_elements",
        [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool],
        ctypes.c_int64,
    ),
    Symbol(
        Library.FARM,
        "ll_debug_get_tensor",
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_bool,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
        ],
        ctypes.c_int64,
    ),
    Symbol(
        Library.FARM,
        "ll_debug_set_tensor",
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_bool,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
        ],
        ctypes.c_int64,
    ),
    Symbol(
        Library.FARM,
        "ll_debug_grad",
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_bool,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
        ],
        ctypes.c_int64,
    ),
    # S1-48: the full-finetune debug accessors -- a flagged BASE weight by its exact ggml name.
    Symbol(
        Library.FARM,
        "ll_debug_base_n_elements",
        [ctypes.c_void_p, ctypes.c_char_p],
        ctypes.c_int64,
    ),
    Symbol(
        Library.FARM,
        "ll_debug_base_grad",
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
        ],
        ctypes.c_int64,
    ),
    Symbol(Library.FARM, "ll_opt_n_params", [ctypes.c_void_p], ctypes.c_int32),
    # S1-17: the activation high-water mark -- the number checkpointing exists to move.
    Symbol(Library.FARM, "ll_compute_buffer_bytes", [ctypes.c_void_p], ctypes.c_int64),
    # S1-10 item 3: the last step's global grad norm, pre and post clip, written into two floats.
    Symbol(
        Library.FARM,
        "ll_grad_norms",
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)],
        ctypes.c_int32,
    ),
    # S1-16: one GRPO step. `weights` is the completion mask and must be exactly 0.0 or 1.0 --
    # ce_sparse multiplies logp_new by it, so a fractional weight corrupts the importance ratio
    # rather than down-weighting the token. The shim rejects it.
    Symbol(
        Library.FARM,
        "ll_train_step_grpo",
        [
            ctypes.c_void_p,  # ctx
            ctypes.POINTER(ctypes.c_int32),  # tokens
            ctypes.POINTER(ctypes.c_int32),  # targets
            ctypes.POINTER(ctypes.c_float),  # weights (the mask)
            ctypes.POINTER(ctypes.c_int32),  # seq_ids
            ctypes.POINTER(ctypes.c_int32),  # positions
            ctypes.c_int32,  # n_tokens
            ctypes.POINTER(ll_grpo_inputs),
            ctypes.c_bool,  # train
            ctypes.POINTER(ctypes.c_float),  # loss_out
        ],
        ctypes.c_int32,
    ),
    # S1-17: layers per gradient-checkpointing segment; 0 = off. Before the first step, or never.
    Symbol(
        Library.FARM,
        "ll_set_grad_checkpointing",
        [ctypes.c_void_p, ctypes.c_int32],
        ctypes.c_int32,
    ),
    # S1-24: chunked attention — the kernel-free long-context path. Pairs with
    # ll_set_grad_checkpointing, and needs it to shrink the backward.
    Symbol(
        Library.FARM,
        "ll_set_chunked_attention",
        [ctypes.c_void_p, ctypes.c_int32],
        ctypes.c_int32,
    ),
    # S1-14: DPO — the pairwise objective, and the reference log-ratio it needs.
    Symbol(
        Library.FARM,
        "ll_train_step_dpo",
        [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),  # tokens
            ctypes.POINTER(ctypes.c_int32),  # targets
            ctypes.POINTER(ctypes.c_float),  # weights: +1 chosen, -1 rejected, 0 elsewhere
            ctypes.POINTER(ctypes.c_int32),  # seq_ids
            ctypes.POINTER(ctypes.c_int32),  # positions
            ctypes.c_int32,  # n_tokens
            ctypes.c_float,  # beta
            ctypes.c_float,  # ref_delta
            ctypes.c_bool,  # train
            ctypes.POINTER(ctypes.c_float),  # loss_out
        ],
        ctypes.c_int32,
    ),
    Symbol(
        Library.FARM,
        "ll_logp_delta",
        [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float),
        ],
        ctypes.c_int32,
    ),
    # S1-11: will this model train, and if not, what stops it?
    #
    # The entry struct is mirrored in preflight.py rather than here, because it is the only ctypes
    # struct in the shim's ABI that a *caller* constructs rather than passes through, and it belongs
    # next to the code that reads it.
    Symbol(
        Library.FARM,
        "ll_preflight_walk",
        [
            ctypes.c_void_p,  # ggml_cgraph *
            ctypes.POINTER(ctypes.c_void_p),  # ggml_tensor ** params
            ctypes.c_int32,  # n_params
            ctypes.c_void_p,  # ll_preflight_entry * out
            ctypes.c_int32,  # max_entries
            ctypes.POINTER(ctypes.c_int32),  # n_blocked
        ],
        ctypes.c_int32,
    ),
    Symbol(
        Library.FARM,
        "ll_preflight",
        [
            ctypes.c_void_p,  # llama_context *
            ctypes.POINTER(ctypes.c_int32),  # tokens
            ctypes.c_int32,  # n_tokens
            ctypes.c_void_p,  # ll_preflight_entry * out
            ctypes.c_int32,  # max_entries
            ctypes.POINTER(ctypes.c_int32),  # n_blocked
        ],
        ctypes.c_int32,
    ),
    # S1-09: the AdamW moments and the iteration counter — what a resume needs.
    Symbol(Library.FARM, "ll_opt_state_count", [ctypes.c_void_p], ctypes.c_int32),
    Symbol(
        Library.FARM,
        "ll_opt_state_info",
        [
            ctypes.c_void_p,
            ctypes.c_int32,  # index
            ctypes.c_char_p,  # name_out
            ctypes.c_int32,  # name_capacity
            ctypes.POINTER(ctypes.c_bool),  # is_v
            ctypes.POINTER(ctypes.c_int64),  # n_elements
        ],
        ctypes.c_int32,
    ),
    Symbol(
        Library.FARM,
        "ll_opt_state_get",
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_bool,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
        ],
        ctypes.c_int64,
    ),
    Symbol(
        Library.FARM,
        "ll_opt_state_set",
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_bool,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
        ],
        ctypes.c_int64,
    ),
    Symbol(Library.FARM, "ll_opt_get_iter", [ctypes.c_void_p], ctypes.c_int64),
    Symbol(Library.FARM, "ll_opt_set_iter", [ctypes.c_void_p, ctypes.c_int64], ctypes.c_int32),
    # S1-08: read the trained adapter tensors back out, so they can be saved.
    Symbol(Library.FARM, "ll_adapter_n_tensors", [ctypes.c_void_p], ctypes.c_int32),
    Symbol(
        Library.FARM,
        "ll_adapter_tensor_info",
        [
            ctypes.c_void_p,  # llama_adapter_lora *
            ctypes.c_int32,  # index
            ctypes.c_char_p,  # name_out
            ctypes.c_int32,  # name_capacity
            ctypes.POINTER(ctypes.c_int64),  # ne_a[4]
            ctypes.POINTER(ctypes.c_int64),  # ne_b[4]
        ],
        ctypes.c_int32,
    ),
    Symbol(
        Library.FARM,
        "ll_adapter_set",
        [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_bool,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
        ],
        ctypes.c_int64,
    ),
    Symbol(
        Library.FARM,
        "ll_adapter_get",
        [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_bool,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
        ],
        ctypes.c_int64,
    ),
    # S1-02: one training step with a per-token weighted (maskable) cross-entropy loss.
    Symbol(
        Library.FARM,
        "ll_train_step",
        [
            ctypes.c_void_p,  # llama_context *
            ctypes.POINTER(ctypes.c_int32),  # tokens
            ctypes.POINTER(ctypes.c_int32),  # targets
            ctypes.POINTER(ctypes.c_float),  # weights (0 masks a token out)
            ctypes.POINTER(ctypes.c_int32),  # seq_ids, or NULL for "all one sequence"
            ctypes.POINTER(ctypes.c_int32),  # positions, or NULL for 0..n-1
            ctypes.c_int32,  # n_tokens
            ctypes.c_bool,  # train
            ctypes.POINTER(ctypes.c_float),  # loss_out
        ],
        ctypes.c_int32,
    ),
]
