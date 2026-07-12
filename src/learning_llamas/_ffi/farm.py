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
        ],
        ctypes.c_int32,
    ),
    Symbol(Library.FARM, "ll_opt_free", [ctypes.c_void_p], ctypes.c_int32),
    Symbol(Library.FARM, "ll_opt_n_params", [ctypes.c_void_p], ctypes.c_int32),
    # S1-02: one training step with a per-token weighted (maskable) cross-entropy loss.
    Symbol(
        Library.FARM,
        "ll_train_step",
        [
            ctypes.c_void_p,  # llama_context *
            ctypes.POINTER(ctypes.c_int32),  # tokens
            ctypes.POINTER(ctypes.c_int32),  # targets
            ctypes.POINTER(ctypes.c_float),  # weights (0 masks a token out)
            ctypes.c_int32,  # n_tokens
            ctypes.c_bool,  # train
            ctypes.POINTER(ctypes.c_float),  # loss_out
        ],
        ctypes.c_int32,
    ),
]
