"""ctypes mirrors of ``include/llama.h``.

Mirrored from llama.cpp at commit ``4f37f519722aa3242eecb7649466b4a4a2d6d6da``. Model,
context, and adapter handles stay opaque (``c_void_p``); only the two parameter structs that
cross the ABI **by value** — ``llama_model_params`` and ``llama_context_params`` — need full
field-by-field layouts, and those are the ones a vendor bump can silently corrupt.

Always construct them through ``llama_model_default_params()`` / ``llama_context_default_params()``
and override individual fields. Zero-initializing them by hand is wrong: several fields have
non-zero defaults (``n_ctx``, ``rope_freq_base``, ``flash_attn_type = AUTO = -1``).
"""

from __future__ import annotations

import ctypes
from enum import IntEnum

from .registry import Library, Symbol

# Opaque handles.
llama_model_p = ctypes.c_void_p
llama_context_p = ctypes.c_void_p
llama_adapter_lora_p = ctypes.c_void_p

llama_token = ctypes.c_int32
llama_pos = ctypes.c_int32
llama_seq_id = ctypes.c_int32


class GGMLType(IntEnum):
    """``enum ggml_type`` (ggml.h). Only the members learning-llamas names are listed.

    The authoritative full table for *file-format* work is ``gguf.GGMLQuantizationType`` in the
    vendored gguf-py, which uses the same numbering.
    """

    F32 = 0
    F16 = 1
    Q8_0 = 8
    Q4_K = 12
    I32 = 26
    BF16 = 30


class SplitMode(IntEnum):
    """``enum llama_split_mode`` (llama.h)."""

    NONE = 0
    LAYER = 1
    ROW = 2
    TENSOR = 3


class AttentionType(IntEnum):
    """``enum llama_attention_type`` (llama.h).

    ``UNSPECIFIED`` defers to the model's own ``hparams.causal_attn``
    (``src/llama-context.cpp:216-220``).
    """

    UNSPECIFIED = -1
    CAUSAL = 0
    NON_CAUSAL = 1


class FlashAttnType(IntEnum):
    """``enum llama_flash_attn_type`` (llama.h)."""

    AUTO = -1
    DISABLED = 0
    ENABLED = 1


class ContextType(IntEnum):
    """``enum llama_context_type`` (llama.h)."""

    DEFAULT = 0
    MTP = 1


class llama_model_params(ctypes.Structure):  # noqa: N801 — mirrors the C name
    """``struct llama_model_params`` (llama.h).

    The booleans are kept together at the end in the C struct "to avoid misalignment during
    copy-by-value"; the mirror must preserve that order exactly.
    """

    _fields_ = [
        ("devices", ctypes.c_void_p),
        ("tensor_buft_overrides", ctypes.c_void_p),
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.POINTER(ctypes.c_float)),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.c_void_p),
        ("vocab_only", ctypes.c_bool),
        ("use_mmap", ctypes.c_bool),
        ("use_direct_io", ctypes.c_bool),
        ("use_mlock", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
        ("use_extra_bufts", ctypes.c_bool),
        ("no_host", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool),
    ]


class llama_context_params(ctypes.Structure):  # noqa: N801 — mirrors the C name
    """``struct llama_context_params`` (llama.h).

    ``type_k`` / ``type_v`` are the KV-cache tensor types. Training forces them to
    :attr:`GGMLType.F32` today, because CPU ``out_prod`` aborts on F16 (S1-18 removes that
    constraint at the source).
    """

    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_rs_seq", ctypes.c_uint32),
        ("n_outputs_max", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("ctx_type", ctypes.c_int),
        ("rope_scaling_type", ctypes.c_int),
        ("pooling_type", ctypes.c_int),
        ("attention_type", ctypes.c_int),
        ("flash_attn_type", ctypes.c_int),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int),
        ("type_v", ctypes.c_int),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("samplers", ctypes.c_void_p),
        ("n_samplers", ctypes.c_size_t),
        ("ctx_other", ctypes.c_void_p),
    ]


# bool (*)(const struct ggml_tensor * tensor, void * userdata) -- llama.h:1564.
# Safe to declare as a CFUNCTYPE: it returns a bool, not a struct.
llama_opt_param_filter = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)


class llama_opt_params(ctypes.Structure):  # noqa: N801 — mirrors the C name
    """``struct llama_opt_params`` (llama.h:1569-1579).

    Mirrored as a reference: learning-llamas forks the training loop (BLUEPRINT D1) rather than
    calling ``llama_opt_epoch``, whose loss is hardcoded and unmaskable. Keeping the mirror
    means the fork can be diffed against the stock path, and the symbol test notices if
    upstream reshapes it.

    ``get_opt_pars`` is ``c_void_p`` for the same reason as in
    :class:`~learning_llamas._ffi.ggml_opt.ggml_opt_params`: its C typedef returns a struct by
    value, and a ctypes callback with that restype is an ABI trap.
    """

    _fields_ = [
        ("n_ctx_train", ctypes.c_uint32),
        ("param_filter", llama_opt_param_filter),
        ("param_filter_ud", ctypes.c_void_p),
        ("get_opt_pars", ctypes.c_void_p),
        ("get_opt_pars_ud", ctypes.c_void_p),
        ("optimizer_type", ctypes.c_int),
    ]


SYMBOLS = [
    # Lifecycle.
    Symbol(Library.LLAMA, "llama_backend_init", []),
    Symbol(Library.LLAMA, "llama_backend_free", []),
    Symbol(Library.LLAMA, "llama_model_default_params", [], llama_model_params),
    Symbol(
        Library.LLAMA,
        "llama_model_load_from_file",
        [ctypes.c_char_p, llama_model_params],
        llama_model_p,
    ),
    Symbol(Library.LLAMA, "llama_model_free", [llama_model_p]),
    Symbol(Library.LLAMA, "llama_context_default_params", [], llama_context_params),
    Symbol(
        Library.LLAMA,
        "llama_init_from_model",
        [llama_model_p, llama_context_params],
        llama_context_p,
    ),
    Symbol(Library.LLAMA, "llama_free", [llama_context_p]),
    # Adapters. llama_set_adapters_lora is the batch attach API present at the pinned commit
    # (llama.h:690); if a vendor bump removes it, the symbol-table test fails loudly, which is
    # the entire point of declaring it here.
    Symbol(
        Library.LLAMA,
        "llama_adapter_lora_init",
        [llama_model_p, ctypes.c_char_p],
        llama_adapter_lora_p,
    ),
    Symbol(Library.LLAMA, "llama_adapter_lora_free", [llama_adapter_lora_p]),
    Symbol(
        Library.LLAMA,
        "llama_set_adapters_lora",
        [
            llama_context_p,
            ctypes.POINTER(llama_adapter_lora_p),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
        ],
        ctypes.c_int32,
    ),
    # Stock training entry points -- mirrored as reference, not used (BLUEPRINT D1 forks them).
    Symbol(
        Library.LLAMA,
        "llama_opt_param_filter_all",
        [ctypes.c_void_p, ctypes.c_void_p],
        ctypes.c_bool,
    ),
    Symbol(Library.LLAMA, "llama_opt_init", [llama_context_p, llama_model_p, llama_opt_params]),
    Symbol(
        Library.LLAMA,
        "llama_opt_epoch",
        [
            llama_context_p,
            ctypes.c_void_p,  # ggml_opt_dataset_t
            ctypes.c_void_p,  # ggml_opt_result_t (train)
            ctypes.c_void_p,  # ggml_opt_result_t (eval)
            ctypes.c_int64,
            ctypes.c_void_p,  # ggml_opt_epoch_callback
            ctypes.c_void_p,
        ],
    ),
]
