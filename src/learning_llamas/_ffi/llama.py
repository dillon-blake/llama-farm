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
llama_vocab_p = ctypes.c_void_p
llama_memory_p = ctypes.c_void_p

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


class llama_batch(ctypes.Structure):  # noqa: N801 — mirrors the C name
    """``struct llama_batch`` (llama.h).

    Returned **by value** from ``llama_batch_get_one`` and passed **by value** to
    ``llama_decode``. Only ``n_tokens`` and ``token`` are set by the helper; llama.cpp fills in
    positions and sequence ids itself, and a NULL ``logits`` means "output logits for the last
    token only".
    """

    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(llama_token)),
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(llama_pos)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(llama_seq_id))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]


# void (*)(enum ggml_log_level, const char * text, void * user_data) -- ggml.h.
# Safe as a CFUNCTYPE: returns void.
ggml_log_callback = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)

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
    Symbol(Library.LLAMA, "llama_log_set", [ggml_log_callback, ctypes.c_void_p]),
    # Model and vocab introspection.
    Symbol(Library.LLAMA, "llama_model_get_vocab", [llama_model_p], llama_vocab_p),
    Symbol(Library.LLAMA, "llama_vocab_n_tokens", [llama_vocab_p], ctypes.c_int32),
    Symbol(Library.LLAMA, "llama_model_n_embd", [llama_model_p], ctypes.c_int32),
    Symbol(Library.LLAMA, "llama_get_model", [llama_context_p], llama_model_p),
    Symbol(Library.LLAMA, "llama_n_ctx", [llama_context_p], ctypes.c_uint32),
    # Decode. llama_batch_get_one returns the batch BY VALUE and llama_decode takes it by
    # value; both are fine through ctypes (it is only *callbacks* returning structs that trap).
    Symbol(
        Library.LLAMA,
        "llama_batch_get_one",
        [ctypes.POINTER(llama_token), ctypes.c_int32],
        llama_batch,
    ),
    Symbol(Library.LLAMA, "llama_decode", [llama_context_p, llama_batch], ctypes.c_int32),
    Symbol(
        Library.LLAMA,
        "llama_get_logits_ith",
        [llama_context_p, ctypes.c_int32],
        ctypes.POINTER(ctypes.c_float),
    ),
    Symbol(Library.LLAMA, "llama_get_memory", [llama_context_p], ctypes.c_void_p),
    Symbol(Library.LLAMA, "llama_memory_clear", [ctypes.c_void_p, ctypes.c_bool]),
    # Hidden states instead of logits (S1-13).
    #
    # With embeddings output on, a decode stops one op short: it hands back the post-final-norm
    # hidden states [n_embd, n_tokens] and never runs the lm_head. That is the whole trick — the
    # [n_tokens, n_vocab] tensor the chunked path exists to avoid is a tensor llama.cpp now never
    # builds, rather than one we build and throw away.
    #
    # `llama_get_embeddings_ith` and not `llama_get_embeddings`: the latter indexes the output
    # buffer, whose row order is the order tokens were *marked for output*, not batch order. The
    # _ith form applies ctx->output_ids, so a batch position means what the caller thinks it means.
    Symbol(Library.LLAMA, "llama_set_embeddings", [llama_context_p, ctypes.c_bool]),
    Symbol(
        Library.LLAMA,
        "llama_get_embeddings_ith",
        [llama_context_p, ctypes.c_int32],
        ctypes.POINTER(ctypes.c_float),
    ),
    # Tokenizer and chat template (S1-06).
    Symbol(
        Library.LLAMA,
        "llama_model_chat_template",
        [llama_model_p, ctypes.c_char_p],
        ctypes.c_char_p,
    ),
    # Two-call convention: pass n_tokens_max=0 to get the required size back as a NEGATIVE number.
    Symbol(
        Library.LLAMA,
        "llama_tokenize",
        [
            llama_vocab_p,
            ctypes.c_char_p,  # text
            ctypes.c_int32,  # text_len
            ctypes.POINTER(llama_token),  # out
            ctypes.c_int32,  # n_tokens_max
            ctypes.c_bool,  # add_special
            ctypes.c_bool,  # parse_special
        ],
        ctypes.c_int32,
    ),
    Symbol(
        Library.LLAMA,
        "llama_detokenize",
        [
            llama_vocab_p,
            ctypes.POINTER(llama_token),
            ctypes.c_int32,
            ctypes.c_char_p,  # out buffer
            ctypes.c_int32,
            ctypes.c_bool,  # remove_special
            ctypes.c_bool,  # unparse_special
        ],
        ctypes.c_int32,
    ),
    Symbol(
        Library.LLAMA,
        "llama_token_to_piece",
        [
            llama_vocab_p,
            llama_token,
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_bool,
        ],
        ctypes.c_int32,
    ),
    Symbol(Library.LLAMA, "llama_vocab_bos", [llama_vocab_p], llama_token),
    Symbol(Library.LLAMA, "llama_vocab_eos", [llama_vocab_p], llama_token),
    Symbol(Library.LLAMA, "llama_vocab_eot", [llama_vocab_p], llama_token),
    Symbol(Library.LLAMA, "llama_vocab_pad", [llama_vocab_p], llama_token),
    Symbol(Library.LLAMA, "llama_vocab_get_add_bos", [llama_vocab_p], ctypes.c_bool),
    Symbol(Library.LLAMA, "llama_vocab_get_add_eos", [llama_vocab_p], ctypes.c_bool),
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
