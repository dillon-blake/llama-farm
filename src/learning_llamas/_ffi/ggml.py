"""ctypes mirrors of the core ``ggml.h`` calls learning-llamas uses.

Mirrored from llama.cpp at commit ``4f37f519722aa3242eecb7649466b4a4a2d6d6da``.

``ggml_quantize_chunk`` (``ggml/include/ggml.h:2789``) is here because K-quants are **not**
writable from pure numpy — gguf-py's quantizer covers the legacy types but not Q4_K. Driving
libggml from Python by ctypes to quantize is llama.cpp's own in-repo pattern
(``gguf-py/tests/test_quants.py``), and it is how the test harness builds its Q4_K fixture
without shelling out to ``llama-quantize``.
"""

from __future__ import annotations

import ctypes

from .registry import Library, Symbol


class ggml_init_params(ctypes.Structure):  # noqa: N801 — mirrors the C name
    """``struct ggml_init_params`` (ggml.h) — passed BY VALUE to ggml_init."""

    _fields_ = [
        ("mem_size", ctypes.c_size_t),
        ("mem_buffer", ctypes.c_void_p),
        ("no_alloc", ctypes.c_bool),
    ]


SYMBOLS = [
    # Enough of ggml to BUILD a graph from Python, which exists for exactly one reason: the S1-11
    # preflight walker is a pure function over (graph, params), and the only way to test it against
    # an op it should REJECT is to hand it a graph containing one. Doing that with a real model
    # would mean finding a model whose architecture is currently untrainable, which is a strange
    # thing to require of a test — and would stop testing the walker the moment that op was
    # implemented.
    Symbol(Library.GGML_BASE, "ggml_init", [ggml_init_params], ctypes.c_void_p),
    # ...and, since S1-13, to RUN one. The chunked-logprob pass builds a two-op graph of its own
    # (mul_mat -> ce_sparse) outside any llama_context, so it needs the ops and a way to execute
    # them. The compute half of that lives in ggml_backend.py.
    Symbol(Library.GGML_BASE, "ggml_free", [ctypes.c_void_p], None),
    Symbol(
        Library.GGML_BASE,
        "ggml_new_tensor_2d",
        [ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_int64],
        ctypes.c_void_p,
    ),
    Symbol(Library.GGML_BASE, "ggml_set_param", [ctypes.c_void_p], None),
    Symbol(Library.GGML_BASE, "ggml_set_name", [ctypes.c_void_p, ctypes.c_char_p], ctypes.c_void_p),
    Symbol(
        Library.GGML_BASE,
        "ggml_mul_mat",
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
        ctypes.c_void_p,
    ),
    # PAD has no backward rule, which is what makes it the right op to test the walker with.
    # (CONCAT used to serve here, and then S1-29 gave it one -- at which point the preflight tests
    # started failing, which is exactly what should happen: the preflight was telling the truth.)
    Symbol(
        Library.GGML_BASE,
        "ggml_pad",
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int],
        ctypes.c_void_p,
    ),
    Symbol(Library.GGML_BASE, "ggml_new_graph", [ctypes.c_void_p], ctypes.c_void_p),
    Symbol(
        Library.GGML_BASE,
        "ggml_build_forward_expand",
        [ctypes.c_void_p, ctypes.c_void_p],
        None,
    ),
    # Raw pointer to a tensor's data. Used instead of mirroring `struct ggml_tensor` — that
    # struct is large, churns, and getting one field's offset wrong reads arbitrary memory.
    # Asking ggml for the pointer costs one call and cannot drift.
    Symbol(Library.GGML_BASE, "ggml_get_data", [ctypes.c_void_p], ctypes.c_void_p),
    Symbol(Library.GGML_BASE, "ggml_nelements", [ctypes.c_void_p], ctypes.c_int64),
    Symbol(Library.GGML_BASE, "ggml_nbytes", [ctypes.c_void_p], ctypes.c_size_t),
    # Bytes needed for one row of `ne` elements of the given type. Quantized rows are not
    # ne * type_size: they are (ne / block_size) blocks, each with its own scale.
    Symbol(Library.GGML_BASE, "ggml_row_size", [ctypes.c_int, ctypes.c_int64], ctypes.c_size_t),
    Symbol(Library.GGML_BASE, "ggml_blck_size", [ctypes.c_int], ctypes.c_int64),
    Symbol(Library.GGML_BASE, "ggml_type_name", [ctypes.c_int], ctypes.c_char_p),
    # Some quant types cannot be produced without an importance matrix -- the IQ family. There is
    # nothing sensible to fall back on for them, so a merged export refuses rather than guessing.
    Symbol(Library.GGML_BASE, "ggml_quantize_requires_imatrix", [ctypes.c_int], ctypes.c_bool),
    Symbol(
        Library.GGML_BASE,
        "ggml_quantize_chunk",
        [
            ctypes.c_int,  # enum ggml_type
            ctypes.POINTER(ctypes.c_float),  # src
            ctypes.c_void_p,  # dst
            ctypes.c_int64,  # start
            ctypes.c_int64,  # nrows
            ctypes.c_int64,  # n_per_row
            ctypes.POINTER(ctypes.c_float),  # imatrix (may be NULL)
        ],
        ctypes.c_size_t,
    ),
    Symbol(
        Library.GGML_BASE,
        "ggml_new_tensor_1d",
        [ctypes.c_void_p, ctypes.c_int, ctypes.c_int64],  # ctx, enum ggml_type, ne0
        ctypes.c_void_p,
    ),
    # loss_i = w_i * (logsumexp_j(z_ij) - z_i[label_i]), unreduced -- so the negation of this IS
    # the per-token logprob of the realized token, with the caller's mask already applied (S1-04,
    # ADR-0003). logit_scale and softcap are handled inside the kernel, in stable-lse math.
    Symbol(
        Library.GGML_BASE,
        "ggml_cross_entropy_loss_sparse",
        [
            ctypes.c_void_p,  # ctx
            ctypes.c_void_p,  # logits  [n_vocab, n_tokens]
            ctypes.c_void_p,  # labels  I32 [n_tokens]
            ctypes.c_void_p,  # weights F32 [n_tokens]
            ctypes.c_float,  # logit_scale (1.0 = off)
            ctypes.c_float,  # softcap     (0.0 = off)
        ],
        ctypes.c_void_p,
    ),
    Symbol(
        Library.GGML_BASE,
        "ggml_scale",
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float],
        ctypes.c_void_p,
    ),
    Symbol(
        Library.GGML_BASE,
        "ggml_add",
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
        ctypes.c_void_p,
    ),
]
