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

SYMBOLS = [
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
]
