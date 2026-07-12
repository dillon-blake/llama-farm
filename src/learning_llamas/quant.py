"""Quantize a float array to a GGML type — by whichever route can actually do it.

This exists because **gguf-py cannot quantize the types people actually use**. It dequantizes
everything, and it quantizes F32, F16, Q8_0 and the legacy Q4_0/Q4_1/Q5_0/Q5_1 — but ask it for
Q4_K, Q5_K or Q6_K, the K-quants that make up nearly every model on Hugging Face, and it raises
``NotImplementedError``. Measured, not assumed::

    F32   OK    F16   OK    Q8_0  OK    Q4_0  OK
    Q4_K  NotImplementedError
    Q6_K  NotImplementedError

So a merged export that re-quantized through gguf-py alone could not round-trip a Q4_K model back
to Q4_K. It would have to silently emit something else — an F16 file four times the size, or a
Q8_0 one the user did not ask for — and the first they would know of it is when the model did not
fit on their machine.

The route that works is libggml's own quantizer, driven from Python by ctypes. That is not a hack:
it is llama.cpp's own in-repo pattern (``gguf-py/tests/test_quants.py`` does exactly this to test
gguf-py's quantizers *against* the reference implementation), and it means a re-quantized tensor is
bit-identical to what ``llama-quantize`` would have produced, because it is the same code.

The IQ family is refused rather than approximated: those types genuinely require an importance
matrix computed from calibration data, and there is nothing honest to substitute for one.
"""

from __future__ import annotations

import ctypes

import numpy as np

from learning_llamas import _ffi

# These live in libggml-BASE, and it matters which handle you ask.
#
# ctypes resolves a symbol through a shared library's whole dependency chain, so
# `libs.ggml.ggml_quantize_chunk` finds the function perfectly well -- it is just NOT BOUND, because
# the signature was declared against libggml-base. An unbound ctypes function defaults to
# `restype = c_int` and no argtypes, which on a 64-bit host truncates every pointer you hand it to
# 32 bits. It does not fail; it corrupts. Ask the handle the symbol was declared on.

try:
    import gguf
except ImportError as exc:  # pragma: no cover - environment problem, not a code path
    raise ImportError("learning-llamas needs the vendored gguf-py; see adapter.py") from exc


class ImatrixRequired(ValueError):
    """The target type cannot be produced without an importance matrix.

    The IQ quants derive their codebooks from calibration statistics. Producing one without an
    imatrix is not "slightly worse", it is not the format. Raised rather than silently downgraded,
    because a user who asked for IQ2_XXS wanted IQ2_XXS.
    """


def quantize(values: np.ndarray, dtype: gguf.GGMLQuantizationType, libs: _ffi.Libraries) -> bytes:
    """Quantize a float32 array to ``dtype``.

    Args:
        values: The data, float32, in numpy (row-major) order — so its last axis is the GGUF
            ``ne[0]`` axis, which is the one quantization blocks run along.
        dtype: The target GGML type.
        libs: The loaded native libraries, for the types gguf-py cannot write.

    Returns:
        The quantized bytes, laid out exactly as a GGUF tensor's data.

    Raises:
        ImatrixRequired: If ``dtype`` needs an importance matrix.
        ValueError: If the row length is not a multiple of the type's block size.
    """
    if libs.ggml_base.ggml_quantize_requires_imatrix(int(dtype)):
        raise ImatrixRequired(
            f"{dtype.name} cannot be produced without an importance matrix, which is computed from "
            f"calibration data. Re-quantizing to it is refused rather than approximated. Export to "
            f"a type that does not need one (Q4_K, Q5_K, Q6_K, Q8_0, F16), or quantize the merged "
            f"F16 model yourself with llama-quantize and your own imatrix."
        )

    data = np.ascontiguousarray(values, dtype=np.float32)

    block = int(libs.ggml_base.ggml_blck_size(int(dtype)))
    n_per_row = int(data.shape[-1])

    if n_per_row % block != 0:
        raise ValueError(
            f"a row of {n_per_row} elements cannot be quantized to {dtype.name}, whose blocks are "
            f"{block} elements wide"
        )

    n_rows = int(data.size // n_per_row)
    n_bytes = int(libs.ggml_base.ggml_row_size(int(dtype), n_per_row)) * n_rows

    dst = ctypes.create_string_buffer(n_bytes)
    src = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    written = libs.ggml_base.ggml_quantize_chunk(int(dtype), src, dst, 0, n_rows, n_per_row, None)

    if written != n_bytes:  # pragma: no cover - would mean ggml disagrees with itself
        raise RuntimeError(f"ggml_quantize_chunk wrote {written} bytes, expected {n_bytes}")

    return dst.raw[:n_bytes]


def dequantize(
    data: np.ndarray, dtype: gguf.GGMLQuantizationType, shape: tuple[int, ...]
) -> np.ndarray:
    """Dequantize a GGUF tensor's raw data to float32.

    Args:
        data: The tensor's data. Either as ``GGUFReader`` hands it over — already shaped
            ``(n_rows, row_bytes)`` — or as a flat buffer straight out of :func:`quantize`. Both
            are normalized here, because getting it wrong produces a shape error two libraries deep
            that says nothing about what you actually did.
        dtype: Its GGML type.
        shape: The numpy shape to return — the GGUF ``ne``, reversed.

    Returns:
        The values as float32.
    """
    arr = np.asarray(data)

    if arr.dtype == np.uint8:
        # gguf's dequantizer wants the BYTE shape: rows on the outside, a row's bytes on the inside.
        # A row of a quantized tensor is not `n * type_size` bytes -- it is blocks, each with its
        # own scale -- so the conversion is not one anybody should be doing by hand.
        arr = arr.reshape(gguf.quants.quant_shape_to_byte_shape(shape, dtype))

    return gguf.quants.dequantize(arr, dtype).astype(np.float32).reshape(shape)
