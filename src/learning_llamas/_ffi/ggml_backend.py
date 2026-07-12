"""ctypes mirrors of the ``ggml-backend.h`` calls learning-llamas uses.

Mirrored from llama.cpp at commit ``4f37f519722aa3242eecb7649466b4a4a2d6d6da``
(``ggml/include/ggml-backend.h:92-93``).

``ggml_backend_tensor_set`` / ``_get`` are the host↔device transfer path (BLUEPRINT D7): they
are how a batch of token ids and a loss mask are uploaded into graph input tensors, and how a
trained LoRA A/B tensor is read back out for saving. They are synchronous and backend-agnostic,
so the same two calls serve CPU, Metal, CUDA and Vulkan.
"""

from __future__ import annotations

import ctypes

from .registry import Library, Symbol

SYMBOLS = [
    Symbol(
        Library.GGML_BASE,
        "ggml_backend_tensor_set",
        [
            ctypes.c_void_p,  # struct ggml_tensor * tensor
            ctypes.c_void_p,  # const void * data
            ctypes.c_size_t,  # offset
            ctypes.c_size_t,  # size
        ],
    ),
    Symbol(
        Library.GGML_BASE,
        "ggml_backend_tensor_get",
        [
            ctypes.c_void_p,  # const struct ggml_tensor * tensor
            ctypes.c_void_p,  # void * data
            ctypes.c_size_t,  # offset
            ctypes.c_size_t,  # size
        ],
    ),
]
