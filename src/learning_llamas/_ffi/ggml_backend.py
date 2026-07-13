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
    # ---------------------------------------------------------------------------------------
    # Running a graph of our own (S1-13).
    #
    # Everything above this line operates on a graph llama.cpp built. The chunked-logprob pass
    # builds its own — a two-op graph, run outside any llama_context — so it needs a backend to
    # run it on and a buffer to put its tensors in.
    #
    # `ggml_backend_init_by_type` rather than a named CPU backend: the graph is sched-agnostic by
    # construction (ggml_mul_mat + ce_sparse, nothing else), so the only thing tying it to the CPU
    # is the device we ask for. Stages 2-4 change one argument.
    # ---------------------------------------------------------------------------------------
    Symbol(
        Library.GGML,
        "ggml_backend_init_by_type",
        [ctypes.c_int, ctypes.c_char_p],  # enum ggml_backend_dev_type, const char * params
        ctypes.c_void_p,  # ggml_backend_t
    ),
    Symbol(Library.GGML_BASE, "ggml_backend_free", [ctypes.c_void_p]),
    Symbol(
        Library.GGML_BASE,
        "ggml_backend_graph_compute",
        [ctypes.c_void_p, ctypes.c_void_p],  # ggml_backend_t, struct ggml_cgraph *
        ctypes.c_int,  # enum ggml_status
    ),
    # Allocates a backend buffer for every tensor in a context, in one go. The context must have
    # been created with no_alloc=true.
    Symbol(
        Library.GGML_BASE,
        "ggml_backend_alloc_ctx_tensors",
        [ctypes.c_void_p, ctypes.c_void_p],  # struct ggml_context *, ggml_backend_t
        ctypes.c_void_p,  # ggml_backend_buffer_t
    ),
    # The zero-copy hinge of S1-13: wrap memory we already have -- the lm_head, sitting in the
    # GGUF's mmap -- as a backend buffer, instead of copying it into one. A 128k-vocab lm_head is
    # hundreds of MB; copying it to compute logprobs would cost more memory than the full-logits
    # path this module exists to avoid.
    Symbol(
        Library.GGML_BASE,
        "ggml_backend_cpu_buffer_from_ptr",
        [ctypes.c_void_p, ctypes.c_size_t],  # void * ptr, size_t size
        ctypes.c_void_p,  # ggml_backend_buffer_t
    ),
    Symbol(
        Library.GGML_BASE,
        "ggml_backend_tensor_alloc",
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],  # buffer, tensor, addr
        ctypes.c_int,  # enum ggml_status
    ),
    Symbol(Library.GGML_BASE, "ggml_backend_buffer_free", [ctypes.c_void_p]),
    # How big a buffer actually turned out to be. The chunked-logprob pass's central claim is a
    # claim about memory, so a test asserts it against ggml's own allocator rather than against a
    # calculation that could be wrong in the same direction as the code.
    Symbol(
        Library.GGML_BASE,
        "ggml_backend_buffer_get_size",
        [ctypes.c_void_p],
        ctypes.c_size_t,
    ),
]
