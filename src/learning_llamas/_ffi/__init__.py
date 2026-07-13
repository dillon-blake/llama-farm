"""Layer 2: the ctypes binding to llama.cpp, ggml-opt, and the learning-llamas shim.

ctypes rather than nanobind, deliberately (BLUEPRINT §3): every symbol learning-llamas needs is
already exported (``LLAMA_API`` / ``GGML_API``), the whole ggml-opt driver lives in
``libggml-base``, and llama.cpp itself ships an in-repo precedent for driving libggml from
Python this way (``gguf-py/tests/test_quants.py``). nanobind is deferred until zero-copy views
or GIL release matter at the shim edge; nothing in stage 0 or 1 needs it.

Typical use::

    from learning_llamas import _ffi

    libs = _ffi.load()
    libs.llama.llama_backend_init()
    params = libs.llama.llama_model_default_params()
    params.use_mmap = True
    model = libs.llama.llama_model_load_from_file(b"model.gguf", params)
"""

from __future__ import annotations

from .farm import LLError, check, ll_opt_params, opt_free, opt_init_lora
from .ggml import ggml_init_params
from .ggml_opt import (
    AdamWParams,
    BuildType,
    LossType,
    OptimizerParams,
    OptimizerType,
    SGDParams,
    ggml_opt_optimizer_params,
    ggml_opt_params,
)
from .llama import (
    AttentionType,
    ContextType,
    FlashAttnType,
    GGMLType,
    SplitMode,
    ggml_log_callback,
    llama_batch,
    llama_context_params,
    llama_model_params,
    llama_opt_param_filter,
    llama_opt_params,
    llama_pos,
    llama_sampler_chain_params,
    llama_sampler_p,
    llama_seq_id,
    llama_token,
)
from .loader import SYMBOLS, Libraries, library_dir, library_filename, load
from .registry import Library, Symbol

__all__ = [
    "SYMBOLS",
    "AdamWParams",
    "LLError",
    "AttentionType",
    "BuildType",
    "ContextType",
    "FlashAttnType",
    "GGMLType",
    "Libraries",
    "Library",
    "LossType",
    "OptimizerParams",
    "OptimizerType",
    "SGDParams",
    "SplitMode",
    "Symbol",
    "check",
    "ggml_log_callback",
    "ggml_opt_optimizer_params",
    "ggml_opt_params",
    "library_dir",
    "library_filename",
    "llama_batch",
    "llama_pos",
    "llama_sampler_chain_params",
    "llama_sampler_p",
    "llama_seq_id",
    "llama_context_params",
    "llama_model_params",
    "llama_opt_param_filter",
    "ggml_init_params",
    "ll_opt_params",
    "opt_free",
    "opt_init_lora",
    "llama_opt_params",
    "llama_token",
    "load",
]
