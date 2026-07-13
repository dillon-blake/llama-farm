"""Load a GGUF, open a context to train it in, and attach the adapter being trained.

This is the library's front door. Everything else — :mod:`~learning_llamas.train`,
:mod:`~learning_llamas.logprobs`, :mod:`~learning_llamas.checkpoint` — takes a loaded
:class:`Model` and the :class:`~learning_llamas._ffi.Libraries` handle from :func:`libraries`,
and none of them own that lifecycle: they do not load, they do not free, and they do not guess at
what a context was configured for.

Two of the constructor's flags are load-bearing rather than cosmetic, and both of them fail
*late* and *unhelpfully* if you get them wrong:

``training=True`` disables ggml-cpu's **extra buffer types**. Those repack quantized weights
(``q4_K_8x8`` and friends) to speed up ``MUL_MAT``, and a repacked tensor's ``supports_op``
returns early for every op whose source lives in one — delegating to a handler that implements
``MUL_MAT`` and *not* ``OUT_PROD``, which is exactly what the backward pass needs. A repacked base
weight therefore makes its own gradient node unschedulable, and ``ggml_backend_sched`` aborts
naming neither the op nor the tensor. It bites Q4_K and not Q8_0, purely because a q4_K repack
variant happens to exist.

``full_finetune=True`` additionally disables mmap, because mmap maps the weights **read-only** and
the AdamW step writes updated weights back in place. A mmap'd full fine-tune segfaults inside
``ggml_compute_forward_opt_step_adamw`` — *after* forward and backward have both already
succeeded. LoRA never writes a base weight, so it keeps mmap, and that is the whole point of
BLUEPRINT D2: the frozen base stays on disk.
"""

from __future__ import annotations

import ctypes
import logging
import pathlib

from . import _ffi
from .data.template import ChatTemplate
from .data.tokenize import Tokenizer

log = logging.getLogger("llama.cpp")

# ggml's log levels, mapped onto Python's. CONT is llama.cpp continuing the previous line (it uses
# it for progress bars), which has no level of its own -- DEBUG keeps it out of the way.
_GGML_LOG_LEVELS = {
    0: logging.NOTSET,  # NONE
    1: logging.DEBUG,
    2: logging.INFO,
    3: logging.WARNING,
    4: logging.ERROR,
    5: logging.DEBUG,  # CONT
}


def _forward_to_logging(level: int, text: bytes | None, user_data: object) -> None:  # noqa: ARG001
    """Hand llama.cpp's log line to :mod:`logging` instead of stderr."""
    if text:
        line = text.decode(errors="replace").rstrip()
        log.log(_GGML_LOG_LEVELS.get(level, logging.INFO), "%s", line)


# Kept alive for the process lifetime, deliberately: llama.cpp stores the raw pointer, and letting
# ctypes garbage collect the thunk would leave it calling into freed memory on the next log line.
_LOG_CALLBACK = _ffi.ggml_log_callback(_forward_to_logging)

_libraries: _ffi.Libraries | None = None


def libraries() -> _ffi.Libraries:
    """Open the native libraries, once, and route llama.cpp's logs into :mod:`logging`.

    llama.cpp writes to stderr by default and is extremely chatty on load. Rather than silence it
    — which would throw away the errors along with the noise — this points it at the ``llama.cpp``
    logger, so it is quiet under a default logging config and its warnings and errors still
    arrive. Turn it up with ``logging.getLogger("llama.cpp").setLevel(logging.INFO)``.

    Returns:
        The library handles, cached for the process. Every public entry point takes this.

    Raises:
        RuntimeError: If a library or a symbol is missing, or the vendored llama.cpp commit does
            not match the one the extension was built against.
    """
    global _libraries

    if _libraries is None:
        handles = _ffi.load()
        handles.llama.llama_log_set(_LOG_CALLBACK, None)
        handles.llama.llama_backend_init()
        _libraries = handles

    return _libraries


class Model:
    """A loaded GGUF and a context to run it in.

    Satisfies :class:`~learning_llamas.train.TrainableModel`, which is all the trainers ask for:
    the three handles, and nothing about who owns them.

    Args:
        path: The base model GGUF. Stays frozen and quantized; nothing here ever writes to it.
        libs: The native libraries. Defaults to :func:`libraries`.
        n_ctx: Context size to request. llama.cpp may pad it — read :attr:`n_ctx` back.
        n_ubatch: Physical batch size. Defaults to ``n_ctx``. For training this is the sequence
            length of a batch: the whole batch must go through the graph in one piece.
        training: Configure the context for training. See the module docstring — this is not a
            hint, and a context without it aborts in the scheduler rather than failing cleanly.
        full_finetune: Additionally disable mmap, because base weights will be written. LoRA does
            **not** need this.
        n_seq_max: How many sequences the context can hold. Packing (S1-07) needs one per packed
            sample plus one for the pads; a GRPO group of G needs G.
        kv_unified: Keep the KV cache single-stream. Packing requires it: it is what makes
            llama.cpp hand the batch to the graph in its original order (``split_simple``) rather
            than regrouping it by sequence (``split_equal``).
        n_threads: CPU threads. Defaults to llama.cpp's own default.

    Attributes:
        ctx: The ``llama_context *``.
        model: The ``llama_model *``.
        adapter: The ``llama_adapter_lora *`` being trained, or 0 before one is attached.
        n_vocab: Vocabulary size.
        n_ctx: The context size llama.cpp actually gave you, which may exceed what you asked for.

    Example:
        >>> with Model("base.gguf", n_ctx=512, n_ubatch=256, training=True) as model:
        ...     model.attach_adapter("adapter.gguf")
        ...     train_sft(libraries(), model, samples, SFTConfig(seq_len=256))
    """

    def __init__(
        self,
        path: str | pathlib.Path,
        libs: _ffi.Libraries | None = None,
        n_ctx: int = 512,
        n_ubatch: int | None = None,
        training: bool = False,
        full_finetune: bool = False,
        n_seq_max: int = 1,
        kv_unified: bool = True,
        n_threads: int | None = None,
    ) -> None:
        self._libs = libs if libs is not None else libraries()
        self.path = pathlib.Path(path)
        self.adapter: int = 0
        self.ctx: int = 0
        self.model: int = 0

        # Whether close() should free the adapter. False when it was handed to us already loaded:
        # a shared adapter (GRPO's two contexts) has exactly one owner, and freeing it twice is a
        # double-free that the *other* context takes the blame for.
        self._owns_adapter = False

        model_params = self._libs.llama.llama_model_default_params()
        model_params.n_gpu_layers = 0  # CPU is the oracle. Stage 2+ makes this a parameter.

        if training or full_finetune:
            model_params.use_extra_bufts = False

        if full_finetune:
            model_params.use_mmap = False

        self.model = self._libs.llama.llama_model_load_from_file(
            str(self.path).encode(), model_params
        )
        if not self.model:
            raise RuntimeError(f"failed to load {self.path}")

        ctx_params = self._libs.llama.llama_context_default_params()
        ctx_params.n_ctx = n_ctx
        ctx_params.n_batch = n_ctx
        ctx_params.n_ubatch = n_ubatch if n_ubatch is not None else n_ctx
        ctx_params.n_seq_max = n_seq_max
        ctx_params.kv_unified = kv_unified

        if n_threads is not None:
            ctx_params.n_threads = n_threads
            ctx_params.n_threads_batch = n_threads

        self.ctx = self._libs.llama.llama_init_from_model(self.model, ctx_params)
        if not self.ctx:
            self._libs.llama.llama_model_free(self.model)
            self.model = 0
            raise RuntimeError(f"failed to create a context for {self.path}")

        self.n_ctx: int = self._libs.llama.llama_n_ctx(self.ctx)
        self.tokenizer = Tokenizer(self._libs, self.model)
        self.n_vocab: int = self.tokenizer.n_tokens

    def attach_adapter(
        self,
        path: str | pathlib.Path | None = None,
        scale: float = 1.0,
        adapter: int | None = None,
    ) -> int:
        """Attach a LoRA adapter to this context, and train through it.

        Pass ``adapter`` to attach one that is **already loaded**, instead of loading the file
        again. That is what lets two contexts share a single set of A/B tensors, and GRPO needs
        it: a training step must mutate the very weights the rollout context is sampling through.

        Attaching the same *file* to two contexts loads it twice and gives you two independent
        adapters that start out equal and diverge the moment a step is taken. The trainer moves
        one; the sampler keeps reading the other. There is no error and no NaN — just a policy
        that never changes, and a reward curve that is flat for no reason anyone can see.

        Args:
            path: The adapter GGUF. Required unless ``adapter`` is given.
            scale: The adapter's scale. 1.0 applies it at full strength.
            adapter: An already-loaded ``llama_adapter_lora *`` to share.

        Returns:
            The adapter handle, to hand to another :class:`Model`'s ``adapter=``.

        Raises:
            ValueError: If neither ``path`` nor ``adapter`` is given.
            RuntimeError: If the adapter fails to load or the model rejects it — most often an
                architecture mismatch between the adapter and this base.
        """
        owned = False
        if adapter is None:
            if path is None:
                raise ValueError("attach_adapter needs either a path or an already-loaded adapter")
            adapter = self._libs.llama.llama_adapter_lora_init(self.model, str(path).encode())
            if not adapter:
                raise RuntimeError(f"failed to load adapter {path}")
            owned = True

        adapters = (ctypes.c_void_p * 1)(adapter)
        scales = (ctypes.c_float * 1)(scale)
        status = self._libs.llama.llama_set_adapters_lora(self.ctx, adapters, 1, scales)
        if status != 0:
            if owned:
                self._libs.llama.llama_adapter_lora_free(adapter)
            raise RuntimeError(f"llama_set_adapters_lora failed with status {status}")

        self.adapter = adapter
        self._owns_adapter = owned
        return adapter

    def chat_template(self, name: str | None = None) -> ChatTemplate:
        """The chat template this GGUF embeds, compiled.

        Args:
            name: A named template variant, or None for the default.

        Raises:
            ValueError: If the model embeds none. That is normal for a *base* model, and the error
                says so — construct a :class:`~learning_llamas.data.ChatTemplate` from the source
                you intend to train against instead.
        """
        return ChatTemplate.from_model(self._libs, self.model, self.tokenizer, name)

    def logits(self, tokens: list[int]) -> list[float]:
        """Decode ``tokens`` from a cleared cache and return the last position's logits.

        Raises:
            RuntimeError: If the decode fails.
        """
        libs = self._libs
        libs.llama.llama_memory_clear(libs.llama.llama_get_memory(self.ctx), True)

        buf = (_ffi.llama_token * len(tokens))(*tokens)
        batch = libs.llama.llama_batch_get_one(buf, len(tokens))

        status = libs.llama.llama_decode(self.ctx, batch)
        if status != 0:
            raise RuntimeError(f"llama_decode failed with status {status}")

        out = libs.llama.llama_get_logits_ith(self.ctx, -1)
        if not out:
            raise RuntimeError("llama_get_logits_ith returned NULL")
        return [out[i] for i in range(self.n_vocab)]

    def close(self) -> None:
        """Free the context, the adapter if this model loaded it, and the model. Idempotent.

        An adapter passed in via ``adapter=`` is left alone: it belongs to whoever loaded it.
        """
        # Context first -- it holds a reference to the adapter -- then the adapter, then the model
        # the adapter was built against.
        if self.ctx:
            self._libs.llama.llama_free(self.ctx)
            self.ctx = 0
        if self.adapter and self._owns_adapter:
            self._libs.llama.llama_adapter_lora_free(self.adapter)
        self.adapter = 0
        if self.model:
            self._libs.llama.llama_model_free(self.model)
            self.model = 0

    def __enter__(self) -> Model:
        """Enter a scope that frees the model on the way out."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Free the context, the adapter, and the model."""
        self.close()
