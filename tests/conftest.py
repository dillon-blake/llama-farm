"""Shared fixtures: the tiny models, and a quiet llama.cpp.

Fixture models are generated on first use and cached under ``tests/.fixtures/`` (gitignored).
The cache is keyed by a content hash of the generator, so editing ``gen_tiny_llama.py`` cannot
leave a stale model behind.
"""

from __future__ import annotations

import ctypes
import pathlib

import pytest

from learning_llamas import _ffi

from .fixtures import gen_tiny_llama

CACHE_DIR = pathlib.Path(__file__).parent / ".fixtures"

# Kept alive for the process lifetime: llama.cpp stores the pointer, and letting ctypes garbage
# collect the thunk would leave it calling into freed memory.
_NULL_LOG_CALLBACK = _ffi.ggml_log_callback(
    lambda level, text, user_data: None  # noqa: ARG005
)


@pytest.fixture(scope="session")
def libs() -> _ffi.Libraries:
    """The native libraries, loaded once, with llama.cpp's log spam silenced."""
    handles = _ffi.load()
    handles.llama.llama_log_set(_NULL_LOG_CALLBACK, None)
    handles.llama.llama_backend_init()
    return handles


@pytest.fixture(scope="session")
def fixture_cache_dir() -> pathlib.Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


@pytest.fixture(scope="session")
def tiny_f32(libs: _ffi.Libraries, fixture_cache_dir: pathlib.Path) -> pathlib.Path:
    path, _ = gen_tiny_llama.build("f32", fixture_cache_dir)
    return path


@pytest.fixture(scope="session")
def tiny_q8_0(libs: _ffi.Libraries, fixture_cache_dir: pathlib.Path) -> pathlib.Path:
    path, _ = gen_tiny_llama.build("q8_0", fixture_cache_dir)
    return path


@pytest.fixture(scope="session")
def tiny_q4_k(libs: _ffi.Libraries, fixture_cache_dir: pathlib.Path) -> pathlib.Path:
    path, _ = gen_tiny_llama.build("q4_k", fixture_cache_dir)
    return path


@pytest.fixture(scope="session", params=gen_tiny_llama.VARIANTS)
def tiny_model(
    request: pytest.FixtureRequest, libs: _ffi.Libraries, fixture_cache_dir: pathlib.Path
) -> pathlib.Path:
    """Every fixture variant in turn: F32, Q8_0, Q4_K.

    Quantized bases are exercised from day one because backward through quantized weights is
    engine-supported but lightly tested upstream (BLUEPRINT §10, risk 2).
    """
    path, _ = gen_tiny_llama.build(request.param, fixture_cache_dir)
    return path


@pytest.fixture
def load_model(libs: _ffi.Libraries):  # noqa: ANN201 - a factory, closed over the libs fixture
    """Factory for :class:`Model`, closing everything it hands out at teardown."""
    opened: list[Model] = []

    def _load(
        path: pathlib.Path,
        n_ctx: int = 512,
        n_ubatch: int | None = None,
        training: bool = False,
        full_finetune: bool = False,
    ) -> Model:
        model = Model(
            libs,
            path,
            n_ctx=n_ctx,
            n_ubatch=n_ubatch,
            training=training,
            full_finetune=full_finetune,
        )
        opened.append(model)
        return model

    yield _load

    for model in opened:
        model.close()


class Model:
    """A loaded model plus a context, with decode reduced to one call.

    Not a public API — just enough to let the tests say what they mean.
    """

    def __init__(
        self,
        libs: _ffi.Libraries,
        path: pathlib.Path,
        n_ctx: int = 512,
        n_ubatch: int | None = None,
        training: bool = False,
        full_finetune: bool = False,
    ) -> None:
        """Load a model and a context to run it in.

        Args:
            libs: The loaded native libraries.
            path: The model GGUF.
            n_ctx: Context size to request (llama.cpp may pad it — read it back).
            n_ubatch: Physical batch size; defaults to ``n_ctx``.
            training: Configure for *any* kind of training — disables extra buffer types.
            full_finetune: Additionally disable mmap, because base weights will be written.
                LoRA does **not** need this, and that is exactly the point of BLUEPRINT D2.
        """
        self._libs = libs

        model_params = libs.llama.llama_model_default_params()
        model_params.n_gpu_layers = 0  # CPU is the oracle

        if training or full_finetune:
            # ggml-cpu's "extra buffer types" repack weights (q4_K_8x8 and friends) to speed up
            # MUL_MAT. But supports_op returns EARLY for any op whose src is in one, delegating
            # to a handler that implements MUL_MAT and *not* OUT_PROD — precisely what the
            # backward pass needs. So a repacked base tensor makes its own gradient node
            # unschedulable and ggml_backend_sched aborts, naming neither op nor tensor.
            # It bites Q4_K and not Q8_0, purely because a q4_K repack variant exists.
            model_params.use_extra_bufts = False

        if full_finetune:
            # mmap maps the weights READ-ONLY, and the AdamW step writes updated weights back in
            # place — so a mmap'd full fine-tune segfaults inside
            # ggml_compute_forward_opt_step_adamw, *after* forward and backward have already
            # succeeded. llama.cpp's finetune example disables mmap for exactly this reason
            # (examples/training/finetune.cpp:28-32).
            #
            # LoRA never writes a base weight, so it keeps mmap — which makes mmap a free oracle
            # for "did anything touch the base model?" (see test_opt_init_lora).
            model_params.use_mmap = False

        self.path = path
        self.model = libs.llama.llama_model_load_from_file(str(path).encode(), model_params)
        if not self.model:
            raise RuntimeError(f"failed to load {path}")

        ctx_params = libs.llama.llama_context_default_params()
        ctx_params.n_ctx = n_ctx
        ctx_params.n_batch = n_ctx
        ctx_params.n_ubatch = n_ubatch if n_ubatch is not None else n_ctx
        ctx_params.n_threads = 2
        ctx_params.n_threads_batch = 2
        self.ctx = libs.llama.llama_init_from_model(self.model, ctx_params)
        if not self.ctx:
            libs.llama.llama_model_free(self.model)
            raise RuntimeError(f"failed to create a context for {path}")

        vocab = libs.llama.llama_model_get_vocab(self.model)
        self.n_vocab = libs.llama.llama_vocab_n_tokens(vocab)

    def logits(self, tokens: list[int]) -> list[float]:
        """Decode ``tokens`` from a cleared cache and return the last token's logits."""
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

    def attach_adapter(self, path: pathlib.Path, scale: float = 1.0) -> None:
        libs = self._libs
        adapter = libs.llama.llama_adapter_lora_init(self.model, str(path).encode())
        if not adapter:
            raise RuntimeError(f"failed to load adapter {path}")

        adapters = (ctypes.c_void_p * 1)(adapter)
        scales = (ctypes.c_float * 1)(scale)
        status = libs.llama.llama_set_adapters_lora(self.ctx, adapters, 1, scales)
        if status != 0:
            raise RuntimeError(f"llama_set_adapters_lora failed with status {status}")

        self.adapter = adapter

    def close(self) -> None:
        if getattr(self, "ctx", None):
            self._libs.llama.llama_free(self.ctx)
            self.ctx = None
        if getattr(self, "model", None):
            self._libs.llama.llama_model_free(self.model)
            self.model = None
