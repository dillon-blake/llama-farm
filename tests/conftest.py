"""Shared fixtures: the tiny models, and a loader that cleans up after itself.

The model class under test here is the **public** one — :class:`learning_llamas.Model`. That is
deliberate: the tests load models exactly the way a user does, so a public API that cannot express
what the tests need is a public API that is missing something.

Fixture models are generated on first use and cached under ``tests/.fixtures/`` (gitignored).
The cache is keyed by a content hash of the generator, so editing ``gen_tiny_llama.py`` cannot
leave a stale model behind.
"""

from __future__ import annotations

import pathlib

import pytest

from learning_llamas import Model, _ffi, libraries

from .fixtures import gen_tiny_llama

CACHE_DIR = pathlib.Path(__file__).parent / ".fixtures"

# The fixture models are tiny and the machine running this is not. Two threads is enough to keep
# the thread-pool path exercised without oversubscribing a CI box running tests in parallel.
TEST_THREADS = 2


@pytest.fixture(scope="session")
def libs() -> _ffi.Libraries:
    """The native libraries, loaded once, with llama.cpp's logs routed into ``logging``."""
    return libraries()


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
    """Factory for :class:`learning_llamas.Model`, closing everything it hands out at teardown."""
    opened: list[Model] = []

    def _load(
        path: pathlib.Path,
        n_ctx: int = 512,
        n_ubatch: int | None = None,
        training: bool = False,
        full_finetune: bool = False,
        n_seq_max: int = 1,
        kv_unified: bool = True,
    ) -> Model:
        model = Model(
            path,
            libs=libs,
            n_ctx=n_ctx,
            n_ubatch=n_ubatch,
            training=training,
            full_finetune=full_finetune,
            n_seq_max=n_seq_max,
            kv_unified=kv_unified,
            n_threads=TEST_THREADS,
        )
        opened.append(model)
        return model

    yield _load

    for model in opened:
        model.close()
