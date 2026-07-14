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


# ---------------------------------------------------------------------------
# --device (S1-12)
#
# ROADMAP §11 makes the convergence gate the phase-exit criterion for every backend stage, so it
# is backend-parameterized from day one and S2-01/S3-01/S4-01 reuse it with one flag rather than
# forking it. A device this build does not have SKIPS -- it does not fail, and it does not quietly
# run on the CPU while reporting success.
# ---------------------------------------------------------------------------

DEVICES = ("cpu", "metal", "cuda", "vulkan")

# What ggml calls each device, so `--device metal` can ask ggml whether this build has one rather
# than guessing from the platform.
_GGML_DEVICE_PREFIX = {"cpu": "CPU", "metal": "Metal", "cuda": "CUDA", "vulkan": "Vulkan"}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        choices=DEVICES,
        help="Compute backend for the convergence gate and the grad-check wrapper.",
    )


def ggml_device_names(libs: _ffi.Libraries) -> dict[str, str]:
    """Map each logical device to the name **ggml actually registered it under**.

    The distinction is not pedantry. ``test-backend-ops -b <name>`` matches with an exact
    ``strcmp`` against ``ggml_backend_dev_name`` (``test-backend-ops.cpp:11214``) — and GPU
    backends **index-suffix** their names: the first CUDA device is ``CUDA0``, not ``CUDA``. So
    ``-b CUDA`` matches nothing, and the harness then prints ``Skipping``, counts it as passed,
    and **exits 0** having run zero cases. Every backend lane would go green while checking
    nothing, exactly like ``grad -o GLU`` did.

    So the real name is asked of ggml rather than assumed. ``CPU`` happens to be unsuffixed, which
    is why this was invisible on the CPU lane.
    """
    registered = [
        libs.ggml_base.ggml_backend_dev_name(libs.ggml.ggml_backend_dev_get(i)).decode()
        for i in range(libs.ggml.ggml_backend_dev_count())
    ]
    found: dict[str, str] = {}
    for device, prefix in _GGML_DEVICE_PREFIX.items():
        for name in registered:
            if name.startswith(prefix):
                found.setdefault(device, name)
    return found


def available_devices(libs: _ffi.Libraries) -> set[str]:
    """The devices this ggml build actually registered."""
    return set(ggml_device_names(libs))


@pytest.fixture(scope="session")
def device(request: pytest.FixtureRequest, libs: _ffi.Libraries) -> str:
    """The requested compute backend, skipping the test if this build has no such device."""
    requested = request.config.getoption("--device")
    if requested not in available_devices(libs):
        pytest.skip(f"this build has no {requested} device")
    return str(requested)


@pytest.fixture(scope="session")
def ggml_device(device: str, libs: _ffi.Libraries) -> str:
    """The requested device under ggml's own name for it (``CUDA0``, not ``CUDA``)."""
    return ggml_device_names(libs)[device]
