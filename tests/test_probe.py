"""S0-03 smoke test: the native libraries load in order and the shim answers.

Deliberately dependency-light — it opens the libraries with raw ctypes rather than going
through ``learning_llamas._ffi``, which does not exist until S0-04. It is what proves the
build, the include paths, the RPATH wiring, and the packaging all work.
"""

import ctypes
import pathlib
import sys

import pytest

import learning_llamas

from .vendor_pin import vendored_commit

# The documented load order (csrc/README.md). libggml-cpu is absent on purpose: it is linked
# PUBLIC into libggml and resolved transitively via RPATH.
LOAD_ORDER = ["ggml-base", "ggml", "llama", "learningllamas"]


def _library_filename(stem: str) -> str:
    if sys.platform == "darwin":
        return f"lib{stem}.dylib"
    if sys.platform == "win32":
        return f"{stem}.dll"
    return f"lib{stem}.so"


def _lib_dir() -> pathlib.Path:
    """Locate the ``lib/`` directory the native libraries were installed into.

    Searched over ``__path__`` rather than next to ``__file__``: under an editable install
    the package is a namespace spread across two directories, and CMake installs the
    libraries into the site-packages one while ``__file__`` still points into ``src/``.
    """
    for entry in learning_llamas.__path__:
        lib_dir = pathlib.Path(entry) / "lib"
        if lib_dir.is_dir():
            return lib_dir
    pytest.fail(
        "learning_llamas/lib/ does not exist — the native build did not run. "
        "Run: pip install -e . --no-build-isolation"
    )


@pytest.fixture(scope="module")
def shim() -> ctypes.CDLL:
    """Open the four libraries in the documented order and return the shim handle."""
    lib_dir = _lib_dir()
    handle = None
    for stem in LOAD_ORDER:
        path = lib_dir / _library_filename(stem)
        assert path.exists(), f"{path} is missing from the wheel"
        # RTLD_GLOBAL so that backends dlopen'd later resolve ggml_* against libggml-base.
        handle = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    assert handle is not None
    return handle


def test_all_libraries_ship(shim: ctypes.CDLL) -> None:
    """Every library the loader names, plus the CPU backend, is in learning_llamas/lib/."""
    lib_dir = _lib_dir()
    for stem in [*LOAD_ORDER, "ggml-cpu"]:
        assert (lib_dir / _library_filename(stem)).exists(), f"{stem} missing"


def test_ll_probe_matches_the_submodule(shim: ctypes.CDLL) -> None:
    """ll_probe() returns the commit the submodule is actually checked out at.

    A mismatch means the native build is stale relative to the vendored tree — which is the
    state in which the hand-written ctypes struct mirrors start misreading memory.
    """
    shim.ll_probe.restype = ctypes.c_char_p
    shim.ll_probe.argtypes = []
    assert shim.ll_probe().decode() == vendored_commit()


def test_ll_version_matches_package(shim: ctypes.CDLL) -> None:
    """ll_version() equals learning_llamas.__version__ — a stale native build is visible."""
    shim.ll_version.restype = ctypes.c_char_p
    shim.ll_version.argtypes = []
    assert shim.ll_version().decode() == learning_llamas.__version__
