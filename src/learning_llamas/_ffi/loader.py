"""Open the native libraries in the documented order and enforce the version lock.

The load-order contract lives in ``csrc/README.md``::

    libggml-base  →  libggml  →  libllama  →  liblearningllamas

Each is opened with ``RTLD_GLOBAL`` so that symbols land in the global namespace and backends
``dlopen``'d later (``GGML_BACKEND_DL``) can resolve ``ggml_*`` against the already-loaded
``libggml-base``. The fifth library, ``libggml-cpu``, is never named here: it is linked into
``libggml`` and the dynamic loader pulls it in transitively via RPATH.

The version lock
----------------
``_ffi`` mirrors llama.cpp struct layouts by hand, so it is correct against exactly one
vendored commit. Drift does not fail to link — it silently misreads memory. So the native
build bakes in the commit it was compiled against (``ll_probe()``), the Python package records
the commit it was generated against (``_version_lock.VENDORED_COMMIT``), and :func:`load`
refuses to run if they disagree. The pinned commit, the vendored gguf-py, and these mirrors are
one atomic version (ADR-0001).
"""

from __future__ import annotations

import ctypes
import pathlib
import sys
from dataclasses import dataclass

from . import farm, ggml, ggml_backend, ggml_opt, llama
from ._version_lock import VENDORED_COMMIT
from .registry import Library, Symbol, bind

# The one declarative table. A single test walks it and asserts every symbol resolves, which is
# the tripwire for vendor drift beyond what the commit lock catches.
SYMBOLS: list[Symbol] = [
    *farm.SYMBOLS,
    *llama.SYMBOLS,
    *ggml.SYMBOLS,
    *ggml_opt.SYMBOLS,
    *ggml_backend.SYMBOLS,
]

# Load order matters: each library depends on the ones before it. RPATH would resolve them
# anyway, but opening them explicitly means a missing library produces an error naming *that*
# library, instead of an unresolved-symbol cascade out of libllama.
LOAD_ORDER = [Library.GGML_BASE, Library.GGML, Library.LLAMA, Library.FARM]


@dataclass(frozen=True)
class Libraries:
    """The four open library handles, with symbols bound.

    Attributes:
        ggml_base: ``libggml-base`` — ggml core *and* the ggml-opt training driver.
        ggml: ``libggml`` — the backend registry.
        llama: ``libllama`` — the model and context layer.
        farm: ``liblearningllamas`` — the learning-llamas shim.
    """

    ggml_base: ctypes.CDLL
    ggml: ctypes.CDLL
    llama: ctypes.CDLL
    farm: ctypes.CDLL


_libraries: Libraries | None = None


def library_filename(stem: str) -> str:
    """Return the platform's filename for a library given its stem (e.g. ``ggml``)."""
    if sys.platform == "darwin":
        return f"lib{stem}.dylib"
    if sys.platform == "win32":
        return f"{stem}.dll"
    return f"lib{stem}.so"


def library_dir() -> pathlib.Path:
    """Locate the directory the native libraries were installed into.

    Searched over ``__path__`` rather than next to ``__file__``: under an editable install the
    package spans two directories, and CMake installs the libraries into the site-packages one
    while ``__file__`` still points into ``src/``.

    Raises:
        RuntimeError: If no ``lib/`` directory exists, which means the native build never ran.
    """
    import learning_llamas

    for entry in learning_llamas.__path__:
        candidate = pathlib.Path(entry) / "lib"
        if candidate.is_dir():
            return candidate

    raise RuntimeError(
        "learning_llamas/lib/ does not exist: the native libraries were never built.\n"
        "  run: pip install -e . --no-build-isolation"
    )


def _open_libraries() -> Libraries:
    lib_dir = library_dir()
    handles: dict[Library, ctypes.CDLL] = {}

    for library in LOAD_ORDER:
        path = lib_dir / library_filename(library.value)
        if not path.exists():
            raise RuntimeError(f"missing native library: {path}")
        handles[library] = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)

    bind(handles, SYMBOLS)

    return Libraries(
        ggml_base=handles[Library.GGML_BASE],
        ggml=handles[Library.GGML],
        llama=handles[Library.LLAMA],
        farm=handles[Library.FARM],
    )


def _verify_commit(libs: Libraries, expected: str) -> None:
    actual = libs.farm.ll_probe().decode()
    if actual != expected:
        raise RuntimeError(
            "vendored llama.cpp commit mismatch — the ctypes struct mirrors do not match the "
            "native build, and running anyway would corrupt memory rather than fail.\n"
            f"  native libraries built against: {actual}\n"
            f"  _ffi mirrors generated against: {expected}\n"
            "  rebuild: pip install -e . --no-build-isolation --force-reinstall"
        )


def load(expected_commit: str | None = None) -> Libraries:
    """Open the native libraries (once) and verify the version lock.

    Args:
        expected_commit: The vendored llama.cpp commit the caller expects. Defaults to the one
            recorded at build time. Tests pass a wrong value to prove the lock bites.

    Returns:
        The four open library handles, with every declared symbol bound.

    Raises:
        RuntimeError: If a library or symbol is missing, or the commit lock does not match.
    """
    global _libraries

    if _libraries is None:
        _libraries = _open_libraries()

    # Re-verified on every call, not just the first: the check is one C call and a string
    # compare, and it lets a test tamper with the expected commit without resetting the cache.
    _verify_commit(_libraries, expected_commit or VENDORED_COMMIT)

    return _libraries
