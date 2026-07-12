"""The declarative symbol table that drives ctypes registration.

Every native symbol learning-llamas uses is declared once, as a :class:`Symbol`, in the module
that mirrors its header. :mod:`learning_llamas._ffi.loader` aggregates them and applies the
``argtypes`` / ``restype`` to the loaded handles.

Declaring symbols in a table rather than at each call site buys one thing that matters: a
single test can walk the table and assert every symbol resolves. That is the tripwire for
vendor drift *beyond* what the commit lock catches — a renamed or removed function fails
loudly at test time instead of at the moment ctypes calls a null pointer.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Library(Enum):
    """The four shared libraries the loader opens, in load order.

    ``libggml-cpu`` is deliberately absent: it is linked into ``libggml`` and resolved
    transitively by the dynamic loader (see ``csrc/README.md``).
    """

    GGML_BASE = "ggml-base"
    GGML = "ggml"
    LLAMA = "llama"
    FARM = "learningllamas"


@dataclass(frozen=True)
class Symbol:
    """One native function: which library exports it, and its ctypes signature.

    Attributes:
        library: The library that exports the symbol.
        name: The exported symbol name.
        argtypes: ctypes argument types, in order.
        restype: ctypes return type; ``None`` means ``void``.
    """

    library: Library
    name: str
    argtypes: list[Any] = field(default_factory=list)
    restype: Any = None


def bind(handles: dict[Library, ctypes.CDLL], symbols: list[Symbol]) -> None:
    """Apply each symbol's signature to its library handle.

    Args:
        handles: Open library handles, keyed by :class:`Library`.
        symbols: The symbols to bind.

    Raises:
        RuntimeError: If a declared symbol is not exported by its library. This is what a
            vendor bump that renames or removes a function looks like.
    """
    for symbol in symbols:
        handle = handles[symbol.library]
        try:
            fn = getattr(handle, symbol.name)
        except AttributeError as exc:
            raise RuntimeError(
                f"lib{symbol.library.value} does not export {symbol.name!r}. "
                "The vendored llama.cpp commit and the _ffi mirrors have drifted apart."
            ) from exc
        fn.argtypes = symbol.argtypes
        fn.restype = symbol.restype
