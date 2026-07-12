"""ctypes mirror of ``csrc/farm_api.h`` — the learning-llamas shim's own flat C ABI.

Two probe functions today; S1-01 adds ``ll_opt_init_lora`` and S1-02 adds ``ll_train_step``.
"""

from __future__ import annotations

import ctypes

from .registry import Library, Symbol

SYMBOLS = [
    Symbol(Library.FARM, "ll_version", [], ctypes.c_char_p),
    Symbol(Library.FARM, "ll_probe", [], ctypes.c_char_p),
]
