"""ctypes mirrors of ``ggml/include/ggml-opt.h``.

Mirrored from llama.cpp at commit ``4f37f519722aa3242eecb7649466b4a4a2d6d6da``. These are
interface declarations, not copied implementation, but the trail matters: ctypes reproduces
struct layouts by hand, so a vendor bump that reshapes one of them corrupts memory instead of
failing to link. The commit lock in :mod:`learning_llamas._ffi.loader` is what stops that.

The whole ggml-opt driver lives in ``libggml-base`` — ``ggml-opt.cpp`` is one of that target's
sources (``ggml/src/CMakeLists.txt:192``), not ``libggml``'s.

The struct-return hazard
------------------------
``ggml_opt_optimizer_params`` is returned **by value** from the per-step optimizer-params
callback (``ggml-opt.h:101``). Struct-by-value returns through *ctypes callbacks* are a known
ABI trap, so learning-llamas never registers a Python callback for it. Instead
:class:`OptimizerParams` passes the exported ``ggml_opt_get_constant_optimizer_params``
(``ggml-opt.h:108``) as the callback, with a **Python-owned** struct as its userdata. Python
mutates that struct between steps, which is how learning-rate schedules work for free.

Calling a *C function* that returns the struct is fine and is exercised by the tests — it is
only the callback direction that is unsafe.
"""

from __future__ import annotations

import ctypes
from enum import IntEnum

from .registry import Library, Symbol

# Opaque handles (ggml-opt.h:26-33). Only ggml owns their layout.
ggml_opt_context_t = ctypes.c_void_p
ggml_opt_dataset_t = ctypes.c_void_p
ggml_opt_result_t = ctypes.c_void_p


class LossType(IntEnum):
    """``enum ggml_opt_loss_type`` (ggml-opt.h)."""

    MEAN = 0
    SUM = 1
    CROSS_ENTROPY = 2
    MEAN_SQUARED_ERROR = 3


class BuildType(IntEnum):
    """``enum ggml_opt_build_type`` (ggml-opt.h). Note the non-contiguous values."""

    FORWARD = 10
    GRAD = 20
    OPT = 30


class OptimizerType(IntEnum):
    """``enum ggml_opt_optimizer_type`` (ggml-opt.h)."""

    ADAMW = 0
    SGD = 1


class AdamWParams(ctypes.Structure):
    """The ``adamw`` member of ``ggml_opt_optimizer_params`` (ggml-opt.h:86-92)."""

    _fields_ = [
        ("alpha", ctypes.c_float),  # learning rate
        ("beta1", ctypes.c_float),
        ("beta2", ctypes.c_float),
        ("eps", ctypes.c_float),
        ("wd", ctypes.c_float),  # weight decay, 0.0 to disable
    ]


class SGDParams(ctypes.Structure):
    """The ``sgd`` member of ``ggml_opt_optimizer_params`` (ggml-opt.h:93-96)."""

    _fields_ = [
        ("alpha", ctypes.c_float),  # learning rate
        ("wd", ctypes.c_float),  # weight decay
    ]


class ggml_opt_optimizer_params(ctypes.Structure):  # noqa: N801 — mirrors the C name
    """``struct ggml_opt_optimizer_params`` (ggml-opt.h:85-97).

    ``csrc/farm_internals.cpp`` static_asserts this struct's size on the C side, so a layout
    change breaks the build rather than the training run.
    """

    _fields_ = [
        ("adamw", AdamWParams),
        ("sgd", SGDParams),
    ]


class ggml_opt_params(ctypes.Structure):  # noqa: N801 — mirrors the C name
    """``struct ggml_opt_params`` (ggml-opt.h:111-131).

    ``get_opt_pars`` is typed ``c_void_p``, not a ``CFUNCTYPE``, and that is deliberate: the C
    typedef returns ``ggml_opt_optimizer_params`` by value, and defining a ctypes callback type
    with that restype is the ABI trap this module exists to avoid. A function pointer is
    pointer-sized either way, so the layout is identical; use :class:`OptimizerParams` to fill
    it with the address of the exported constant-params function.
    """

    _fields_ = [
        ("backend_sched", ctypes.c_void_p),
        ("ctx_compute", ctypes.c_void_p),
        ("inputs", ctypes.c_void_p),
        ("outputs", ctypes.c_void_p),
        ("loss_type", ctypes.c_int),
        ("build_type", ctypes.c_int),
        ("opt_period", ctypes.c_int32),
        ("get_opt_pars", ctypes.c_void_p),
        ("get_opt_pars_ud", ctypes.c_void_p),
        ("optimizer", ctypes.c_int),
    ]


class OptimizerParams:
    """A Python-owned ``ggml_opt_optimizer_params`` plus the callback pair that reads it.

    This is the sanctioned way to drive optimizer hyperparameters, and the only one this
    project uses. Rather than registering a Python callback (which would have to return the
    struct by value — the ABI trap), it hands ggml the *exported* C function
    ``ggml_opt_get_constant_optimizer_params``, whose entire job is to cast its userdata back to
    a ``ggml_opt_optimizer_params`` and return it. The userdata is this object's struct.

    The consequence is that a learning-rate schedule is just an attribute assignment between
    steps — no callback, no GIL, no struct-return marshalling:

        >>> params = OptimizerParams(libs)     # doctest: +SKIP
        >>> params.adamw.alpha = 1e-4          # doctest: +SKIP

    Attributes:
        raw: The owned struct. Mutating it changes the next step's hyperparameters.
    """

    def __init__(self, libs: object) -> None:
        """Initialize from ggml's own defaults.

        Args:
            libs: The loaded library namespace from :func:`learning_llamas._ffi.loader.load`.
        """
        base = libs.ggml_base  # type: ignore[attr-defined]
        self.raw = base.ggml_opt_get_default_optimizer_params(None)
        self._get_opt_pars = ctypes.cast(
            base.ggml_opt_get_constant_optimizer_params, ctypes.c_void_p
        ).value

    @property
    def adamw(self) -> AdamWParams:
        """The AdamW hyperparameters (learning rate, betas, epsilon, weight decay)."""
        return self.raw.adamw

    @property
    def sgd(self) -> SGDParams:
        """The SGD hyperparameters."""
        return self.raw.sgd

    def callback(self) -> tuple[int | None, ctypes.c_void_p]:
        """Return the ``(get_opt_pars, get_opt_pars_ud)`` pair to store in a params struct."""
        return self._get_opt_pars, ctypes.cast(ctypes.byref(self.raw), ctypes.c_void_p)


SYMBOLS = [
    # Dataset. ggml-opt owns the storage; ggml_get_data() hands back the pointer to fill.
    Symbol(
        Library.GGML_BASE,
        "ggml_opt_dataset_init",
        [
            ctypes.c_int,  # type_data
            ctypes.c_int,  # type_label
            ctypes.c_int64,  # ne_datapoint
            ctypes.c_int64,  # ne_label
            ctypes.c_int64,  # ndata
            ctypes.c_int64,  # ndata_shard
        ],
        ggml_opt_dataset_t,
    ),
    Symbol(Library.GGML_BASE, "ggml_opt_dataset_free", [ggml_opt_dataset_t]),
    Symbol(Library.GGML_BASE, "ggml_opt_dataset_ndata", [ggml_opt_dataset_t], ctypes.c_int64),
    Symbol(Library.GGML_BASE, "ggml_opt_dataset_data", [ggml_opt_dataset_t], ctypes.c_void_p),
    Symbol(Library.GGML_BASE, "ggml_opt_dataset_labels", [ggml_opt_dataset_t], ctypes.c_void_p),
    # Lifecycle.
    Symbol(Library.GGML_BASE, "ggml_opt_init", [ggml_opt_params], ggml_opt_context_t),
    Symbol(Library.GGML_BASE, "ggml_opt_free", [ggml_opt_context_t]),
    Symbol(Library.GGML_BASE, "ggml_opt_reset", [ggml_opt_context_t, ctypes.c_bool]),
    # Optimizer parameters. Both return the struct BY VALUE; that is safe for a C function
    # called from Python (only the callback direction is the ABI trap).
    Symbol(
        Library.GGML_BASE,
        "ggml_opt_get_default_optimizer_params",
        [ctypes.c_void_p],
        ggml_opt_optimizer_params,
    ),
    Symbol(
        Library.GGML_BASE,
        "ggml_opt_get_constant_optimizer_params",
        [ctypes.c_void_p],
        ggml_opt_optimizer_params,
    ),
    # Gradient accumulators — how a trainer reads the gradient of one parameter tensor.
    Symbol(
        Library.GGML_BASE,
        "ggml_opt_grad_acc",
        [ggml_opt_context_t, ctypes.c_void_p],
        ctypes.c_void_p,
    ),
    # Results.
    Symbol(Library.GGML_BASE, "ggml_opt_result_init", [], ggml_opt_result_t),
    Symbol(Library.GGML_BASE, "ggml_opt_result_free", [ggml_opt_result_t]),
    Symbol(Library.GGML_BASE, "ggml_opt_result_reset", [ggml_opt_result_t]),
    Symbol(
        Library.GGML_BASE,
        "ggml_opt_result_loss",
        [ggml_opt_result_t, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)],
    ),
    Symbol(
        Library.GGML_BASE,
        "ggml_opt_result_ndata",
        [ggml_opt_result_t, ctypes.POINTER(ctypes.c_int64)],
    ),
    # Computation. prepare_alloc -> alloc -> eval is the per-ubatch cycle S1-02 forks.
    Symbol(
        Library.GGML_BASE,
        "ggml_opt_prepare_alloc",
        [ggml_opt_context_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
    ),
    Symbol(Library.GGML_BASE, "ggml_opt_alloc", [ggml_opt_context_t, ctypes.c_bool]),
    Symbol(Library.GGML_BASE, "ggml_opt_eval", [ggml_opt_context_t, ggml_opt_result_t]),
]
