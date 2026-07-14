"""The ops this project added a backward for, fixed a backward for, or made grad-checkable.

**This is the single registry.** The list used to live inline in `.github/workflows/ci-cpu.yml`,
where it went stale the moment a kernel ticket landed: by S1-31 it still read
``CROSS_ENTROPY_LOSS,...,CONCAT`` and knew nothing about MUL_MAT_ID, ADD_ID, GLU, SSM_CONV or
SSM_SCAN — every op the MoE and Mamba work added. CI was green while checking a third of what it
claimed to.

Deliberately import-free, so CI can read it without building the package:

    python -c "import sys; sys.path.insert(0, 'tests'); \
               from project_ops import PROJECT_ADDED_OPS; print(','.join(PROJECT_ADDED_OPS))"

**A green ``grad -o <op>`` is not evidence that the op has a backward.** MODE_GRAD only checks a
gradient if the test class calls ``ggml_set_param``; otherwise it prints ``OK`` having compared
nothing at all. Every op below was confirmed to produce *real* cases —
``tests/test_backend_ops_grad.py::test_every_op_reports_real_cases`` enforces that, so an op cannot
be added here on the strength of a vacuous pass.

What is deliberately **not** here, and why, is in ``docs/dev/backward-coverage.md``.
"""

from __future__ import annotations

PROJECT_ADDED_OPS: tuple[str, ...] = (
    # S1-04 — the sparse CE the entire loss is built on, and the dense one it replaced.
    "CROSS_ENTROPY_LOSS",
    "CROSS_ENTROPY_LOSS_SPARSE",
    # S0-09 — every norm in the model.
    "RMS_NORM",
    # S1-19 — small VJPs. (ggml_clamp itself was broken: it returned a view, so any clamp whose
    # gradient was requested aborted.)
    "TANH",
    "SIGMOID",
    "CLAMP",
    # S1-20 (ALiBi's bogus max_bias assert) and S1-34 (attention sinks — and the discovery that
    # sum(out) makes a softmax's grad check compare zero against zero).
    "SOFT_MAX",
    # S1-29.
    "CONCAT",
    # S1-25 — MoE routing. The LoRA A/B tensors *are* the 3D expert operand, which is why
    # MUL_MAT_ID needed a backward at all.
    "MUL_MAT_ID",
    "ADD_ID",
    # S1-28 — all six GLU variants. SwiGLU is the FFN.
    "GLU",
    # S1-30 / S1-31 — Mamba.
    "SSM_CONV",
    "SSM_SCAN",
)
