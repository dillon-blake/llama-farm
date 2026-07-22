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
``tests/test_backend_ops_grad.py::test_the_gradient_of_a_project_op_is_checked_and_correct``
enforces that (and ``::test_the_vacuity_guard_can_actually_detect_vacuity`` mutation-checks the
enforcement), so an op cannot be added here on the strength of a vacuous pass.

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
    # S1-28 — the GLU family. SwiGLU is the FFN.
    #
    # NOT "GLU". `-o` filters on ggml_op_desc(out), and for a GGML_OP_GLU node that returns the
    # *variant* name, not "GLU" (ggml.c:1398-1400 — same as GGML_OP_UNARY returning "SILU"). So
    # `grad -o GLU` matches ZERO cases: it runs nothing, prints "Backend CPU: OK", and exits 0.
    # It sat in this list checking nothing until the grad-check wrapper was fixed to parse the
    # GRADIENT verdict rather than the support line.
    "SWIGLU",
    "GEGLU",
    "REGLU",
    "GEGLU_ERF",
    "GEGLU_QUICK",
    "SWIGLU_OAI",
    # S1-30 / S1-31 — Mamba.
    "SSM_CONV",
    "SSM_SCAN",
)
