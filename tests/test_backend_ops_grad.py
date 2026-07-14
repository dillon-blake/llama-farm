"""Run the vendored ``test-backend-ops`` MODE_GRAD cases for every op this project touched.

The op list is :mod:`tests.project_ops` — one registry, read by both this test and CI, so the two
cannot drift apart again. (They had: CI's inline list still named eight ops when the project had
thirteen.)

The interesting test here is the second one. ``grad -o <op>`` printing ``OK`` is **not** evidence
that the op has a backward — MODE_GRAD only checks a gradient if the test class called
``ggml_set_param``, and otherwise reports success having compared nothing.
:func:`test_every_op_reports_real_cases` is what stops an op joining the registry on the strength
of a vacuous pass.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

import pytest

from .project_ops import PROJECT_ADDED_OPS

# Where S0-07's CI puts it, and where docs/dev/backward-coverage.md says to build it.
_BUILD_DIRS = ("build/vendor-tests/bin", "build/vendor-tests")

_GGML_DEVICE = {"cpu": "CPU", "metal": "Metal", "cuda": "CUDA", "vulkan": "Vulkan"}


def _binary() -> pathlib.Path:
    if override := os.environ.get("TEST_BACKEND_OPS"):
        return pathlib.Path(override)

    root = pathlib.Path(__file__).resolve().parents[1]
    for d in _BUILD_DIRS:
        candidate = root / d / "test-backend-ops"
        if candidate.exists():
            return candidate

    if found := shutil.which("test-backend-ops"):
        return pathlib.Path(found)

    pytest.skip(
        "test-backend-ops is not built. Build it with:\n"
        "  cmake -S vendor/llama.cpp -B build/vendor-tests -DCMAKE_BUILD_TYPE=Release "
        "-DLLAMA_BUILD_TESTS=ON\n"
        "  cmake --build build/vendor-tests --target test-backend-ops -j2\n"
        "or point TEST_BACKEND_OPS at it."
    )
    raise AssertionError("unreachable")


@pytest.mark.parametrize("op", PROJECT_ADDED_OPS)
def test_the_gradient_of_a_project_op_matches_finite_differences(op: str, device: str) -> None:
    """``test-backend-ops grad -o <op>``, under ADR-0002's tolerances (the harness's own).

    Failure prints the harness's output, which names the failing case and its ``MAA``.
    """
    result = subprocess.run(
        [str(_binary()), "grad", "-b", _GGML_DEVICE[device], "-o", op],
        capture_output=True,
        text=True,
        timeout=900,
    )

    if result.returncode != 0:
        failures = "\n".join(
            line for line in result.stdout.splitlines() if "FAIL" in line or "MAA" in line
        )
        pytest.fail(f"grad -o {op} failed on {device}:\n{failures or result.stdout[-3000:]}")


def test_every_op_reports_real_cases(device: str) -> None:
    """The vacuity guard: an op that checks *nothing* must not sit here reporting OK.

    ``grad -o MUL_MAT_ID`` printed ``OK — 16829 tests passed`` for months while ``MUL_MAT_ID`` had
    no backward at all, because ``test_mul_mat_id`` never called ``ggml_set_param``. The count is
    not the verdict. So this asserts each op produces at least one case that actually ran — not
    ``not supported``, not ``skipping large tensors``.
    """
    empty = []
    for op in PROJECT_ADDED_OPS:
        result = subprocess.run(
            [str(_binary()), "grad", "-b", _GGML_DEVICE[device], "-o", op],
            capture_output=True,
            text=True,
            timeout=900,
        )
        ran = [
            line
            for line in result.stdout.splitlines()
            if "OK" in line and "not supported" not in line and "skipping" not in line
        ]
        if not ran:
            empty.append(op)

    assert not empty, (
        f"these ops report no real gradient cases at all: {empty}. Either the test class does not "
        "call ggml_set_param (so MODE_GRAD checks nothing and still prints OK), or every case is "
        "above grad_nmax() and silently skipped. See docs/dev/backward-coverage.md."
    )
