"""Run the vendored ``test-backend-ops`` MODE_GRAD cases for every op this project touched.

The op list is :mod:`tests.project_ops` — one registry, read by both this test and CI, so the two
cannot drift apart again.

**The exit code is not the verdict, and neither is the word OK.** This file exists because
``grad -o <op>`` reports success in at least three ways that mean nothing:

1. **The test class never called ``ggml_set_param``.** MODE_GRAD then checks no gradient at all and
   still prints ``Backend CPU: OK``. This is how ``MUL_MAT_ID`` reported *"OK — 16829 tests
   passed"* for months while having no backward whatsoever.
2. **Every case sits above ``grad_nmax()`` (10000)** and is silently skipped.
3. **The op name matches nothing.** ``-o`` filters on ``ggml_op_desc(out)``, which for a
   ``GGML_OP_GLU`` node returns the *variant* name — ``SWIGLU``, ``GEGLU``, … — not ``GLU``
   (``ggml.c:1398-1400``, exactly as ``GGML_OP_UNARY`` returns ``SILU``). So ``grad -o GLU``
   matches **zero cases**, runs nothing, prints ``OK``, and exits 0. That is not hypothetical: it
   is what this file did on the day it was written.

So the verdict is parsed, not assumed. The harness prints **two lines per case** — first whether
the backend *supports* the op, then the *gradient* verdict:

    OUT_PROD(...): OK                        <- support. Says nothing about gradients.
    OUT_PROD(...): not supported [OUT_PROD]  <- the gradient verdict: no params, nothing checked.

Reading the first line is what made the original guard vacuous. :func:`_grad_verdicts` reads the
second.
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess

import pytest

from .project_ops import PROJECT_ADDED_OPS

# Where S0-07's CI puts it, and where docs/dev/backward-coverage.md says to build it.
_BUILD_DIRS = ("build/vendor-tests/bin", "build/vendor-tests")

_GGML_DEVICE = {"cpu": "CPU", "metal": "Metal", "cuda": "CUDA", "vulkan": "Vulkan"}

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# `  OP(vars): result`. The op name on the gradient line is not necessarily the filtered op: once
# MODE_GRAD wraps the output in its objective, it prints the *loss node's* name (SUM).
_CASE = re.compile(r"^\s*[A-Z_0-9]+\((.*)\):\s*(.*?)\s*$")


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
    raise AssertionError("unreachable")  # pragma: no cover - pytest.skip raises


def _run(op: str, device: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(_binary()), "grad", "-b", _GGML_DEVICE[device], "-o", op],
        capture_output=True,
        text=True,
        timeout=900,
    )


def _grad_verdicts(stdout: str) -> list[str]:
    """The **gradient** verdict of each case — the second of its two lines.

    The two lines are emitted back to back, so the verdicts are simply the odd-indexed ones. Pairing
    them by their ``vars`` string instead looks more careful and is *wrong*: two distinct cases can
    stringify to the same vars, which mis-pairs the run and lets a support line be counted as a
    gradient verdict. That mistake alone made a fully-vacuous ``OUT_PROD`` report one "real" case.
    """
    lines = [m.group(2) for line in stdout.splitlines() if (m := _CASE.match(_ANSI.sub("", line)))]
    return lines[1::2]


def _checked(verdicts: list[str]) -> list[str]:
    """Verdicts from cases that actually compared a gradient."""
    return [v for v in verdicts if not v.startswith(("not supported", "skipping"))]


@pytest.mark.parametrize("op", PROJECT_ADDED_OPS)
def test_the_gradient_of_a_project_op_is_checked_and_correct(op: str, device: str) -> None:
    """For each project op: MODE_GRAD must actually check it, **and** it must pass.

    Both halves matter, and the first is the one that bites. An op whose cases all bail out — no
    ``ggml_set_param``, or a name that matches nothing — exits 0 and looks green forever.
    """
    result = _run(op, device)

    checked = _checked(_grad_verdicts(result.stdout))
    assert checked, (
        f"`grad -o {op}` compared ZERO gradients — it exited {result.returncode} having checked "
        f"nothing. Either the test class never calls ggml_set_param, or every case is above "
        f"grad_nmax() (10000), or `{op}` is not what ggml_op_desc() calls this op and the filter "
        f"matched nothing at all (GLU's nodes report SWIGLU/GEGLU/...). "
        f"See docs/dev/backward-coverage.md."
    )

    if result.returncode != 0:
        failures = "\n".join(
            line for line in result.stdout.splitlines() if "FAIL" in line or "MAA" in line
        )
        pytest.fail(f"grad -o {op} failed on {device}:\n{failures or result.stdout[-3000:]}")


def test_the_vacuity_guard_can_actually_detect_vacuity(device: str) -> None:
    """The guard above is only worth having if it can fail. So point it at ops that check nothing.

    ``OUT_PROD`` and ``FLASH_ATTN_EXT`` both report ``Backend CPU: OK`` and exit 0 in MODE_GRAD
    while comparing **not one gradient** — ``test_out_prod`` and ``test_flash_attn_ext`` never call
    ``ggml_set_param`` (and ``ggml_flash_attn_back``'s first statement is a ``GGML_ABORT``). If the
    guard ever stops flagging them, it has stopped working, and every op in the registry is
    unverified.

    This is the test that would have caught ``GLU`` sitting in the registry checking nothing.
    """
    for vacuous in ("OUT_PROD", "FLASH_ATTN_EXT"):
        verdicts = _grad_verdicts(_run(vacuous, device).stdout)
        assert verdicts, f"expected `grad -o {vacuous}` to emit cases at all"
        assert not _checked(verdicts), (
            f"`grad -o {vacuous}` now reports genuinely-checked gradients. Either upstream added "
            f"ggml_set_param to its test class — good news, update this test — or the vacuity "
            f"guard is broken and every op in PROJECT_ADDED_OPS is being taken on trust."
        )
