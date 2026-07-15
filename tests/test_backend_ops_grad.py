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

# Where S0-07's CI puts it, and where docs/dev/backward-coverage.md says to build it. The
# `bin/Release` entry and the `.exe` name are for Windows (S1-43): MSVC's multi-config generators
# nest the binary under the build type (`bin/Release/test-backend-ops.exe`), where single-config
# generators (Ninja, Make) drop it straight in `bin/`. CI points TEST_BACKEND_OPS at the exact
# path, so this fallback only matters for a bare local run — but a Windows dev deserves one too.
_BUILD_DIRS = ("build/vendor-tests/bin", "build/vendor-tests/bin/Release", "build/vendor-tests")
_EXE_NAMES = ("test-backend-ops", "test-backend-ops.exe")

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# `  OP(vars): result`. The op name on the gradient line is not necessarily the filtered op: once
# MODE_GRAD wraps the output in its objective, it prints the *loss node's* name (SUM).
_CASE = re.compile(r"^\s*[A-Z_0-9]+\((.*)\):\s*(.*?)\s*$")


def _discover_in(root: pathlib.Path) -> pathlib.Path | None:
    """The built binary under ``root``, across generators and platforms.

    Single-config generators (Ninja, Make) drop it in ``bin/``; MSVC's multi-config generators nest
    it under the build type (``bin/Release/``), and Windows suffixes it ``.exe``. So the search is
    the cross product of :data:`_BUILD_DIRS` and :data:`_EXE_NAMES`, not the one Linux path.
    """
    for d in _BUILD_DIRS:
        for name in _EXE_NAMES:
            candidate = root / d / name
            if candidate.exists():
                return candidate
    return None


def _binary() -> pathlib.Path:
    if override := os.environ.get("TEST_BACKEND_OPS"):
        return pathlib.Path(override)

    root = pathlib.Path(__file__).resolve().parents[1]
    if found := _discover_in(root):
        return found

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


def _run(op: str, ggml_device: str) -> subprocess.CompletedProcess[str]:
    """``ggml_device`` must be the name **ggml registered** (``CUDA0``, not ``CUDA``).

    ``-b`` is an exact ``strcmp`` against ``ggml_backend_dev_name``
    (``test-backend-ops.cpp:11214``). A name that matches nothing makes the harness print
    ``Skipping``, count it as passed, and **exit 0** having run zero cases.
    """
    return subprocess.run(
        [str(_binary()), "grad", "-b", ggml_device, "-o", op],
        capture_output=True,
        text=True,
        timeout=900,
    )


def _bailed(tail: str) -> bool:
    """A verdict that compared no gradient."""
    return tail.startswith(("not supported", "skipping"))


def _n_gradients_compared(stdout: str) -> int:
    """How many cases MODE_GRAD actually compared a gradient for.

    The output has to be walked **in order**, and the reason is a trap worth naming. Per case,
    ``eval_grad`` prints:

    * a non-F32 *output* bails at ``test-backend-ops.cpp:1746`` and prints **one** line;
    * every other case prints an **info line first** (``:1754``) — which ``print_operation``
      renders as a bare ``OK``, *before a single thing has been checked* — and then exactly one
      real verdict.

    So the count is **not** two lines per case, and "take every second line" is wrong. ``CLAMP``
    emits **nine** lines today — an odd number, which is definitionally impossible under that
    assumption — and the mis-alignment makes support lines get read as gradient verdicts. An
    earlier version of this file did exactly that.

    Walking in order is unambiguous: a leading ``not supported``/``skipping`` is a one-line bail;
    anything else is an info line whose verdict is the line after it.
    """
    tails = [m.group(2) for line in stdout.splitlines() if (m := _CASE.match(_ANSI.sub("", line)))]

    compared = 0
    i = 0
    while i < len(tails):
        if _bailed(tails[i]):
            i += 1  # a one-line bail: no info line was printed for this case
            continue
        # tails[i] is the info line. Its verdict is the next line.
        verdict = tails[i + 1] if i + 1 < len(tails) else ""
        if not _bailed(verdict):
            compared += 1
        i += 2

    return compared


@pytest.mark.parametrize("op", PROJECT_ADDED_OPS)
def test_the_gradient_of_a_project_op_is_checked_and_correct(op: str, ggml_device: str) -> None:
    """For each project op: MODE_GRAD must actually compare a gradient, **and** it must pass.

    Both halves matter, and the first is the one that bites. An op whose cases all bail out — no
    ``ggml_set_param``, every case above ``grad_nmax()``, or a name that matches nothing — exits 0
    and looks green forever.
    """
    result = _run(op, ggml_device)

    compared = _n_gradients_compared(result.stdout)
    assert compared > 0, (
        f"`grad -o {op}` compared ZERO gradients on {ggml_device} — it exited "
        f"{result.returncode} having checked nothing. Either the test class never calls "
        f"ggml_set_param, or every case is above grad_nmax() (10000), or `{op}` is not what "
        f"ggml_op_desc() calls this op so the filter matched nothing at all (a GLU node reports "
        f"SWIGLU/GEGLU/..., never 'GLU'). See docs/dev/backward-coverage.md."
    )

    if result.returncode != 0:
        failures = "\n".join(
            line for line in result.stdout.splitlines() if "FAIL" in line or "MAA" in line
        )
        pytest.fail(f"grad -o {op} failed on {ggml_device}:\n{failures or result.stdout[-3000:]}")


def test_the_vacuity_guard_can_actually_detect_vacuity(ggml_device: str) -> None:
    """The guard above is only worth having if it can fail. So point it at ops that check nothing.

    ``OUT_PROD`` and ``FLASH_ATTN_EXT`` both report ``Backend CPU: OK`` and exit 0 in MODE_GRAD
    while comparing **not one gradient** — ``test_out_prod`` and ``test_flash_attn_ext`` never call
    ``ggml_set_param`` (and ``ggml_flash_attn_back``'s first statement is a ``GGML_ABORT``). If the
    guard ever stops flagging them, it has stopped working, and every op in the registry is being
    taken on trust.

    This is the test that would have caught ``GLU`` sitting in the registry checking nothing.
    """
    for vacuous in ("OUT_PROD", "FLASH_ATTN_EXT"):
        result = _run(vacuous, ggml_device)
        assert result.returncode == 0, f"`grad -o {vacuous}` was expected to exit 0 and be useless"
        assert _n_gradients_compared(result.stdout) == 0, (
            f"`grad -o {vacuous}` now compares real gradients. Either upstream added "
            f"ggml_set_param to its test class — good news, update this test — or the vacuity "
            f"guard is broken and every op in PROJECT_ADDED_OPS is being taken on trust."
        )


def test_the_harness_runs_on_the_device_we_asked_for(ggml_device: str) -> None:
    """``-b`` is an exact ``strcmp``, and the device names are not what you would guess.

    ``ggml_backend_dev_name`` returns ``MTL0`` (not ``Metal``), ``CUDA0`` / ``ROCm0`` / ``MUSA0``
    (not ``CUDA``), and ``Vulkan0`` (not ``Vulkan``) — ``ggml-metal-device.m:858``,
    ``ggml-cuda.cu:5178``, ``ggml-vulkan.cpp:6459``, and the filter at
    ``test-backend-ops.cpp:11214``. A name that matches no device makes the harness print
    ``Skipping``, **count it as passed, and exit 0** having run nothing at all. Demonstrated:
    ``test -b NOPE0 -o RMS_NORM`` prints ``1/1 backends passed``, ``OK``, and returns 0.

    **Assert the positive.** The obvious guard — "``Skipping`` must not appear" — is WRONG, and it
    failed in CI on the very first macOS runner it met: that box has *two* devices, so ``-b CPU``
    skips the **Metal** one entirely legitimately and says so. The guard fired on a perfectly
    healthy run. What actually matters is that the device we asked for *did something*.
    """
    result = _run(PROJECT_ADDED_OPS[0], ggml_device)

    assert _n_gradients_compared(result.stdout) > 0, (
        f"test-backend-ops compared no gradients on {ggml_device!r}. If it printed "
        f"'Skipping {ggml_device}', the name matched no device and the run is worthless: -b takes "
        f"ggml_backend_dev_name()'s exact string (MTL0, not Metal; CUDA0, not CUDA)."
    )


def test_the_binary_override_is_respected_verbatim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The override is returned untouched — ``.exe``, ``bin/Release/`` and all.

    CI hands ``_binary()`` the exact built path, so it must not re-derive a layout of its own.
    This is precisely what lets the Windows lane (S1-43) point ``TEST_BACKEND_OPS`` at
    ``build/vendor-tests/bin/Release/test-backend-ops.exe`` without ``_binary()`` needing to know a
    thing about MSVC's multi-config output tree.
    """
    exe = tmp_path / "anywhere" / "test-backend-ops.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    monkeypatch.setenv("TEST_BACKEND_OPS", str(exe))

    assert _binary() == exe


def test_the_windows_multiconfig_binary_is_discovered(tmp_path: pathlib.Path) -> None:
    """With no override, discovery must still find MSVC's ``bin/Release/test-backend-ops.exe``.

    A local ``cmake --build --config Release`` on Windows produces exactly that layout — nested
    under the build type and suffixed ``.exe`` — where the Linux build produces ``bin/test-backend-
    ops``. Before S1-43 the fallback knew only the Linux path, so a Windows dev running the wrapper
    bare would have been told the binary "is not built" while it sat right there.
    """
    exe = tmp_path / "build" / "vendor-tests" / "bin" / "Release" / "test-backend-ops.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("")

    assert _discover_in(tmp_path) == exe
