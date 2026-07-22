"""Guards on the CI workflows and the packaging config.

These are the checks nothing else can make: a workflow is only "run" by pushing to GitHub, so a
false-green step or a self-cancelling concurrency group is invisible until the day it costs you a
regression. Most of what is here is a real 2026-07-22 audit failure and fails again if the fix is
reverted; ``test_archive_artifacts_are_not_in_the_packaged_component`` is the exception and says so
in its own comment — that finding was rejected, and the test guards a property that already held.

Deliberately parsed as TEXT, not with PyYAML: the ci-cpu `test` job installs
`scikit-build-core cmake ninja pytest` and nothing else, so a test that needs yaml would be a test
that silently does not run in CI.

This file runs on the ci-windows lane too (that lane's pytest invocation collects all of `tests/`,
and nothing here is marked slow), so it may not assume a POSIX shell or a UTF-8 locale: see
:func:`_read` and :func:`_bash`, both of which exist for that reason and for no other.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
LANES = ["ci-cpu.yml", "ci-windows.yml", "ci-metal.yml"]


def _read(path: pathlib.Path) -> str:
    """Read a repo file as UTF-8, explicitly, because ci-windows runs this file too.

    ``Path.read_text()`` with no encoding is ``locale.getpreferredencoding(False)``, which on
    GitHub's ``windows-latest`` is cp1252 — and ``ci-metal.yml`` contains ``⚠️`` (U+26A0 U+FE0F).
    Its third UTF-8 byte, 0x8F, is undefined in cp1252, so the bare call raises UnicodeDecodeError
    at byte 6559 and takes the only automated coverage of the concurrency group and the pinned seed
    down with it. Verified by decoding all three lanes as cp1252: the two ASCII ones survive (as
    mojibake), ci-metal.yml does not.
    """
    return path.read_text(encoding="utf-8")


def _bash(*args: str, **kwargs) -> subprocess.CompletedProcess:  # noqa: ANN003
    """Run a repo script through ``bash`` by name, never as ``argv[0]``.

    Windows ``subprocess`` goes through ``CreateProcess``, which executes only ``.exe``/``.com``
    (and ``.bat``/``.cmd`` via cmd). It does not read shebangs and does not consult file
    associations, so ``["./scripts/apply-patches.sh"]`` raises OSError [WinError 193] on the
    ci-windows runner — unconditionally, for every PR. ``bash`` is present on that runner, but only
    if it is the thing being executed.
    """
    return subprocess.run(["bash", *args], cwd=REPO_ROOT, **kwargs)


def _steps(text: str) -> list[str]:
    """Split a workflow into step blocks, keyed on the 6-space `- name:` every step uses."""
    parts = re.split(r"\n      - name:", text)
    return ["- name:" + p for p in parts[1:]]


def _run_body(step: str) -> str:
    """The commands a step actually runs, comments dropped.

    Matching on the raw block would match the workflow's own prose: these files explain themselves
    at length, and several comments name `tests/test_backend_ops_grad.py` without running it.
    """
    if "run:" not in step:
        return ""
    body = step.split("run:", 1)[1]
    return "\n".join(ln for ln in body.splitlines() if not ln.strip().startswith("#"))


# ---------------------------------------------------------------------------------------------
# The nightly must not be cancellable by a push (audit G4).
#
# `schedule` and `push` to main both resolve to refs/heads/main, so a group keyed on the ref alone
# put them in the SAME group with cancel-in-progress: true -- a merge landing inside the nightly
# window silently killed the only run of the slow suite, the full sweep and the MUL_MAT grad.
# ---------------------------------------------------------------------------------------------
def test_concurrency_group_separates_scheduled_runs_from_pushes():
    for lane in LANES:
        text = _read(WORKFLOWS / lane)
        group = re.search(r"^concurrency:\n(?:.*\n)*?\s*group:(.*)$", text, re.MULTILINE)
        assert group, f"{lane}: no concurrency group"
        assert "github.event_name" in group.group(1), (
            f"{lane}: the concurrency group is not keyed on the event, so a push to main and the "
            f"nightly share a group and cancel each other: {group.group(1).strip()}"
        )


# ---------------------------------------------------------------------------------------------
# Every MODE_GRAD lane pins the draw (audit G6). An unseeded run makes a marginal SSM_SCAN FD
# failure unreproducible, which is the one thing you need when it fires.
# ---------------------------------------------------------------------------------------------
def test_every_grad_step_pins_the_seed():
    for lane in LANES:
        for step in _steps(_read(WORKFLOWS / lane)):
            if "test_backend_ops_grad.py" not in _run_body(step):
                continue
            name = step.splitlines()[0]
            assert "GGML_TEST_SEED" in step, f"{lane}: grad step without a pinned seed: {name}"


# ---------------------------------------------------------------------------------------------
# The Windows wheel check must be able to fail (audit G1).
#
# Under pwsh only the LAST native command's exit code propagates, and the final import was
# satisfiable by the editable install the previous step left behind -- so the step certified a
# wheel it had never loaded.
# ---------------------------------------------------------------------------------------------
def test_windows_wheel_step_cannot_be_satisfied_by_the_editable_install():
    steps = _steps(_read(WORKFLOWS / "ci-windows.yml"))
    wheel = [s for s in steps if "python -m build --wheel" in _run_body(s)]
    assert len(wheel) == 1, "ci-windows: expected exactly one wheel-building step"
    step = wheel[0]
    assert "shell: bash" in step, (
        "ci-windows: the wheel step runs under pwsh, where a failing pip install is invisible "
        "because only the last native command's exit code reaches the runner"
    )
    assert re.search(r"pip uninstall -y learning[-_]llamas", _run_body(step)), (
        "ci-windows: the wheel step must uninstall the editable install first, or the final "
        "import resolves out of src/ and passes whatever the wheel contains"
    )


def test_windows_multi_command_steps_do_not_swallow_failures():
    text = _read(WORKFLOWS / "ci-windows.yml")
    # Job blocks, because bash can be set once as a job default (`ops` does) or per step.
    jobs = re.split(r"\n  (?=[A-Za-z_][\w-]*:\n)", text.split("\njobs:\n", 1)[1])
    for job in jobs:
        job_default_bash = re.search(r"defaults:\n\s+run:\n\s+shell: bash", job) is not None
        for step in _steps("\n" + job):
            if "run: |" not in step:
                continue
            commands = [ln for ln in _run_body(step).splitlines() if ln.strip()]
            if len(commands) < 2:
                continue
            name = step.splitlines()[0]
            assert job_default_bash or "shell: bash" in step, (
                f"ci-windows: multi-command step under pwsh, where only the last exit code "
                f"propagates: {name}"
            )


# ---------------------------------------------------------------------------------------------
# CI verifies the patch queue, it does not apply it (audit G5). Applying in the wheel-building job
# alone would ship a vendor tree no other job tested, under a commit hash that exists on no remote.
# ---------------------------------------------------------------------------------------------
def test_ci_only_verifies_the_patch_queue():
    text = _read(WORKFLOWS / "ci-cpu.yml")
    calls = re.findall(r"^\s*run: (\./scripts/apply-patches\.sh.*)$", text, re.MULTILINE)
    assert calls, "ci-cpu no longer runs the patch-queue script at all"
    for call in calls:
        assert "--check" in call, f"ci-cpu APPLIES the patch queue rather than verifying it: {call}"


# The two check-mode tests below drive a real `git worktree` inside the submodule, so they need it
# to be a checkout. It always is in CI (every lane checks out with submodules: recursive) and in a
# dev tree; an unpacked sdist has no .git anywhere and there is nothing there to verify.
needs_vendor_checkout = pytest.mark.skipif(
    not (REPO_ROOT / "vendor" / "llama.cpp" / ".git").exists(),
    reason="vendor/llama.cpp is not a git checkout (sdist build): --check has nothing to verify",
)


def _vendor_head() -> str:
    return subprocess.run(
        ["git", "-C", "vendor/llama.cpp", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _vendor_worktrees() -> list[str]:
    out = subprocess.run(
        ["git", "-C", "vendor/llama.cpp", "worktree", "list", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [ln.split(" ", 1)[1] for ln in out.splitlines() if ln.startswith("worktree ")]


def test_apply_patches_rejects_unknown_arguments_on_an_empty_queue():
    """The argument parser, and nothing else — which is all an empty queue can reach.

    `patches/` holds only a README in steady state, so the script returns at its emptiness check
    before check mode has a worktree to build. A "--check changed nothing" assertion written
    against the real queue is therefore true for the uninteresting reason: nothing ran, and it is
    equally true of a script that ignores its arguments. The mechanism is exercised by
    `test_apply_patches_check_mode_never_touches_the_submodule` below, against a synthesized queue.
    """
    assert _bash("scripts/apply-patches.sh", "--check", capture_output=True).returncode == 0
    # An unknown flag must not be silently treated as "apply".
    assert _bash("scripts/apply-patches.sh", "--nonsense", capture_output=True).returncode == 2
    # ...and neither must a --patch-dir with nothing after it.
    assert _bash("scripts/apply-patches.sh", "--patch-dir", capture_output=True).returncode == 2


def _format_patch(tmp_path: pathlib.Path, filename: str, body: str) -> pathlib.Path:
    """A genuine `git format-patch` file that adds ``filename``, built in a throwaway repo.

    Hand-written patches are not equivalent: `git am` reads the `index` line, and a wrong blob hash
    changes which code path it takes. Generating it with git means the queue under test is the same
    kind of object `patches/` holds.
    """
    src = tmp_path / "patch-src"
    src.mkdir()
    git = ["git", "-C", str(src), "-c", "user.name=t", "-c", "user.email=t@invalid"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "base"], check=True)
    (src / filename).write_text(body, encoding="utf-8")
    subprocess.run([*git, "add", filename], check=True)
    subprocess.run([*git, "commit", "-q", "-m", f"probe: add {filename}"], check=True)

    queue = tmp_path / "queue"
    queue.mkdir()
    subprocess.run([*git, "format-patch", "-1", "-o", str(queue), "--quiet"], check=True)
    return queue


@needs_vendor_checkout
def test_apply_patches_check_mode_never_touches_the_submodule(tmp_path):
    """The four properties G5's rewrite actually claims, against a queue that is not empty.

    A synthesized one-patch queue adds a file `vendor/llama.cpp` does not have, so check mode has
    to build the throwaway worktree, run the same sequential `git am` loop in it, and tear it down.
    What is pinned: it exits 0, the submodule HEAD does not move, the patch's file never appears in
    the submodule's working tree, and no worktree entry outlives the run.
    """
    queue = _format_patch(tmp_path, "ll-patch-queue-probe.txt", "probe\n")
    probe = REPO_ROOT / "vendor" / "llama.cpp" / "ll-patch-queue-probe.txt"

    before, worktrees_before = _vendor_head(), _vendor_worktrees()
    done = _bash(
        "scripts/apply-patches.sh", "--check", "--patch-dir", str(queue), capture_output=True
    )

    assert done.returncode == 0, done.stderr.decode()
    assert _vendor_head() == before, "--check moved the submodule HEAD; it must change nothing"
    assert not probe.exists(), "--check applied the patch to the submodule instead of a worktree"
    assert _vendor_worktrees() == worktrees_before, "--check left its throwaway worktree behind"


@needs_vendor_checkout
def test_apply_patches_check_mode_fails_on_a_stale_patch_and_still_cleans_up(tmp_path):
    """The other half: a queue that does NOT apply has to be loud, and still leave no trace.

    A `--check` that cannot distinguish these two cases is the one CI failure mode that matters
    here — the whole point of running it is to learn that the queue went stale.
    """
    queue = _format_patch(tmp_path, "ll-patch-queue-probe.txt", "probe\n")
    # Retarget the patch at a file that DOES exist, with context that does not match it.
    patch = next(queue.glob("*.patch"))
    patch.write_text(
        patch.read_text(encoding="utf-8").replace("ll-patch-queue-probe.txt", "README.md"),
        encoding="utf-8",
    )

    before, worktrees_before = _vendor_head(), _vendor_worktrees()
    done = _bash(
        "scripts/apply-patches.sh", "--check", "--patch-dir", str(queue), capture_output=True
    )

    assert done.returncode == 1, done.stdout.decode() + done.stderr.decode()
    assert b"does not apply cleanly" in done.stderr
    assert _vendor_head() == before
    assert _vendor_worktrees() == worktrees_before, "the failure path leaked a worktree"


# ---------------------------------------------------------------------------------------------
# Import libraries stay out of the wheel (audit G2, REJECTED as a defect and kept as a guard).
#
# The audit claimed a blanket leading COMPONENT dragged the ARCHIVE kind into the wheel. It did
# not: `cmake --help-command install` says that on DLL platforms, specifying a RUNTIME destination
# and no ARCHIVE destination installs the RUNTIME component and NOT the ARCHIVE one -- which is
# exactly what the pre-audit rule did. Nothing leaked, and no wheel was ever inspected showing one.
#
# The rule now names ARCHIVE explicitly anyway, and this test is what makes that explicit form
# load-bearing rather than decorative: it pins the .libs to a component the wheel does not install
# (pyproject: install.components), so the guarantee no longer depends on a defaulting rule that
# only holds while every target is shared and a RUNTIME destination is present.
# ---------------------------------------------------------------------------------------------
def test_archive_artifacts_are_not_in_the_packaged_component():
    text = _read(REPO_ROOT / "CMakeLists.txt")
    block = re.search(r"install\(TARGETS \$\{LL_VENDOR_LIBS\}[^)]*\)", text)
    assert block, "the wheel install(TARGETS) rule moved"
    body = block.group(0)
    archive = re.search(r"ARCHIVE[^\n]*(?:\n\s+(?!LIBRARY|RUNTIME|ARCHIVE)[^\n]*)*", body)
    assert archive, (
        "install(TARGETS) no longer names the ARCHIVE kind. CMake's own defaulting still keeps the "
        "import libraries out of the wheel while every target is shared and a RUNTIME destination "
        "is named -- but that is two conditions held in someone's head instead of one line here"
    )
    assert "COMPONENT learningllamas\n" not in archive.group(0), (
        "the import libraries are in the packaged component"
    )
