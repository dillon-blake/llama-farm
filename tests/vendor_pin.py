"""The vendored llama.cpp commit, read from the submodule rather than hardcoded.

The version lock has three parties that must agree:

1. the **submodule** gitlink — what is actually checked out;
2. the **native build** — ``ll_probe()``, baked in at CMake configure time;
3. the **Python package** — ``_ffi._version_lock.VENDORED_COMMIT``, generated at the same time.

A test that hardcodes a literal commit only checks (2) against (3), and has to be edited by hand
on every vendor bump — which is exactly the moment you least want a test that people are used to
editing. Reading (1) from git instead makes the tests verify all three against the one thing that
is unambiguously true, and a bump needs no test edits at all.
"""

from __future__ import annotations

import functools
import pathlib
import subprocess

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# The upstream commit learning-llamas-base was branched from. Every `file:line` anchor cited in
# the tickets and docs is valid at THIS commit. The pin below advances past it as fork commits
# land (ADR-0001); those commits only add, so the anchors stay valid.
UPSTREAM_BASE_COMMIT = "4f37f519722aa3242eecb7649466b4a4a2d6d6da"


@functools.cache
def vendored_commit() -> str:
    """Return the commit ``vendor/llama.cpp`` is actually checked out at."""
    result = subprocess.run(
        ["git", "-C", "vendor/llama.cpp", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()
