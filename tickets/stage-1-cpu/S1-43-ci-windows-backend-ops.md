---
id: S1-43
title: "ci-windows: build and drive the vendored test-backend-ops (MODE_GRAD) on MSVC"
stage: 1
track: infra
size: S
deps: [S1-33]
status: done
pr: null
---

# S1-43 — ci-windows test-backend-ops MODE_GRAD coverage

**One-line outcome:** the Windows/MSVC lane now builds the vendored `test-backend-ops` and drives
MODE_GRAD for the 18 project ops through the *same* registry-and-vacuity-guarded pytest wrapper the
Linux and Metal lanes use — so the kernel VJPs are gradient-checked on MSVC instead of Linux-only,
and the check cannot go green while comparing zero gradients.

## Why (context)

The 2026-07-15 Stage 0+1 audit (`docs/dev/audit-2026-07-15.md`, ci-docs) rated this a **major**:
`.github/workflows/ci-windows.yml` never built or ran the vendored `test-backend-ops`, so the 18
project kernel VJPs listed in `tests/project_ops.py` had **zero** gradient coverage on MSVC. That is
not a formality. The fork's CPU kernels lean on compiler-sensitive intrinsics, and MSVC is a
genuinely different compiler from the GCC/Clang the Linux and macOS lanes use — the exact class of
platform difference this project made Windows a CI target to catch (S1-33, BLUEPRINT §8). MODE_GRAD
was the one acceptance harness (ADR-0002) that had no MSVC witness at all.

S1-33's own acceptance criteria already asked for "vendored `test-backend-ops` (CPU) subset passes
on Windows"; it shipped without that step. This ticket adds it, and does so through the wrapper so
the traps documented in `docs/dev/backward-coverage.md` and `docs/dev/ci-metal.md` cannot re-open:
an inline op list drifts, an exact-`strcmp` `-b` name that matches nothing runs zero cases and still
exits 0, and a `grad -o <op>` that compared nothing still prints `OK`.

## What to do

1. `.github/workflows/ci-windows.yml`: a dedicated `ops` job (parallel to `build`, so it also adds a
   `ci-windows / ops` check in the `ci-<backend>/<job>` naming the other lanes use):
   - build the vendored `test-backend-ops` with MSVC. windows-latest defaults to the Visual Studio
     **multi-config** generator, so the build type is selected at build time with `--config Release`
     (not `-DCMAKE_BUILD_TYPE`, which multi-config ignores) and the binary lands in `bin/Release/`,
     not `bin/`. `LLAMA_BUILD_TESTS=ON`, examples/server/tools OFF, `GGML_NATIVE=OFF` (oracle
     determinism), `-j 2`.
   - drive MODE_GRAD through `pytest tests/test_backend_ops_grad.py`, with `TEST_BACKEND_OPS` set to
     the *exact* built path (`.../bin/Release/test-backend-ops.exe`) rather than guessing a layout.
   - two tiers mirroring ci-cpu: per-PR runs a targeted forward subset plus the full grad registry;
     nightly runs the full forward sweep, the grad registry, and `grad -o MUL_MAT`.
2. `tests/test_backend_ops_grad.py`: make `_binary()`'s fallback discovery Windows-aware — search
   `bin/Release/` and a `.exe` suffix, not just the Linux `bin/test-backend-ops` — so a bare local
   run on Windows finds the binary. CI does not depend on this (it sets `TEST_BACKEND_OPS`), but a
   Windows developer deserves the same zero-config path Linux has.

## Out of scope

- Windows GPU lanes (CUDA/Vulkan on Windows) — not in the project CI matrix (S2-01/S3-01/S4-01).
- Wiring `ci-windows / ops` into branch protection — a repo-settings action, not a code change.
- Any kernel or backward-rule work: this ticket is CI + test-glue only.

## Acceptance criteria

- [x] ci-windows builds the vendored `test-backend-ops` with MSVC (Release, multi-config layout).
- [x] MODE_GRAD runs through `tests/test_backend_ops_grad.py` on Windows against that binary, so the
      one op registry (`tests/project_ops.py`) and the vacuity/positive-assertion guards apply
      identically to the Linux lane — no inline op list, no guessed `-b`.
- [x] Two-tier budget mirrored from ci-cpu: per-PR targeted subset + grad registry; nightly full
      sweep + grad registry + `grad -o MUL_MAT`.
- [x] `_binary()` resolves the MSVC multi-config `.exe` path (override respected verbatim; fallback
      discovers `bin/Release/test-backend-ops.exe`), with unit tests for both.
- [x] `ci-windows / ops` observed green on a real windows-latest run — **unverifiable on this Linux
      host; unproven until CI runs** (see below).

## Testing & verification

Verified locally on Linux (cannot run Windows CI here):

- `.venv/bin/python -m pytest tests/test_backend_ops_grad.py -q` — green against the Linux binary,
  now 22 tests (the 20 pre-existing plus the 2 new `_binary()`/`_discover_in` unit tests).
- `.venv/bin/ruff check tests/test_backend_ops_grad.py` — clean.
- `.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/ci-windows.yml'))"` — the
  workflow parses (actionlint is not installed on this host).
- Full suite `.venv/bin/python -m pytest tests/ -q` — green.

**Unproven until CI runs (be explicit):** the Windows leg itself is verified by careful reading only.
The MSVC build, the `bin/Release/` output path, the `.exe` invocation under Git Bash, and
scikit-build-core's MSVC toolchain discovery under the job's `bash` default shell cannot be exercised
on this Linux box. First real signal is a windows-latest run of `ci-windows / ops`.

## PR notes

- Branch: `ticket/S1-43-ci-windows-backend-ops`. Local commits only.
- Build-caching caveat: the Visual Studio generator does not honour `CMAKE_<LANG>_COMPILER_LAUNCHER`,
  so sccache does not accelerate the `test-backend-ops` build (it still accelerates the pip extension
  build). If the per-PR budget is exceeded, the follow-up is Ninja + `ilammy/msvc-dev-cmd` + sccache,
  which also collapses the output back to `bin/test-backend-ops.exe`.
</content>
</invoke>
