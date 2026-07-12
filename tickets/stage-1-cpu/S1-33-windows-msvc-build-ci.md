---
id: S1-33
title: "Windows/MSVC build of liblearningllamas + ci-windows lane (CPU)"
stage: 1
track: infra
size: M
deps: [S0-03, S0-07]
status: open
pr: null
---

# S1-33 — Windows/MSVC build of liblearningllamas + ci-windows lane (CPU)

**One-line outcome:** the shim, bindings, and Python package build and pass the CPU
test suite under MSVC on Windows, enforced by a `ci-windows` GitHub Actions lane —
so the private-internals shim never silently rots on the one platform most likely
to break it.

## Why (context)

BLUEPRINT §8 lists Windows as "expected to work, untested" with the explicit
instruction "the shim builds against private C++ internals — make MSVC a CI target
from P1", and BLUEPRINT §9 names Windows/MSVC CI a P1 deliverable. The risk is
specific: `liblearningllamas` compiles against vendored `src/` internals
(`vendor/llama.cpp/src/llama-context.h`, `src/llama-graph.h`), not the stable C
API, so MSVC-specific breakage (symbol visibility, `__declspec` export rules,
MSVC's stricter C++ conformance in private headers, CRT linkage between the
libraries and the ctypes-loaded shim) will not be caught by the Linux/macOS lanes.
llama.cpp itself builds cleanly under MSVC in upstream CI, so failures here will
be in *our* shim and packaging, which is exactly what this lane must catch.

Scope is CPU-only Windows: the four GPU backend stages target Linux/macOS VMs;
Windows CUDA/Vulkan lanes are not part of this plan (the project's backend CI
matrix is defined by S2-01/S3-01/S4-01).

## What to do

1. Fix whatever MSVC surfaces in `csrc/`: explicit `LL_API` export macro
   (`__declspec(dllexport)`/`visibility("default")`) on every `farm_api.h` symbol;
   no GNU-isms; correct import-lib generation for the shim DLL.
2. CMake: Windows branch of the S0-03 build (MSVC generator, `/utf-8`, matching
   CRT across libggml/libllama/liblearningllamas); document any vendored-build flags
   Windows needs (e.g. `GGML_NATIVE=OFF` for reproducible CI).
3. `_ffi` loader: Windows DLL search-path handling (`os.add_dll_directory` rather
   than PATH mutation), `.dll` naming, and the load-order contract from S0-04.
4. Workflow `ci-windows.yml`: `windows-latest`, per-PR quick lane (build wheel +
   pytest CPU subset + targeted vendored `test-backend-ops` run) and nightly full
   lane mirroring `ci-cpu` (S0-07 two-tier pattern), with sccache/ccache-equivalent
   caching.
5. Wheel: confirm the scikit-build-core wheel builds and imports on Windows;
   record any packaging deltas in `docs/dev/building.md` (S0-08).

## Out of scope

- Windows GPU lanes (CUDA/Vulkan on Windows) — not in the project CI matrix; file
  a backlog stub if demand appears.
- Release/publish wheel automation (cibuildwheel matrix) — BLUEPRINT §3
  distribution work, unscheduled.
- Any training-correctness work: this ticket changes build/packaging/CI only.

## Acceptance criteria

- [ ] `pip install -e .` and wheel build succeed under MSVC on `windows-latest`.
- [ ] Full CPU pytest suite (S0-06 harness, incl. the no-op adapter smoke test and
      the P0 proof-of-gradient test if merged) passes on Windows in CI.
- [ ] Vendored `test-backend-ops` (CPU) targeted subset passes on Windows.
- [ ] `ci-windows / build` and `ci-windows / test` are required checks wired into
      the repo's branch protection alongside `ci-cpu`.
- [ ] `docs/dev/building.md` gains a Windows section (S0-08 doc updated).

## Testing & verification

The lane itself is the deliverable: per-PR quick + nightly full on
`windows-latest`, mirroring S0-07's tiering. Local reproduction path documented
for a Windows VM (`docs/dev/vm-playbooks.md` addendum).

## PR notes

- Branch: `ticket/S1-33-windows-msvc-build-ci`.
- Single-repo PR (build/CI/bindings only) unless an MSVC fix requires touching
  vendored code — then the S0-02 two-repo flow applies and the fix is
  **upstream-early** (mainline MSVC users benefit).
- Upstreaming disposition: fork-local (project packaging).
