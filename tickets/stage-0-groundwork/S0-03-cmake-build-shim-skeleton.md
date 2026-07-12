---
id: S0-03
title: "CMake + scikit-build-core build: vendored llama.cpp (CPU) + shim skeleton liblearningllamas"
stage: 0
track: infra
size: M
deps: ["S0-02"]
status: pr-open
pr: https://github.com/dillon-blake/llama-farm/pull/3
---

# S0-03 — CMake + scikit-build-core build: vendored llama.cpp (CPU) + shim skeleton liblearningllamas

**One-line outcome:** `pip install -e .` builds vendored llama.cpp (CPU backend) plus an
empty-but-real `csrc/` shim library `liblearningllamas` that compiles against private `src/`
internals and exports a probe symbol through a flat C ABI (`farm_api.h`).

## Why (context)

The shim is Layer 1 of the architecture and the key new native code (BLUEPRINT §3). It cannot be
avoided: the per-ubatch training loop's dependencies — `graph_params`, `llm_graph_result`,
`balloc`, `memory` — are private C++ (the loop is declared in
`vendor/llama.cpp/src/llama-context.h:207`, `opt_epoch_iter`), and the public `llama_opt_epoch`
hardcodes the wrong loss with no masking. So learning-llamas ships a thin C++ library compiled
**against vendored `src/` internals** (private headers such as `src/llama-context.h` and
`src/llama-graph.h`), exposing everything Python needs through a flat C ABI in `csrc/farm_api.h`
(BLUEPRINT §4). ggml-opt's own header sanctions copying its high-level pieces into user code
(`vendor/llama.cpp/ggml/include/ggml-opt.h:191-194`). This ticket builds only the skeleton — the
probe symbol proves the toolchain, include paths, and packaging; S1-01/S1-02 fill in the real
entry points.

Packaging follows llama-cpp-python's proven approach without depending on it (BLUEPRINT §3,
"Packaging"): scikit-build-core drives CMake over the vendored submodule, and the wheel carries
the native libraries. The vendored build produces three libraries whose targets are defined at
`vendor/llama.cpp/ggml/src/CMakeLists.txt:192` (`ggml-base` — note it contains `ggml-opt.cpp`,
i.e. the training driver lives in **libggml-base**),
`vendor/llama.cpp/ggml/src/CMakeLists.txt:242` (`ggml`), and
`vendor/llama.cpp/src/CMakeLists.txt:11` (`llama`). The documented load order — `libggml-base →
libggml → libllama → liblearningllamas`, opened with `RTLD_GLOBAL` so future `GGML_BACKEND_DL`
dlopen'd backends resolve symbols (BLUEPRINT §3) — is fixed here and consumed by the S0-04
loader.

Stage 0 builds CPU-only: the CPU backend is the correctness oracle for the whole kernel plan
(ROADMAP §4) and stages 2-4 add GPU backends behind their own CI lanes.

## What to do

1. Replace the S0-01 placeholder `CMakeLists.txt` with the real top-level build:
   `add_subdirectory(vendor/llama.cpp)` with CPU-only cache defaults (`GGML_METAL=OFF`,
   `GGML_CUDA=OFF`, `GGML_VULKAN=OFF`; leave the CPU backend on; `LLAMA_BUILD_TESTS=OFF`,
   `LLAMA_BUILD_EXAMPLES=OFF`, but keep `LLAMA_BUILD_TOOLS` available — S0-06 may want
   `llama-quantize`). Build shared libs (`BUILD_SHARED_LIBS=ON`) so the load-order contract is
   real.
2. Create `csrc/farm_api.h`: flat C ABI skeleton — `extern "C"`, `LL_API` export macro, and two
   functions: `const char * ll_version(void)` (learning-llamas package version) and `const char *
   ll_probe(void)` (returns the vendored llama.cpp commit hash the library was built from).
3. Create `csrc/farm_probe.cpp` implementing both. Bake the commit hash at configure time:
   `execute_process(git -C vendor/llama.cpp rev-parse HEAD)` → `configure_file` → generated
   `farm_version.h`. Fail configuration if the submodule is uninitialized.
4. Define the `learningllamas` shared-library target: link `llama` and `ggml`;
   `target_include_directories` including `vendor/llama.cpp/src` (private headers). To prove
   private-internal access now (S1-01/S1-02 depend on it), have one TU `#include
   "llama-context.h"` and `#include "llama-graph.h"` and reference a private type (e.g.
   `sizeof`-check a struct) behind no public symbol.
5. Wire scikit-build-core in `pyproject.toml`: install `libggml-base`, `libggml`, `libllama`,
   `liblearningllamas` (platform-appropriate names/sonames) into `learning_llamas/lib/` inside the wheel;
   set RPATH/`@loader_path` so the libraries resolve each other from that directory. Document
   the editable-build workflow (`pip install -e . --no-build-isolation` + rebuild command) in
   `csrc/README.md`.
6. Document the load-order contract in `csrc/README.md`: `libggml-base → libggml → libllama →
   liblearningllamas`, each opened `RTLD_GLOBAL`; note this anticipates `GGML_BACKEND_DL` dlopen'd
   backend packages later. S0-04 implements the loader against this text.
7. Smoke test `tests/test_probe.py`: locate `learning_llamas/lib/` from the installed package,
   `ctypes.CDLL` the four libraries in order, call `ll_probe()`, assert it equals the submodule
   commit (`4f37f519722aa3242eecb7649466b4a4a2d6d6da`). Keep it dependency-light; S0-06
   formalizes the harness and markers.
8. Add a `ccache`-friendly setup (compiler launcher variables honored, not required) so S0-07 CI
   caching works without changes here.

## Out of scope

- Any real shim functionality — adapter APIs, `ll_opt_init_lora`, `ll_train_step` are
  S1-01/S1-02.
- The Python `_ffi` loader/mirrors and version-lock enforcement (S0-04 — this ticket only
  *exposes* `ll_probe`).
- GPU backend builds and their flags (stages 2-4; flag documentation is S0-08).
- Prebuilt-wheel distribution/cibuildwheel matrix (BLUEPRINT §3 "Distribution" — future infra
  ticket if needed; `tickets/backlog/` candidate).
- Windows/MSVC support — S1-33 (BLUEPRINT §8 defers it to P1; that ticket owns the MSVC
  build and the `ci-windows` lane).

## Acceptance criteria

- [ ] In a clean venv on Linux x86_64, `pip install -e .` compiles vendored llama.cpp (CPU) +
      `liblearningllamas` and succeeds end-to-end.
- [ ] `python -m build --wheel` produces a wheel whose `learning_llamas/lib/` contains the four
      libraries (`ggml-base`, `ggml`, `llama`, `learningllamas` platform equivalents).
- [ ] `pytest tests/test_probe.py` passes: `ll_probe()` returns
      `4f37f519722aa3242eecb7649466b4a4a2d6d6da`.
- [ ] `ll_version()` returns the `learning_llamas.__version__` string.
- [ ] The shim contains a TU that includes `llama-context.h` and `llama-graph.h` from
      `vendor/llama.cpp/src` and compiles (private-internals access proven).
- [ ] Configure fails with an actionable message when `vendor/llama.cpp` is uninitialized.
- [ ] `csrc/README.md` documents the load order + RTLD_GLOBAL contract and the editable-build
      workflow.

## Testing & verification

- `tests/test_probe.py` (new), run locally on the Linux CPU dev VM; it becomes part of the
  default pytest suite that S0-06 formalizes and `ci-cpu` (S0-07) runs per-PR on ubuntu-latest
  and macos-14 — macOS arm64 portability is verified there, not gated here.
- Manual: the clean-venv install and wheel-content listing go in the PR description.
- No MODE_GRAD applicability: this ticket adds no ops/kernels.

## PR notes

- Branch: `ticket/S0-03-cmake-build-shim-skeleton`.
- Single learning-llamas PR. No vendored llama.cpp source changes — if a build flag turns out to
  require a vendor patch, that patch goes through the S0-02 two-repo flow (fork PR + submodule
  bump), not inline here.
- Upstreaming disposition: **fork-local** (build glue and shim skeleton are
  learning-llamas-specific).
- `csrc/` files copied/adapted from llama.cpp later must carry provenance headers per S0-01
  policy; the skeleton itself is original code.
