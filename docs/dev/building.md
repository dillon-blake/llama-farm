# Building learning-llamas

## Prerequisites

| Platform | Needs |
|---|---|
| Linux | a C++17 compiler (GCC ≥ 11 or Clang ≥ 14), CMake ≥ 3.21, Python ≥ 3.10, git |
| macOS | Xcode command-line tools (`xcode-select --install`), CMake ≥ 3.21, Python ≥ 3.10 |
| Windows | **not yet supported** — S1-33 owns the MSVC build and the `ci-windows` lane |

`ccache` (or `sccache`) is picked up automatically if it is on `PATH`, and is never required.
On a first build it saves nothing; on every subsequent one it saves most of the wall time.

`ninja` is optional but worth having — it parallelizes better than Make on this tree.

## The build

```bash
git clone --recurse-submodules https://github.com/dillon-blake/llama-farm.git
cd llama-farm

python3 -m venv .venv && source .venv/bin/activate
pip install scikit-build-core cmake ninja pytest ruff numpy

pip install -e . --no-build-isolation        # compiles vendored llama.cpp + the shim
pip install -e vendor/llama.cpp/gguf-py      # the file-format code
```

If you cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
./scripts/apply-patches.sh                   # exits 0 on the empty queue (normal)
```

CMake configuration **fails with an actionable message** if the submodule is missing, rather
than producing a confusing compile error.

`gguf-py` comes from the vendored tree, not PyPI, on purpose: the pinned llama.cpp commit, the
vendored gguf-py, and the `_ffi` struct mirrors are **one atomic version** (ADR-0001). A PyPI
`gguf` that drifts from the pinned commit is exactly the failure mode the version lock exists to
prevent.

### Verify

```bash
pytest tests/ -m "not slow"
```

52 tests should pass in about a second. If `learning_llamas/lib/` does not exist, the native
build did not run — re-run the editable install.

### On a memory-constrained machine

ggml's CPU kernels are heavy translation units. `-j$(nproc)` can push a 4 GB box into swap and
make the build *slower*:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=2 pip install -e . --no-build-isolation
```

For reference, a full clean build (vendored llama.cpp CPU + the shim) takes roughly **4 minutes
at `-j3` on an Intel N100**.

### Rebuilding

Editable installs rebuild on import when sources change. To force one after touching `csrc/` or
the vendored tree:

```bash
pip install -e . --no-build-isolation --no-deps --force-reinstall
```

## Backend flags

Stage 0 and stage 1 are **CPU-only**: the CPU backend is the correctness oracle for the entire
kernel plan (ROADMAP §4), and stages 2-4 add the GPU backends behind their own CI lanes. The
top-level `CMakeLists.txt` therefore forces the GPU backends off.

The switches are plain CMake options in the vendored tree, and they are what the stage-2/3/4 VMs
turn on:

| Option | Vendor anchor | Default |
|---|---|---|
| `GGML_CUDA` | `vendor/llama.cpp/ggml/CMakeLists.txt:199` | `OFF` |
| `GGML_VULKAN` | `vendor/llama.cpp/ggml/CMakeLists.txt:224` | `OFF` |
| `GGML_METAL` | `vendor/llama.cpp/ggml/CMakeLists.txt:239` | **ON for Apple builds** |

The Metal default is a trap worth knowing about: on macOS, a build that says nothing about
Metal *gets* Metal. The `ci-cpu` lane passes `-DGGML_METAL=OFF` explicitly so that its macOS
entry is genuinely CPU-only.

Pass flags through scikit-build-core with `CMAKE_ARGS`:

```bash
CMAKE_ARGS="-DGGML_CUDA=ON"    pip install -e . --no-build-isolation
CMAKE_ARGS="-DGGML_VULKAN=ON"  pip install -e . --no-build-isolation
CMAKE_ARGS="-DGGML_METAL=OFF"  pip install -e . --no-build-isolation   # macOS, CPU-only
```

## The vendored test binaries

`test-backend-ops` is the acceptance harness for every kernel ticket (ADR-0002), and it is built
from the vendored tree, not from ours:

```bash
cmake -S vendor/llama.cpp -B build/vendor-tests \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=ON \
      -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF \
      -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF
cmake --build build/vendor-tests --target test-backend-ops -j2
```

Add the backend you are working on (`-DGGML_CUDA=ON`, …) to test its kernels. Usage is in
[`testing.md`](testing.md).

`LLAMA_BUILD_TESTS` is `vendor/llama.cpp/CMakeLists.txt:107`; the target is registered at
`vendor/llama.cpp/tests/CMakeLists.txt:243`.

## What gets built

The CPU build produces **five** shared libraries, all installed into `learning_llamas/lib/`
inside the wheel with RPATH pointing at their own directory:

```
libggml-base   ggml core + ggml-opt.cpp (the training driver lives here, not in libggml)
libggml-cpu    the CPU backend
libggml        the backend registry
libllama       the model and context layer
liblearningllamas   the C shim (csrc/)
```

The load-order contract — and why `libggml-cpu` ships but is never opened by name — is in
[`csrc/README.md`](../../csrc/README.md).

## Debug builds

```bash
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Debug" pip install -e . --no-build-isolation
```

Debug ggml is *slow* — 10-30× on the kernels. Use it to attach a debugger to a specific failure,
not to run the suite. For a middle ground, `RelWithDebInfo` keeps the optimizations and the
symbols.


## Windows / MSVC (S1-33)

```
pip install -e ".[dev]" --no-build-isolation
```

Two things are genuinely different on Windows, and **both fail at import rather than at build**,
which is what makes them worth knowing about in advance.

**Dependency resolution.** Windows resolves a DLL's imports through the process's DLL search path,
and **since Python 3.8 that path no longer includes the directory the DLL was loaded from**. So
`learningllamas.dll` — which imports `llama.dll`, which imports `ggml.dll`, all of them sitting in
the same folder — fails on the very first load. And the error names the *dependency*, not the file
you actually asked for.

`_ffi.loader` handles this with `os.add_dll_directory`, held for the life of the process (ggml's
backend registry can load a backend DLL lazily, long after the initial load). It does **not** edit
`PATH`: that is process-wide, order-dependent, and shared with everything else in the interpreter.

**`GGML_NATIVE=OFF` in CI.** With it on, ggml compiles for whatever the runner happens to be — and a
cached object file from one runner generation crashes on another. Reproducibility beats the few
percent.

`ci-windows.yml` is a separate workflow rather than a matrix entry on `ci-cpu`, because almost
nothing is shared: a different compiler, a different linker, a different DLL model, a different cache
tool. Folding it in would mean an `if: runner.os == 'Windows'` on every step — a matrix in a trench
coat.
