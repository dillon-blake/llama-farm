# `csrc/` — the C shim (`liblearningllamas`)

Layer 1 of the architecture (BLUEPRINT §3): a thin C++ library compiled **against
llama.cpp's private `src/` internals**, exposing everything Python needs through a flat C
ABI in [`farm_api.h`](farm_api.h).

## Why the shim is not optional

The training loop learning-llamas needs is `llama_context::opt_epoch_iter`
(`vendor/llama.cpp/src/llama-context.h:207`). It is private C++, and its dependencies —
`llm_graph_params`, `llm_graph_result`, the batch allocator, the memory module — are private
too. The *public* entry point, `llama_opt_epoch`, hardcodes the wrong loss and offers no
masking, so it cannot train an instruction-tuned model. learning-llamas therefore forks the
loop (BLUEPRINT D1) and must compile against those private headers to do it.

ggml's own header sanctions this shape of reuse: "The high-level functions start here. They
do not depend on any private functions or structs and can be copied to and adapted for user
code" (`vendor/llama.cpp/ggml/include/ggml-opt.h:191-194`).

[`farm_internals.cpp`](farm_internals.cpp) exists to keep that access honest. It compiles
nothing useful — it includes `llama-context.h` and `llama-graph.h` and `static_assert`s on
the private types. If an upstream rebase walls them off, the build breaks on the `ci-cpu`
lane immediately, rather than in the middle of S1-02.

| Ticket | Adds |
|---|---|
| S0-03 | The build, `farm_api.h`, `ll_version()` / `ll_probe()` |
| S1-01 | `ll_opt_init_lora` — `ggml_set_param` on the adapter A/B tensors |
| S1-02 | `ll_train_step` — the forked per-ubatch loop with a pluggable loss |

## The load-order contract

The CPU build produces **five** shared libraries. `_ffi/loader.py` (S0-04) opens four of them
explicitly, in this order, each with `RTLD_GLOBAL`:

```
libggml-base  →  libggml  →  libllama  →  liblearningllamas
```

- **`libggml-base`** — ggml core *and `ggml-opt.cpp`*, so the training driver lives here, not
  in `libggml` (`vendor/llama.cpp/ggml/src/CMakeLists.txt:192`).
- **`libggml`** — the backend registry (`vendor/llama.cpp/ggml/src/CMakeLists.txt:242`).
- **`libllama`** — the model and context layer (`vendor/llama.cpp/src/CMakeLists.txt:11`).
- **`liblearningllamas`** — this shim.

The fifth, **`libggml-cpu`**, is the CPU backend. With `GGML_BACKEND_DL` off (our build) it
is linked `PUBLIC` into `libggml`, so the dynamic loader pulls it in transitively via RPATH
and the Python loader never names it. It still ships in the wheel: without it there is no
backend to register.

Two properties of the contract are load-bearing:

- **Order.** Each library depends on the ones before it. RPATH would resolve them anyway, but
  opening them explicitly and in order means a *missing* library produces an error naming
  that library, instead of an unresolved-symbol cascade out of `libllama`.
- **`RTLD_GLOBAL`.** Symbols land in the global namespace, so backends `dlopen`'d later can
  resolve `ggml_*` against the already-loaded `libggml-base`. Nothing in stage 0 needs this;
  it costs nothing now and is required the moment `GGML_BACKEND_DL` backend packages arrive.

Inside the wheel all five libraries sit in `learning_llamas/lib/` with RPATH pointing at
their own directory (`$ORIGIN`, `@loader_path` on macOS), so they resolve each other with no
`LD_LIBRARY_PATH`. They are installed under a dedicated CMake component (`learningllamas`),
which is how llama.cpp's own install rules — headers, CMake config, pkgconfig — stay out of
the wheel.

## Building

The native build runs through scikit-build-core; there is no separate CMake invocation to
remember.

```bash
git submodule update --init --recursive
pip install -e . --no-build-isolation          # first build compiles all of llama.cpp
```

To force a rebuild after touching `csrc/` or the vendored tree:

```bash
pip install -e . --no-build-isolation --no-deps --force-reinstall
```

On a memory-constrained machine, cap the compile jobs — ggml's CPU kernels are heavy
translation units and `-j$(nproc)` can push a 4 GB box into swap:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=2 pip install -e . --no-build-isolation
```

`ccache` or `sccache` is used automatically when it is on `PATH`, and is never required.
