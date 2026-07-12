---
id: S0-04
title: "ctypes binding layer (_ffi): load order, struct mirrors, version lock"
stage: 0
track: python
size: M
deps: ["S0-03"]
status: pr-open
pr: https://github.com/dillon-blake/llama-farm/pull/4
---

# S0-04 — ctypes binding layer (_ffi): load order, struct mirrors, version lock

**One-line outcome:** `src/learning_llamas/_ffi/` loads the native libraries in the documented order,
mirrors the llama.h/ggml-opt.h structs and functions later tickets need, and refuses to run
against a mismatched vendored commit.

## Why (context)

Layer 2 of the architecture is ctypes-first (BLUEPRINT §3): every symbol learning-llamas needs is
already exported (`LLAMA_API`/`GGML_API`), the entire ggml-opt driver lives in `libggml-base`
(its sources are part of the `ggml-base` target,
`vendor/llama.cpp/ggml/src/CMakeLists.txt:192-210`), and llama.cpp itself ships an in-repo
precedent for driving libggml from Python via ctypes:
`vendor/llama.cpp/gguf-py/tests/test_quants.py` (BLUEPRINT §1.5), which `CDLL`s libggml and
calls `ggml_quantize_chunk` with explicit argtypes (`test_quants.py:44-55`). nanobind is
deliberately deferred until zero-copy views or GIL release matter at the shim edge; nothing in
stage 0/1 needs it.

Because ctypes mirrors struct layouts by hand, the binding is only correct against the exact
vendored commit — struct drift produces silent memory corruption, not errors. Hence the
**version lock**: S0-03's `ll_probe()` exports the commit hash baked into the native build, and
`_ffi` hard-errors at import if it differs from the hash recorded when the Python package was
generated. Pinned commit + gguf-py + struct mirrors form one atomic version (BLUEPRINT §3,
"Packaging").

One hazard is designed around up front (BLUEPRINT §3, "Why ctypes first"): ggml-opt's per-step
optimizer-params hook is a callback **returning `struct ggml_opt_optimizer_params` by value**
(`vendor/llama.cpp/ggml/include/ggml-opt.h:101`); struct-by-value returns through ctypes
callbacks are a known ABI trap. The policy: never register a Python callback for it. Instead
pass the exported `ggml_opt_get_constant_optimizer_params` (`ggml-opt.h:108`) as `get_opt_pars`,
with a **Python-owned** `ggml_opt_optimizer_params` struct (mirror of `ggml-opt.h:85-97`) as the
`userdata`; Python mutates that struct between steps, which is how LR schedules work for free
(BLUEPRINT §1.1).

## What to do

1. `src/learning_llamas/_ffi/loader.py`: locate `learning_llamas/lib/` in the installed package;
   `ctypes.CDLL(..., mode=RTLD_GLOBAL)` in the S0-03 contract order `libggml-base → libggml →
   libllama → liblearningllamas` (platform-appropriate names; on macOS use the dylib naming). Cache
   handles as module singletons; expose `load()` returning a namespace of the four handles.
2. `src/learning_llamas/_ffi/farm.py`: mirror `farm_api.h` (`ll_version`, `ll_probe`), restype
   `c_char_p`.
3. `src/learning_llamas/_ffi/llama.py`: mirror the llama.h surface later tickets need — opaque
   pointers only where possible (model/context/adapter are opaque `c_void_p`-style handles):
   - lifecycle: `llama_backend_init`, `llama_model_default_params`, `llama_model_load_from_file`
     (`vendor/llama.cpp/include/llama.h:493`), `llama_model_free` (`llama.h:516`),
     `llama_context_default_params`, `llama_init_from_model` (`llama.h:518`), `llama_free` —
     with full `ctypes.Structure` mirrors for `llama_model_params` and `llama_context_params`;
   - adapters: `llama_adapter_lora_init`, `llama_adapter_lora_free`, and the fork/pinned-commit
     batch attach `llama_set_adapters_lora(ctx, adapters**, n, scales*)` (`llama.h:690-694` —
     present at `4f37f51`; the mirror test must fail loudly if a vendor bump removes/renames
     it);
   - the stock training entry points, mirrored as reference/template even though D1 forks the
     loop: `llama_opt_param_filter` typedef (`llama.h:1564`), `llama_opt_init` (`llama.h:1581`),
     `llama_opt_epoch` (`llama.h:1583`), and the `llama_opt_params` struct.
4. `src/learning_llamas/_ffi/ggml_opt.py`: mirrors for `ggml_opt_optimizer_params`
   (`ggml-opt.h:85-97`), `ggml_opt_params`, and the driver functions `ggml_opt_init`
   (`ggml-opt.h:138`), `ggml_opt_free` (`:139`), `ggml_opt_grad_acc` (`:156`),
   `ggml_opt_result_init`/`_free`/`_loss` (`:164-170`), `ggml_opt_prepare_alloc` (`:177`),
   `ggml_opt_alloc`/`ggml_opt_eval` (`:186-189`), plus `ggml_opt_get_default_optimizer_params`
   (`:105`) and `ggml_opt_get_constant_optimizer_params` (`:108`).
5. `src/learning_llamas/_ffi/ggml_backend.py`: `ggml_backend_tensor_set` / `ggml_backend_tensor_get`
   (`vendor/llama.cpp/ggml/include/ggml-backend.h:92-93`) — the D7 batch-upload path.
6. Optimizer-params policy, enforced in code: provide `OptimizerParams` (the Python-owned struct
   wrapper) and a helper that wires `get_opt_pars = ggml_opt_get_constant_optimizer_params,
   get_opt_pars_ud = byref(params)`; do **not** define any `CFUNCTYPE` whose restype is
   `ggml_opt_optimizer_params`. Add a unit test that introspects `_ffi` and asserts no such
   CFUNCTYPE exists.
7. Version lock: at build time (extend the S0-03 CMake/scikit-build step) generate
   `src/learning_llamas/_ffi/_version_lock.py` containing `VENDORED_COMMIT = "4f37f51..."`; on first
   `load()`, compare `ll_probe()` against it and raise `RuntimeError` naming both hashes on
   mismatch.
8. Central symbol table: one declarative list of (library, symbol, argtypes, restype) driving
   registration, so a single test can iterate it and confirm every symbol resolves — this is the
   tripwire for vendor drift beyond the commit lock.
9. Tests `tests/test_ffi.py`: load-order smoke; probe==lock; symbol-table resolution;
   `ggml_opt_get_default_optimizer_params(None)` called with the mirrored struct as restype
   returns plausible defaults (finite floats, `adamw.alpha > 0`) — a partial struct-layout
   check; monkeypatched wrong `VENDORED_COMMIT` raises `RuntimeError`; no struct-return
   CFUNCTYPE exists.

## Out of scope

- Binding the shim's real training/adapter entry points — they do not exist yet (S1-01/S1-02
  extend `_ffi` as they add symbols).
- Tokenization/decode bindings beyond lifecycle (S1-06 adds what the data layer needs).
- nanobind promotion (revisit when profiling shows ctypes overhead at the shim edge;
  `tickets/backlog/` candidate).
- Auto-generation of mirrors from headers (nice-to-have; hand-written + symbol-table test is
  stage-0 scope).
- The tiny fixture models and any test requiring a real GGUF (S0-06).

## Acceptance criteria

- [ ] `pytest tests/test_ffi.py` passes on the Linux CPU dev VM against the S0-03 build.
- [ ] All four libraries load in the documented order with `RTLD_GLOBAL`; a comment/test
      documents why the order matters (BLUEPRINT §3).
- [ ] The symbol-table test resolves every declared symbol, including `llama_set_adapters_lora`,
      `llama_opt_init`, `llama_opt_epoch`, `ggml_opt_prepare_alloc`,
      `ggml_opt_get_constant_optimizer_params`, `ggml_backend_tensor_set/get`.
- [ ] `ggml_opt_get_default_optimizer_params` round-trips through the mirrored
      `ggml_opt_optimizer_params` with plausible field values.
- [ ] Import against a tampered `VENDORED_COMMIT` raises `RuntimeError` naming both hashes
      (tested).
- [ ] A test asserts `_ffi` defines no ctypes callback type returning
      `ggml_opt_optimizer_params` by value.

## Testing & verification

- `tests/test_ffi.py` (new), pytest, local on the Linux CPU VM; runs in `ci-cpu / test` per-PR
  on ubuntu-latest and macos-14 once S0-07 lands (S0-06 formalizes markers; these tests need no
  fixture models, only the built libraries).
- No MODE_GRAD applicability: no ops/kernels added.

## PR notes

- Branch: `ticket/S0-04-ctypes-ffi-binding-layer`.
- Single learning-llamas PR; no vendored llama.cpp changes.
- Upstreaming disposition: **fork-local** (Python bindings are the product, not upstream
  material).
- Struct mirrors adapted from llama.h/ggml-opt.h declarations: cite the header path + pinned
  commit in each `_ffi` module docstring per S0-01 provenance policy (declarations are
  interface, not copied implementation, but keep the trail).
