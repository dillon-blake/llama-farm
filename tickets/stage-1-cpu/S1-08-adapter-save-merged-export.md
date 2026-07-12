---
id: S1-08
title: "Adapter save-from-training + merged-model export with re-quantize"
stage: 1
track: python
size: M
deps: ["S1-01", "S0-05"]
status: pr-open
pr: https://github.com/dillon-blake/llama-farm/pull/22
---

# S1-08 — Adapter save-from-training + merged-model export with re-quantize

**One-line outcome:** live trained A/B tensors can be saved to a stock-loadable adapter
GGUF, and a merged-model exporter re-quantizes merged tensors back to the base's original
quant types instead of forcing F16.

## Why (context)

BLUEPRINT G2 identifies the missing half of the adapter lifecycle: llama.cpp's loader is
file-path-only (`llama_adapter_lora_init`, `vendor/llama.cpp/src/llama-adapter.cpp:420`)
and the public C API offers no save, no in-place update, and no tensor enumeration — the
`llama.h` adapter surface is init/meta/free only (`vendor/llama.cpp/include/llama.h:
657-685`). The trained weights live in `llama_adapter_lora::ab_map`
(`vendor/llama.cpp/src/llama-adapter.h:67`, in the struct spanning `llama-adapter.h:
48-88`), private C++ — which is why the enumeration/get API belongs in the shim
(BLUEPRINT §1.2). D3 makes the payoff explicit: the adapter GGUF is the *interchange*
format, so a save must load in stock llama.cpp/llama-server/ollama with zero conversion.
One footgun to carry forward from S0-05: `alpha == 0` in metadata silently drops the
`alpha/rank` factor (the ternary at `vendor/llama.cpp/src/llama-adapter.h:55`) — always
write the real alpha from the training config.

For merged export, `tools/export-lora` already has the right merge graph — per-tensor
`delta = ggml_mul_mat(...)`, `ggml_scale`, `ggml_add`
(`vendor/llama.cpp/tools/export-lora/export-lora.cpp:353-366`, inside `merge_tensor` at
`:282`), including dequantizing the base tensor to F32 first (`:316-322`). Its one gap is
the output type: it forces F16 (`export-lora.cpp:183`, "output is forced to f16 for now"
at `:190-191`), which bloats a Q4_K base ~4x on export. BLUEPRINT §1.3/Appendix A says
copy the merge graph and extend it to re-quantize to the base's original types. That
needs `ggml_quantize_chunk` (`vendor/llama.cpp/ggml/include/ggml.h:2789`): gguf-py's
numpy quantizer cannot write K-quants (`Q4_K`/`Q5_K`/`Q6_K` implement only
`dequantize_blocks`, `vendor/llama.cpp/gguf-py/gguf/quants.py:476,553`, so
`gguf.quants.quantize` raises `NotImplementedError` for them, `quants.py:57-65`); the
in-repo precedent for calling `ggml_quantize_chunk` from Python via ctypes is
`vendor/llama.cpp/gguf-py/tests/test_quants.py:44-55`. The fidelity bar is upstream's
own: mirror `vendor/llama.cpp/tests/test-lora-conversion-inference.sh`, which compares
base vs adapter vs merged outputs (BLUEPRINT Appendix A).

## What to do

1. Shim (`csrc/farm_adapter.cpp`, `csrc/farm_api.h`): the G2 enumeration/get C ABI —
   `ll_adapter_n_tensors(adapter)`, `ll_adapter_tensor_info(adapter, i, ...)` (base
   name, role a|b, shape, type) walking `ab_map`, and `ll_adapter_get(adapter, i, buf,
   nbytes)` reading tensor data via `ggml_backend_tensor_get` (adapter tensors may live
   on non-CPU bufts). Read-only in this ticket; no set/update API.
2. `src/learning_llamas/_ffi/`: bind the new symbols (extend the S0-04 symbol table + tests).
3. `src/learning_llamas/adapter.py`: `save_adapter(adapter_handle, out_path, alpha, meta...)`
   — pull A/B via the new ABI into numpy and write the adapter GGUF with the S0-05
   writer (same four KVs as `vendor/llama.cpp/convert_lora_to_gguf.py:422-428`, same
   shape conventions incl. the flipped `token_embd` case). Validate `alpha > 0` at the
   API boundary. Round-trip test: save from a live training context, reload with stock
   `llama_adapter_lora_init`, assert tensor-exact equality against `ll_adapter_get`.
4. Merged export `src/learning_llamas/export.py` + `src/learning_llamas/quant.py`: reimplement the
   export-lora merge flow over gguf-py (read base + adapter, dequantize base tensor to
   F32 via `gguf.quants.dequantize`, compute `merged = base + scale · (BA)` in numpy —
   the numpy transcription of the graph at `export-lora.cpp:353-366` — with per-file
   provenance header per S0-01), then **re-quantize to each base tensor's original
   type**: `gguf.quants.quantize` where numpy supports it, else `ggml_quantize_chunk`
   via ctypes (`quant.py`, patterned on `test_quants.py:44-55`). Fall back to F16 with a
   warning for types where re-quantization is unsupported or requires an importance
   matrix (`ggml_quantize_requires_imatrix`, `ggml.h:2786`); write non-tensor KVs
   through unchanged and set `general.file_type` to match the output. Reject quantized
   adapters like upstream does (`export-lora.cpp:304-306`).
5. Fidelity test mirroring `test-lora-conversion-inference.sh`, on S0-06 fixture models
   (no HF downloads): (a) merged-at-scale-0 equals a re-quantize round-trip of the base
   (regression guard for the quant path); (b) with a nonzero trained/seeded adapter,
   greedy generations from base+adapter and from the merged model agree on the fixture
   prompt set within a documented tolerance (quantization noise is real; compare
   token-level with a small allowed divergence tail, as the upstream script does via
   prefix comparison).
6. CLI entry points: `python -m learning_llamas.export merge ...` and `... save-adapter ...`
   documented in `docs/dev/`.

## Out of scope

- In-place adapter set/update from Python (needed for weight-averaging workflows later;
  `tickets/backlog/` candidate).
- Optimizer-state sidecar and resume (S1-09 — the adapter GGUF stays interchange-only,
  D3).
- Merged-export imatrix computation for IQ-class targets (fallback-to-F16 covers v1).
- Norm-vector adapters and MoE expert (`build_lora_mm_id`) targets (loader ignores
  norms; MoE training is stage-1 MoE tickets).

## Acceptance criteria

- [ ] `pytest tests/test_adapter_save.py` passes: save-from-training round-trip is
      tensor-exact and stock-loadable (loader path
      `vendor/llama.cpp/src/llama-adapter.cpp:420`).
- [ ] Saved adapters carry `adapter.lora.alpha > 0` from the training config; `alpha=0`
      is rejected (test).
- [ ] `pytest tests/test_export_merge.py` passes: merged output tensors have the base's
      original quant types for at least Q4_K and Q8_0 fixture bases (not F16), verified
      by reading the output GGUF's tensor types.
- [ ] The fidelity test (mirroring `test-lora-conversion-inference.sh`) passes on the
      fixture models with the documented tolerance.
- [ ] The `_ffi` symbol-table test resolves `ll_adapter_n_tensors` /
      `ll_adapter_tensor_info` / `ll_adapter_get`.
- [ ] `ci-cpu / test` green per-PR with the new tests.

## Testing & verification

- `tests/test_adapter_save.py` and `tests/test_export_merge.py` (new), pytest with S0-06
  fixture models (F32, Q8_0, and a Q4_K variant — extend the fixture generator if Q4_K
  is missing); `ci-cpu / test` per-PR, full suite nightly (S0-07). The `llama-cli
  --lora` end-to-end load is already exercised by S1-03's gate; this ticket's round-trip
  uses the library loader directly.
- No MODE_GRAD applicability: no ops/kernels; quantization correctness is covered by the
  round-trip and fidelity tests.

## PR notes

- Branch: `ticket/S1-08-adapter-save-merged-export`.
- Single learning-llamas PR (shim + Python); no vendored llama.cpp changes, so no two-repo
  flow.
- Upstreaming disposition: **fork-local** for the shim/Python; the re-quantizing merge
  is a plausible **upstream-later** contribution to `tools/export-lora` once proven —
  note it in the PR description but do not block on it.
- Provenance: the numpy merge flow is adapted from `tools/export-lora/export-lora.cpp`
  (MIT) — per-file provenance header in `export.py` per S0-01 policy.
