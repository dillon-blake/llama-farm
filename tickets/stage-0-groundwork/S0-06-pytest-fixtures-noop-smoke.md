---
id: S0-06
title: "Test harness: pytest + tiny fixture GGUF models + no-op adapter smoke test"
stage: 0
track: python
size: M
deps: [S0-04, S0-05]
status: open
pr: null
---

# S0-06 — Test harness: pytest + tiny fixture GGUF models + no-op adapter smoke test

**One-line outcome:** a pytest suite with generated tiny llama-arch fixture GGUFs (F32 + Q4_K + Q8_0) and a passing no-op adapter smoke test: logits with a zero-B adapter attached at scale 1 equal logits without it.

## Why (context)

Every later stage lands its Python-visible behavior as tests in this harness, and the CPU CI
lane (S0-07) runs it per-PR — so the harness must exist before CI and before any training
code. BLUEPRINT §4 reserves `tests/` for exactly this, including the adapter round-trip and
finite-difference checks that later tickets add.

The first real test is dictated by BLUEPRINT D3: the adapter GGUF format is the checkpoint
format, and a zero-initialized adapter (`A ~ N(0,σ)`, `B = 0`) is a *provable no-op* — the
LoRA delta `scale·B(A·x)` is exactly zero when B is zero. D3 explicitly says to ship this as
a smoke test ("logits with adapter@scale=1 == logits without"). It end-to-end validates the
S0-05 writer (KV keys, tensor naming, shapes, the alpha caveat from
`vendor/llama.cpp/src/llama-adapter.h:48-88` where `alpha == 0` silently drops the
`alpha/rank` factor) against the stock loader, including the flipped/A-transposed
`token_embd` convention enforced at `vendor/llama.cpp/src/llama-adapter.cpp:356-368`.

Fixtures must cover quantized bases from day one: BLUEPRINT §10 (risk 2) notes that backward
through quantized weights is engine-supported but lightly exercised, and recommends
validating early per quant type. Hence three tiny fixture variants — F32, Q8_0 (writable by
gguf-py's pure-numpy quantizer), and Q4_K (K-quants are *not* writable by pure numpy;
BLUEPRINT §1.5 points at the in-repo ctypes precedent
`vendor/llama.cpp/gguf-py/tests/test_quants.py:112`, which calls `ggml_quantize_chunk`).
Fixtures are synthesized, cached per session, and never committed.

## What to do

1. **Pytest configuration.** Add a `[tool.pytest.ini_options]` block to `pyproject.toml`
   registering markers: `slow` (nightly-only), `cuda`, `metal`, `vulkan`
   (backend-specific). Default per-PR selection is `-m "not slow"` plus deselecting
   backend markers; S0-07 encodes that in CI. Add `tests/conftest.py` with shared fixtures.
2. **Fixture generator** `tests/fixtures/gen_tiny_llama.py`: synthesize a tiny
   random-weight llama-arch GGUF purely in Python via gguf-py `GGUFWriter` (suggested
   shape: `n_layer=2`, `n_embd=64`, `n_head=4`, `n_head_kv=2`, `n_ff=128`, `n_vocab=256`;
   seeded RNG so fixtures are deterministic). Write all hparam/rope KV the llama loader
   requires. Emit the F32 variant first.
3. **Tokenizer fixture:** embed a minimal SPM-style vocab in the same GGUF —
   `tokenizer.ggml.model = "llama"` plus tokens/scores/token-type arrays and BOS/EOS/UNK
   ids, sized to `n_vocab`. Use `vendor/llama.cpp/models/ggml-vocab-llama-spm.gguf` as the
   reference for the exact KV set llama.cpp's vocab loader expects (read it with
   `GGUFReader`; do not commit or ship it).
4. **Quantized variants:** produce Q8_0 via gguf-py's numpy quantizer; produce Q4_K via
   ctypes `ggml_quantize_chunk` (declared at `vendor/llama.cpp/ggml/include/ggml.h:2789`)
   loaded from the built `libggml-base` — this is the BLUEPRINT `quant.py` note — with a
   documented fallback to the `llama-quantize` binary from the vendor build
   (`vendor/llama.cpp/tools/quantize/CMakeLists.txt:17`) if the symbol path fails.
   Keep norm/embedding tensors in the types the quantization tooling produces by default.
5. **Session-scoped caching:** cache generated fixtures in a gitignored directory keyed by
   a content hash of the generator source + gguf-py version, via a session-scoped fixture;
   second `pytest` run must not re-generate. Assert no `*.gguf` is ever tracked by git.
6. **Loadability test** `tests/test_fixtures.py`: each fixture variant loads through the
   S0-04 `_ffi` layer (`llama_model_load` → context → decode a fixed token sequence) and
   produces finite logits.
7. **No-op smoke test** `tests/test_adapter_noop.py`: for each fixture variant, create a
   zero-B adapter with S0-05 `adapter.py`; load it via `llama_adapter_lora_init`
   (`vendor/llama.cpp/include/llama.h:657`) and attach with `llama_set_adapters_lora`
   (`vendor/llama.cpp/include/llama.h:690`) at scale 1.0; decode the same fixed token
   sequence on the CPU backend and compare logits exactly against the no-adapter run.
8. Document how to run and extend the suite in `tests/README.md` (markers, fixture cache,
   regeneration).

## Out of scope

- CI wiring and lane definitions — S0-07.
- Developer-facing testing docs beyond `tests/README.md` — S0-08.
- Finite-difference gradient tests and any training-step tests — stage 1 (the tiny-model
  convergence gate is S1-12).
- MODE_GRAD/test-backend-ops invocation — vendored harness, exercised from S0-07/S0-08.
- Adapter round-trip fidelity against `convert_lora_to_gguf.py` outputs — later ticket
  mirroring `tests/test-lora-conversion-inference.sh` (BLUEPRINT Appendix A).

## Acceptance criteria

- [ ] `pytest tests/ -m "not slow"` passes on a Linux x86_64 CPU-only build from a fresh
      checkout, generating all fixtures on first run.
- [ ] Fixture variants F32, Q4_K, and Q8_0 all exist after a run, and
      `git ls-files '*.gguf'` returns nothing (no model binaries committed).
- [ ] `tests/test_adapter_noop.py` passes for all three variants with exact logit equality
      (zero tolerance) at adapter scale 1.0 on CPU.
- [ ] A second `pytest` invocation reuses the cached fixtures (generator log line absent /
      cache-hit assertion in a dedicated test).
- [ ] Markers `slow`, `cuda`, `metal`, `vulkan` are registered (pytest runs with
      `--strict-markers` clean) and `tests/README.md` documents them.

## Testing & verification

This ticket *is* the test harness; verification is running it. Locally: fresh venv,
`pip install -e .`, `pytest tests/ -m "not slow"` twice (second run proves caching). CI:
this suite becomes the `ci-cpu / test` job payload when S0-07 lands (per-PR: `-m "not
slow"`; nightly: full suite including `slow`). No kernel changes, so no test-backend-ops
MODE_GRAD cases here; the harness's fixture models are what later MODE_GRAD-adjacent
Python tests build on.

## PR notes

- Branch: `ticket/S0-06-pytest-fixtures-noop-smoke`.
- One PR; pure learning-llamas Python/tests — no vendored llama.cpp changes, so no two-repo
  flow. Upstreaming disposition: **fork-local**.
- Soft coordination: S0-07 consumes the marker names and the `-m "not slow"` convention;
  keep them stable or update S0-07's workflow in the same PR window.
- If any helper is adapted from `gguf-py/tests/test_quants.py`, carry the per-file
  provenance header (source path, commit `4f37f51`, MIT) per S0-01 policy.
