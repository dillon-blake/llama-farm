# `tests/` — the pytest suite

```bash
pip install -e . --no-build-isolation      # builds the native libraries
pip install -e vendor/llama.cpp/gguf-py    # the file-format code, pinned with the submodule
pytest tests/ -m "not slow"
```

`ci-cpu` (S0-07) runs exactly that per PR, and the full suite (including `slow`) nightly.

## Markers

Registered in `pyproject.toml`; `--strict-markers` is on, so a typo is an error rather than a
silently-skipped test.

| Marker | Meaning |
|---|---|
| `slow` | Nightly only — convergence gates (S1-12), full sweeps |
| `cuda` / `metal` / `vulkan` | Needs that backend compiled in |

The per-PR selection is `-m "not slow"`, with backend markers deselected on lanes that cannot
run them.

## Fixture models

`tests/fixtures/gen_tiny_llama.py` synthesizes a tiny llama-arch model in three variants —
**F32**, **Q8_0**, **Q4_K** — on first use, and caches them under `tests/.fixtures/`
(gitignored). Nothing is committed: `git ls-files '*.gguf'` returns nothing, and a test asserts
that.

Quantized bases are covered from day one on purpose. BLUEPRINT §10 (risk 2) notes that backward
through quantized weights is engine-supported but *lightly exercised* upstream — an F32-only
fixture set would hide exactly the class of bug this project is most exposed to.

Two things about the fixture are load-bearing rather than arbitrary:

- **`n_embd = 256`, `n_ff = 512`.** Q4_K's superblock is 256 elements, so every quantized row
  must be a multiple of 256. A smaller `n_embd` cannot be Q4_K-quantized at all.
  `TinyLlamaHParams.validate()` enforces this so a future edit cannot quietly reintroduce it.
- **The vocab is the first 512 entries of llama.cpp's own reference SPM vocab**
  (`vendor/llama.cpp/models/ggml-vocab-llama-spm.gguf`), which covers the three special tokens
  *plus the complete 256-token byte-fallback range*. Truncating the byte range would still load
  but would not tokenize, and S1-06's data layer needs it to tokenize.

Q4_K cannot be written from numpy — gguf-py's quantizer does not implement K-quants — so both
quantized variants go through `ggml_quantize_chunk` over ctypes. That is llama.cpp's own in-repo
pattern (`gguf-py/tests/test_quants.py`), and it means the fixtures are quantized by exactly the
code that dequantizes them at runtime.

### Regenerating

The cache is keyed by a content hash of the generator, its hyperparameters, and the gguf-py
version, so editing `gen_tiny_llama.py` invalidates it automatically. To force a rebuild:

```bash
rm -rf tests/.fixtures
```

## The no-op gate

`test_adapter_noop.py` is the project's first end-to-end correctness gate (BLUEPRINT D3). A
zero-initialized adapter has `B = 0`, so its delta `scale·B(A·x)` is **exactly** zero — for any
A, at any scale. Attaching it must leave the logits *bit-identical*.

The exactness is the point. It validates, against the **stock loader**, everything the adapter
writer produces: the four KVs, the tensor names, both shape conventions (including the flipped
`token_embd` one), the F32 dtypes, and the alpha caveat. Approximate equality would still pass
if the adapter were being applied with a tiny but non-zero scale; exact equality would not.

The scale=1000 variant separates "the adapter is zero" from "the adapter is not being applied at
all" — a scale that large would turn any non-zero delta into something impossible to miss.

## Writing tests

`conftest.py` provides:

- `libs` — the native libraries, loaded once per session with llama.cpp's log output silenced.
- `tiny_f32` / `tiny_q8_0` / `tiny_q4_k` — one fixture variant each.
- `tiny_model` — parametrized over all three, so a test written once runs against every variant.
- `load_model` — a factory that loads a model + context and frees both at teardown.
