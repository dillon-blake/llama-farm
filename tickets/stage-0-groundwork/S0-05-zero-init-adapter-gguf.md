---
id: S0-05
title: "adapter.py: zero-init LoRA adapter GGUF via gguf-py (create/read/enumerate)"
stage: 0
track: python
size: M
deps: ["S0-01"]
status: pr-open
pr: null
---

# S0-05 — adapter.py: zero-init LoRA adapter GGUF via gguf-py (create/read/enumerate)

**One-line outcome:** pure-Python creation of a zero-initialized LoRA adapter GGUF from
base-model metadata alone — loadable by stock llama.cpp — plus read/enumerate helpers.

## Why (context)

The LoRA adapter GGUF format is a complete, stable interchange contract, and learning-llamas adopts
it as both its adapter format and its checkpoint format (BLUEPRINT §1.3, D3): anything we write
must load in stock llama.cpp/llama-server/ollama with zero conversion. gguf-py alone can create
a zero-init adapter from the base model's metadata — no C code needed (BLUEPRINT §0 fact 5, §6.1
init recipe): read the base GGUF with `GGUFReader` (mmap), enumerate LoRA-targetable tensors,
and write `A ~ N(0,σ)`, `B = 0` pairs plus four KVs with `GGUFWriter` (which is already
adapter-aware: `GGUFType.ADAPTER` at `vendor/llama.cpp/gguf-py/gguf/constants.py:400-402`,
`Keys.Adapter.TYPE`/`LORA_ALPHA` at `constants.py:303-305`, and `.lora_a`/`.lora_b`-aware
parameter counting in `gguf_writer.py:121-137`). The stock converter
`vendor/llama.cpp/convert_lora_to_gguf.py:422-428` writes exactly these KVs and is the format
reference to mirror.

Zero-init B makes the step-0 adapter a **provable no-op**: `W·x + scale·B(A·x) = W·x` exactly,
for any A. That property is this project's first end-to-end correctness gate (BLUEPRINT D3) —
the executable test (attach at scale=1, compare logits) lands in S0-06; this ticket must
document the property and produce adapters that satisfy it.

Two loader subtleties are load-bearing, both verified against the pinned commit. First,
**alpha**: the effective scale is `user_scale * alpha / rank` with `rank = b->ne[0]`, but `alpha
== 0` in metadata **silently drops the `alpha/rank` factor** (scale becomes user scale alone —
`vendor/llama.cpp/src/llama-adapter.h:48-88`, the ternary at `llama-adapter.h:55`). Always write
a real alpha. Second, **shapes**: the loader validates `model.ne[0]==a.ne[0] &&
model.ne[1]==b.ne[1] && a.ne[1]==b.ne[0]` for normal targets, but `token_embd.weight` uses a
**flipped, A-transposed convention** (`model.ne[0]==b.ne[1] && model.ne[1]==a.ne[1]`) —
`vendor/llama.cpp/src/llama-adapter.cpp:356-368`. The loader also rejects adapters whose
`general.architecture` differs from the base model's
(`vendor/llama.cpp/src/llama-adapter.cpp:202-218`).

## What to do

1. Consume gguf-py **from the vendored tree** (`vendor/llama.cpp/gguf-py`, pinned by the
   submodule) — e.g. a path dependency or documented `pip install -e vendor/llama.cpp/gguf-py`
   dev step. Soft coordination: the submodule lands in S0-02; if developing before it merges,
   use a llama.cpp checkout at `4f37f519722aa3242eecb7649466b4a4a2d6d6da` and switch the path
   before this PR merges. Do not pin PyPI `gguf` — the mirrors must match the vendored commit
   atomically (BLUEPRINT §3).
2. `src/learning_llamas/adapter.py` — enumeration: `enumerate_targets(base_gguf_path, preset=DEFAULT,
   include_output=False, include_token_embd=False)` reads the base GGUF via `GGUFReader` (mmap)
   and returns the LoRA target list: tensor name, shape `(n_in, n_out)` in GGUF `ne` terms, and
   dtype. Default preset per BLUEPRINT D5: suffix-match `attn_q`, `attn_k`, `attn_v`,
   `attn_qkv`, `attn_output`, `ffn_up`, `ffn_gate`, `ffn_down` (`.weight` tensors, all layers);
   opt-in extras: `output.weight`, `token_embd.weight`. Pure name/shape logic — no graph walk
   (that is S1-11's preflight).
3. `create_zero_adapter(base_gguf_path, out_path, r, alpha, sigma, seed, preset...)` — write the
   adapter GGUF:
   - KVs: `general.type="adapter"` (`GGUFWriter.add_type`, `gguf_writer.py:496`),
     `general.architecture=<base model's arch>` (`add_architecture`, `gguf_writer.py:499` — must
     equal the base arch or the loader throws), `adapter.type="lora"`,
     `adapter.lora.alpha=<f32>`. **Validate `alpha > 0`** at the API boundary (reject 0 with an
     error citing the scale-drop behavior).
   - Per target: `<name minus .weight>.lora_a` with ne `[n_in, r]` (numpy array shape `(r,
     n_in)`), values `N(0, sigma)` from a seeded `numpy.random.Generator`, F32; `<name minus
     .weight>.lora_b` with ne `[r, n_out]` (numpy `(n_out, r)`), zeros, F32.
   - `token_embd.weight` targets use the flipped convention so the shape checks at
     `llama-adapter.cpp:356-360` pass: write A/B such that `b.ne[1] == n_embd` and `a.ne[1] ==
     n_vocab` (derive from the base tensor; add an explicit unit test for it).
   - Sensible defaults documented: `r=16`, `alpha=r` (scale 1.0 at user_scale 1), `sigma`
     default documented in the docstring (any small σ is correct for the no-op property since
     B=0; pick and document one, e.g. `1/√r`).
4. Read helpers: `read_adapter(path)` returns metadata (arch, alpha, per-target rank, tensor
   names/shapes/dtypes) by walking the KVs and `.lora_a`/`.lora_b` pairs; validate pairing
   (every `lora_a` has its `lora_b`) and F32 dtype; raise on `alpha == 0` with a warning-grade
   message when reading foreign adapters (do not hard-fail — stock files may legitimately carry
   user-scale-only semantics).
5. Document the zero-B no-op property in the module docstring, stating that the executable smoke
   test lives in S0-06 and that trained-adapter fidelity (load in `llama-cli --lora`) is
   exercised from S1-03 onward.
6. Unit tests `tests/test_adapter.py`, **no native build required**: synthesize a minimal fake
   base GGUF in-test with `GGUFWriter` (a few `blk.N.*` tensors + `token_embd.weight` +
   `output.weight`, tiny dims, plus `general.architecture="llama"`); assert enumeration matches
   the preset exactly; create an adapter; re-read it and assert all KVs, shapes (both
   conventions), dtypes, `B == 0`, `A` reproducible under the seed; assert `alpha=0` is rejected
   at create time.

## Out of scope

- The no-op logits smoke test against a real loaded model (S0-06 owns it; needs S0-04's bindings
  and fixture models).
- Adapter save-from-training, in-place update, and merged-model export (S1-08).
- Optimizer-state sidecar/checkpoint format (S1-09; BLUEPRINT G14 — the adapter GGUF is
  interchange, not resume).
- Trainability preflight / graph walk and the mirrored `LLM_TENSOR_INFOS` op-classification
  table (S1-11; BLUEPRINT D5).
- MoE expert targets (`build_lora_mm_id` operands) and norm-vector adapters (loader ignores
  norms today, BLUEPRINT §1.3).

## Acceptance criteria

- [ ] `pytest tests/test_adapter.py` passes in a pure-Python environment (no compiled learning-llamas
      extension) using vendored gguf-py.
- [ ] A created adapter contains exactly the four KVs with `general.type="adapter"`,
      `adapter.type="lora"`, base-matching `general.architecture`, and `adapter.lora.alpha > 0`
      (F32).
- [ ] For every non-embedding target: `.lora_a` ne `[n_in, r]`, `.lora_b` ne `[r, n_out]`, both
      F32, B all-zeros — satisfying the loader checks at `llama-adapter.cpp:362-367`.
- [ ] A `token_embd.weight` adapter tensor pair satisfies the flipped checks at
      `llama-adapter.cpp:356-360` (dedicated test).
- [ ] `create_zero_adapter(..., alpha=0)` raises with a message referencing the alpha/rank
      scale-drop.
- [ ] `enumerate_targets` on the fixture returns the default preset exactly;
      `include_output`/`include_token_embd` extend it as documented.
- [ ] Same seed ⇒ byte-identical adapter file (reproducibility test).

## Testing & verification

- `tests/test_adapter.py` (new), plain pytest, runnable on any dev machine; joins the S0-06
  harness unchanged and runs in `ci-cpu / test` per-PR (S0-07). The definitive cross-check —
  stock `llama_adapter_lora_init` (`vendor/llama.cpp/src/llama-adapter.cpp:420`) loading our
  file and the zero-B no-op logits comparison — is S0-06's smoke test and S1-03's `llama-cli
  --lora` gate; this ticket's tests must make those unlikely to fail (shape/KV/dtype exactness).
- No MODE_GRAD applicability: pure-Python file format work.

## PR notes

- Branch: `ticket/S0-05-zero-init-adapter-gguf`.
- Single learning-llamas PR; no vendored llama.cpp changes.
- Upstreaming disposition: **fork-local** (Python product code; the file format is already
  upstream's).
- Format-writing logic is informed by `convert_lora_to_gguf.py` (MIT): add a provenance note in
  `adapter.py`'s docstring per S0-01 policy (pattern reference, not code copy).
