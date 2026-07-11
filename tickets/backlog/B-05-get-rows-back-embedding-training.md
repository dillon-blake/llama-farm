---
id: B-05
title: "GET_ROWS_BACK generalization + embedding/lm_head training"
stage: backlog
track: kernels
size: L
deps: [S3-02]
status: open
pr: null
---

# B-05 — GET_ROWS_BACK generalization + embedding/lm_head training

**One-line outcome:** **DEFERRED** — batch-dim + scatter `GET_ROWS_BACK` on CUDA, the Metal M9
port, and the llama-context embedding FIXMEs unpicked, enabling token_embd/lm_head (LoRA) training.

**Activation trigger:** trainable embeddings or lm_head enter product scope — a committed need
such as new-token/vocab adaptation or embedding-LoRA training. The op is **dormant for
quantized-base LoRA** (embeddings frozen; ROADMAP §5 C5, BLUEPRINT §8/§9 P4) — do not schedule
without a concrete need.

## Why (context)

`GET_ROWS_BACK` scatters token-embedding gradients back into the `[vocab, n_embd]` grad table.
It is training-blocked in three places. CUDA supports only F32 with no batch dims — `supports_op`
requires `ne[2] == 1 && ne[3] == 1`
(`vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:4709-4711`) — and the kernel is an
O(vocab × tokens) scan: `k_get_rows_back_float` loops over *every* grad row per destination row
(`vendor/llama.cpp/ggml/src/ggml-cuda/getrows.cu:80-104`), catastrophic at 128k vocab. Metal has
no kernel at all (ROADMAP §6 M9). Vulkan has a deterministic scan (ROADMAP §2) whose batch-dim
behavior is unaudited.

Above the kernels, `llama_set_param` hard-excludes `token_embd.weight` and `rope_freqs.weight`
with explicit FIXMEs (`vendor/llama.cpp/src/llama-context.cpp:3207-3212`); BLUEPRINT §8 notes
token_embd/lm_head LoRA is inference-format-supported but training it needs those FIXMEs
revisited, and the embedding LoRA path uses a flipped A/B convention the shim must honor
(`vendor/llama.cpp/src/llama-adapter.cpp:356-368`). Determinism policy applies (ADR-0002 /
gate G-B): the scatter must default to a deterministic scheme (sort-by-destination-row +
segmented reduction, in the spirit of the mmid compaction reused by E2), atomicAdd opt-in only.

## What to do

1. **CUDA:** replace the scan (`getrows.cu:80-104`) with a deterministic scatter (sort/compact
   token indices per destination row + segmented reduction); add batch dims; widen the
   `supports_op` gate (`ggml-cuda.cu:4709-4711`); atomic variant behind the G-B opt-in flag.
2. **Metal M9:** deterministic per-dst-row scan kernel first, patterned on the `kernel_set_rows`
   family (`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:9621-9682`); atomic-float
   variant pending MSL verification, opt-in only.
3. **Vulkan audit:** confirm the existing scan handles batch dims; extend if not.
4. **Unpick the FIXMEs:** decide param handling for `token_embd.weight`/`rope_freqs.weight`
   (`llama-context.cpp:3207-3212`); extend llama-farm's S1-01 filter to offer embedding-LoRA A/B,
   honoring the flipped convention (`llama-adapter.cpp:356-368`).
5. **Tests:** extend `test-backend-ops` GET_ROWS_BACK cases with batch dims and wide-vocab shapes
   (MODE_GRAD vs CPU oracle); add an e2e tiny-model run training token_embd + lm_head LoRA whose
   loss falls and whose adapter round-trips through stock `llama-cli --lora`.

## Out of scope

- Vocab resize / new special tokens (data-layer + adapter-format work; separate product ticket).
- Full (non-LoRA) embedding fine-tuning on quantized bases — explicit non-goal.
- Sparse-CE / lm_head chunking changes — S1-04/S1-13.

## Acceptance criteria

- [ ] `test-backend-ops` MODE_GRAD `GET_ROWS_BACK` (incl. new batch-dim cases) passes vs the CPU
      oracle within the ADR-0002 tolerance on CUDA, Metal, and Vulkan.
- [ ] Determinism: bitwise-identical grad tables across reruns on the default path.
- [ ] CUDA scatter beats the old scan at 128k-vocab shapes (benchmark in the PR).
- [ ] e2e: tiny-model token_embd/lm_head LoRA run — loss falls; adapter loads in stock
      `llama-cli --lora` (flipped-convention round-trip asserted).
- [ ] `ci-cpu`, `ci-cuda`, `ci-metal` green on the targeted sweeps.

## Testing & verification

Vendored `tests/test-backend-ops` MODE_GRAD vs the CPU oracle (ADR-0002): `ci-cuda` and `ci-metal`
targeted per-PR, full nightly; `ci-vulkan` covers the batch-dim audit. The e2e embedding-training
test lives in llama-farm `tests/` and runs in `ci-cpu` per-PR.

## PR notes

- Branch: `ticket/B-05-get-rows-back-embedding-training`.
- Two-repo flow per S0-02: fork PR + llama-farm submodule bump.
- Upstreaming disposition: **upstream-early** for the kernel generalizations (pure improvements to
  an existing upstream op with existing tests, ROADMAP §11 triage a); the llama-context FIXME
  unpick goes upstream with it (mainline carries the same FIXMEs).
