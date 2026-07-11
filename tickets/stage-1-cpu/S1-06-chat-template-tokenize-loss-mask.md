---
id: S1-06
title: "Data layer: chat templating, tokenization, loss-mask round-trip"
stage: 1
track: python
size: M
deps: ["S0-04"]
status: open
pr: null
---

# S1-06 — Data layer: chat templating, tokenization, loss-mask round-trip

**One-line outcome:** `src/llama_farm/data/` renders chats through the GGUF-embedded
template with jinja2, tokenizes via the llama.cpp C API, and computes loss-mask boundaries
at tokenization time — proven by a round-trip test that masks align with actual template
tokens.

## Why (context)

BLUEPRINT D7 puts the entire data pipeline in Python and bypasses `ggml_opt_dataset`
(which is a fixed tensor pair with no masks/packing — BLUEPRINT G10): chat templating is
jinja2 over the template embedded in the GGUF, tokenization uses the model's own
tokenizer via the C API, and loss-mask boundaries are computed **at tokenization time**.
The mask is what makes SFT correct — prompt tokens get weight 0, completion tokens weight
1 — and it feeds the `ce_sparse` weights input (S1-04/S1-05), where weight 0 produces
exactly-zero gradient.

The critical correctness constraint comes from BLUEPRINT §8: mask boundaries must be
computed against the model's **actual template-token behavior**, never string offsets.
Tokenizers merge characters across text boundaries, so "tokenize the prompt string,
count tokens, mask that many" silently misaligns whenever the completion's first token
would have merged with the prompt's tail. The defense is incremental rendering: render
the conversation prefix-by-prefix, tokenize each rendered prefix, and derive span
boundaries from token counts — then verify each prefix's token sequence is an exact
prefix of the next (the round-trip test this ticket must land).

Everything needed is exported C API, extending the S0-04 `_ffi` layer: the embedded
template via `llama_model_chat_template` (`vendor/llama.cpp/include/llama.h:621`;
implementation `vendor/llama.cpp/src/llama-model.cpp:2645`; stored under the GGUF key
`tokenizer.chat_template`, `vendor/llama.cpp/gguf-py/gguf/constants.py:285`),
`llama_tokenize` (`llama.h:1142`), `llama_token_to_piece` (`llama.h:1156`),
`llama_detokenize` (`llama.h:1170`), and the vocab special-token accessors incl.
`llama_vocab_get_add_bos`/`llama_vocab_get_add_eos` (`llama.h:1084-1094`). One v1 hard
rule (BLUEPRINT §8): **no vocab resize and no new special tokens** — base embeddings are
frozen and quantized, so adapters must use the base vocab; the data layer validates and
errors rather than silently emitting unknown-token garbage.

## What to do

1. Extend `src/llama_farm/_ffi/llama.py` (S0-04 symbol table) with the tokenizer/template
   surface: `llama_model_chat_template`, `llama_tokenize`, `llama_token_to_piece`,
   `llama_detokenize`, `llama_model_get_vocab`, `llama_vocab_n_tokens`, the special-token
   getters (`llama_vocab_bos/eos/eot/pad`, `llama.h:1084-1089`) and
   `llama_vocab_get_add_bos`/`add_eos` (`llama.h:1092-1093`).
2. `src/llama_farm/data/template.py`: extract the embedded template (name arg `NULL` for
   the default; error clearly if the GGUF has none, suggesting an explicit
   `--chat-template` override which the API also accepts as a string); render with
   jinja2 in a sandboxed environment mirroring the common HF conventions (`messages`,
   `add_generation_prompt`, `bos_token`/`eos_token` context vars). Note in the docstring
   that llama.cpp renders with its own minja engine — jinja2 output equivalence for the
   supported model set is exactly what the round-trip test checks.
3. `src/llama_farm/data/tokenize.py`: `tokenize(text, add_special, parse_special)`
   wrapping `llama_tokenize` with the two-call length-negotiation convention (negative
   return = required size) and `detokenize(tokens)`. Templated text is tokenized with
   `parse_special=True` (templates embed special-token text) and `add_special=False`
   (the template supplies BOS/EOS; assert against `llama_vocab_get_add_bos` and document
   the interaction).
4. Loss-mask computation in `src/llama_farm/data/mask.py`: for a chat sample, render the
   incremental prefixes (per message, and within the final assistant message the
   pre-completion prefix), tokenize each, verify the token-prefix property, and emit
   `(tokens, weights)` with weight 0 on prompt/template tokens and weight 1 on completion
   tokens (assistant content the sample trains on). If the prefix property fails for a
   boundary, raise a descriptive error naming the offending boundary and the merged
   tokens — do not silently shift the mask.
5. Vocab guard: walk the rendered template's special-token strings and any user-supplied
   extra tokens; if any does not tokenize to existing vocab ids, raise the v1
   no-new-special-tokens error (BLUEPRINT §8).
6. Round-trip tests `tests/test_data_roundtrip.py` on the S0-06 fixture models (which
   must embed a chat template — extend the fixture generator if needed): for several chat
   structures (single-turn, multi-turn, system prompt, empty system), assert (a) the
   token-prefix property holds across all boundaries; (b) detokenizing the weight-1 span
   reproduces the expected completion text exactly; (c) masks are invariant to prompt
   content sharing a tokenizer merge boundary with the completion (adversarial case with
   no leading space); (d) the vocab guard raises on an unknown special token.

## Out of scope

- Sample packing, boundary masking across packed samples, and fixed-shape collators
  (S1-07 builds directly on this ticket's `(tokens, weights)` output).
- The trainer that consumes the masks (S1-05) and batching into ggml input tensors
  (S1-02/S1-05 via `ggml_backend_tensor_set`).
- Vocab resize / new special tokens / embedding training (deferred to P4; BLUEPRINT §8).
- Dataset formats/loaders beyond a plain list-of-chats interface (later product work;
  `tickets/backlog/` candidate if needed).

## Acceptance criteria

- [ ] `pytest tests/test_data_roundtrip.py` passes on the Linux CPU VM against the S0-03
      build, covering at least four chat structures.
- [ ] The detokenized weight-1 span equals the expected completion string for every test
      case (exact match, no whitespace slack).
- [ ] The adversarial merge-boundary case passes (mask lands on actual token boundaries,
      not string offsets).
- [ ] A template-less GGUF and an unknown special token each produce the documented
      errors (tested).
- [ ] The `_ffi` symbol-table test (S0-04) resolves all newly added symbols.
- [ ] `ci-cpu / test` green per-PR with the new tests.

## Testing & verification

- `tests/test_data_roundtrip.py` (new) plus small unit tests for `template.py`/
  `tokenize.py`; pytest with S0-06 fixture models; runs in `ci-cpu / test` per-PR
  (S0-07), full suite nightly.
- No MODE_GRAD applicability: pure Python data-layer work over exported C API.

## PR notes

- Branch: `ticket/S1-06-chat-template-tokenize-loss-mask`.
- Single llama-farm PR; no vendored llama.cpp changes.
- Upstreaming disposition: **fork-local** (product code; the template/tokenizer APIs are
  already upstream's).
- Soft coordination: S1-05 consumes `(tokens, weights)`; S1-07 consumes the same plus
  per-sample lengths — keep the sample record a small typed structure so both plug in
  without reshaping.
