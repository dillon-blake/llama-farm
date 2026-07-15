---
id: S1-45
title: "Data layer: the S1-06 vocab guard (no new special tokens) + de-vacuify the special-token test"
stage: 1
track: python
size: S
deps: [S1-06]
status: done
pr: null
---

# S1-45 — Data layer vocab guard

**One-line outcome:** the chat-template path now validates the special tokens it depends on
against the model's actual vocabulary and refuses, loudly and by name, a token the base vocab
lacks — closing S1-06 item 5 / acceptance criterion (d); and the special-token parse test that
could not fail on the fixtures now can.

## Why (context)

The Stage 0+1 audit (2026-07-15, data-pipeline section) rated this a **major defect**: S1-06
item 5 and acceptance criterion (d) require a vocab guard — walking the rendered template's
special tokens plus any user-supplied extras and raising the v1 no-new-special-tokens error
(BLUEPRINT §8) when one is not in the vocab — and no such guard existed anywhere in `src/` or
`tests/` (grep for `vocab.guard` / `no.new.special` / `unknown.special`: zero hits).

The rule it enforces is BLUEPRINT §8's one v1 hard constraint: **no vocab resize and no new
special tokens.** The base token embeddings are frozen and quantized, so a control token the base
vocab lacks has no embedding to train. When a template names such a token, `parse_special` cannot
match it and the tokenizer byte-falls-back into a dozen per-character tokens — and the model is
trained to *spell out* `< | i m _ s t a r t | >` where it should emit one control token. That is
silent: the loss still falls, and (because the S1-06 mask lands on token boundaries, not string
offsets) the mask still lines up. The guard is the only thing that says so.

The same audit flagged a related **minor/vacuous-test**:
`test_special_token_text_is_parsed_as_one_token` asserted
`len(encode('<|im_start|>', parse_special=True)) <= len(encode(..., False))`. The fixture's
512-token vocab has no `<|im_start|>`, so both sides byte-fall-back to the *same* 11 tokens and
`11 <= 11` holds no matter what `parse_special` does — even if it did nothing. There was no input
on the current fixtures that could make it fail.

## What was done

- `src/learning_llamas/data/template.py`:
  - `UnknownSpecialTokenError(ValueError)` — the named no-new-special-tokens error.
  - `guard_special_tokens(tokenizer, special_tokens)` — for each declared special-token string,
    tokenize under `parse_special=True` and require **exactly one** id. A token the vocab has
    collapses to its single id; one it lacks byte-falls-back to many, so `len != 1` is precisely
    the failure. The message names the offending token and what it byte-fell-back to. Empty
    strings are skipped (a model with no BOS reports its BOS text as `""` — "no token", not "an
    unknown one").
  - `ChatTemplate.from_model(..., require_special=())` runs the guard over the model's own
    BOS/EOS text plus any `require_special` the caller declares, on the normal load path — so the
    guard is wired, not decorative. Exported from `learning_llamas.data`.
- `tests/test_data_roundtrip.py`:
  - `test_special_token_text_is_parsed_as_one_token` rewritten to use the model's own BOS (`<s>`,
    genuinely in the fixture vocab): the special path returns the single BOS id, the plain path
    spells out its bytes, and the assertion is now strict (`==` on the id and `<` on the length).
  - `test_the_special_token_test_is_not_vacuous` — the repo-idiom mutation check: runs the
    parse_special-off configuration and asserts both of the sibling's load-bearing checks go red.
  - `test_vocab_guard_accepts_the_models_own_special_tokens`,
    `test_vocab_guard_raises_on_a_token_absent_from_the_vocab` (acceptance (d)),
    `test_vocab_guard_is_not_vacuous` (both directions pinned on one tokenizer), and
    `test_from_model_wires_the_vocab_guard` (proves the guard is reachable from `from_model`).

## Design choice (spec was ambiguous — conservative reading, documented per instructions)

S1-06 item 5 says "walk the rendered template's special-token strings." Taken literally as "scan
rendered output for `<|...|>`-shaped substrings and reject any that byte-fall-back," this is
**unsound and would break every existing round-trip test**: the tiny fixtures' own ChatML markers
(`<|im_start|>`, `<|im_end|>`) are byte-fallback on the 512-token vocab and *correct* — their
mask lands on the trailing newline. A tokenizer's byte-fallback of a control string is
indistinguishable from a legitimately literal one, so such a scan would reject a valid template.

The conservative, sound reading — the one implemented — validates a **declared** set: the
special-token strings the model exposes to the template (its BOS/EOS text) plus any the caller
names via `require_special` (the "user-supplied extra tokens" the ticket also lists). That is the
protective surface that matters in practice: a user who brings a `--chat-template` override naming
control tokens the base vocab lacks gets a named error at load time, not garbage 50 steps in. The
core check (`parse_special` must collapse the string to a single id) is exactly the ticket's
"does not tokenize to existing vocab ids → raise."

The fixture generator was **not** touched: the vacuous test was fixed by pointing it at a token
the fixture vocab already has (its BOS), which is smaller and more honest than extending the
vocab — and avoids any `gen_tiny_llama.cache_key` / CRLF-normalization concern.

## Acceptance criteria

- [x] `guard_special_tokens` raises `UnknownSpecialTokenError`, naming the token, when a declared
      special token is not a single vocab id; passes silently when it is.
- [x] The guard is reachable from `ChatTemplate.from_model` (via `require_special`), not a
      standalone helper nothing calls.
- [x] `test_special_token_text_is_parsed_as_one_token` can now fail — proven by a mutation test
      that exercises the parse_special-off configuration.
- [x] `pytest tests/test_data_roundtrip.py` green; ruff clean; full suite green.

## Out of scope

- Vocab resize / new special tokens / embedding training (still deferred to P4; BLUEPRINT §8).
- Per-message incremental-prefix masking (a separate S1-06 minor the audit lists in mask.py).
