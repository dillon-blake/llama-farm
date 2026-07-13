---
id: S1-35
title: "Public API: promote the model loader, export the surface, write the quickstart"
stage: 1
track: python
size: S
deps: [S1-05, S1-08, S1-16]
status: pr-open
pr: null
---

# S1-35 — Public API: promote the model loader, export the surface, write the quickstart

**One-line outcome:** `import learning_llamas` binds a surface a user can actually train with,
and `docs/quickstart.md` walks a GGUF to a `--lora`-loadable adapter without touching a private
name.

## Why (context)

The library had 249 passing tests, a working SFT/DPO/GRPO stack, and **no way for a user to load a
model**:

- `import learning_llamas` bound exactly one name — `__version__`.
- Every public entry point takes `libs: _ffi.Libraries`, and `_ffi` is private with no public
  accessor.
- Every trainer takes a `TrainableModel`, which is a `Protocol` **with no implementation anywhere
  in `src/`**. The only implementation was `tests/conftest.py:Model`, whose own docstring said
  "Not a public API".
- Nothing in `src/` ever called `llama_model_load_from_file`. The one occurrence was a docstring
  example in `_ffi/__init__.py`.

That is not a documentation gap; it is a missing module. It also explains why no usage
documentation existed — `train_sft` and `train_dpo` appeared in zero `.md` files repo-wide.

Two further copies of the loader had accreted in `benches/` (`cpu_train_step.py` carried its own
`Model` class; `grad_checkpointing.py` imported the one from `tests.conftest`), which is what
duplication looks like when the thing being duplicated should have been public all along.

## What to do

1. `src/learning_llamas/model.py` — a public `Model` (loader + context + adapter lifecycle) and a
   public `libraries()` accessor. `Model` satisfies `TrainableModel` structurally.
2. Route llama.cpp's logs into `logging` under the `llama.cpp` logger rather than silencing them
   or letting them spam stderr. Silencing throws the errors away with the noise.
3. Adapter ownership: `close()` frees the adapter **only if this `Model` loaded it**. A shared
   adapter (GRPO's two contexts) has exactly one owner; freeing it twice is a double free.
4. `Model.chat_template()` — the GGUF's embedded template, so the quickstart does not have to
   hand-write one.
5. Re-export the real surface from `learning_llamas/__init__.py`.
6. Rewire `tests/conftest.py` and both benches onto the public `Model`, deleting the duplicates.
7. `docs/quickstart.md` — SFT and GRPO, end to end, runnable.

## Out of scope

- Publishing to PyPI. The install path stays "clone and `pip install -e .`".
- A CLI. `export.py` already has a `_main`; a general CLI is backlog.
- Changing `save_adapter`'s signature. It reads a live adapter handle, which carries no
  architecture or alpha metadata, so the `read_adapter` round-trip is the honest way to supply
  them.

## Acceptance criteria

- [x] `import learning_llamas` exposes `Model` and `libraries`; `__all__` has 26 entries and every
      one of them resolves.
- [x] A test trains an adapter end to end **importing no private name**, and the loss falls.
- [x] A trained adapter changes the logits, reloads, and is attachable to a fresh context.
- [x] Two `Model`s sharing one adapter can both be closed without a double free (checked out of
      process, because the failure mode is an abort).
- [x] `tests/conftest.py` and `benches/` contain no `Model` class of their own.
- [x] The suite is green: 261 tests.

## Testing & verification

`tests/test_model.py` (12 tests). The load-bearing one is the end-to-end quickstart run: if the
public API were missing a step, that test would not be writable at all. The double-free check runs
in a subprocess so an abort is a readable failure rather than a dead test session.

The GRPO half of the quickstart was executed against the tiny fixture, which is how its `n_seq_max`
sizing got fixed — the first draft was wrong, and only running it said so.

## PR notes

- Branch: `ticket/S1-35-public-api`.
- Upstreaming disposition: `n/a` — no vendored llama.cpp changes.
