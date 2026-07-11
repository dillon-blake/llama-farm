---
id: S0-01
title: "Repo scaffolding, license, packaging skeleton, provenance policy"
stage: 0
track: infra
size: S
deps: []
status: open
pr: null
---

# S0-01 — Repo scaffolding, license, packaging skeleton, provenance policy

**One-line outcome:** an empty-but-buildable repo skeleton — pyproject (scikit-build-core),
`src/llama_farm/` package stub, MIT LICENSE, NOTICE, a written provenance-header policy,
lint/format/pre-commit config, and `.gitignore` — so every later ticket has a place to land and
a license story to follow.

## Why (context)

llama-farm is a four-layer system (BLUEPRINT §3): vendored llama.cpp (Layer 0, arrives in
S0-02), a C shim `libllamafarm` in `csrc/` (Layer 1, arrives in S0-03), a ctypes binding `_ffi`
(Layer 2, S0-04), and the Python library (Layer 3). This ticket creates the repo tree from
BLUEPRINT §4 so those tickets have stable paths to fill in, and pins the packaging approach:
scikit-build-core + CMake over a vendored submodule, deliberately *not* depending on
llama-cpp-python (BLUEPRINT §3, "Packaging").

Licensing must be settled before the first line of copied code. llama-farm itself is **MIT** —
chosen so code we later upstream into llama.cpp (itself MIT, `vendor/llama.cpp/LICENSE`) moves
without friction (BLUEPRINT §7, "Licensing & provenance"; ROADMAP §11 plans substantial
upstreaming). Two provenance regimes apply to our two reference codebases. Copied llama.cpp code
is MIT: retain copyright notices and add per-file provenance headers. unsloth is **ideas-only**:
its repo default is Apache-2.0, but the LICENSE carve-out (unsloth `LICENSE:190-191`) puts
`studio/*` and `unsloth_cli/*` under AGPLv3 (root `COPYING` is the AGPL text), `kernels/moe/**`
is AGPL, one function-level AGPL marker sits inside an otherwise-Apache file
(`unsloth/models/rl_replacements.py:1191`), and four files are LGPL-3.0+
(`unsloth/utils/packing.py`, `unsloth/utils/attention_dispatch.py`, `unsloth/utils/__init__.py`,
`unsloth/kernels/rope_embedding.py`) — a tier a naive AGPL grep misses (ROADMAP §13). The policy
this ticket writes down: **math and design may be reimplemented from any unsloth file; code may
be copied from none of the AGPL/LGPL files, and Apache-side code only with per-file
attribution.**

Every path in this ticket was re-verified against the checkouts at
`/home/dillon/Desktop/llama-farm/llama.cpp` (clean at `4f37f51`) and
`/home/dillon/Desktop/llama-farm/unsloth` during ticket authoring.

## What to do

1. Create the directory skeleton per BLUEPRINT §4, each with a one-paragraph `README.md` (or
   `.gitkeep`) stating which ticket fills it: `csrc/`, `src/llama_farm/`, `tests/`, `benches/`,
   `patches/`, `docs/adr/`, `docs/dev/`. Do **not** create `vendor/` — S0-02 owns it.
2. `src/llama_farm/__init__.py` with `__version__ = "0.0.1"` and a module docstring naming the
   project goal (one sentence).
3. `pyproject.toml`: `build-backend = "scikit_build_core.build"`; project metadata (name
   `llama-farm`, MIT, Python ≥3.10); a **minimal placeholder** `CMakeLists.txt` (project
   declaration only, no targets) so `pip install -e .` already produces an importable
   `llama_farm` — S0-03 replaces it with the real vendored build. Configure the `wheel.packages`
   entry for `src/llama_farm`.
4. `LICENSE`: MIT, copyright the project owner, year 2026.
5. `NOTICE`: names llama.cpp (MIT, `vendor/llama.cpp/LICENSE`, pinned commit `4f37f51`) as a
   source of copied/adapted code, and unsloth (Apache-2.0 top-level with AGPLv3/LGPL carve-outs;
   **ideas/math only, no code copied**) as a design source.
6. `docs/PROVENANCE.md` — the policy every later PR cites:
   - every file in `csrc/` containing copied/adapted llama.cpp code carries a header: source
     path, source commit (`4f37f51...`), license (MIT), summary of changes;
   - unsloth: reimplementation of math/design permitted from any file; code copying permitted
     from **no** AGPL/LGPL path; list the excluded paths verbatim (the AGPL and LGPL lists from
     the Why section, with `rl_replacements.py:1191` called out as a function-level marker);
   - reviewers block PRs whose diffs contain copied code without a header.
7. Lint/format config: `ruff` (lint + format) configured in `pyproject.toml`; `.clang-format` at
   repo root (LLVM base style, 4-space indent — matches llama.cpp conventions);
   `.pre-commit-config.yaml` running ruff, ruff-format, clang-format, and
   end-of-file/trailing-whitespace hooks.
8. `CONTRIBUTING.md` stub: points at `tickets/README.md` for the workflow, `docs/PROVENANCE.md`
   for licensing, and the ADR directory for binding decisions.
9. `.gitignore`: Python (`__pycache__`, `.venv`, `dist/`, `*.egg-info`), CMake/scikit-build
   (`build/`, `_skbuild/`, `.cache/`), editor cruft, `*.gguf` test artifacts.

## Out of scope

- The `vendor/llama.cpp` submodule and `patches/` apply mechanism (S0-02 — this ticket only
  creates the empty `patches/` dir).
- Any real CMake build of native code (S0-03).
- CI workflows (S0-07); developer build docs (S0-08); the numerics ADR (S0-09).
- Any code under `csrc/` or `src/llama_farm/` beyond the package stub.

## Acceptance criteria

- [ ] In a clean venv, `pip install -e .` succeeds and `python -c "import llama_farm;
      print(llama_farm.__version__)"` prints `0.0.1`.
- [ ] `python -m build --wheel` produces a wheel containing `llama_farm/__init__.py`.
- [ ] `LICENSE` (MIT) and `NOTICE` exist; `NOTICE` names both llama.cpp (MIT) and unsloth
      (ideas-only) with the pinned commit.
- [ ] `docs/PROVENANCE.md` exists, contains the per-file header template, and lists every
      excluded unsloth path named in this ticket (AGPL: `kernels/moe/**`, `studio/`,
      `unsloth_cli/`, `cli.py`, `models/rl_replacements.py:1191` function marker; LGPL: the four
      files above).
- [ ] `pre-commit run --all-files` exits 0 on the committed tree.
- [ ] `CONTRIBUTING.md` exists and links `tickets/README.md`.
- [ ] Directory skeleton `csrc/ src/llama_farm/ tests/ benches/ patches/ docs/adr/ docs/dev/`
      exists in git.

## Testing & verification

No pytest harness exists yet (S0-06). Verification is command-level, recorded in the PR
description: the clean-venv `pip install -e .` + import check, `python -m build --wheel`, and
`pre-commit run --all-files` output. Once S0-07 lands, `ci-cpu / build` re-verifies the install
path on ubuntu-latest and macos-14 per-PR; no action needed here beyond keeping the skeleton
green.

## PR notes

- Branch: `ticket/S0-01-repo-scaffolding-license-provenance`.
- One PR; no vendored llama.cpp changes, so single-repo flow.
- Upstreaming disposition: **fork-local** (project-repo-only content; nothing to upstream).
- This PR *defines* the provenance-header policy; it contains no copied code itself.
