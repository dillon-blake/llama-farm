---
id: S0-02
title: "Vendor llama.cpp as pinned submodule + fork + patch queue"
stage: 0
track: infra
size: S
deps: ["S0-01"]
status: open
pr: null
---

# S0-02 — Vendor llama.cpp as pinned submodule + fork + patch queue

**One-line outcome:** `vendor/llama.cpp` exists as a git submodule pinned at `4f37f51`, pointing
at the project's GitHub fork branch `llama-farm-base`, with a `patches/` queue mechanism and
ADR-0001 recording lineage and rebase cadence.

## Why (context)

Layer 0 of the architecture is vendored llama.cpp (BLUEPRINT §3): the C shim compiles against
**private `src/` internals** (S0-03), so the exact llama.cpp commit is part of llama-farm's ABI,
and every kernel ticket (track `kernels`) lands its changes in llama.cpp core code. A long-lived
private fork would fight ggml's fast-moving op table, so ROADMAP §11 prescribes the strategy
this ticket implements: a GitHub fork with a `llama-farm-base` branch pinned at `4f37f51`,
upstream-friendly changes PR'd to mainline early, fork-first changes carried as
op-enum-tail/new-file diffs, a **monthly rebase cadence**, and a **full `test-backend-ops`
MODE_GRAD rerun on every vendor bump**.

There is a lineage question to settle and record (BLUEPRINT §3, "Which llama.cpp is Layer 0?").
The blueprint flagged the local checkout as possibly a drifted fork because it carries
`src/llama-ext.h` (a staging header explicitly marked "new llama.cpp API ... considered WIP",
`vendor/llama.cpp/src/llama-ext.h:1-5`) and the batch adapter API `llama_set_adapters_lora`
(`vendor/llama.cpp/include/llama.h:690`). Verification during ticket authoring found: the
checkout at `/home/dillon/Desktop/llama-farm/llama.cpp` is **clean** (`git status` empty) at
commit `4f37f519722aa3242eecb7649466b4a4a2d6d6da` ("server: accept null sampling params
(#25538)") on branch `master` with sole remote `https://github.com/ggml-org/llama.cpp.git`, and
`src/llama-ext.h` is a **tracked file with upstream-PR history** (e.g. #24506) — i.e. the
checkout appears to be genuine upstream, and `llama-ext.h` landed upstream. ADR-0001 must
confirm this independently (fetch upstream, check `4f37f51` is reachable from
`ggml-org/llama.cpp` master) and record the answer, because it determines the upstreaming path
and the ctypes struct mirrors (S0-04).

The `patches/` queue exists for fork-local diffs that must apply *before* a corresponding fork
PR merges (e.g. an urgent build fix while a fork PR is in review). Steady-state, the queue
should be empty: real changes live as commits on `llama-farm-base`.

## What to do

1. Create the GitHub fork of `ggml-org/llama.cpp` under the project owner's account. Create
   branch `llama-farm-base` pointing exactly at `4f37f519722aa3242eecb7649466b4a4a2d6d6da`;
   protect it (no force-push).
2. Verify lineage and record evidence: from the local checkout, `git fetch` upstream and confirm
   `4f37f51` is an ancestor of `ggml-org/llama.cpp` `master` (`git merge-base --is-ancestor`);
   confirm `src/llama-ext.h` exists in the upstream tree at that commit. Capture command output
   for the ADR.
3. In llama-farm, `git submodule add <fork-url> vendor/llama.cpp`, checked out at `4f37f51` and
   tracking `llama-farm-base`. Commit `.gitmodules` + the gitlink.
4. `patches/` mechanism: `patches/README.md` (format: numbered `git format-patch` files, applied
   in order onto the submodule HEAD) and `scripts/apply-patches.sh` — idempotent, applies every
   `patches/*.patch` with `git am` (or `git apply --check` first), exits 0 on an empty queue,
   fails loudly on conflict. Document that a patch must be deleted once its content merges into
   `llama-farm-base`.
5. Write `docs/adr/ADR-0001-vendor-lineage.md`: the lineage finding from step 2 (with evidence);
   the decision to pin the fork's `llama-farm-base` at `4f37f51`; the ROADMAP §11 triage classes
   for future changes (upstream-early / in-fork-first / rebase-patch fallback); **monthly rebase
   cadence** onto upstream master; and the rule that **every vendor bump reruns the full
   MODE_GRAD suite** before the submodule-bump PR merges.
6. Document the two-repo PR flow for kernel tickets in ADR-0001 (matching `tickets/README.md`):
   implementation PR against the fork's `llama-farm-base` with the ticket ID in the title, then
   a trivial llama-farm PR bumping the `vendor/llama.cpp` gitlink and referencing the same
   ticket ID.
7. Update the placeholder `CMakeLists.txt`/README from S0-01 only insofar as pointing at
   `vendor/llama.cpp` for S0-03; do not build it yet.

## Out of scope

- Building the vendored tree (S0-03 owns CMake/scikit-build integration).
- Any actual patch content — the queue ships empty.
- gguf-py packaging/pinning decisions beyond noting the vendored copy exists (consumed by
  S0-05).
- The numerics/determinism ADR (ADR-0002, S0-09).

## Acceptance criteria

- [ ] `git submodule status` in a fresh clone (after `git submodule update --init`) shows
      `vendor/llama.cpp` at `4f37f519722aa3242eecb7649466b4a4a2d6d6da`.
- [ ] `.gitmodules` URL points at the project fork (not `ggml-org/llama.cpp`), and branch
      `llama-farm-base` exists on the fork at that commit (URL recorded in ADR-0001).
- [ ] `docs/adr/ADR-0001-vendor-lineage.md` exists and contains: lineage verification evidence,
      rebase cadence, MODE_GRAD-on-bump rule, and the two-repo PR flow.
- [ ] `scripts/apply-patches.sh` exits 0 on the empty queue; `patches/README.md` documents the
      format and lifecycle.
- [ ] `vendor/llama.cpp/src/llama-ext.h` and `vendor/llama.cpp/gguf-py/` exist in the
      initialized submodule (spot-check that the pin is the expected tree).

## Testing & verification

No automated harness yet. Verification is the fresh-clone sequence recorded in the PR
description: `git clone && git submodule update --init --depth 1 && ./scripts/apply-patches.sh
&& git -C vendor/llama.cpp rev-parse HEAD`. When S0-07 lands, `ci-cpu` initializes the submodule
on every PR (with caching), which continuously re-verifies the pin; the MODE_GRAD-on-bump rule
is enforced procedurally via ADR-0001 until the nightly lane exists.

## PR notes

- Branch: `ticket/S0-02-vendor-llama-cpp-submodule-fork`.
- One PR in llama-farm; the fork/branch creation itself happens on GitHub and is referenced, not
  diffed.
- Upstreaming disposition: **fork-local** (infrastructure; nothing upstreamable).
- This ticket *establishes* the two-repo flow all later kernel tickets follow; it performs no
  fork-side code changes itself.
