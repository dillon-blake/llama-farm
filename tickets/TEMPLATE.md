---
id: SX-00                # unique ticket ID: S<stage>-<nn> or B-<nn> (backlog)
title: "Short imperative title"
stage: 0                 # 0 groundwork · 1 CPU · 2 Metal · 3 CUDA · 4 Vulkan · backlog
track: infra             # infra | python | shim | kernels | docs
size: M                  # S ≤2 days · M ≤1 week · L 2–3 weeks · XL 4+ weeks
deps: []                 # ticket IDs that must be DONE before this starts
status: open             # open | in-progress | pr-open | done
pr: null                 # PR URL once opened
---

# SX-00 — Title

**One-line outcome:** what exists and works when this ticket is done.

## Why (context)

Enough background that an implementer can work from this ticket alone: what problem
this solves, where it sits in the plan, and the key design constraints. Cite the
design docs (`docs/GGUF-LORA-TRAINING-BLUEPRINT.md`, `docs/KERNEL-ROADMAP.md`) by
section, and code by `path:line` (line anchors are valid at llama.cpp commit
`4f37f51`, the pinned vendor commit).

## What to do

Concrete, ordered work items. Name the files to create/change. For vendored
llama.cpp code use `vendor/llama.cpp/...` paths (the vendor tree lands in S0-02).

## Out of scope

Explicit exclusions and deferrals, with a pointer to the ticket that owns them
(or `tickets/backlog/`).

## Acceptance criteria

- [ ] Every item objectively verifiable: a test that passes, a CI lane that is
      green, an artifact that exists. No "works well" criteria.

## Testing & verification

Which tests to add/extend, where they live, and where they run (local/VM + which
GitHub Actions lane). Kernel tickets: `test-backend-ops` MODE_GRAD cases against
the CPU oracle; training tickets: unit tests + (where applicable) the tiny-model
convergence gate (S1-12).

## PR notes

- Branch: `ticket/<id>-<slug>`.
- One PR per ticket; keep diffs reviewable (split only if the ticket says so).
- Kernel tickets touching vendored llama.cpp: PR against the project's llama.cpp
  fork branch (see S0-02), plus a trivial submodule-bump PR here referencing the
  ticket ID.
- Upstreaming disposition: `fork-local` | `upstream-early` | `upstream-later`
  (see docs/KERNEL-ROADMAP.md §11).
- Copied/adapted code must carry per-file provenance headers (source path,
  commit, license) per S0-01 policy.
