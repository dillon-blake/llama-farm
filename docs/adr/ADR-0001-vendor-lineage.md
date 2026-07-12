# ADR-0001 — Vendored llama.cpp: lineage, pin, rebase cadence, two-repo PR flow

- **Status:** Accepted
- **Date:** 2026-07-12
- **Ticket:** S0-02
- **Supersedes:** nothing

## Context

Layer 0 of learning-llamas is vendored llama.cpp (BLUEPRINT §3). This is not a loose
dependency: the C shim in `csrc/` compiles against llama.cpp's **private `src/` internals**,
so the exact llama.cpp commit is part of learning-llamas's ABI. Every ticket on the `kernels`
track lands its changes *inside* llama.cpp core code — new ggml ops, new backward kernels,
new op-table entries. A ctypes binding that mirrors struct layouts by hand (S0-04) makes the
coupling total: a struct that changes shape between commits produces silent memory
corruption, not a link error.

Two things therefore had to be settled before any code was written.

### Which llama.cpp is Layer 0?

BLUEPRINT §3 flagged the reference checkout as *possibly a drifted fork* rather than genuine
upstream, on two pieces of evidence: it carries `src/llama-ext.h`, a staging header whose own
comment marks it "new llama.cpp API ... considered WIP" (`src/llama-ext.h:1-5`), and it
exposes the batch adapter API `llama_set_adapters_lora` (`include/llama.h:690`). If the
checkout were a drifted fork, the upstreaming plan in ROADMAP §11 would be built on sand and
the S0-04 struct mirrors would be mirroring the wrong structs.

It is genuine upstream. Verified independently, from the vendored tree:

```
$ git merge-base --is-ancestor 4f37f519722aa3242eecb7649466b4a4a2d6d6da upstream/master
$ echo $?
0                              # 4f37f51 is an ancestor of ggml-org/llama.cpp master

$ git cat-file -e 4f37f51:src/llama-ext.h && echo present
present                        # llama-ext.h is a tracked file in the 4f37f51 tree

$ git log --oneline -3 upstream/master -- src/llama-ext.h
d78952748 spec : Support Step3.5/3.7 flash mtp3 (#24340)
02182fc5b fit : avoid including llama-ext.h in fit.h (#24506)
88a39274e spec: add EAGLE3 speculative decoding support (#18039)

$ git log -1 --format='%H %ad %s' --date=short 4f37f51
4f37f519722aa3242eecb7649466b4a4a2d6d6da 2026-07-10 server: accept null sampling params (#25538)
```

`src/llama-ext.h` has ordinary upstream PR history (#18039, #24506); it is upstream's own
staging header, not fork drift. At the time of writing, `ggml-org/llama.cpp` master is 9
commits ahead of the pin and 0 behind. The "WIP" comment is a stability warning about that
header's API, which we honor by not depending on it — it is not a lineage signal.

**Consequences of the finding:** the upstreaming path in ROADMAP §11 is real (our kernels can
go to mainline), and the S0-04 struct mirrors are mirroring upstream structs at a known
commit — including `llama_set_adapters_lora`, which exists at the pin and which the mirror
test must fail loudly on if a vendor bump ever removes it.

### Why a fork at all, rather than tracking upstream directly?

Because kernel tickets must be able to commit to llama.cpp before (or without) upstream
accepting them, while a long-lived private fork would fight ggml's fast-moving op table.
ROADMAP §11 prescribes the middle path this ADR adopts.

## Decision

### 1. The pin

`vendor/llama.cpp` is a git submodule tracking branch **`learning-llamas-base`** on the project
fork **<https://github.com/dillon-blake/llama.cpp>**. The branch is protected against
force-push: rewriting it would silently change the meaning of every gitlink in this repository's
history.

Two commits matter and they are not the same one:

- **The upstream base: `4f37f519722aa3242eecb7649466b4a4a2d6d6da`.** This is the upstream commit
  `learning-llamas-base` was branched from, and **every `file:line` anchor cited in the tickets
  and docs is valid at this commit.**
- **The pin: whatever `learning-llamas-base` currently points at.** It *advances* past the
  upstream base as fork commits land (S0-10 was the first). Those commits only add, so the
  anchors above stay valid.

Nothing hardcodes the pin. The native build bakes it in from `git rev-parse HEAD` at configure
time, CMake generates `_ffi/_version_lock.py` from the same value, and the tests read it back out
of the submodule (`tests/vendor_pin.py`) — so all three parties to the version lock are checked
against the one thing that is unambiguously true, and a bump requires **no edits to test
literals**.

The pinned commit, the vendored `gguf-py`, and the S0-04 ctypes struct mirrors form **one
atomic version**. They are bumped together or not at all.

### 2. Change triage (ROADMAP §11)

Every change to llama.cpp code falls into one of three classes, declared in the ticket's PR
notes as its *upstreaming disposition*:

| Class | Meaning | Examples |
|---|---|---|
| **upstream-early** | Generally useful, small, no learning-llamas-specific semantics. PR to `ggml-org/llama.cpp` mainline *first*; carry on `learning-llamas-base` only until it merges. | Small VJPs (TANH/SIGMOID/CLAMP, S1-19), CUDA `OUT_PROD` (S3-02), the `SOFT_MAX_BACK` ALiBi lift (S1-20) |
| **in-fork-first** | Needed now, upstream shape not yet settled, or too large to land as one upstream PR. Lives on `learning-llamas-base`; upstreamed later in digestible pieces. | Sparse CE (S1-04), flash-attention backward (S1-23), `OUT_PROD_ID` (S1-26) |
| **fork-local** | learning-llamas-specific; never upstreamed. | The training-graph KV-cache bypass (S1-00) in its project-specific form |

To keep the carried diff rebasable, fork-local changes follow two rules: **new op enums are
appended at the tail of their table**, never inserted in the middle (an insert renumbers every
op after it and turns every rebase into a merge conflict across the whole backend matrix); and
new functionality goes in **new files** wherever it plausibly can, rather than as hunks inside
files upstream is actively editing.

### 3. Rebase cadence

`learning-llamas-base` is rebased onto upstream `master` **monthly**. A rebase is not complete
until the **full `test-backend-ops` MODE_GRAD suite** has been re-run against the new base:

```bash
test-backend-ops grad          # no -o filter: every op, every backend built
```

### 4. Every vendor bump reruns MODE_GRAD

**No submodule-bump PR merges without a full MODE_GRAD run attached**, whether the bump comes
from a monthly rebase or from a kernel ticket landing on the fork. This is the rule that keeps
"CPU is the oracle" (ROADMAP §4) from quietly decaying: an upstream change to a shared kernel
can shift the oracle itself, and the only thing that notices is the finite-difference suite.
Until the nightly lane exists (S0-07), this is enforced procedurally, by review.

### 5. The two-repo PR flow for `kernels` tickets

A ticket that modifies vendored llama.cpp code produces **two** pull requests:

1. **The real PR**, against the fork's `learning-llamas-base` branch (or against
   `ggml-org/llama.cpp` mainline for an `upstream-early` change, then cherry-picked onto
   `learning-llamas-base`). Its title carries the ticket ID: `[S1-04] ggml_cross_entropy_loss_sparse`.
2. **A trivial PR here**, bumping the `vendor/llama.cpp` gitlink to the merged commit and
   flipping the ticket's `status`. Same ticket ID in the title. This PR carries the MODE_GRAD
   evidence required by decision 4.

The learning-llamas PR is where CI proves the bump is safe; the fork PR is where the code is
reviewed. Splitting them keeps the kernel diff reviewable against llama.cpp's own history
instead of appearing as an opaque gitlink change.

### 6. The patch queue is an exception, not a workflow

`patches/` (see `patches/README.md`) applies fork-local diffs on top of the submodule for the
narrow case where a change must apply before its fork PR merges. **A patch is deleted in the
same PR that bumps the submodule to a commit containing its content.** Steady state: empty.

## Consequences

- A fresh clone needs `git submodule update --init --recursive`; CI (S0-07) does this on every
  run, which continuously re-verifies the pin.
- Bumping the submodule is never routine. It is a PR with a MODE_GRAD suite attached, and it
  may require regenerating the S0-04 struct mirrors.
- We accept a monthly rebase cost in exchange for being able to land kernels immediately, and
  we pay that cost down by upstreaming aggressively (ROADMAP §11) — every accepted upstream PR
  is one less hunk to rebase forever.
- The `4f37f51` line anchors cited throughout `tickets/` and `docs/` are valid exactly as long
  as the pin holds. A rebase invalidates them; the rebase PR updates the ones it breaks.
