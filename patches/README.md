# `patches/` — the fork-local patch queue

Diffs applied on top of the `vendor/llama.cpp` submodule by
[`scripts/apply-patches.sh`](../scripts/apply-patches.sh).

**In steady state this queue is empty.** Real changes to llama.cpp live as commits on the
project fork's `learning-llamas-base` branch and arrive here through a submodule bump — that
is the two-repo flow every `kernels` ticket follows (see
[ADR-0001](../docs/adr/ADR-0001-vendor-lineage.md)). The queue exists for the narrow case
where a diff must apply *before* its fork PR merges: an urgent build fix, or a change whose
fork PR is still in review while a dependent learning-llamas PR needs to be tested.

## Format

Numbered `git format-patch` files, applied in lexical order:

```
patches/0001-short-slug.patch
patches/0002-another-slug.patch
```

Produce one from a commit on the submodule:

```bash
git -C vendor/llama.cpp format-patch -1 -o ../../patches
```

## Lifecycle

A patch has exactly one job: to bridge the gap until its content is on
`learning-llamas-base`. **Delete the patch file in the same PR that bumps the submodule to a
commit containing it.** A patch that outlives its merge is a silent double-apply hazard.

`apply-patches.sh` is idempotent — it skips patches already present in the submodule HEAD, so
a stale patch will *look* harmless right up until a rebase makes it conflict. Do not rely on
that; delete it.

## Applying

```bash
git submodule update --init --recursive
./scripts/apply-patches.sh
```

Exits 0 on an empty queue. Fails loudly, without leaving the submodule mid-`am`, if a patch no
longer applies.
