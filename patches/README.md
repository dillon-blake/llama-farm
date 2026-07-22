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

## Verifying (what CI does)

```bash
./scripts/apply-patches.sh --check
```

`--check` runs the same `git am` sequence in a throwaway worktree — sequential patches are checked
in order, as they will really be applied — and leaves `vendor/llama.cpp` untouched.

There is also `--patch-dir DIR`, which reads the queue from somewhere other than this directory. It
exists for the tests, not for daily use: this directory is empty in steady state, so every code
path past the emptiness check is unreachable from anything that can only point at `patches/`, and
"`--check` left the submodule alone" is then true because nothing ran.
`tests/test_ci_config.py` synthesizes a real one-patch queue and a deliberately stale one, and
asserts what `--check` claims: exit 0 / exit 1, HEAD unmoved, the patch's file absent from the
submodule, and no worktree left behind on either path.

**CI only ever verifies.** Applying is a local/fork action, deliberately. Only `ci-cpu / build`
builds a wheel; every other job in every lane builds the pin as checked out. Were CI to apply the
queue, that one job would ship a wheel from a vendor tree nothing else tested, and would bake a
submodule commit created on the runner — a hash that exists on no remote — into `farm_version.h`
and `_ffi/_version_lock.py`, which is precisely the divergence the version lock exists to catch.
A patch that needs to be in a build is a patch that needs to be a commit on
`learning-llamas-base` and a submodule bump.
