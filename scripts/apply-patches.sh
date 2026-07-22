#!/usr/bin/env bash
#
# Apply the fork-local patch queue on top of the vendor/llama.cpp submodule.
#
# Patches live in patches/ as numbered `git format-patch` files and are applied in
# lexical order with `git am`. The queue is expected to be EMPTY in steady state --
# real changes live as commits on the fork's learning-llamas-base branch (see
# docs/adr/ADR-0001-vendor-lineage.md). It exists only for diffs that must apply
# before their fork PR merges.
#
# The script is idempotent: a patch whose content is already present in the submodule
# HEAD is skipped, so re-running on an already-patched tree is a no-op. It exits 0 on
# an empty queue and fails loudly on a genuine conflict.
#
# Usage: ./scripts/apply-patches.sh [--check] [--patch-dir DIR]
#
#   (no argument)   apply the queue to vendor/llama.cpp, creating submodule commits.
#   --check         verify the queue still applies and change NOTHING. This is what CI
#                   runs: applying in one job would give that job's build a vendor tree
#                   no other job tested, and bake a commit hash created seconds earlier
#                   on the runner -- present on no remote -- into the version lock whose
#                   entire job is to make that kind of mismatch detectable.
#   --patch-dir DIR read the queue from DIR instead of patches/. This exists for ONE
#                   reason: the queue is empty in steady state, so every path below the
#                   emptiness check is unreachable from a test that can only use
#                   patches/ -- and "--check leaves the submodule alone" is then true
#                   for the uninteresting reason that nothing ran. tests/test_ci_config.py
#                   synthesizes a real one-patch queue and points this at it.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vendor="${repo_root}/vendor/llama.cpp"
patch_dir="${repo_root}/patches"

usage() {
    echo "usage: $(basename "${BASH_SOURCE[0]}") [--check] [--patch-dir DIR]" >&2
}

mode="apply"
while [ $# -gt 0 ]; do
    case "$1" in
        --check) mode="check" ;;
        --patch-dir)
            shift
            if [ $# -eq 0 ]; then
                usage
                exit 2
            fi
            patch_dir="$1"
            ;;
        --patch-dir=*) patch_dir="${1#--patch-dir=}" ;;
        *)
            usage
            exit 2
            ;;
    esac
    shift
done

if [ ! -d "${vendor}/.git" ] && [ ! -f "${vendor}/.git" ]; then
    echo "error: vendor/llama.cpp is not initialized." >&2
    echo "       run: git submodule update --init --recursive" >&2
    exit 1
fi

if [ ! -d "${patch_dir}" ]; then
    echo "error: patch directory does not exist: ${patch_dir}" >&2
    exit 1
fi

shopt -s nullglob
patches=("${patch_dir}"/*.patch)
shopt -u nullglob

if [ ${#patches[@]} -eq 0 ]; then
    if [ "${mode}" = "check" ]; then
        echo "patch queue is empty (expected in steady state) -- nothing to check"
    else
        echo "patch queue is empty (expected in steady state) -- nothing to apply"
    fi
    exit 0
fi

if [ "${mode}" = "check" ]; then
    # Verify in a throwaway worktree rather than against HEAD in place, because the queue is
    # SEQUENTIAL: 0002 may well only apply on top of 0001, so checking each patch against the
    # current HEAD in isolation would reject a queue that is perfectly fine. A detached worktree
    # gets the real thing -- the same `git am` loop, in the same order -- while vendor/llama.cpp
    # itself is never touched and no submodule commit outlives the check.
    scratch="$(mktemp -d)"
    trap 'git -C "${vendor}" worktree remove --force "${scratch}/tree" >/dev/null 2>&1 || true
          rm -rf "${scratch}"' EXIT
    git -C "${vendor}" worktree add --detach --quiet "${scratch}/tree" HEAD
    target="${scratch}/tree"
    # `git am` refuses to commit without an author identity and a CI runner has none configured,
    # which would fail the check for a reason that has nothing to do with the patches. The commits
    # die with the worktree, so any identity does; what is under test is whether the diffs apply.
    git_am=(git -C "${target}" -c user.name="patch-queue check" -c user.email="ci@invalid" am)
    verb="check"
else
    target="${vendor}"
    git_am=(git -C "${target}" am)
    verb="apply"
fi

# `git am` leaves the tree mid-rebase on failure; make sure we start clean.
git -C "${target}" am --abort 2>/dev/null || true

for patch in "${patches[@]}"; do
    name="$(basename "${patch}")"

    # Already applied? `git apply --reverse --check` succeeds only if the patched
    # content is present, which is what makes re-runs a no-op.
    if git -C "${target}" apply --reverse --check "${patch}" 2>/dev/null; then
        echo "skip  ${name} (already applied)"
        continue
    fi

    if ! git -C "${target}" apply --check "${patch}" 2>/dev/null; then
        echo "error: ${name} does not apply cleanly to vendor/llama.cpp HEAD" >&2
        echo "       the patch is stale -- rebase it onto learning-llamas-base or delete it" >&2
        echo "       if its content has merged (see patches/README.md)" >&2
        exit 1
    fi

    echo "${verb} ${name}"
    "${git_am[@]}" --keep-non-patch "${patch}"
done

if [ "${mode}" = "check" ]; then
    echo "patch queue verified: ${#patches[@]} patch(es) apply; vendor/llama.cpp left untouched"
else
    echo "patch queue applied: ${#patches[@]} patch(es)"
fi
