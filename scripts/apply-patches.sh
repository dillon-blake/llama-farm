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
# Usage: ./scripts/apply-patches.sh

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vendor="${repo_root}/vendor/llama.cpp"
patch_dir="${repo_root}/patches"

if [ ! -d "${vendor}/.git" ] && [ ! -f "${vendor}/.git" ]; then
    echo "error: vendor/llama.cpp is not initialized." >&2
    echo "       run: git submodule update --init --recursive" >&2
    exit 1
fi

shopt -s nullglob
patches=("${patch_dir}"/*.patch)
shopt -u nullglob

if [ ${#patches[@]} -eq 0 ]; then
    echo "patch queue is empty (expected in steady state) -- nothing to apply"
    exit 0
fi

# `git am` leaves the tree mid-rebase on failure; make sure we start clean.
git -C "${vendor}" am --abort 2>/dev/null || true

for patch in "${patches[@]}"; do
    name="$(basename "${patch}")"

    # Already applied? `git apply --reverse --check` succeeds only if the patched
    # content is present, which is what makes re-runs a no-op.
    if git -C "${vendor}" apply --reverse --check "${patch}" 2>/dev/null; then
        echo "skip  ${name} (already applied)"
        continue
    fi

    if ! git -C "${vendor}" apply --check "${patch}" 2>/dev/null; then
        echo "error: ${name} does not apply cleanly to vendor/llama.cpp HEAD" >&2
        echo "       the patch is stale -- rebase it onto learning-llamas-base or delete it" >&2
        echo "       if its content has merged (see patches/README.md)" >&2
        exit 1
    fi

    echo "apply ${name}"
    git -C "${vendor}" am --keep-non-patch "${patch}"
done

echo "patch queue applied: ${#patches[@]} patch(es)"
