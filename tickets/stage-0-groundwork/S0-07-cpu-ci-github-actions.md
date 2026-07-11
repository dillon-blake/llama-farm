---
id: S0-07
title: "CPU CI: GitHub Actions build + test on Linux x86 and macOS arm (CPU-only)"
stage: 0
track: infra
size: M
deps: [S0-06]
status: open
pr: null
---

# S0-07 — CPU CI: GitHub Actions build + test on Linux x86 and macOS arm (CPU-only)

**One-line outcome:** a per-PR GitHub Actions lane that builds the wheel and runs pytest plus a targeted vendored `test-backend-ops` (CPU) subset on ubuntu-latest and macos-14 (arm64, CPU-only), with a nightly full run.

## Why (context)

The ROADMAP §11 CI matrix makes CPU the always-on tier: MODE_GRAD per-op runs per-PR with
CPU as the oracle, while heavier tiers (coopmat Vulkan, ROCm, convergence gates) run
nightly or per-phase. This ticket builds the CPU column of that matrix — the only column
that needs no special hardware — covering both ISAs the matrix names ("CPU (x86 + ARM)")
via ubuntu-latest (x86_64) and macos-14 (Apple Silicon arm64, Metal disabled). BLUEPRINT §8
confirms CPU is the fully-working v1 baseline platform, so a green CPU lane is a meaningful
correctness signal, not just a build check.

The lane also establishes the **two-tier philosophy** every later backend lane copies:
per-PR jobs stay fast (< 20 min) by running pytest's default selection and a targeted
`test-backend-ops` subset; a scheduled nightly job runs the full suites. Later
self-hosted lanes (ci-metal / ci-cuda / ci-vulkan, provisioned per the S0-08 playbooks)
reuse this workflow's structure and check-naming so branch-protection rules stay uniform.

The vendored `test-backend-ops` harness supports exactly the modes we need: positional
mode `test` (compare against the CPU backend), `grad` (MODE_GRAD finite differences),
`perf`, and `support`, with `-o <op,..>` and `-b <backend>` filters
(`vendor/llama.cpp/tests/test-backend-ops.cpp:10075-10091`). It already generates
quantized `OUT_PROD` cases (`vendor/llama.cpp/tests/test-backend-ops.cpp:8780-8806`) that
kernel tickets will later light up on GPU backends — running the CPU oracle continuously
from day one protects those tickets' baseline.

## What to do

1. **Create `.github/workflows/ci-cpu.yml`** with triggers: `pull_request`, `push` to the
   default branch, `schedule` (nightly cron), and `workflow_dispatch`. Put the two-tier
   philosophy (per-PR quick vs nightly detailed; later backend lanes follow the same
   pattern) in a header comment in the workflow itself, per the manifest.
2. **Matrix:** `ubuntu-latest` (x86_64) and `macos-14` (arm64). On macOS pass
   `-DGGML_METAL=OFF` so the lane is genuinely CPU-only (Metal defaults on for Apple
   builds via `GGML_METAL_DEFAULT`, `vendor/llama.cpp/ggml/CMakeLists.txt:239`).
3. **Build job (`build`):** checkout with `submodules: recursive`; cache ccache and pip;
   build the wheel from the S0-03 scikit-build-core setup; upload the wheel as an actions
   artifact per matrix entry.
4. **Test job (`test`):** install the built wheel (or editable install), run
   `pytest tests/ -m "not slow"` using the S0-06 markers (backend-marked tests deselected
   on this lane).
5. **Vendored op tests:** configure the vendored tree with `-DLLAMA_BUILD_TESTS=ON`
   (`vendor/llama.cpp/CMakeLists.txt:107`; target defined at
   `vendor/llama.cpp/tests/CMakeLists.txt:243`), build the `test-backend-ops` binary, and
   run per-PR a **sanity subset** on the CPU backend, e.g.
   `test-backend-ops test -b CPU -o MUL_MAT,OUT_PROD,SOFT_MAX,RMS_NORM,CROSS_ENTROPY_LOSS`
   and `test-backend-ops grad -o OUT_PROD,CROSS_ENTROPY_LOSS`. Pick the final op list for
   run time; record it in the workflow comment.
6. **Nightly job:** on `schedule`/`workflow_dispatch` only, run the full
   `test-backend-ops test` and `test-backend-ops grad` (no `-o` filter) plus
   `pytest tests/` including `slow`.
7. **Check naming convention:** name jobs so required checks read `ci-cpu / build` and
   `ci-cpu / test`; document the convention in the workflow header — subsequent tickets'
   acceptance criteria reference these names, and future lanes follow
   `ci-<backend> / <job>`.
8. Keep total per-PR wall time under 20 minutes; trim the pytest/op subset before
   loosening the budget.

## Out of scope

- GPU lanes (ci-metal / ci-cuda / ci-vulkan) and self-hosted runner registration — later
  stage tickets, provisioning documented in S0-08.
- Release/publish wheels, cibuildwheel matrix — BLUEPRINT §3 "Distribution"; unscheduled.
- Windows/MSVC CI — S1-33 owns the MSVC build and `ci-windows` lane (BLUEPRINT §8/P1).
- The tiny-model convergence gate — S1-12 (nightly per ROADMAP §11).
- Branch-protection configuration itself (repo settings, not code) — note the required
  check names in the PR description for the maintainer to apply.

## Acceptance criteria

- [ ] `ci-cpu.yml` exists; a test PR shows `ci-cpu / build` and `ci-cpu / test` green on
      both matrix entries (ubuntu-latest x86_64, macos-14 arm64).
- [ ] The per-PR run includes a passing targeted `test-backend-ops` CPU invocation in
      both `test` and `grad` modes (visible in job logs).
- [ ] A nightly-path run (triggered once via `workflow_dispatch`) completes the full
      `test-backend-ops` and full pytest suite green.
- [ ] Wheel artifacts for both platforms are downloadable from the run summary.
- [ ] Recorded per-PR wall time for the slowest matrix entry is under 20 minutes.
- [ ] The workflow file contains the two-tier philosophy comment and the
      `ci-<backend> / <job>` naming convention.

## Testing & verification

The CI lane is verified by exercising it: open a scratch PR from the ticket branch and
confirm both jobs pass on both platforms; trigger `workflow_dispatch` once to prove the
nightly path. Locally, dry-run the workflow steps as shell commands (build wheel, pytest,
`test-backend-ops test -b CPU -o ...`) before pushing. This ticket runs the S0-06 pytest
suite and the vendored `test-backend-ops` harness; it adds no new test code of its own.

## PR notes

- Branch: `ticket/S0-07-cpu-ci-github-actions`.
- One PR; workflow + docs comments only — no vendored llama.cpp changes, so no two-repo
  flow. Upstreaming disposition: **fork-local**.
- Soft coordination: consumes S0-06 marker names and the S0-03 build entry points; if
  either changes later, this workflow is the single place to update.
- Expect several force-pushes while iterating on Actions; squash-merge to keep history
  clean.
