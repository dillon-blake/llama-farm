---
id: S4-01
title: "Vulkan CI: lavapipe software lane (hosted) + native GPU lane (self-hosted)"
stage: 4
track: infra
size: M
deps: ["S0-07"]
status: open
pr: null
---

# S4-01 — Vulkan CI: lavapipe software lane (hosted) + native GPU lane (self-hosted)

**One-line outcome:** `ci-vulkan.yml` exists: per-PR correctness on ubuntu-latest with Mesa
lavapipe (software Vulkan — real driver semantics, no GPU needed) plus a nightly/label-gated
native-GPU lane on a self-hosted runner from the project Vulkan VM, running `test-backend-ops`
MODE_GRAD and the e2e convergence gate.

## Why (context)

Stage 4 lands the Vulkan kernel suite (S4-02..S4-08), and every one of those tickets phrases
acceptance as "green in ci-vulkan" — so the lane must exist before the first shader PR opens.
The ROADMAP §11 CI matrix defines the Vulkan column: MODE_GRAD per-op runs per-PR on the
scalar path, coopmat/MoltenVK nightly, and the tiny-model e2e convergence gate (S1-12) as the
per-phase exit, across "coopmat NV, scalar AMD/Intel, MoltenVK" drivers.

Vulkan is unique among the GPU backends in having a genuinely useful *hosted* tier: Mesa's
lavapipe (llvmpipe Vulkan) is a conformant software driver, so a plain ubuntu-latest runner
exercises real Vulkan driver semantics — descriptor limits, `maxStorageBufferRange`, subgroup
feature queries — with no GPU. llama.cpp's own CI proves the setup: its `ubuntu-llvmpipe` job
(`vendor/llama.cpp/.github/workflows/build-vulkan.yml:76`) installs `mesa-vulkan-drivers` from
the kisak-mesa PPA plus the LunarG Vulkan SDK (`vendor/llama.cpp/.github/workflows/build-vulkan.yml:87-89`),
builds with `-DGGML_VULKAN=ON`, and runs ctest under `GGML_VK_VISIBLE_DEVICES=0` /
`GGML_VK_DISABLE_F16=1` / `GGML_VK_DISABLE_COOPMAT=1` — while explicitly skipping
`test-backend-ops` as too slow on llvmpipe
(`vendor/llama.cpp/.github/workflows/build-vulkan.yml:129-134`). This ticket deliberately
diverges from that last point: correctness on lavapipe is exactly what per-PR kernel review
needs, so we run a *targeted* `test-backend-ops` subset per-PR (bounded op list, generous
timeout) and push the full sweep to nightly. Lavapipe is acceptable for correctness only —
perf numbers come exclusively from the native lane.

The native lane follows the S3-01 pattern: a self-hosted runner registered from the project
Vulkan VM (real GPU — NVIDIA coopmat and/or AMD/Intel scalar per the ROADMAP §11 matrix;
provisioning per the S0-08 playbook), label-gated so kernel PRs get real-driver coverage and
docs PRs never queue on hardware. Because driver capability decides which shader variants run
(coopmat/coopmat2/scalar), every GPU job starts with a probe: the backend's init log prints
the decisive caps per device — `uma | fp16 | bf16 | warp size | shared memory | int dot |
matrix cores` (`vendor/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:6808`) — and the
`GGML_VK_DISABLE_COOPMAT` / `GGML_VK_DISABLE_COOPMAT2` env switches
(`vendor/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:5816`, `:5824`) let one NVIDIA runner
also cover the scalar path.

## What to do

1. **Create `.github/workflows/ci-vulkan.yml`** modeled on `ci-cpu.yml`/`ci-metal.yml`/
   `ci-cuda.yml` (triggers: `pull_request`, `push` to default, nightly `schedule`,
   `workflow_dispatch`). Required-check names per the S0-07 convention and the manifest:
   `ci-vulkan / lavapipe` and `ci-vulkan / gpu`. State the two-tier philosophy and the
   lavapipe-is-correctness-only rule in the workflow header comment.
2. **Hosted `lavapipe` job (per-PR):** ubuntu-latest; install the Vulkan SDK and Mesa
   lavapipe per the vendored precedent (kisak-mesa PPA + `mesa-vulkan-drivers`,
   `vendor/llama.cpp/.github/workflows/build-vulkan.yml:87-89`; SDK cache per `:96-108`);
   configure with `-DGGML_VULKAN=ON` (`vendor/llama.cpp/ggml/CMakeLists.txt:224`) and
   `-DLLAMA_BUILD_TESTS=ON`; build the wheel and `test-backend-ops`. Run a targeted
   correctness subset per-PR: `test -b Vulkan0 -o <list>` and `grad -b Vulkan0 -o <list>`
   (backend devices are named `Vulkan<idx>`,
   `vendor/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:6459`; modes/filters per usage,
   `vendor/llama.cpp/tests/test-backend-ops.cpp:10075-10091`). Derive `<list>` from changed
   paths under `vendor/llama.cpp/ggml/src/ggml-vulkan/**` via the shared
   `.github/scripts/backend-changed-ops.sh` (generalized in S3-01), defaulting to the
   dense-training set. Document the measured subset runtime; lavapipe is slow — trim the
   default list before loosening the time budget.
3. **Nightly lavapipe full sweep:** on `schedule`/`workflow_dispatch`, run the full
   `test-backend-ops test` and `grad` (no `-o`) on lavapipe with an explicit generous
   timeout, acknowledging the vendored skip-for-speed precedent in a comment.
4. **Native `gpu` job:** runner labels `[self-hosted, vulkan]` (registered from the Vulkan
   VM per S0-08), label-gated per-PR — runs only when a PR touches
   `vendor/llama.cpp/ggml/src/ggml-vulkan/**`, `vendor/llama.cpp/ggml/src/ggml.c`, or
   `csrc/` — and in full nightly. Where the VM's GPU supports coopmat, run the targeted/full
   suites twice: once natively (coopmat) and once with `GGML_VK_DISABLE_COOPMAT=1`
   (`ggml-vulkan.cpp:5816`) to cover the scalar path the AMD/Intel column of ROADMAP §11
   needs. Document the VM's actual GPU and driver in `docs/dev/`.
5. **Probe step (start of every Vulkan job, both lanes):** run
   `test-backend-ops support -b Vulkan0` and capture the device-properties log line
   (`ggml-vulkan.cpp:6808` — uma/fp16/warp size/shared memory/matrix cores) plus
   `vulkaninfo --summary` if available; upload as an artifact. The probe records
   coopmat/coopmat2/subgroup caps into the CI log for triage of shader-variant failures.
6. **Nightly e2e:** run the S1-12 convergence gate with `--device vulkan` on the lavapipe
   lane (correctness) and on the native lane (correctness + timing); skip-with-notice if the
   gate is absent on the branch. Capture the sched fallback report with `GGML_SCHED_DEBUG=2`
   (`vendor/llama.cpp/ggml/src/ggml-backend.cpp:1740-1741`; printer at `:945`) using the
   shared parser from S2-01/S3-01; the e2e step fails if the report is missing — not if
   fallbacks exist (fallback-forbidden arrives with S4-09).
7. **Document** the lane, the label gating, the coopmat/scalar double-run, and the
   lavapipe-vs-native division of labor in `docs/dev/` (extending the S0-08 Vulkan playbook
   section); record check names in the workflow header.

## Out of scope

- The Vulkan shaders themselves — S4-02..S4-08.
- Flipping e2e to fallback-forbidden and the perf snapshot — S4-09.
- MoltenVK CI coverage (ROADMAP §11 nightly tier; §12 Q10 benchmark decision) — revisit once
  a Mac runner exists; note it in `docs/dev/` as an open lane.
- Vulkan VM provisioning and runner registration procedure — S0-08 playbook (consumed here).
- CUDA/Metal/CPU lanes — S3-01/S2-01/S0-07.

## Acceptance criteria

- [ ] `ci-vulkan.yml` exists; a scratch PR shows `ci-vulkan / lavapipe` green on
      ubuntu-latest with the probe artifact proving a lavapipe/llvmpipe device was used.
- [ ] The per-PR lavapipe log shows targeted `test -b Vulkan0 -o <list>` and
      `grad -b Vulkan0 -o <list>` invocations passing.
- [ ] `ci-vulkan / gpu` runs green on the self-hosted runner; its probe artifact contains
      the device-properties line including the matrix-cores (coopmat) mode; a docs-only PR
      does not queue the gpu job.
- [ ] On a coopmat-capable runner, the gpu job shows both the native and the
      `GGML_VK_DISABLE_COOPMAT=1` passes in its logs.
- [ ] One `workflow_dispatch` run completes the nightly path: full lavapipe `test` + `grad`
      sweep, the S1-12 gate invoked with `--device vulkan` (passing or skip-with-notice),
      and the sched fallback report uploaded; the e2e step is red when the report is absent.
- [ ] `docs/dev/` documents the runner labels, the VM's GPU/driver, and the
      lavapipe-correctness / native-perf division; the workflow header states the two-tier
      philosophy and check names.

## Testing & verification

The lane is verified by exercising it: a scratch PR for the lavapipe tier and the
kernel-gated gpu tier, one `workflow_dispatch` for the nightly tier, one deliberately
report-less run for the red path. Local dry-run: execute the workflow's build and
`test-backend-ops` commands on the Vulkan VM per the S0-08 playbook, and under lavapipe on
any Linux box with Mesa installed. This lane (ci-vulkan) then *is* the harness every S4
kernel ticket cites: `grad` per-PR on lavapipe (targeted) and native (kernel-gated), full
sweeps + gate `--device vulkan` nightly.

## PR notes

- Branch: `ticket/S4-01-vulkan-ci-lavapipe-gpu-lanes`.
- One PR; workflow + scripts + docs only — no vendored llama.cpp changes, so no two-repo
  flow. Upstreaming disposition: **fork-local** (project CI).
- Soft coordination: consumes S0-07's check-naming/two-tier conventions, the S0-08 Vulkan
  playbook, S1-12's `--device` option, and the shared `backend-changed-ops.sh` + fallback
  parser from S2-01/S3-01 (extend, don't fork). S4-09 later flips this lane's e2e to
  fallback-forbidden — keep the report parser reusable.
