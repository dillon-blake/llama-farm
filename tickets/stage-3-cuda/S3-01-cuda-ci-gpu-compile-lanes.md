---
id: S3-01
title: "CUDA CI: GPU lane (hosted GPU runner or self-hosted VM) + per-PR compile lane"
stage: 3
track: infra
size: M
deps: [S0-07]
status: open
pr: null
---

# S3-01 — CUDA CI: GPU lane (hosted GPU runner or self-hosted VM) + per-PR compile lane

**One-line outcome:** `ci-cuda.yml` exists: a per-PR nvcc build lane on ubuntu-latest (no GPU)
plus a GPU test lane (GitHub GPU runner if available to the org, else a self-hosted runner
registered from the project CUDA VM) running `test-backend-ops` MODE_GRAD and the e2e gate.

## Why (context)

Stage 3 lands the CUDA kernel suite (S3-02..S3-09) and the FA5 flagship, and every one of
those tickets phrases acceptance as "green in ci-cuda" — so the lane must exist before the
first kernel PR opens (the README stage-ordering exception allows this ticket any time after
S0-07). The ROADMAP §11 CI matrix defines what the lane must run: MODE_GRAD per-op per-PR on
CUDA with **two arch targets** (its column header reads "CUDA (sm_70 + sm_90)"), and the
tiny-model e2e convergence gate (S1-12) as the per-phase exit. The project goal is detailed
GitHub Actions tests per backend, which is only auditable if each run reports which ops
executed on the GPU versus falling back to CPU via `ggml_backend_sched` (ROADMAP §11
scheduler note).

Unlike ci-cpu and ci-metal, no standard GitHub-hosted runner has an NVIDIA GPU: GPU "larger
runners" are an org-level paid feature that may or may not be available. The lane therefore
splits in two. A **compile-only lane** runs per-PR on ubuntu-latest for cheap universal
coverage — llama.cpp's own CI proves this works GPU-less by building inside an
`nvidia/cuda:12.6.2-devel-ubuntu24.04` container on ubuntu-24.04
(`vendor/llama.cpp/.github/workflows/build-cuda-ubuntu.yml:40-41`) with an explicit
`-DCMAKE_CUDA_ARCHITECTURES=89-real` (`vendor/llama.cpp/.github/workflows/build-cuda-ubuntu.yml:68`).
The **GPU test lane** prefers a GitHub-hosted GPU larger runner if the org has one, and
otherwise runs on a self-hosted runner registered from the project CUDA VM (provisioning and
registration playbook: S0-08), label-gated so the workflow never queues on hardware that
does not exist.

The fallback report reuses the mechanism S2-01 established: `GGML_SCHED_DEBUG` env-gated
assignment printing (`vendor/llama.cpp/ggml/src/ggml-backend.cpp:1740-1741`; printer
`ggml_backend_sched_print_assignments` at `vendor/llama.cpp/ggml/src/ggml-backend.cpp:945`).
Until S3-10 flips the switch, e2e runs tolerate CPU fallbacks but must always publish the
GPU-vs-CPU-fallback op report — a run without the report fails.

## What to do

1. **Create `.github/workflows/ci-cuda.yml`** modeled on `ci-cpu.yml`/`ci-metal.yml`
   (triggers: `pull_request`, `push` to default, nightly `schedule`, `workflow_dispatch`).
   Check names per the S0-07 convention: `ci-cuda / compile`, `ci-cuda / gpu`,
   `ci-cuda / e2e`. State the two-tier philosophy in the workflow header comment.
2. **Per-PR `compile` job (no GPU):** ubuntu-latest with a pinned `nvidia/cuda:*-devel`
   container (toolkit ships in the image — cheap and cache-friendly; ccache on top, as in
   the vendored precedent above). Configure with `-DGGML_CUDA=ON`
   (`vendor/llama.cpp/ggml/CMakeLists.txt:199`), `-DGGML_NATIVE=OFF`,
   `-DLLAMA_BUILD_TESTS=ON`; build the wheel and `test-backend-ops`. **Matrix over two arch
   targets** per ROADMAP §11: one sm_70-class and one sm_90-class
   `CMAKE_CUDA_ARCHITECTURES` value (e.g. `70-real` and `90-real`), so every PR gets compile
   coverage on both ends of the supported-arch range.
3. **GPU lane runner selection:** a workflow-level variable holds the runner labels —
   default `[self-hosted, cuda]` (runner registered from the project CUDA VM per the S0-08
   playbook), overridable to the org's hosted GPU runner label if one exists. Document both
   paths in the workflow header. Probe step at the start of every GPU job:
   `nvidia-smi` plus `test-backend-ops support -b CUDA0` (modes/filters per usage,
   `vendor/llama.cpp/tests/test-backend-ops.cpp:10075-10091`), uploaded as an artifact.
   **Document what arch the VM actually has** (in `docs/dev/` and the workflow comment) and
   gate arch-specific steps accordingly — the sm_70/sm_90 pair is a compile-matrix
   requirement; the GPU lane tests whatever silicon the runner offers.
4. **Per-PR `gpu` quick subset, kernel-gated:** run the GPU job on PRs only when changed
   paths touch `vendor/llama.cpp/ggml/src/ggml-cuda/**`, `vendor/llama.cpp/ggml/src/ggml.c`,
   or `csrc/` (path filter), with a targeted `-o <op,..>` list derived from changed files —
   generalize S2-01's `metal-changed-ops.sh` into a shared
   `.github/scripts/backend-changed-ops.sh` rather than forking it. Run both
   `test -b CUDA0 -o <list>` and `grad -b CUDA0 -o <list>`.
5. **Nightly `e2e` job (detailed):** full `test-backend-ops test` and `grad` sweeps on
   CUDA, then the S1-12 convergence gate with `--device cuda` (skip-with-notice if absent on
   the branch), then a memory/perf report: `test-backend-ops perf` over the training op set
   plus the gate's JSON report (device, fallback indicator, peak-memory numbers) uploaded as
   artifacts.
6. **Fallback report (auditable resident-ness):** run the e2e training step with
   `GGML_SCHED_DEBUG=2` (`vendor/llama.cpp/ggml/src/ggml-backend.cpp:1740-1741`), parse
   GPU-vs-CPU op assignments with the S2-01 parser (keep it shared), and write the list to
   the job summary and an artifact. The e2e job fails if the report is missing — not if
   fallbacks exist (fallback-forbidden arrives with S3-10).
7. **Document** the lane, runner-label switch, and the VM's actual GPU/arch in `docs/dev/`
   (extending the S0-08 CUDA playbook section); record check names in the workflow header.

## Out of scope

- The CUDA kernels themselves — S3-02..S3-09.
- Flipping e2e to fallback-forbidden and publishing the perf snapshot — S3-10.
- ROCm/HIP nightly tier (ROADMAP §11 matrix) — follow-up once a HIP-capable runner exists;
  S3-02 notes the hipBLAS ride-along concern.
- Vulkan/Metal lanes — S4-01 / S2-01.
- CUDA VM provisioning and runner registration procedure — S0-08 playbook (consumed here).

## Acceptance criteria

- [ ] `ci-cuda.yml` exists; a scratch PR shows `ci-cuda / compile` green on **both** arch
      matrix entries (sm_70-class and sm_90-class) on ubuntu-latest without a GPU.
- [ ] The `ci-cuda / gpu` job runs green on the registered GPU runner; its probe artifact
      contains `nvidia-smi` and `test-backend-ops support -b CUDA0` output.
- [ ] A PR touching `vendor/llama.cpp/ggml/src/ggml-cuda/**` triggers the GPU quick subset
      with a targeted `-o` list (visible in logs); a docs-only PR does not queue the GPU job.
- [ ] One `workflow_dispatch` run completes the nightly path: full CUDA `test` + `grad`
      sweeps, the S1-12 gate invoked with `--device cuda` (passing or skip-with-notice), and
      the memory/perf artifacts uploaded.
- [ ] The e2e job summary contains the GPU-vs-CPU-fallback op report, and the job is red
      when the report artifact is absent.
- [ ] `docs/dev/` documents the runner-label switch and the CUDA VM's actual GPU arch; the
      workflow header states the two-tier philosophy and check names.

## Testing & verification

The lane is verified by exercising it: a scratch PR for the compile tier and the
kernel-gated GPU tier, one `workflow_dispatch` for the nightly tier, one deliberately
report-less run to prove the red path. Local dry-run: execute the workflow's build and
`test-backend-ops` commands on the CUDA VM per the S0-08 playbook. This lane (ci-cuda) then
*is* the harness every S3 kernel ticket cites: `grad` per-PR (kernel-gated), full sweep +
gate `--device cuda` nightly.

## PR notes

- Branch: `ticket/S3-01-cuda-ci-gpu-compile-lanes`.
- One PR; workflow + scripts + docs only — no vendored llama.cpp changes, so no two-repo
  flow. Upstreaming disposition: **fork-local** (project CI).
- Soft coordination: consumes S0-07's check-naming/two-tier conventions, the S0-08 CUDA
  playbook, and S1-12's `--device` option; if the changed-ops script is generalized from
  S2-01's, update ci-metal in the same PR so both lanes share it. S3-10 later flips this
  lane's e2e to fallback-forbidden — keep the report parser reusable.
