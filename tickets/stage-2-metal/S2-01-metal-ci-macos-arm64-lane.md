---
id: S2-01
title: "Metal CI: GitHub Actions lane on macOS arm64 (MODE_GRAD + e2e)"
stage: 2
track: infra
size: M
deps: ["S0-07"]
status: pr-open
pr: https://github.com/dillon-blake/llama-farm/pull/44
---

# S2-01 — Metal CI: GitHub Actions lane on macOS arm64 (MODE_GRAD + e2e)

> **⚠️ THE TICKET'S OWN `-b Metal` WOULD HAVE MADE THIS LANE GREEN WHILE RUNNING ZERO TESTS.**
>
> `test-backend-ops` filters devices with an **exact `strcmp`** against `ggml_backend_dev_name`
> (`test-backend-ops.cpp:11214`), and **Metal registers itself as `MTL0`** —
> `ggml-metal-device.m:858` formats it as `"MTL%d"`. Not `Metal`. The ticket specifies `-b Metal` in
> three places.
>
> A name that matches nothing makes the harness print `Skipping`, **count it as passed, and exit 0**.
> Demonstrated: `test-backend-ops test -b NOPE0 -o RMS_NORM` prints `1/1 backends passed`, `OK`, and
> returns **0**.
>
> So the lane **resolves** the device name from the binary's own output and never hardcodes it, and
> every GPU step asserts the device actually reached a verdict. Note the guard is a *positive*
> assertion, not a `grep Skipping` — with `-b MTL0` the CPU device is skipped **legitimately** and
> says so, so a naive grep fails every healthy run.
>
> **S3-01 and S4-01 inherit this**: `CUDA0` (or `ROCm0`/`MUSA0`) and `Vulkan0`. It also fixed a live
> bug in S1-12's `conftest`, which mapped `metal -> "Metal"` and would have reported "no metal
> device" *on an actual Mac*.
>
> **One deliberate deviation:** the ticket's `metal-changed-ops.sh` (a changed-files -> op-list
> mapper with its own default list) is **not** implemented. That is a *second op registry*, and a
> second op registry is exactly the drift S1-12 had to fix. One registry: `tests/project_ops.py`.

**One-line outcome:** `ci-metal.yml` exists: a per-PR quick lane plus a nightly detailed lane on
macOS arm64 runners building `GGML_METAL=ON`, running `test-backend-ops` on the Metal backend
(including the MODE_GRAD subset) and, as ops land, the tiny-model training e2e with the sched
CPU-fallback report.

## Why (context)

Stage 2 lands ~8 new Metal kernels (ROADMAP §6), and every one of their tickets phrases
acceptance as "green in ci-metal" — so the lane must exist before the first kernel PR opens.
The ROADMAP §11 CI matrix requires MODE_GRAD per-op runs per-PR on Metal (Apple7+) and the
tiny-model e2e convergence gate as the per-phase exit; the project goal is "detailed tests on
each backend", which is only auditable if each run reports which ops actually executed on the
GPU versus falling back to CPU via `ggml_backend_sched` (ROADMAP §11 scheduler note).

GitHub-hosted macOS arm64 runners expose Metal through a paravirtualized GPU, and llama.cpp's
own CI relies on this: its `macos-latest-arm64` job builds with Metal enabled (Apple builds
default `GGML_METAL=ON`, `vendor/llama.cpp/ggml/CMakeLists.txt:239`) and runs GPU-offloaded
tests — `test-thread-safety ... -ngl 99` plus `ctest -L main`
(`vendor/llama.cpp/.github/workflows/build-apple.yml:40-74`; the `-ngl 99` GPU run at `:68`,
ctest at `:74`). The paravirtual device's capability set is not guaranteed to match real Apple Silicon,
so the lane must probe it: the backend detects `has_simdgroup_reduction`/`has_simdgroup_mm`
from the MTLGPUFamily (`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m:696-699`) and
logs both at init (`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m:901-902`); every
stage-2 kernel gates on those flags in `supports_op`
(`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m:1051-1368`). If the hosted GPU lacks
them, the lane must skip-with-report, and the work moves to a self-hosted Apple Silicon runner
provisioned per the S0-08 playbook.

The lane copies the two-tier structure and `ci-<backend> / <job>` check naming established by
S0-07's `ci-cpu.yml`: per-PR jobs stay fast and targeted; the nightly runs the full op sweep
and the S1-12 convergence gate with `--device metal`. The fallback semantics are deliberate:
until S2-10 flips the switch, the e2e job tolerates CPU fallbacks but must always publish the
fallback report — a run without the report fails.

## What to do

1. **Create `.github/workflows/ci-metal.yml`** modeled on `ci-cpu.yml` (triggers:
   `pull_request`, `push` to default, nightly `schedule`, `workflow_dispatch`). Runner:
   GitHub-hosted macOS arm64 (`macos-15`, with `macos-14` as documented alternative). Name jobs
   so required checks read `ci-metal / build`, `ci-metal / grad`, `ci-metal / e2e`.
2. **`build` job:** checkout with submodules, build the wheel plus the vendored test binaries
   with `-DGGML_METAL=ON -DLLAMA_BUILD_TESTS=ON`; upload the wheel and `test-backend-ops` as
   artifacts; ccache as in ci-cpu.
3. **Probe step (start of every GPU job):** run `test-backend-ops support -b Metal` (modes and
   filters per usage, `vendor/llama.cpp/tests/test-backend-ops.cpp:10075-10091`) and capture
   the backend init log lines for `simdgroup reduction` / `simdgroup matrix mul.`
   (`ggml-metal-device.m:901-902`). Upload the output as an artifact. If the device is absent
   or lacks simdgroup reduction/matrix-mul, **skip** the remaining GPU steps with a prominent
   job-summary notice (not a failure), so hosted-runner capability regressions are visible, not
   red.
4. **`grad` job (per-PR):** targeted `test-backend-ops` on Metal for ops named in changed
   files: a small script (`.github/scripts/metal-changed-ops.sh`) maps changed paths under
   `vendor/llama.cpp/ggml/src/ggml-metal/` and `csrc/` to a `-o <op,..>` list, defaulting to
   the dense-training set (`MUL_MAT,OUT_PROD,SOFT_MAX,RMS_NORM,SILU,GLU,REPEAT,ADD,MUL`) when
   the mapping is empty. Run both modes: `test -b Metal -o <list>` (forward parity vs CPU) and
   `grad -b Metal -o <list>` (MODE_GRAD finite differences).
5. **`e2e` job (nightly + `workflow_dispatch`):** full Metal sweep (`test` and `grad`, no `-o`)
   followed by the S1-12 convergence gate with `--device metal` (pytest `--device` option; skip
   cleanly with a notice if the gate is not yet present on the branch). Fallback lane spec:
   the job accepts a runner-label input so the same workflow runs on a self-hosted
   Apple Silicon runner (registration per the S0-08 playbook) for anything the paravirtual GPU
   cannot do.
6. **Fallback report (the auditability requirement):** run the e2e training step with
   `GGML_SCHED_DEBUG=2` (env-gated assignment printing,
   `vendor/llama.cpp/ggml/src/ggml-backend.cpp:1740-1741`; printer at `:945`), parse which ops
   ran on Metal vs CPU, and write the list into the job summary plus an artifact. The e2e job
   **fails if the report is missing**, not if fallbacks exist (fallback-forbidden for dense ops
   arrives with S2-10).
7. **Document** the lane, the probe semantics, and self-hosted registration in `docs/dev/`
   (extending the S0-08 testing/CI docs); record the check-name convention in the workflow
   header comment.

## Out of scope

- The Metal kernels themselves — S2-02..S2-09.
- Flipping e2e to fallback-forbidden for dense-LoRA ops and the perf snapshot — S2-10.
- CUDA/Vulkan lanes — stage-3/4 tickets (same structure, per S0-07's convention).
- Self-hosted runner *provisioning* (hardware, OS setup) — S0-08 playbook; this ticket only
  consumes the registration procedure and adds the runner-label switch.

## Acceptance criteria

- [ ] `ci-metal.yml` exists; a scratch PR shows `ci-metal / build` and `ci-metal / grad` green
      on a GitHub-hosted macOS arm64 runner.
- [ ] The probe artifact from that run contains the `test-backend-ops support -b Metal` output
      and the two simdgroup capability log lines; the workflow's skip-with-report path is
      exercised once (e.g. by forcing the probe condition) and produces a skipped job with a
      job-summary notice, not a failure.
- [ ] The per-PR `grad` job log shows both `test -b Metal` and `grad -b Metal` invocations with
      a targeted `-o` list derived from changed files (or the documented default list).
- [ ] One `workflow_dispatch` run completes the nightly path: full Metal `test` + `grad` sweep,
      and the S1-12 gate invoked with `--device metal` (passing, or skipping with notice if the
      gate is absent at merge time).
- [ ] The e2e job summary contains the GPU-vs-CPU-fallback op report, and the job is
      red when the report artifact is absent.
- [ ] `docs/dev/` documents the self-hosted fallback runner registration and the workflow's
      runner-label switch; the workflow header states the two-tier philosophy and check names.

## Testing & verification

The lane is verified by exercising it: a scratch PR for the per-PR tier, one
`workflow_dispatch` for the nightly tier, and one forced-skip run for the probe path. Local
dry-run: build with `-DGGML_METAL=ON` on any Apple Silicon machine and run the exact
`test-backend-ops` invocations from the workflow. This ticket adds the `metal-changed-ops.sh`
script and the fallback-report parser; both get shellcheck/pytest coverage in the repo's
existing lint job. Everything else reuses S0-06/S0-07 test assets. This lane (ci-metal) then
*is* the harness that every S2 kernel ticket cites: `grad` per-PR, full sweep + e2e nightly.

## PR notes

- Branch: `ticket/S2-01-metal-ci-macos-arm64-lane`.
- One PR; workflow + scripts + docs only — no vendored llama.cpp changes, so no two-repo flow.
- Upstreaming disposition: **fork-local** (project CI).
- Soft coordination: consumes S0-07's check-naming and two-tier conventions, the S0-08
  registration playbook, and the S1-12 `--device` option; the fallback-report format defined
  here is what S2-10 later flips to fallback-forbidden — keep the parser reusable.
