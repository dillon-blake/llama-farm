---
id: S0-08
title: "Developer docs: build guide + per-backend VM playbooks"
stage: 0
track: docs
size: S
deps: [S0-01]
status: pr-open
pr: null
---

# S0-08 — Developer docs: build guide + per-backend VM playbooks

**One-line outcome:** `docs/dev/` contains a local build guide, a test-running guide, and command-level idempotent VM provisioning playbooks for every backend stage (Linux CPU, macOS Metal, Linux CUDA, Linux Vulkan including the lavapipe software path).

## Why (context)

Stages 1-4 of this project are implemented by autonomous agents working on VMs that must
be able to build the tree and run its tests unattended. That only works if provisioning
and build/test invocation are written down as exact, re-runnable commands — a human
"install the CUDA toolkit" bullet is useless to an agent mid-ticket. These docs are the
contract between the infra tickets (S0-03 build system, S0-07 CI) and every kernel ticket
that follows.

The per-backend split mirrors the BLUEPRINT §8 platform table: CPU is the fully-working
baseline; Linux CUDA works with CPU fallback for quantized backward; macOS Metal runs
forward on GPU only until the Stage-2 kernel suite lands; Vulkan is the Stage-4 target,
where the **lavapipe** software rasterizer gives a GPU-less validation path so Vulkan
shader work can be smoke-tested on any Linux VM before touching real hardware. The GPU
playbooks also cover registering self-hosted GitHub Actions runners, because the ROADMAP
§11 CI matrix needs Metal/CUDA/Vulkan lanes that GitHub-hosted runners cannot provide.

The backend switches are plain CMake options in the vendored tree: `GGML_CUDA`
(`vendor/llama.cpp/ggml/CMakeLists.txt:199`), `GGML_VULKAN`
(`vendor/llama.cpp/ggml/CMakeLists.txt:224`), and `GGML_METAL`
(`vendor/llama.cpp/ggml/CMakeLists.txt:239`, default ON for Apple builds). The vendored
`test-backend-ops` harness self-documents its modes — `test`, `grad` (MODE_GRAD
finite-difference gradient checking), `perf`, `support`, with `-o <op,..>` and
`-b <backend>` filters (`vendor/llama.cpp/tests/test-backend-ops.cpp:10075-10091`) — and
the testing guide must teach exactly those invocations since every kernel ticket's
acceptance criteria are phrased in terms of them.

## What to do

1. **`docs/dev/building.md`:** OS prerequisites per platform (compilers, CMake, Python,
   ccache); submodule init; editable install via scikit-build-core
   (`pip install -e .` with `CMAKE_ARGS` passthrough, per the S0-03 setup); per-backend
   configure flags with the verified option names `-DGGML_CUDA=ON`, `-DGGML_VULKAN=ON`,
   `-DGGML_METAL=ON/OFF` (cite the three `ggml/CMakeLists.txt` anchors above); how to
   build the vendored test binaries (`-DLLAMA_BUILD_TESTS=ON`); debug vs release notes.
2. **`docs/dev/testing.md`:** the pytest suite and marker conventions from S0-06 (`slow`,
   `cuda`, `metal`, `vulkan`; per-PR selection `-m "not slow"`); vendored
   `test-backend-ops` usage including MODE_GRAD — mode is positional (`test-backend-ops
   grad`) with `-o <OP>` to filter ops and `-b <backend>` to pin the backend under test;
   worked examples (e.g. `test-backend-ops grad -o OUT_PROD -b CUDA0`); pointer to the
   ADR-0002 numerics/parity policy (S0-09) for tolerances; pointer to the tiny-model
   convergence gate (S1-12) for end-to-end training acceptance.
3. **`docs/dev/vm-playbooks.md`:** one playbook per backend stage, each a numbered list
   of copy-pasteable, idempotent shell commands (safe to re-run on a half-provisioned
   VM), ending with a verification command whose expected output is stated:
   - **Linux CPU:** baseline toolchain; build + `pytest` + `test-backend-ops test -b CPU`.
   - **macOS Metal:** Xcode command-line tools; note Metal requires a real macOS host or
     Apple-virtualized macOS (no GPU passthrough on Linux hypervisors); verification via
     `test-backend-ops support -b Metal`.
   - **Linux CUDA:** NVIDIA driver + CUDA toolkit install (pin a known-good version
     range), `nvidia-smi` check, build with `-DGGML_CUDA=ON`, verification via
     `test-backend-ops support -b CUDA0`.
   - **Linux Vulkan:** Vulkan SDK (incl. `glslc`/`vulkaninfo`) plus **lavapipe**
     (Mesa software implementation) so shader compilation and op tests run without a GPU;
     document how to select the lavapipe ICD via environment variable; note lavapipe is a
     functional path, not a performance path.
   - **Self-hosted runners:** steps to register a VM as a GitHub Actions runner with
     backend-specific labels, matching the `ci-<backend> / <job>` naming from S0-07.
4. Cross-link the three files and add a short `docs/dev/README.md` index.

## Out of scope

- The CI workflows themselves — S0-07 (CPU) and later per-backend lane tickets.
- The build system being documented — S0-03 owns it; this ticket documents, and any
  mismatch discovered is fixed in docs or filed against S0-03.
- ADR content (numerics, determinism) — S0-09; testing.md only links to it.
- Windows/MSVC developer docs — S1-33 owns the Windows build/CI and updates building.md
  when it lands; note the gap explicitly in building.md until then.

## Acceptance criteria

- [ ] `docs/dev/building.md`, `docs/dev/testing.md`, `docs/dev/vm-playbooks.md`, and
      `docs/dev/README.md` exist and are linked from the repo README.
- [ ] building.md names all three backend CMake options with the exact spellings
      `GGML_CUDA`, `GGML_VULKAN`, `GGML_METAL` and the editable-install command.
- [ ] testing.md contains at least one complete MODE_GRAD invocation example
      (`test-backend-ops grad -o <OP> -b <backend>`) and links to ADR-0002 and S1-12.
- [ ] vm-playbooks.md contains all four backend playbooks plus the self-hosted-runner
      section, each step a fenced command block, each playbook ending with a verification
      command and its expected output.
- [ ] Every playbook is idempotency-reviewed: re-running any step on an already-provisioned
      VM is a no-op or safe overwrite (stated per step where non-obvious).
- [ ] The Linux CPU playbook is executed top-to-bottom on a fresh VM/container and the
      transcript (or CI log) is attached to the PR.

## Testing & verification

Docs-only ticket: verification is executing the Linux CPU playbook end-to-end on a clean
VM or container (attach transcript to the PR) and dry-reviewing the GPU playbooks against
the BLUEPRINT §8 platform table. Markdown link check runs in the `ci-cpu` lane once S0-07
lands (soft coordination — if S0-07 merges first, add the docs check there; otherwise note
it as follow-up). No test-backend-ops or pytest changes.

## PR notes

- Branch: `ticket/S0-08-dev-docs-vm-playbooks`.
- One PR; documentation only — no vendored llama.cpp changes, no two-repo flow.
  Upstreaming disposition: **fork-local**.
- Soft coordination: command examples must match S0-03's build entry points and S0-06's
  marker names; if those tickets are still in flight, sync with their open PRs before
  merging.
