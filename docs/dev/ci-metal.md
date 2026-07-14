# The Metal lane (S2-01)

`ci-metal.yml`. Two tiers, the same shape as `ci-cpu` (S0-07), with check names `ci-metal / build`,
`ci-metal / grad`, `ci-metal / e2e` so branch protection can require them by name.

| tier | when | what |
|---|---|---|
| **per-PR** | `pull_request`, `push` | build with `GGML_METAL=ON`, probe the device, `test` + `grad` for `tests/project_ops.py` |
| **nightly** | `schedule`, `workflow_dispatch` | full sweep, the S1-12 convergence gate with `--device metal`, and the fallback report |

## ⚠️ `-b Metal` matches nothing. Metal calls itself `MTL0`.

This is the single most important thing on this page, and it is the trap S2-01's own ticket walked
into — it specifies `-b Metal` in three places.

`test-backend-ops` filters devices with an **exact `strcmp`** against `ggml_backend_dev_name`
(`test-backend-ops.cpp:11214`). And the device names are not what you would guess:

| backend | registers as | where |
|---|---|---|
| CPU | `CPU` | — |
| **Metal** | **`MTL0`** | `ggml-metal-device.m:858` — `snprintf(..., "%s%d", "MTL", device)` |
| CUDA | `CUDA0` (or `ROCm0`, `MUSA0`) | `ggml-cuda.cu:5178`, `GGML_CUDA_NAME` + index |
| Vulkan | `Vulkan0` | `ggml-vulkan.cpp:6459`, `GGML_VK_NAME` + index |

Pass a name that matches nothing and the harness prints `Skipping`, **counts it as passed, and
exits 0** — having run no tests whatsoever. A fully green Metal lane that checked nothing at all.
It is the same failure mode as `grad -o GLU` (see `docs/dev/backward-coverage.md`): green, and
worthless.

So the lane **resolves** the device name from the binary's own `Backend i/N: <name>` output and
never hardcodes it, and every GPU step greps for `Skipping` and fails if it finds it. On the pytest
side, the `ggml_device` fixture asks ggml for the name it actually registered — which is why the
workflow drives MODE_GRAD through `pytest tests/test_backend_ops_grad.py --device metal` rather than
shelling out with a guessed `-b`.

**S3-01 and S4-01 inherit this.** `CUDA` and `Vulkan` are both wrong; you want `CUDA0` and
`Vulkan0`, and on a HIP or MUSA build `CUDA0` is wrong too.

## The probe, and why a bad GPU is a skip and not a failure

GitHub's hosted macOS arm64 runners expose Metal through a **paravirtual GPU**, whose capability set
is not guaranteed to match real Apple Silicon. Every Stage-2 kernel gates on
`has_simdgroup_reduction` / `has_simdgroup_mm` in `supports_op`
(`ggml-metal-device.m:1051-1368`), and the backend logs both at init (`:901-902`).

If the device is absent, or lacks them, the lane **skips with a job-summary notice** rather than
going red. A hosted-runner capability regression is not a code regression, and dressing it up as one
trains people to ignore the lane. The work then moves to a self-hosted Apple Silicon box:

```
gh workflow run ci-metal.yml -f runner=self-hosted-apple-silicon
```

The `runner` input is a plain label, so registration follows the S0-08 playbook unchanged.

## The fallback report — acceptable, but never hidden

ROADMAP §11's scheduler note: until Stage 2's kernels are complete, `ggml_backend_sched`
transparently runs the gaps on the CPU. So training **works** on Metal from day one, at reduced
speed. That is the right behaviour — and it is also a way to ship a green GPU lane that is barely
touching the GPU.

The rule is therefore: **a fallback does not fail the build; a run with no report does.**

`.github/scripts/sched_fallback_report.py` parses `GGML_SCHED_DEBUG=2` output
(`ggml-backend.cpp:1740`; printer at `:945`) and writes a table of which ops ran on the GPU and
which fell back, into the job summary and an artifact. The `upload-artifact` step is
`if-no-files-found: error`, so a missing report is red.

Each op in the fallback table is a kernel this backend still owes. S2-10 flips the dense-LoRA ops to
**fallback-forbidden** once S2-02..S2-09 have landed them.

### The parser has a trap in it, and it is tested

The printer's backend field is `%5.5s` — **truncated to five characters**. `Vulkan0` prints as
`Vulka`. A parser comparing against the full device name would find nothing, report **0% GPU on a
perfectly healthy run**, and the honest-looking conclusion would be "the Vulkan lane is doing
nothing on the GPU". `tests/test_sched_report.py` pins that case.

## One deviation from the ticket, on purpose

S2-01 asks for `.github/scripts/metal-changed-ops.sh` — a changed-files → `-o` list mapper, with a
hardcoded default op list.

**It is not here, and it should not be.** That is a *second op registry*, and a second op registry
is exactly what S1-12 had to fix: CI's inline grad list still named eight ops when the project had
thirteen, so MUL_MAT_ID, ADD_ID, GLU, SSM_CONV and SSM_SCAN — every op the MoE and Mamba work
added — were **not being grad-checked at all**, and CI was green the entire time.

One registry: `tests/project_ops.py`. It is 18 ops, it runs in well under the per-PR budget, and it
cannot drift from the CPU lane because it *is* the CPU lane's list.
