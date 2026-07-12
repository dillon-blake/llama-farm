# Testing learning-llamas

Two suites, with different jobs.

- **pytest** (`tests/`) tests the Python library and the shim end to end: the bindings, the
  adapter format, the trainers, the convergence gate.
- **`test-backend-ops`** (vendored) tests *kernels*. It is the acceptance harness every kernel
  ticket is measured by (ADR-0002), and MODE_GRAD is the part that matters.

## pytest

```bash
pytest tests/ -m "not slow"      # the per-PR selection
pytest tests/                    # everything, including the convergence gate
```

Markers are registered in `pyproject.toml` with `--strict-markers` on, so a typo is an error
rather than a silently-skipped test.

| Marker | Meaning |
|---|---|
| `slow` | Nightly only — convergence gates (S1-12), full sweeps |
| `cuda` / `metal` / `vulkan` | Needs that backend compiled in |

`ci-cpu` runs `-m "not slow"` per PR and the full suite nightly. Fixture models are generated on
first use and cached; see [`tests/README.md`](../../tests/README.md).

## `test-backend-ops`

Build it first (see [`building.md`](building.md)). The **mode is positional**:

```bash
./build/vendor-tests/bin/test-backend-ops <mode> [-o <OP,..>] [-b <backend>]
```

| Mode | What it does |
|---|---|
| `test` | Compare a backend's output against the **CPU backend** (correctness) |
| `grad` | **MODE_GRAD** — compare backpropagated gradients against finite differences |
| `perf` | Performance |
| `support` | Probe which ops a backend claims to support |

(`vendor/llama.cpp/tests/test-backend-ops.cpp:10075-10091`.)

### ⚠️ Before you trust a MODE_GRAD result, read this

Two things about this harness will fool you. Both are measured in
[`backward-coverage.md`](backward-coverage.md).

**1. A MODE_GRAD case checks nothing unless the test calls `ggml_set_param`.** If nothing in the
graph is a parameter, `eval_grad` prints `not supported [<OP>]`, checks zero gradients, and the
run still ends in `Backend CPU: OK`. **52 of the 100 test classes are in that state**, including
`test_out_prod`, `test_flash_attn_ext`, `test_mul_mat_id`, `test_ssm_scan` and `test_glu` — very
nearly the exact set of ops this project is about.

> `test-backend-ops grad -o OUT_PROD` reports `Backend CPU: OK` while checking **zero** gradients.

So a kernel ticket does not discharge its acceptance criterion by adding a test case. It must make
the `test_case` call `ggml_set_param` on the input whose gradient it means to check, and say in
the PR how many cases were actually grad-checked (ADR-0002).

**2. The `N/M tests passed` number is global, not per-filter.**

```
$ test-backend-ops grad -o ADD                -> 16817/16817 tests passed
$ test-backend-ops grad -o OUT_PROD           -> 16817/16817 tests passed
$ test-backend-ops grad -o CROSS_ENTROPY_LOSS -> 16817/16817 tests passed
```

Same denominator every time. Never quote it as evidence. **The signal is the `Backend CPU: OK` /
`FAIL` verdict and the per-case lines.**

### MODE_GRAD — the one that matters

Every new or ported kernel must pass MODE_GRAD against the CPU oracle. Worked examples:

```bash
# The CPU oracle for one op. This is what a stage-1 kernel ticket runs.
./build/vendor-tests/bin/test-backend-ops grad -o OUT_PROD -b CPU

# A GPU port, checked against finite differences.
./build/vendor-tests/bin/test-backend-ops grad -o OUT_PROD -b CUDA0

# Several ops at once.
./build/vendor-tests/bin/test-backend-ops grad -o MUL_MAT,SOFT_MAX,CROSS_ENTROPY_LOSS

# Everything. Required on every vendor bump (ADR-0001).
./build/vendor-tests/bin/test-backend-ops grad
```

Two useful flags when adding an op: `--list-ops` (is my op registered?) and `--show-coverage`
(is it actually being exercised?).

### Which backend name do I pass to `-b`?

```bash
./build/vendor-tests/bin/test-backend-ops support     # lists the backends this build has
```

`CPU`, `CUDA0`, `Metal`, `Vulkan0`. A GPU name only exists if you built with that backend on.

## Tolerances — read ADR-0002, do not invent them

[ADR-0002](../adr/ADR-0002-numerics-determinism-parity.md) is normative. The two bounds it
defines are different things and are routinely conflated:

- **Per-op finite-difference bound.** Within one backend. `max_maa_err()`, default `1e-4`
  (`test-backend-ops.cpp:1158-1160`), overridable per test case. For *discontinuous* gradients
  (ReLU-like kinks, where a finite-difference estimate straddling the kink is simply wrong), use
  the harness's expected-value filtering (`test-backend-ops.cpp:319-321`) — **not** a loosened
  bound. The right fix for a discontinuity is to not sample across it.

- **Cross-backend parity criterion.** Between backends: **max-abs gradient error ≤ 0.05 at
  fp16**, GPU vs the CPU oracle, on identical inputs. This is the bar for calling a backend's
  kernel "at parity", and what the milestone tickets (S2-10, S3-10, S4-09) measure.

ADR-0002 also fixes two things your kernel PR must state:

- which **determinism scheme** it implements (gate G-B: deterministic by default; atomics only
  as a benchmarked, opt-in variant);
- that it contains **neither forbidden pattern** — CUDA's F16 cuBLAS traits, or Vulkan's
  `f16acc` `mul_mm` variants — on a gradient path.

## End-to-end training acceptance

The per-op suites can all be green while training still fails to converge. The gate that
catches that is the **tiny-model convergence gate**, S1-12: a tiny-model SFT run compared
against a recorded PEFT reference loss curve. It is marked `slow` and runs nightly, and it is
the exit criterion for stage 1 and the acceptance criterion for every backend milestone
(`--device metal`, `--device cuda`, …).

## Before you open a PR

```bash
pre-commit run --all-files
pytest tests/ -m "not slow"
./build/vendor-tests/bin/test-backend-ops grad -o <YOUR_OP> -b CPU     # kernel tickets
```
