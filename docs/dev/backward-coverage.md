# MODE_GRAD backward coverage, measured

*Measured on the CPU backend at vendor commit `a089900d` (= upstream `4f37f51` + the S0-10
harness fix). Reproduce with the script in the "How this was measured" section.*

ROADMAP §2 has a coverage matrix. This is the *empirical* version — what `test-backend-ops grad`
actually does today, run op by op. It exists because two assumptions the plan rests on turned out
to be wrong, and both were only visible by running the thing.

## The headline: a MODE_GRAD case is vacuous unless the test calls `ggml_set_param`

`test_case::eval_grad` checks gradients only if the test's `build_graph` marked something as a
parameter. If nothing is a parameter, it prints `not supported [<OP>]` and **checks nothing** —
while still ending in `Backend CPU: OK`.

**52 of the 100 `test_case` classes never call `ggml_set_param`.** And the list is not a random
tail — it is, almost exactly, *the ops this project is about*:

| Uncovered test class | Owned by |
|---|---|
| `test_out_prod` | every backward path (S1-18, S2-05/06, S3-02, S4-02/03) |
| `test_flash_attn_ext` | S1-21, S1-23, S2-13, S3-05/06, S4-08 |
| `test_mul_mat_id` | S1-25, S1-26, S1-27 |
| `test_ssm_scan`, `test_ssm_conv` | S1-30, S1-31 |
| `test_glu` | S1-28 |
| `test_clamp` | S1-19 |
| `test_soft_max_back`, `test_rms_norm_back`, `test_silu_back`, `test_repeat_back`, `test_get_rows_back`, `test_cross_entropy_loss_back` | the `*_BACK` ops themselves |

> **`test-backend-ops grad -o OUT_PROD` reports `Backend CPU: OK` while checking exactly zero
> gradients.**

The `*_BACK` entries are fine and expected — you do not take the gradient of a gradient op; they
are forward-tested in `test` mode. The rest are real gaps.

**Consequence, and it binds every kernel ticket:** "add a MODE_GRAD case for the new op" is not
enough. A kernel ticket must ensure the corresponding `test_case` **calls `ggml_set_param` on the
input whose gradient it means to check**, or its acceptance criterion passes trivially and proves
nothing. ADR-0002 now says so normatively.

### Read the verdict, not the count

`test-backend-ops` prints `N/M tests passed` where **M is the global case count of the whole
binary**, not of the `-o` filter:

```
$ test-backend-ops grad -o ADD                 -> 16817/16817 tests passed
$ test-backend-ops grad -o OUT_PROD            -> 16817/16817 tests passed
$ test-backend-ops grad -o CROSS_ENTROPY_LOSS  -> 16817/16817 tests passed
```

Identical denominators. The number is not evidence of anything about the filtered op. **The
signal is the `Backend CPU: OK` / `FAIL` line, and the per-case lines.**

## Ops with no backward rule (MODE_GRAD aborts)

These abort in `ggml_compute_backward` because the op has no VJP. That is ggml behaving
correctly — it is telling you the rule does not exist yet.

| Op(s) | Abort site | Owner |
|---|---|---|
| `REGLU`, `GEGLU`, `GEGLU_ERF`, `GEGLU_QUICK`, `SWIGLU_OAI` | `ggml.c:6897` "unsupported glu op for backward pass" | **S1-28** |
| `CEIL`, `FLOOR`, `ROUND`, `TRUNC`, `XIELU` | `ggml.c:6875` (unary, no VJP) | not scheduled — rounding ops have zero gradient a.e.; not on any training path |
| `CUMSUM`, `DIAG`, `FILL`, `IM2COL_3D`, `POOL_1D`, `SOLVE_TRI`, `TRI` | `ggml.c:6906` | not scheduled — not on any training path |

Plain **`SWIGLU` has a backward rule and passes**, which is why dense llama (SwiGLU FFN) trains
today while gpt-oss-style `SWIGLU_OAI` and gated `GEGLU`/`REGLU` architectures do not. That is
precisely the gap S1-28 closes.

**So the unfiltered `test-backend-ops grad` sweep still cannot complete**, and will not until the
stage-1 backward-rule tickets land. MODE_GRAD is not a working baseline that stage 1 extends —
it is a baseline that stage 1 *creates*. ADR-0001's "full MODE_GRAD on every vendor bump" is
therefore aspirational today; the bump gate runs the allowlist below, which grows ticket by
ticket until it is the whole table and the qualifier can be deleted.

## Ops whose backward exists but does not match finite differences

| Op | Symptom | Assessment |
|---|---|---|
| `MUL_MAT` | a handful of cases marginally over the bound (`MAA = 0.0001004 > 0.0001`) | Finite-difference precision at large `k`, not a gradient bug. 360 F32 cases pass. Candidate for a per-op `max_maa_err()` override. |
| `CPY`, `SCALE`, `SUM` | marginal | Not investigated. None is on the LoRA training path. |

### Two entries this table got wrong, and what they cost

**`SOFT_MAX` — the gradient was never wrong. The test never checked it.** (S1-34, fork PR #13.)

This table used to say sinks were a *missing term* in `SOFT_MAX_BACK` and that "attention-sink
models would train on a wrong gradient". Both claims were false, and the reason is worth carrying:

MODE_GRAD's objective is `sum(out)`, and **a softmax's rows sum to one by construction**. So
`sum(out)` is identically 1 whatever the input, `d(sum(out))/dx` is *exactly zero*, and every
`sinks=0` case was comparing zero against zero and reporting `OK`. The test had never exercised
`SOFT_MAX_BACK` at all. A sink breaks the conservation — it takes part in the normalization but
emits no output, so the rows sum to `1 − p_sink`, which *does* depend on the input. The `sinks=1`
cases were the only ones in the whole sweep with a nonzero gradient, hence the only ones that
*could* fail. And they failed on the **finite difference**, not on the kernel: against a float64
FD with mask, sink, scale and ALiBi slope all present, the worst relative error is `3.1e-6`.

The lesson generalizes past softmax: **an op whose output sum is conserved is invisible to a
`sum(out)` objective.** `test_case` now takes a `grad_loss` hook so such an op can supply an
objective that actually depends on its input.

What *was* genuinely missing is `dL/d(sinks)` — silently ignored rather than unimplemented, so a
full fine-tune would have trained its attention sinks **frozen** and nothing would have said so.
It now aborts with the formula in the message: `dL/ds = -(1 - sum(y)) * dot(y, dy)`.

**`EXPM1` — "not on the LoRA training path" was wrong, and it hid a real numerical bug.**

`EXPM1` is on the GRPO training path: the k3 KL estimator is `expm1(d) − d`. And ggml's
`op_expm1` was implemented as `expf(x) - 1.0f` — precisely the catastrophic cancellation the op
exists to avoid, and precisely the regime a GRPO run lives in (`d ≈ 0` on every on-policy step, by
design). Measured against the true value:

| d | ggml's k3 | true | |
|---|---|---|---|
| 1e-6 | 7.29e-8 | 5.0e-13 | 145,000× too large |
| 5e-5 | **−5.13e-8** | +1.25e-9 | **negative** |

A KL penalty that goes negative does not penalize divergence; it **pays for it**. Fixed in the
fork (`op_expm1` → `expm1f`, which was already used twice in the same file — see
[`fork-changes.md`](fork-changes.md)).

Dismissing an op as "low priority, not on the training path" is a claim about the *whole* library,
including the parts not written yet. This table now says what was checked, not what was assumed.

## ⚠️ `grad -o <op>` reporting OK does not mean the op has a backward

`test-backend-ops grad` only checks a gradient if the test class asks for one, by calling
`ggml_set_param`. If it never does, MODE_GRAD builds a graph with no parameters, requests no
gradients, compares nothing, and prints **`Backend CPU: OK`**.

Measured today, on the ops stage 1 still has to implement:

| test class | `ggml_set_param` calls | `grad -o` says | has a backward in `ggml.c`? |
|---|---|---|---|
| `test_mul_mat_id` | **0** | `OK` — 16829 tests passed | **no** |
| `test_add_id` | **0** | `OK` | **no** |
| `test_ssm_conv` | **0** | `OK` | **no** |
| `test_ssm_scan` | **0** | `OK` | **no** |
| `test_flash_attn_ext` | **0** | `OK` | **no** — and `ggml_flash_attn_back` is a stub whose *first statement* is `GGML_ABORT("TODO: adapt to ggml_flash_attn_ext() changes")` |
| `test_out_prod` | **0** | `OK` | n/a (it *is* a backward) |
| `test_glu` | **0** | `OK` | partially — SWIGLU only |

Every one of those ops falls through `ggml_compute_backward`'s `default:` case, which is a
`GGML_ABORT`. **Every one of them reports `OK` under `grad`.**

This matters beyond bookkeeping, because the remaining stage-1 kernel tickets (S1-21 … S1-31) each
name *"`test-backend-ops grad -o <op>` green"* as an acceptance criterion — **and that criterion
passes right now, with nothing implemented.** It cannot fail. A ticket closed against it would ship
a backward that had never once been differentiated.

So a kernel PR here is **not** done when the VJP is written. It is done when:

1. the test class calls `ggml_set_param` on the inputs whose gradients the VJP produces, **and**
2. the objective actually depends on those inputs — see the SOFT_MAX entry above; an op whose
   output sum is conserved needs `test_case::grad_loss`, or it is checking zero against zero, **and**
3. `grad -o <op>` is green *after* (1) and (2), which is the first point at which its greenness
   means anything.

## The allowlist: what the vendor-bump gate runs today

Genuinely grad-checked (the test class calls `ggml_set_param`) **and** green:

```
test-backend-ops grad -o CROSS_ENTROPY_LOSS,CROSS_ENTROPY_LOSS_SPARSE,RMS_NORM,TANH,SIGMOID,CLAMP
```

**S1-19 added TANH, SIGMOID and CLAMP.** All three now grad-check (6, 6 and 5 cases). Note that
`test_clamp` already *declared* `grad_eps()` and `grad_expect() = {0, 1}` — as if it were being
gradient-checked — while never calling `ggml_set_param`. It looked like a gradient test and
verified nothing. Exactly the trap this document exists to name.

`MUL_MAT` is genuinely checked and *nearly* green, but it takes **3m20s** and has the marginal
FD failures above, so it is a nightly candidate rather than a per-PR one. It matters more than it
looks: **`MUL_MAT`'s backward is built from `ggml_out_prod`** (`ggml.c:6594-6630`), so
`grad -o MUL_MAT` is the only thing that exercises `OUT_PROD`'s gradient path at all — indirectly,
since `grad -o OUT_PROD` checks nothing.

This list grows as stage 1 lands. Each kernel ticket adds its op here.

## How this was measured

```bash
cmake -S vendor/llama.cpp -B build/vendor-tests -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON \
      -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF \
      -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF
cmake --build build/vendor-tests --target test-backend-ops -j2

TBO=./build/vendor-tests/bin/test-backend-ops
$TBO --list-ops | tr ',' '\n' | tr -d ' ' | grep -E '^[A-Z]' | sort -u | while read -r op; do
    out=$(timeout 300 "$TBO" grad -o "$op" 2>&1)
    case $? in
        134) echo "$op ABORT $(echo "$out" | grep -oE 'ggml\.c:[0-9]+: .*' | head -1)" ;;
        *)   echo "$out" | grep -q 'Backend CPU: .*FAIL' && echo "$op FAIL" || echo "$op OK" ;;
    esac
done
```

For the `ggml_set_param` audit, parse `tests/test-backend-ops.cpp` for `struct test_* : public
test_case` blocks and check each body for `ggml_set_param`. **Do this before trusting any `OK`.**
