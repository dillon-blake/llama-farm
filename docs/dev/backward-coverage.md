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

### There is a THIRD way to check nothing: the op name matches no case at all

`-o` filters on `ggml_op_desc(out)` — and for a `GGML_OP_GLU` node that returns the **variant**
name, not `"GLU"` (`ggml.c:1398-1400`, exactly as `GGML_OP_UNARY` returns `"SILU"` rather than
`"UNARY"`):

```
$ test-backend-ops grad -b CPU -o GLU
Testing 1 devices
Backend 1/1: CPU
  Backend CPU: OK          # ...having matched ZERO cases and run nothing. Exit code 0.
```

`GLU` sat in `tests/project_ops.py` on the strength of that, checking nothing, until the guard was
fixed. The real names are `SWIGLU`, `GEGLU`, `REGLU`, `GEGLU_ERF`, `GEGLU_QUICK`, `SWIGLU_OAI` —
8 genuinely-checked cases each.

### How to tell, mechanically

Reading the harness's output is trickier than it looks, and **both** obvious approaches are wrong.

Per case, `eval_grad` prints:

* a **non-F32 output** bails at `test-backend-ops.cpp:1746` and prints **one** line;
* every other case prints an **info line first** (`:1754`) — which `print_operation` renders as a
  bare `OK`, *before a single thing has been checked* — and then exactly one real verdict.

```
OUT_PROD(...): OK                        <- the INFO line. Printed before any check. Means nothing.
OUT_PROD(...): not supported [OUT_PROD]  <- the GRADIENT verdict: no params, nothing compared.
```

Reading the **first** line is what made the first version of the guard vacuous. But "take every
second line" is **also** wrong, because of the one-line bail: `CLAMP` emits **nine** lines today —
an odd number, which is definitionally impossible under two-per-case — and the misalignment makes
info lines get read as gradient verdicts.

`tests/test_backend_ops_grad.py::_n_gradients_compared` walks the output **in order**: a leading
`not supported`/`skipping` is a one-line bail; anything else is an info line whose verdict is the
line after it. An op counts as checked only if some case's *verdict* is neither `not supported` nor
`skipping`.

And `test_the_vacuity_guard_can_actually_detect_vacuity` points the guard at `OUT_PROD` and
`FLASH_ATTN_EXT` — both of which must keep reporting **zero** compared gradients, or the guard has
stopped working and every op in the registry is being taken on trust.

### A fourth way to check nothing: `-b` takes ggml's device name, and GPU names are index-suffixed

`test-backend-ops -b <name>` matches with an exact `strcmp` against `ggml_backend_dev_name`
(`test-backend-ops.cpp:11214`). GPU backends **index-suffix** their names — the first CUDA device is
`CUDA0`, not `CUDA`. So `-b CUDA` matches nothing: the harness prints `Skipping`, counts it as
passed, and **exits 0** having run zero cases. Every backend lane would go green while checking
nothing.

`CPU` happens to be unsuffixed, which is exactly why this stays invisible until a GPU lane exists.
The `ggml_device` fixture asks ggml for the real name rather than assuming it, and
`test_the_harness_runs_on_the_device_we_asked_for` asserts the run was not skipped.

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

## ⚠️ Your VJP will be handed a TRANSPOSED grad. Read `src->nb[0]`, never `float*[j]`

ggml's autodiff produces non-contiguous gradients as a matter of course. `GGML_OP_TRANSPOSE`'s
backward is `ggml_add_or_set(..., ggml_transpose(ctx, grad))`, and the MUL_MAT backward passes
`ggml_transpose(grad)` **straight into `ggml_out_prod`** with no `ggml_cont`. That is precisely why
`ggml_compute_forward_out_prod_f32` reads `src1` through `i1*nb10` instead of indexing a `float *`.

Both S1-26/S1-27 kernels dropped that and indexed `grad` as `g_col[j]` — a hard-coded 4-byte
stride. A TRANSPOSE node reaching them has `nb[0] == 8`. Measured: **24 of 36 output elements
wrong, max abs error 9.2.** No assert fires. The run just trains on a wrong gradient.

The same applies to **index tensors**. `ggml_mul_mat_id` places *no* contiguity constraint on `ids`,
and the forward reads it through `nb[0]`
(`*(int32_t *)(ids->data + iid1*nb[1] + id*nb[0])`). So a strided `ids` is a legal tensor that the
forward computes **correctly** and a `int32_t*[i]` backward misreads — crediting the wrong experts
with each other's gradients.

**And no test in this repo could see either bug.** `test-backend-ops` only ever builds a contiguous
grad. A hand-written numerical reference does not help either, because it constructs *its own*
contiguous tensors: a verification that builds its own inputs cannot discover an input shape you did
not think of.

So the guard has to force the shape. `test_mul_mat_id::grad_transposed` routes the objective through
`cont(transpose(out))`, which makes `ggml_build_backward_expand` hand the backward a TRANSPOSE-op
grad. With the bug reintroduced:

| cases | verdict |
|---|---|
| contiguous grad — **the entire pre-existing suite** | **PASS. Invisible.** |
| transposed grad — the new case | **FAIL, MAA 21.0 / 2.57 / 2.55** |

**Every new VJP needs an equivalent case.** Contiguity is an assumption, and in ggml it is usually
the wrong one. Assert contiguity only where the arithmetic genuinely requires it — e.g. the *vector*
operand of `ggml_vec_mad_f32`, where a strided read is not merely wrong but unrepresentable — and
read everything else through its strides.

## ⚠️ `mean_abs_asymm` divides by `(gn + ga)`, not `(|gn| + |ga|)`

```c
const float asymm = (a[i] - b[i]) / (a[i] + b[i]);
```

So **any output element whose true gradient is near zero sends that ratio to infinity**, and MAA is
a *mean* — one bad element out of ninety is enough to fail a case whose kernel is exact.

This is not theoretical. It is what `MUL_MAT_ID`'s grad test did (S1-26/S1-27). `d_as[k,j,e]` sums
only over the slots that routed to expert `e`, so with 5 tokens and 3 experts each output element is
a sum of **one or two** random products — which lands near zero often. Measured, with the kernel
verified exact against a double-precision reference the entire time:

| tokens | outcome |
|---|---|
| 5 | **3/10 runs fail**, MAA up to **7.9** |
| 32 | **0/30 runs fail** |

**Condition the test; do not widen the tolerance until the flapping stops.** An op whose output
element is a sum over a *selected subset* (MoE routing, gathers, anything index-driven) needs enough
terms per element that the sum is reliably far from zero. If you find yourself raising
`max_maa_err()` to silence an intermittent failure, check the conditioning first — you may be hiding
a real defect behind a bound wide enough to fit one.

When a looser bound genuinely *is* right, measure both sides of it and say so. S1-26's:

| | MAA |
|---|---|
| worst FD noise, 40 runs | 4.5e-4 |
| kernel ignores `grad` entirely | 9.21 |
| kernel ignores the broadcast | 3.39 |
| kernel reads `grad` row 0 always | 4.41 |

`5e-3` sits 10x above the noise and 200-1800x below every real defect. That is a tolerance with an
argument behind it, not a number chosen to make a test go green.

## ⚠️ MODE_GRAD cannot check a gradient that flows through a QUANTIZED weight

ggml's quantized matmul **quantizes the activations on the fly** to `vec_dot_type` before the dot
product. The forward is therefore a *staircase* in its activation input, and a finite difference of
it measures quantization edges rather than a derivative. The analytic gradient is the gradient of
the intended *smooth* function; the harness evaluates the *actual quantized* one. They disagree by
construction — for `OUT_PROD_ID`, MAA 0.028-0.071 with the kernel bit-exact.

Upstream knows, and encoded it with no comment at all (`test_mul_mat`):

```c
if (!ggml_is_quantized(type_a)) {
    ggml_set_param(b);        // b is a param ONLY when the weight is not quantized
}
```

So don't try. Verify the dequantize path by **direct equivalence** instead: run the op with the
quantized weight, run it again with that same weight dequantized to F32, and require they agree.
For `OUT_PROD_ID` they agree **bit-exactly** on q8_0, q4_K and q4_0. That is a stronger check than a
finite difference, not a weaker one.

## ⚠️ A MODE_GRAD **failure** is not proof of a bug either. The FD is cancellation-limited.

The trap above is a false *pass*. This is the false *fail*, and it wasted an afternoon in S1-12.

`test-backend-ops grad -o SILU` reports **FAIL, MAA 0.31**. SILU is on the LoRA training path (it is
the activation in the SwiGLU FFN), so this looks alarming. Checked against the exact derivative,
`ggml_silu_back` is **correct to 7.1e-07** — float32 rounding. The kernel is fine. The *harness* is
measuring noise.

**MODE_GRAD finite-differences a scalar objective that it reads back as a float32.** Look at what
that costs:

| | |
|---|---|
| `test_unary` initializes in **[-150, 150]** | (`test-backend-ops.cpp:2114`) |
| `silu(x) ≈ x` for `x > 0` — **unbounded** | so `sum(silu(x))` over 5005 elements is **187,860** |
| float32 ulp at 187,860 | **0.0156** |
| `grad_eps` = 0.1, so the FD quantum is `ulp / 2·eps` | **0.078** |
| elements whose true gradient is below 1e-3 | **47%** — and each scores asymm = **1.000** |

That is the entire MAA. The general rule, which is worth stating because nothing in the harness
says it:

> **MODE_GRAD's finite-difference noise floor is `ulp(|objective|) / (2 · grad_eps)`.** It fails
> whenever the objective's *magnitude* is large relative to the gradient signal — **whether or not
> the kernel is correct**.

Bounded-output ops (TANH, SIGMOID, SOFT_MAX) never trip it: their summed objective stays small.
SILU trips it because `silu` is unbounded, and SCALE trips it because its `bias=1` inflates the sum.
Upstream already knows, without saying so: `test_sum` overrides `grad_eps()` to `0.1·sqrt(n)`
*precisely* to fight this, and `test_unary` does not.

Two consequences:

* **A MODE_GRAD failure is a hypothesis, not a verdict.** Confirm it against an exact oracle before
  touching a kernel. (SILU, SCALE, SUM and CPY are all in this category and are all correct.)
* **A MODE_GRAD pass on a large-magnitude objective is weak**, for the same reason — the FD it is
  agreeing with is quantized.

Neatly, the weighted `grad_loss` hook added for the *vacuity* trap cures this one too: zero-mean
weights make the objective `O(sqrt(n)·sigma)` instead of `O(n·mu)`, so the cancellation goes away.
That is why the ops which override `grad_loss` (SOFT_MAX, MUL_MAT_ID, SSM_CONV) are green.

This is also why S1-12's convergence gate exists, and why it is a **float64 reference** rather than
a tolerance band: MODE_GRAD's arithmetic simply cannot resolve a gradient to better than a few
percent on a large objective, and a real gradient bug lives well inside that.

## The allowlist: what the vendor-bump gate runs today

**The list is now `tests/project_ops.py`, and CI reads it from there.** It used to be typed inline
in `.github/workflows/ci-cpu.yml`, where it went stale the moment a kernel ticket landed — by S1-31
it still named eight ops while the project had thirteen, so MUL_MAT_ID, ADD_ID, GLU, SSM_CONV and
SSM_SCAN (every op the MoE and Mamba work added) were not grad-checked in CI at all. CI was green.

All thirteen are genuinely grad-checked (the test class calls `ggml_set_param`) **and** green, with
real cases rather than skips — `test_backend_ops_grad.py::test_every_op_reports_real_cases` enforces
that, so an op cannot join the list on the strength of a vacuous pass:

| op | real cases | added by |
|---|---|---|
| `CROSS_ENTROPY_LOSS` | 5 | S1-04 |
| `CROSS_ENTROPY_LOSS_SPARSE` | 17 | S1-04 |
| `RMS_NORM` | 33 | S0-09 |
| `TANH` / `SIGMOID` / `CLAMP` | 8 / 8 / 7 | S1-19 |
| `SOFT_MAX` | 293 | S1-20, S1-34 |
| `CONCAT` | 46 | S1-29 |
| `MUL_MAT_ID` | 840 | S1-25 |
| `ADD_ID` | 64 | S1-25 |
| `SWIGLU` / `GEGLU` / `REGLU` / `GEGLU_ERF` / `GEGLU_QUICK` / `SWIGLU_OAI` | 8 each | S1-28 |
| `SSM_CONV` | 75 | S1-30 |
| `SSM_SCAN` | 10 | S1-31 |

Deliberately **not** on the list:

* `MUL_MAT` — genuinely checked and nearly green, but it takes **3m20s**, so it is nightly. It
  matters more than it looks: **`MUL_MAT`'s backward is built from `ggml_out_prod`**
  (`ggml.c:6594-6630`), so it is the only thing that exercises `OUT_PROD`'s gradient path at all —
  `grad -o OUT_PROD` checks zero gradients, because `test_out_prod` never calls `ggml_set_param`.
* `SILU`, `SCALE`, `SUM`, `CPY` — FD-cancellation-limited, see the section above. All correct.

**S1-19 added TANH, SIGMOID and CLAMP.** All three now grad-check. Note that
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
