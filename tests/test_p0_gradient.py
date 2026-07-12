"""S1-03: the gradient reaching the adapter is the *right* gradient.

Everything before this proves that a gradient arrives — the graph builds, the loss falls, the
adapter changes. None of that proves the gradient is **correct**. A gradient that is wrong by a
constant factor, or transposed, or missing a term, still makes a loss curve go down; it just
converges to somewhere else, slowly, and nothing complains. (It found exactly such a bug: see
``build_masked_ce`` in csrc/farm_train.cpp.)

So this closes the loop: for individual elements of the adapter's A and B tensors, compare the
analytic gradient ggml computed against a **central finite difference of the whole graph** —
perturb one weight, re-run the real forward pass, and see how the real loss moved.

This is deliberately *not* the per-op MODE_GRAD check of ADR-0002. That validates one kernel in
isolation. This validates the entire composition — embedding, attention, the LoRA injection, the
loss — end to end, through the shim.


Why the finite difference runs on an F32 base and not the quantized one
----------------------------------------------------------------------

The ticket asks for the finite-difference check on a **quantized** base. That is not possible, and
the reason is worth stating because it looks like a tolerance problem and is not.

llama.cpp does not do a quantized matmul by dequantizing the weights. It quantizes the
**activations** to the weight type's ``vec_dot_type`` — Q8_K for a Q4_K weight — and dot-products
in the integer domain. So a small change to the adapter makes a small change to the activations,
which usually does not change their 8-bit codes at all, and occasionally flips one by a whole
quantum. The forward is a **step function** of the adapter weights, with steps of order 1e-3 in
the loss. Measured, sweeping one element of B on this fixture:

    B[0]      Q4_K base        F32 base
   -0.040   6.3610424995    6.2726659775
   -0.020   6.3612122536    6.2726297379
    0.000   6.3611059189    6.2725958824
   +0.020   6.3608088493    6.2725639343
   +0.040   6.3606944084    6.2725348473

The F32 column is a straight line. The Q4_K column has no trend at all — the true signal (a slope
of ~1e-3 across the whole sweep) is entirely buried under the quantization steps. No choice of eps
recovers it: shrink eps and the signal shrinks while the steps do not.

The backward, meanwhile, differentiates the *smooth dequantized* function (the MUL_MAT gradient is
``ggml_out_prod(W, grad)``, which dequantizes W), and that is the right thing for it to do — it is
the standard straight-through treatment of a non-differentiable quantizer. So the analytic gradient
and the finite difference are computing different things on a quantized base, and neither is wrong.

Hence the split below:

  * the finite-difference check runs on **F32**, where the forward is genuinely differentiable, and
    proves the composition is differentiated correctly;
  * a separate test proves the **quantized** backward agrees with the F32 one, which is the claim
    the ticket actually cares about ("gradients through a quantized base are numerically right").
"""

import ctypes
import math

import numpy as np
import pytest

from learning_llamas import _ffi
from learning_llamas.adapter import _write_adapter_gguf, create_zero_adapter

N_CTX = 64
N_UBATCH = 32
RANK = 4
SEED = 7

# Elements whose analytic and numeric gradients must agree to this relative error.
#
# The worst observed on this fixture is 0.009. 3e-2 leaves room for a different SIMD reduction
# order on another host without leaving room for an actually-wrong gradient: a transposed,
# mis-scaled or missing-term gradient misses by tens of percent to orders of magnitude, as the
# S1-02 backward bug did (it sat at rel 0.4-1.4).
FD_TOLERANCE = 3e-2

# Large enough that the loss moves well above float32 rounding, small enough that the second-order
# term stays negligible. The check is insensitive to it: 1e-3 and 1e-1 both pass on F32.
FD_EPS = 1e-2


def _arrays(tokens, targets, weights):
    n = len(tokens)
    return (
        (ctypes.c_int32 * n)(*tokens),
        (ctypes.c_int32 * n)(*targets),
        (ctypes.c_float * n)(*weights),
    )


def _batch(n: int = 32):
    """A fixed batch whose first half is masked out, as a real SFT sample's prompt would be."""
    tokens = [7, 11, 13, 17] * (n // 4)
    targets = tokens[1:] + [tokens[0]]
    weights = [0.0] * (n // 2) + [1.0] * (n // 2)
    return tokens, targets, weights


class Harness:
    """A LoRA-initialized model with a fixed batch, and the debug accessors to poke at it."""

    def __init__(self, libs, model, batch, targets):
        self.libs = libs
        self.model = model
        self.targets = targets
        self.names = [t.name for t in targets]
        self.tok, self.tgt, self.wts = _arrays(*batch)
        self.n = len(batch[0])

    def step(self, train: bool) -> float:
        loss = ctypes.c_float()
        _ffi.check(
            self.libs.farm.ll_train_step(
                self.model.ctx, self.tok, self.tgt, self.wts, self.n, train, ctypes.byref(loss)
            ),
            "ll_train_step",
        )
        return loss.value

    def n_elements(self, name: str, is_b: bool) -> int:
        return self.libs.farm.ll_debug_n_elements(self.model.ctx, name.encode(), is_b)

    def get(self, name: str, is_b: bool):
        n = self.n_elements(name, is_b)
        buf = (ctypes.c_float * n)()
        assert self.libs.farm.ll_debug_get_tensor(self.model.ctx, name.encode(), is_b, buf, n) == n
        return buf

    def set(self, name: str, is_b: bool, values) -> None:
        n = len(values)
        buf = values if isinstance(values, ctypes.Array) else (ctypes.c_float * n)(*values)
        assert self.libs.farm.ll_debug_set_tensor(self.model.ctx, name.encode(), is_b, buf, n) == n

    def grad(self, name: str, is_b: bool) -> list[float]:
        n = self.n_elements(name, is_b)
        buf = (ctypes.c_float * n)()
        got = self.libs.farm.ll_debug_grad(self.model.ctx, name.encode(), is_b, buf, n)
        assert got == n, f"ll_debug_grad returned {got}"
        return list(buf)

    def snapshot(self) -> dict[tuple[str, bool], list[float]]:
        """Every A and B of every target, so a step can be rewound exactly."""
        return {(n, b): list(self.get(n, b)) for n in self.names for b in (False, True)}

    def restore(self, snap: dict[tuple[str, bool], list[float]]) -> None:
        for (name, is_b), saved in snap.items():
            self.set(name, is_b, saved)

    def grad_at(self, snap: dict[tuple[str, bool], list[float]], name: str, is_b: bool):
        """The gradient at exactly the weights in ``snap``, leaving the model back at them.

        A training step moves *every* trainable tensor, and a gradient is only comparable with a
        finite difference taken at the same point in parameter space. Restoring only the tensor
        under test would leave the other fourteen at their post-step values -- a different loss
        surface, and a mismatch that looks exactly like a wrong gradient.
        """
        self.restore(snap)
        self.step(train=True)
        g = self.grad(name, is_b)
        self.restore(snap)
        return g


def _make(libs, load_model, base, tmp_path, tag: str, alpha: float = 1e-3):
    """A rank-4 zero-init adapter on ``base``, with the optimizer initialized."""
    adapter_path = tmp_path / f"adapter-{tag}.gguf"
    targets = create_zero_adapter(base, adapter_path, r=RANK, seed=SEED)

    model = load_model(base, n_ctx=N_CTX, n_ubatch=N_UBATCH, training=True)
    model.attach_adapter(adapter_path, scale=1.0)

    # ggml_opt asserts alpha > 0, so the weights cannot be frozen by zeroing the learning rate.
    # Tests that need a fixed evaluation point snapshot and restore instead -- which is correct
    # anyway: a backward pass produces the gradient AT THE PRE-STEP WEIGHTS, so that is where a
    # finite difference must be taken.
    params = _ffi.ll_opt_params(alpha=alpha, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0)
    _ffi.opt_init_lora(libs, model.ctx, model.model, [model.adapter], params)

    return Harness(libs, model, _batch(), targets), params


@pytest.fixture
def f32_harness(tiny_f32, tmp_path, load_model, libs: _ffi.Libraries):
    """An F32 base -- the only one whose forward is a differentiable function of the adapter."""
    h, params = _make(libs, load_model, tiny_f32, tmp_path, "f32")
    yield h, params
    _ffi.opt_free(libs, h.model.ctx)


@pytest.fixture
def q4_k_harness(tiny_q4_k, tmp_path, load_model, libs: _ffi.Libraries):
    """A Q4_K base, memory-mapped -- the configuration the project actually ships."""
    h, params = _make(libs, load_model, tiny_q4_k, tmp_path, "q4_k")
    yield h, params
    _ffi.opt_free(libs, h.model.ctx)


def _fd_check(h: Harness, name: str, is_b: bool, snap, eps: float = FD_EPS) -> list[str]:
    """Central finite differences against the analytic gradient, on samples across the tensor."""
    analytic = h.grad_at(snap, name, is_b)
    base = snap[(name, is_b)]

    failures = []
    stride = max(1, len(base) // 8)

    for i in range(0, len(base), stride):
        perturbed = (ctypes.c_float * len(base))(*base)

        perturbed[i] = base[i] + eps
        h.set(name, is_b, perturbed)
        loss_up = h.step(train=False)

        perturbed[i] = base[i] - eps
        h.set(name, is_b, perturbed)
        loss_down = h.step(train=False)

        h.set(name, is_b, base)

        numeric = (loss_up - loss_down) / (2.0 * eps)
        denom = max(abs(analytic[i]), abs(numeric), 1e-8)
        rel = abs(analytic[i] - numeric) / denom

        if rel > FD_TOLERANCE:
            which = "B" if is_b else "A"
            failures.append(
                f"{name}.{which}[{i}]: analytic={analytic[i]:.6g} numeric={numeric:.6g} "
                f"rel={rel:.3g}"
            )

    return failures


def test_dL_dB_matches_finite_differences(f32_harness) -> None:
    """The check that closes the loop, on the tensor that carries the gradient at initialization.

    B is where a zero-init adapter's gradient lives: dL/dB is nonzero even at B = 0 (that is what
    lets the adapter learn at all), while dL/dA is exactly zero there. Two different layers, so a
    bug that happens to cancel in one is not hidden by it.
    """
    h, _ = f32_harness
    snap = h.snapshot()

    names = [n for n in h.names if n.startswith(("blk.0.attn_q", "blk.1.ffn_down"))]
    assert len(names) >= 2, names

    failures = [f for name in names for f in _fd_check(h, name, True, snap)]

    assert not failures, "dL/dB disagrees with finite differences:\n" + "\n".join(failures)


# dL/dA is proportional to B, so at B = 0 it is not merely small but exactly zero, and just off
# zero it is too small to finite-difference: the loss moves by a couple of float32 ULPs and the
# numeric derivative comes back quantized to multiples of one ULP. Measured on this fixture, the
# error is pure measurement resolution and falls away as soon as the signal clears the floor:
#
#     B = 0.02, eps = 0.01   |dL/dA|max = 5.5e-04   worst rel = 0.42
#     B = 0.02, eps = 0.10   |dL/dA|max = 5.5e-04   worst rel = 0.032
#     B = 0.10, eps = 0.05   |dL/dA|max = 3.0e-03   worst rel = 0.0017
#     B = 0.50, eps = 0.05   |dL/dA|max = 2.6e-02   worst rel = 0.0003
#
# So drive B well off zero before asking A for a derivative. This is a property of float32, not of
# the gradient -- but a test that ignored it would fail for a reason that has nothing to do with
# correctness, and would be "fixed" by loosening the tolerance until it hid a real bug.
DL_DA_B = 0.5
DL_DA_EPS = 5e-2


def test_dL_dA_matches_finite_differences_once_B_is_nonzero(f32_harness) -> None:
    """A's gradient is only *reachable* once B is off zero -- and then it must be right too.

    dL/dA travels a different path than dL/dB: it is the gradient of the *inner* projection, so a
    transposed or mis-scaled out_prod there would be entirely invisible to the B check above.
    """
    h, _ = f32_harness

    name = next(n for n in h.names if n.startswith("blk.0.attn_q"))

    b = h.get(name, is_b=True)
    for i in range(len(b)):
        b[i] = DL_DA_B
    h.set(name, True, b)

    snap = h.snapshot()

    failures = _fd_check(h, name, False, snap, eps=DL_DA_EPS)

    assert not failures, "dL/dA disagrees with finite differences:\n" + "\n".join(failures)


def test_dL_dA_is_zero_at_B_zero_and_becomes_nonzero_after_B_moves(f32_harness) -> None:
    """A's gradient is exactly zero while B is zero — and that is not a bug.

    A reaches the loss only through B: the LoRA delta is ``scale · B(A·x)``. With B = 0 the
    derivative w.r.t. every element of A is exactly 0, which is precisely why a zero-init adapter
    is a no-op and why B must be the thing that moves first. Getting this backwards — initializing
    B random and A zero — gives an adapter whose A never learns. The test pins the direction.
    """
    h, _ = f32_harness
    name = next(n for n in h.names if n.startswith("blk.0.attn_q"))

    h.step(train=True)
    assert all(g == 0.0 for g in h.grad(name, is_b=False)), "dL/dA must be exactly 0 while B is 0"

    b = h.get(name, is_b=True)
    for i in range(len(b)):
        b[i] = 0.01
    h.set(name, True, b)

    h.step(train=True)
    assert any(g != 0.0 for g in h.grad(name, is_b=False)), "dL/dA must be nonzero once B is not"


def test_gradients_do_not_accumulate_across_steps(f32_harness) -> None:
    """Two backward passes at the same weights must give the same gradient, not twice it.

    ggml-opt allocates its gradient accumulators once and, with **dynamic graphs**, always builds
    the backward to *add* into them (``accumulate`` is forced on whenever the graph is rebuilt each
    step). It zeroes them on the first build and then only at the start of a new accumulation
    window -- and it detected that window with ``opt_period > 1``, which is never true for us.

    So every step's gradient was added to the running sum of every previous step's, and the
    optimizer descended on that sum. The loss still falls, which is exactly what makes it quiet.
    Fixed in ggml_opt_alloc; this is the regression test.
    """
    h, _ = f32_harness
    name = next(n for n in h.names if n.startswith("blk.0.attn_q"))

    snap = h.snapshot()

    first = h.grad_at(snap, name, True)
    second = h.grad_at(snap, name, True)

    assert any(g != 0.0 for g in first), "the gradient is zero; this test would prove nothing"

    worst = max(abs(a - b) for a, b in zip(first, second, strict=True))
    scale = max(abs(g) for g in first)

    assert worst <= 1e-6 * scale, (
        f"the same backward pass at the same weights gave a different gradient the second time "
        f"(worst delta {worst:.6g}, gradient scale {scale:.6g}). The accumulators are not being "
        f"zeroed between steps."
    )


def test_the_quantized_backward_tracks_the_f32_backward(
    tiny_f32, tiny_q8_0, tiny_q4_k, tmp_path, load_model, libs: _ffi.Libraries
) -> None:
    """A quantized backward diverges from the F32 one in proportion to the quantization.

    Eight bits barely, four bits more — and in that order.

    This is the claim the P0 gate actually rests on: that training a LoRA on a **quantized** GGUF
    descends the model's real loss, and is not an artifact of the quantizer. It cannot be checked
    by finite differences (see this module's docstring — the quantized forward is a step function),
    so it is checked against the F32 gradient, which the tests above have already tied to one.

    The trend is the evidence, not any single number. Quantizing the base genuinely moves the loss
    surface, so the gradients are *not* equal and no threshold on their difference means much on
    its own. But a backward that dequantizes correctly must degrade **gracefully and monotonically**
    with bit width, because that is the only thing that changed. Measured here:

        q8_0   cosine 0.99991   0.8° from the F32 gradient   magnitude x0.998
        q4_k   cosine 0.97791  12.1° from the F32 gradient   magnitude x1.019

    A backward that dequantized the wrong way, or transposed, would not produce that ordering — it
    would be wrong at 8 bits and wrong at 4, with no reason to be three orders of magnitude tighter
    at 8. (The fixture is a *random* tiny model, which is the worst case for Q4_K: random weights
    have no structure for 4 bits to exploit. 12° is unremarkable here and would be large on a real
    model.)
    """
    grads = {}
    for tag, base in (("f32", tiny_f32), ("q8_0", tiny_q8_0), ("q4_k", tiny_q4_k)):
        # Same rank, same seed, same base tensor shapes -> byte-identical adapters. The only
        # difference between the three runs is the precision of the frozen weights.
        h, _params = _make(libs, load_model, base, tmp_path, tag)
        name = next(n for n in h.names if n.startswith("blk.0.attn_q"))

        grads[tag] = h.grad_at(h.snapshot(), name, True)
        _ffi.opt_free(libs, h.model.ctx)

    ref = grads["f32"]
    n_ref = math.sqrt(sum(x * x for x in ref))
    assert n_ref > 0.0, "the reference gradient is identically zero; this test proves nothing"

    measured = {}
    for tag in ("q8_0", "q4_k"):
        g = grads[tag]
        n_g = math.sqrt(sum(x * x for x in g))
        assert n_g > 0.0, f"the {tag} gradient is identically zero"

        dot = sum(x * y for x, y in zip(ref, g, strict=True))
        measured[tag] = (dot / (n_ref * n_g), n_g / n_ref)

    for tag, floor in (("q8_0", 0.999), ("q4_k", 0.95)):
        cosine, ratio = measured[tag]
        degrees = math.degrees(math.acos(min(1.0, cosine)))

        assert cosine >= floor, (
            f"the {tag} gradient points {degrees:.1f}° away from the F32 gradient "
            f"(cosine {cosine:.5f}, floor {floor}). Training on a quantized base is not descending "
            f"the model's loss."
        )

        # Direction is not enough: a systematically mis-scaled gradient sails through a cosine
        # check while training at silently the wrong learning rate.
        assert 0.8 <= ratio <= 1.25, f"{tag} gradient magnitude is {ratio:.3f}x the F32 one"

    # The ordering is the part that says the dequantization is *right* rather than merely close.
    assert measured["q8_0"][0] > measured["q4_k"][0], (
        f"the 8-bit gradient ({measured['q8_0'][0]:.5f}) is no closer to F32 than the 4-bit one "
        f"({measured['q4_k'][0]:.5f}). The disagreement is not tracking the quantization, so it is "
        f"not coming from the quantization."
    )


def test_a_fully_masked_batch_gives_finite_zero_gradients(q4_k_harness) -> None:
    """The masked-token path must produce a bitwise zero, not a multiplied-by-zero NaN.

    A weight of 0 means the token contributes nothing. If that were implemented as "compute the
    gradient, then scale by 0", an inf or NaN anywhere in the computation would survive as NaN —
    and one such token would poison every adapter gradient in the batch. ADR-0003 requires an exact
    zero instead, and ce_sparse delivers one because the weight multiplies the loss *before* it is
    differentiated.

    This is also what caught the S1-02 backward bug: the dense weighted-one-hot stopgap gave a
    masked row a gradient of softmax/nr rather than zero, so this test failed while the loss curve
    looked perfectly healthy.
    """
    h, _ = q4_k_harness
    name = next(n for n in h.names if n.startswith("blk.0.attn_q"))

    for i in range(h.n):
        h.wts[i] = 0.0

    loss = h.step(train=True)
    assert loss == 0.0, f"a fully masked batch must have zero loss, got {loss}"

    for is_b in (False, True):
        grads = h.grad(name, is_b=is_b)
        assert all(g == g for g in grads), "gradient went NaN on a fully masked batch"  # noqa: PLR0124
        assert all(g == 0.0 for g in grads), "a fully masked batch must give zero gradients"


def test_the_trained_adapter_loads_into_a_stock_inference_context(
    tiny_q4_k, tmp_path, load_model, libs: _ffi.Libraries
) -> None:
    """Train, save, and load the result the way the rest of the ecosystem would — the P0 milestone.

    "The trained adapter loads in stock ``llama-cli --lora``" is the last clause of the P0 exit
    criterion, and it is checked here at the library level rather than by shelling out, because
    ``llama-cli --lora`` *is* ``llama_adapter_lora_init`` + ``llama_set_adapters_lora`` — the exact
    two calls below — wrapped in an argument parser. Testing the loader directly tests the same code
    path, deterministically, with no subprocess and no second build.

    What makes it a real check is the **context it loads into**: a plain inference context, with
    none of training's concessions. Training mode bypasses the KV cache (S1-00), disables the
    repacking buffer types (so the backward's OUT_PROD is schedulable), and needs a forked step
    loop. Inference has all of that back on. If the adapter we produce only worked under our own
    training context, it would be an artifact, not a model.

    So: train it, write it out, and load it into a context that has never heard of any of this.
    The logits must move.
    """
    h, _params = _make(libs, load_model, tiny_q4_k, tmp_path, "milestone", alpha=1e-2)

    for i in range(h.n):
        h.wts[i] = 1.0

    for _ in range(16):
        h.step(train=True)

    # Read the trained weights back out and write a real adapter GGUF from them. numpy shapes are
    # the reverse of GGUF ne, so A (ne = [n_in, r]) is a numpy array of shape (r, n_in).
    pairs = []
    for t in h.targets:
        a = np.array(h.get(t.name, is_b=False), dtype=np.float32).reshape(RANK, t.n_in)
        b = np.array(h.get(t.name, is_b=True), dtype=np.float32).reshape(t.n_out, RANK)
        pairs.append((t.name, a, b))

    # Finiteness FIRST, and explicitly. `b.any()` is true for a NaN, so a "did it train?" check
    # written as `any(b.any() ...)` accepts a diverged adapter and passes -- which it did, once.
    assert all(np.isfinite(a).all() and np.isfinite(b).all() for _, a, b in pairs), (
        "training diverged: the adapter contains NaN or inf"
    )
    assert max(abs(b).max() for _, _, b in pairs) > 0.0, (
        "training left every B at zero; nothing was learned and the check below would be vacuous"
    )

    trained = tmp_path / "trained.gguf"
    _write_adapter_gguf(trained, "llama", float(RANK), pairs)

    _ffi.opt_free(libs, h.model.ctx)

    # A stock inference context: KV cache on, repacking buffer types on, mmap on, training off.
    stock = load_model(tiny_q4_k, n_ctx=64)
    prompt = [7, 11, 13, 17]

    before = stock.logits(prompt)
    stock.attach_adapter(trained, scale=1.0)
    after = stock.logits(prompt)

    assert all(x == x for x in after), "the trained adapter produced NaN logits in stock inference"  # noqa: PLR0124

    moved = max(abs(x - y) for x, y in zip(before, after, strict=True))
    assert moved > 1e-4, (
        f"stock inference loaded the trained adapter and produced identical logits (max delta "
        f"{moved:.3g}). It is being loaded but not applied, or it never actually trained."
    )


def test_loss_falls_on_a_quantized_memory_mapped_base(
    tiny_q4_k, tmp_path, load_model, libs: _ffi.Libraries
) -> None:
    """The end-to-end curve, on the configuration the project ships — the P0 gate.

    Q4_K, memory-mapped, LoRA-only. mmap stays on precisely because LoRA never writes a base
    weight: the read-only mapping is a free assertion that nothing did.
    """
    h, _params = _make(libs, load_model, tiny_q4_k, tmp_path, "curve", alpha=1e-2)

    for i in range(h.n):
        h.wts[i] = 1.0  # train on the whole sequence: this is a convergence check, not a mask one

    losses = [h.step(train=True) for _ in range(32)]

    first = sum(losses[:4]) / 4
    last = sum(losses[-4:]) / 4

    assert all(x == x for x in losses), f"loss went NaN: {losses}"  # noqa: PLR0124
    assert last < 0.8 * first, f"loss did not fall enough: {first:.4f} -> {last:.4f}"

    _ffi.opt_free(libs, h.model.ctx)
