"""S1-04 forward-parity: summed sparse CE equals the dense ``ggml_cross_entropy_loss``.

The sparse op (ADR-0003) exists because the dense op takes an ``[n_vocab, n_tokens]`` one-hot label
MATRIX and mean-reduces over every row, which is a gigabyte of zeros at a real vocab and offers no
way to mask a token. The two must nonetheless compute the *same number* on the same inputs, or the
sparse kernel has changed what "cross-entropy" means. This pins that identity.

The ticket's acceptance line is "summed sparse loss matches the dense ``ggml_cross_entropy_loss``
within F32 round-off." Two things are checked here, not one:

* the ggml sparse op against the ggml dense op, in a single hand-built graph run on the CPU backend
  (an op-vs-op check that a shared f32 reduction quirk could, in principle, hide); and
* both of them against an independent **float64** numpy reference. That is the oracle: it is what
  makes the op-vs-op agreement mean "both are right" rather than merely "both are equal". A wrong
  sparse kernel that a wrong dense kernel happened to match would still fail the float64 leg.

The construction ties the two ops together: with per-token weights ``1/n_tokens`` and a one-hot
label matrix, ``sum_i w_i * ce_i`` (summed sparse) is exactly ``mean_i ce_i`` (dense). The last
test proves this comparison can fail, by feeding the discriminating mutation (weights of 1, whose
summed sparse loss is ``n_tokens`` times the dense mean).

This is a Python raw-graph test rather than a ``test-backend-ops`` case on purpose: that harness
compares a graph across two backends, which on a CPU-only build is CPU-vs-CPU and cannot see a
sparse-vs-dense value disagreement at all. A float64 reference can.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from learning_llamas import _ffi

GGML_TYPE_F32 = 0
GGML_TYPE_I32 = 26
GGML_BACKEND_DEVICE_TYPE_CPU = 0

# Two shapes: a tiny one, and a wide-vocab one whose logsumexp reduces over 4096 terms -- the case
# where an f32 sparse kernel and an f32 dense kernel could most plausibly drift apart.
SHAPES = (
    pytest.param(30, 5, id="30x5"),
    pytest.param(4096, 4, id="4096x4"),
)


def _dense_ce_reference(logits: np.ndarray, labels: np.ndarray) -> float:
    """The dense op's value in float64: mean over tokens of ``-log_softmax(logits)[label]``."""
    x = logits.astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    logp = x - np.log(np.exp(x).sum(axis=1, keepdims=True))
    ce = -logp[np.arange(len(labels)), labels]
    return float(ce.mean())


@pytest.mark.parametrize(("n_vocab", "n_tokens"), SHAPES)
def test_summed_sparse_ce_matches_dense_and_float64(
    n_vocab: int, n_tokens: int, libs: _ffi.Libraries
) -> None:
    ggml, base = libs.ggml, libs.ggml_base

    rng = np.random.default_rng(4)
    # ggml stores [n_vocab, n_tokens] with n_vocab as the contiguous axis, so numpy shape
    # (n_tokens, n_vocab) in C order lands exactly right when flattened.
    logits = rng.uniform(-2.0, 2.0, size=(n_tokens, n_vocab)).astype(np.float32)
    labels = rng.integers(0, n_vocab, size=n_tokens).astype(np.int32)
    weights = np.full(n_tokens, 1.0 / n_tokens, dtype=np.float32)

    onehot = np.zeros((n_tokens, n_vocab), dtype=np.float32)
    onehot[np.arange(n_tokens), labels] = 1.0

    params = _ffi.ggml_init_params(mem_size=64 * 1024 * 1024, mem_buffer=None, no_alloc=True)
    ctx = base.ggml_init(params)
    assert ctx, "ggml_init failed"
    backend = ggml.ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, None)
    assert backend, "no CPU backend"
    buf = None
    try:
        t_logits = base.ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_vocab, n_tokens)
        t_labels = base.ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens)
        t_weights = base.ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_tokens)
        t_onehot = base.ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_vocab, n_tokens)

        sparse = base.ggml_cross_entropy_loss_sparse(ctx, t_logits, t_labels, t_weights, 1.0, 0.0)
        dense = base.ggml_cross_entropy_loss(ctx, t_logits, t_onehot)

        gf = base.ggml_new_graph(ctx)
        base.ggml_build_forward_expand(gf, sparse)
        base.ggml_build_forward_expand(gf, dense)

        buf = base.ggml_backend_alloc_ctx_tensors(ctx, backend)
        assert buf, "ggml_backend_alloc_ctx_tensors failed"

        def upload(tensor: int, arr: np.ndarray) -> None:
            arr = np.ascontiguousarray(arr)
            base.ggml_backend_tensor_set(tensor, arr.ctypes.data_as(ctypes.c_void_p), 0, arr.nbytes)

        upload(t_logits, logits)
        upload(t_labels, labels)
        upload(t_weights, weights)
        upload(t_onehot, onehot)

        status = base.ggml_backend_graph_compute(backend, gf)
        assert status == 0, f"graph compute failed with status {status}"

        per_token = (ctypes.c_float * n_tokens)()
        base.ggml_backend_tensor_get(sparse, per_token, 0, ctypes.sizeof(per_token))
        summed_sparse = float(np.sum(np.array(per_token, dtype=np.float64)))

        dense_scalar = ctypes.c_float()
        base.ggml_backend_tensor_get(
            dense, ctypes.byref(dense_scalar), 0, ctypes.sizeof(dense_scalar)
        )
        dense_value = float(dense_scalar.value)
    finally:
        if buf:
            base.ggml_backend_buffer_free(buf)
        # ggml_backend_free is exported by libggml-base, so it must go through `base`. The `ggml`
        # handle resolves the symbol too (via the library dependency) but with no argtypes attached,
        # so ctypes marshals the 64-bit backend pointer as a 32-bit C int and truncates it -- a wild
        # free() on any build whose allocator hands back an address above 4 GiB (ASan, and CI).
        base.ggml_backend_free(backend)
        base.ggml_free(ctx)

    reference = _dense_ce_reference(logits, labels)

    # Both ggml ops against the float64 oracle -- "within F32 round-off". At n_vocab=4096 the f32
    # logsumexp is a reduction over 4096 terms; a few ulps per term put the observed gap ~1e-5.
    assert summed_sparse == pytest.approx(reference, rel=2e-5, abs=2e-6), (
        f"summed sparse {summed_sparse:.8f} vs float64 reference {reference:.8f}"
    )
    assert dense_value == pytest.approx(reference, rel=2e-5, abs=2e-6), (
        f"dense ggml {dense_value:.8f} vs float64 reference {reference:.8f}"
    )
    # ...and, the ticket's exact wording, the two ggml ops against each other.
    assert summed_sparse == pytest.approx(dense_value, rel=2e-5, abs=2e-6), (
        f"summed sparse {summed_sparse:.8f} != dense {dense_value:.8f} within F32 round-off"
    )


def test_the_parity_check_is_not_vacuous(libs: _ffi.Libraries) -> None:
    """Prove the parity comparison discriminates, via the mutation it must catch.

    Unit weights make the summed sparse loss ``n_tokens`` x the dense mean. If the
    reference/comparison could not tell those apart, the parity test above would pass on a sparse
    kernel that forgot to weight, so the separation is asserted rather than assumed.
    """
    ggml, base = libs.ggml, libs.ggml_base

    n_vocab, n_tokens = 30, 5
    rng = np.random.default_rng(4)
    logits = rng.uniform(-2.0, 2.0, size=(n_tokens, n_vocab)).astype(np.float32)
    labels = rng.integers(0, n_vocab, size=n_tokens).astype(np.int32)
    weights = np.ones(n_tokens, dtype=np.float32)  # the mutation: NOT 1/n_tokens

    params = _ffi.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None, no_alloc=True)
    ctx = base.ggml_init(params)
    backend = ggml.ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, None)
    buf = None
    try:
        t_logits = base.ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_vocab, n_tokens)
        t_labels = base.ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens)
        t_weights = base.ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_tokens)
        sparse = base.ggml_cross_entropy_loss_sparse(ctx, t_logits, t_labels, t_weights, 1.0, 0.0)
        gf = base.ggml_new_graph(ctx)
        base.ggml_build_forward_expand(gf, sparse)
        buf = base.ggml_backend_alloc_ctx_tensors(ctx, backend)

        for tensor, arr in ((t_logits, logits), (t_labels, labels), (t_weights, weights)):
            arr = np.ascontiguousarray(arr)
            base.ggml_backend_tensor_set(tensor, arr.ctypes.data_as(ctypes.c_void_p), 0, arr.nbytes)
        assert base.ggml_backend_graph_compute(backend, gf) == 0

        per_token = (ctypes.c_float * n_tokens)()
        base.ggml_backend_tensor_get(sparse, per_token, 0, ctypes.sizeof(per_token))
        summed = float(np.sum(np.array(per_token, dtype=np.float64)))
    finally:
        if buf:
            base.ggml_backend_buffer_free(buf)
        # ggml_backend_free is exported by libggml-base, so it must go through `base`. The `ggml`
        # handle resolves the symbol too (via the library dependency) but with no argtypes attached,
        # so ctypes marshals the 64-bit backend pointer as a 32-bit C int and truncates it -- a wild
        # free() on any build whose allocator hands back an address above 4 GiB (ASan, and CI).
        base.ggml_backend_free(backend)
        base.ggml_free(ctx)

    reference_mean = _dense_ce_reference(logits, labels)
    # Unit weights make the summed sparse loss n_tokens times the dense mean -- far outside the
    # 2e-5 band the parity test asserts. The comparison discriminates.
    assert summed == pytest.approx(n_tokens * reference_mean, rel=1e-5)
    assert abs(summed - reference_mean) > 1.0, (
        "unit-weighted summed sparse loss is indistinguishable from the dense mean; the parity "
        "test could not catch an unweighted kernel"
    )
