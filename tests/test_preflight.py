"""S1-11: will this model train, and if not, what exactly stops it?

Two claims, and the second is the one that is easy to get wrong.

**A blocked op on the gradient path must be reported.** Otherwise the user finds out when ggml
aborts inside ``ggml_compute_backward`` naming an op enum and nothing else — after the model has
loaded, the data has tokenized, and they have waited.

**A blocked op OFF the gradient path must NOT be reported.** A model is full of ops with no backward
rule that the backward pass never touches. A preflight that flagged them would be noise, and noise
trains people to ignore the report — at which point it is worse than not having one, because now
there is a green check next to a model that does not train.

Both are tested against a **synthetic graph**, fed straight to the walker. That is why the walker is
a pure function over ``(graph, params)`` rather than a method on a model: testing it against a real
model would mean finding one whose architecture is currently untrainable, and the test would then
stop testing anything the moment that op was implemented.
"""

import ctypes

import numpy as np
import pytest

from learning_llamas import _ffi
from learning_llamas.adapter import create_zero_adapter, enumerate_targets
from learning_llamas.preflight import MAX_ENTRIES, Status, ll_preflight_entry, preflight
from learning_llamas.train import TrainConfig, Trainer

RANK = 4
N_CTX = 64
SEQ_LEN = 32

GGML_TYPE_F32 = 0


class Graph:
    """A hand-built ggml graph, for feeding the walker things a real model would not give it."""

    def __init__(self, libs: _ffi.Libraries) -> None:
        self.libs = libs
        params = _ffi.ggml_init_params(
            mem_size=64 * 1024 * 1024,
            mem_buffer=None,
            no_alloc=True,  # metadata only: the walker reads the graph, it never runs it
        )
        self.ctx = libs.ggml_base.ggml_init(params)
        assert self.ctx

    def tensor(self, name: str, ne0: int = 4, ne1: int = 4) -> int:
        t = self.libs.ggml_base.ggml_new_tensor_2d(self.ctx, GGML_TYPE_F32, ne0, ne1)
        self.libs.ggml_base.ggml_set_name(t, name.encode())
        return t

    def param(self, name: str, ne0: int = 4, ne1: int = 4) -> int:
        t = self.tensor(name, ne0, ne1)
        self.libs.ggml_base.ggml_set_param(t)
        return t

    def mul_mat(self, a: int, b: int, name: str) -> int:
        t = self.libs.ggml_base.ggml_mul_mat(self.ctx, a, b)
        self.libs.ggml_base.ggml_set_name(t, name.encode())
        return t

    def undifferentiable(self, a: int, name: str) -> int:
        """PAD has no backward rule — which is exactly why it is here.

        CONCAT used to play this part, and then S1-29 gave it a backward, at which point these tests
        started failing. That is precisely what should happen: the preflight was telling the truth,
        and the test that said otherwise was the one that was wrong.
        """
        t = self.libs.ggml_base.ggml_pad(self.ctx, a, 1, 0, 0, 0)
        self.libs.ggml_base.ggml_set_name(t, name.encode())
        return t

    def build(self, output: int) -> int:
        gf = self.libs.ggml_base.ggml_new_graph(self.ctx)
        self.libs.ggml_base.ggml_build_forward_expand(gf, output)
        return gf

    def walk(self, gf: int, params: list[int]):
        entries = (ll_preflight_entry * MAX_ENTRIES)()
        n_blocked = ctypes.c_int32()

        n = self.libs.farm.ll_preflight_walk(
            gf,
            (ctypes.c_void_p * len(params))(*params),
            len(params),
            entries,
            MAX_ENTRIES,
            ctypes.byref(n_blocked),
        )
        assert n >= 0, f"ll_preflight_walk returned {n}"

        return [
            (
                entries[i].node.decode(),
                entries[i].op.decode(),
                Status(entries[i].status),
                entries[i].detail.decode(),
            )
            for i in range(n)
        ], n_blocked.value

    def close(self) -> None:
        self.libs.ggml_base.ggml_free(self.ctx)


@pytest.fixture
def graph(libs: _ffi.Libraries):
    g = Graph(libs)
    yield g
    g.close()


# ---------------------------------------------------------------------------
# The walker, on graphs a real model would not hand it.
# ---------------------------------------------------------------------------


def test_a_differentiable_graph_is_clean(graph: Graph) -> None:
    """MUL_MAT has a backward rule, so nothing is blocked."""
    w = graph.param("w")
    x = graph.tensor("x")

    findings, n_blocked = graph.walk(graph.build(graph.mul_mat(w, x, "y")), [w])

    assert findings == []
    assert n_blocked == 0


def test_an_undifferentiable_op_on_the_gradient_path_is_blocked(graph: Graph) -> None:
    """PAD has no backward rule, and it is downstream of the parameter — so it blocks."""
    w = graph.param("w")
    x = graph.tensor("x")

    y = graph.mul_mat(w, x, "y")  # on the gradient path
    z = graph.undifferentiable(y, "the_problem")  # ...and so is this

    findings, n_blocked = graph.walk(graph.build(z), [w])

    assert n_blocked == 1
    assert len(findings) == 1

    node, op, status, detail = findings[0]
    assert node == "the_problem"
    assert op == "PAD"
    assert status is Status.BLOCKED
    assert "no rule" in detail, f"the message must say what is wrong, got: {detail}"


def test_an_undifferentiable_op_OFF_the_gradient_path_is_not_reported(graph: Graph) -> None:
    """The claim that makes this a check rather than a lint.

    The same PAD, on a tensor the parameter never reaches. The backward pass will never touch it,
    so it cannot block anything — and a preflight that reported it would be crying wolf about every
    ARGSORT and ARGMAX in the model, which is how a report gets ignored.
    """
    w = graph.param("w")
    x = graph.tensor("x")

    y = graph.mul_mat(w, x, "y")  # this IS on the gradient path

    a = graph.tensor("a")
    unreachable = graph.undifferentiable(a, "not_my_problem")  # ...this is not

    # Both are in the graph. Only one of them matters.
    gf = graph.build(y)
    graph.libs.ggml_base.ggml_build_forward_expand(gf, unreachable)

    findings, n_blocked = graph.walk(gf, [w])

    assert n_blocked == 0, f"a node off the gradient path was reported: {findings}"
    assert findings == []


def test_the_gradient_path_propagates_through_several_hops(graph: Graph) -> None:
    """`needs a gradient` is transitive, exactly as ggml_build_backward_expand makes it."""
    w = graph.param("w")

    h = graph.mul_mat(w, graph.tensor("x1"), "h1")
    h = graph.mul_mat(h, graph.tensor("x2"), "h2")
    h = graph.mul_mat(h, graph.tensor("x3"), "h3")

    blocked = graph.undifferentiable(h, "four_hops_downstream")

    findings, n_blocked = graph.walk(graph.build(blocked), [w])

    assert n_blocked == 1
    assert findings[0][0] == "four_hops_downstream"


def test_a_graph_with_no_parameters_blocks_nothing(graph: Graph) -> None:
    """With nothing to differentiate, nothing is on the gradient path."""
    a = graph.tensor("a")

    findings, n_blocked = graph.walk(graph.build(graph.undifferentiable(a, "c")), [])

    assert n_blocked == 0
    assert findings == []


# ---------------------------------------------------------------------------
# ...and on a real model.
# ---------------------------------------------------------------------------


@pytest.fixture
def trainable(tiny_q4_k, tmp_path, load_model, libs: _ffi.Libraries):
    adapter_path = tmp_path / "adapter.gguf"
    create_zero_adapter(tiny_q4_k, adapter_path, r=RANK, seed=7)

    model = load_model(tiny_q4_k, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True)
    model.attach_adapter(adapter_path, scale=1.0)

    return model


def test_the_llama_fixture_is_reported_as_trainable(trainable, libs: _ffi.Libraries) -> None:
    """It had better be — the rest of the suite trains it.

    Which is also what makes this worth asserting: if the preflight ever said a model was
    untrainable while the tests were happily training it, the preflight would be the thing that is
    wrong, and it would be wrong in the direction that stops people using a model that works.
    """
    model = trainable

    with Trainer(libs, model, TrainConfig(lr=1e-4)):
        report = preflight(libs, model.ctx, [7, 11, 13, 17] * 8)

    assert report.trainable, report.summary()
    assert report.n_blocked == 0
    assert not report.warnings, report.summary()
    assert "trainable" in report.summary()


def test_preflight_needs_a_prepared_context(trainable, libs: _ffi.Libraries) -> None:
    """The walk is seeded from the trainable tensors, so there has to be a set of them."""
    with pytest.raises(RuntimeError, match="NOT_INITIALIZED"):
        preflight(libs, trainable.ctx, [7, 11, 13, 17])


# ---------------------------------------------------------------------------
# ...and that the walk does not cost the caller the context it walked.
# ---------------------------------------------------------------------------


def _step(libs, model, n: int, *, train: bool) -> float:
    tokens = [7, 11, 13, 17] * (n // 4)
    targets = tokens[1:] + [tokens[0]]

    loss = ctypes.c_float()
    _ffi.check(
        libs.farm.ll_train_step(
            model.ctx,
            (ctypes.c_int32 * n)(*tokens),
            (ctypes.c_int32 * n)(*targets),
            (ctypes.c_float * n)(*([1.0] * n)),
            None,
            None,
            n,
            train,
            ctypes.byref(loss),
        ),
        "ll_train_step",
    )
    return float(loss.value)


def _first_lora_grad(libs, model, tiny_q4_k) -> list[float]:
    """The gradient of the first adapted tensor's B, straight out of ggml-opt's accumulator."""
    name = enumerate_targets(tiny_q4_k)[0].name.encode()

    n = libs.farm.ll_debug_n_elements(model.ctx, name, True)
    assert n > 0, f"ll_debug_n_elements returned {n}"

    buf = (ctypes.c_float * n)()
    assert libs.farm.ll_debug_grad(model.ctx, name, True, buf, n) == n

    return list(buf)


def _prepared(libs, tiny_q4_k, tmp_path, load_model, tag: str):
    adapter_path = tmp_path / f"{tag}.gguf"
    create_zero_adapter(tiny_q4_k, adapter_path, r=RANK, seed=7)

    model = load_model(tiny_q4_k, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True)
    model.attach_adapter(adapter_path, scale=1.0)

    params = _ffi.ll_opt_params(alpha=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0)
    _ffi.opt_init_lora(libs, model.ctx, model.model, [model.adapter], params)

    return model, params


def test_a_preflight_does_not_disturb_the_training_state_it_walked(
    tiny_q4_k, tmp_path, load_model, libs: _ffi.Libraries
) -> None:
    """``init -> preflight -> train`` is the order farm_api.h documents, so it has to be safe.

    It was not. The walk used to run its forward pass on the SHARED training ggml-opt context, and
    a llama.cpp context uses dynamic ggml-opt graphs — so ``ggml_opt_alloc`` re-runs
    ``ggml_opt_build`` every call, and the early return that would stop a ``train=false`` build
    after the forward pass is gated on ``build_type_alloc``, not on this call's ``build_type``.
    With a fresh OPT context that gate is open: the preflight's graph sized ``grad_accs`` for good,
    and it is one loss node shorter than a training step's. The next real step then indexed that
    array past its end inside ``ggml_build_backward_expand`` and used whatever it found there as
    the loss's gradient accumulator.

    That entry is a wild pointer, so the failure without the fix is an abort or a segfault rather
    than a clean assertion — and, being heap-contents-dependent, it is not even guaranteed to be
    either: a zero there falls into ggml's "allocate one for the loss" branch and the run survives.
    So this test fails by dying, on the runs where it fails at all. What it can also catch, and
    asserts, is the quieter half: that the step which follows a preflight computes the same numbers
    as one that never saw a preflight.

    Not bitwise. The two runs are separate contexts, and a preflight sizes the shared backend
    scheduler's compute buffer with a graph of its own — an allocation shift, which moves SIMD
    reduction splits by an ulp on macos-14 arm64 (ADR-0002; measured at ~1e-7 relative in
    ``test_grad_clip``, twice, by S1-49). 1e-6 sits above that residue and four orders of magnitude
    below anything a corrupted accumulator would do.
    """
    baseline, _keep_baseline = _prepared(libs, tiny_q4_k, tmp_path, load_model, "baseline")
    walked, _keep_walked = _prepared(libs, tiny_q4_k, tmp_path, load_model, "walked")

    try:
        # The one difference between the two runs.
        report = preflight(libs, walked.ctx, [7, 11, 13, 17] * (SEQ_LEN // 4))
        assert report.trainable, report.summary()

        loss_baseline = _step(libs, baseline, SEQ_LEN, train=True)
        loss_walked = _step(libs, walked, SEQ_LEN, train=True)

        assert loss_baseline > 0.0, "the baseline step has no loss; nothing is being compared"
        assert loss_walked == pytest.approx(loss_baseline, rel=1e-6), (
            f"a preflight changed the loss of the step that followed it: {loss_walked} vs "
            f"{loss_baseline}"
        )

        grad_walked = _first_lora_grad(libs, walked, tiny_q4_k)
        grad_baseline = _first_lora_grad(libs, baseline, tiny_q4_k)

        assert any(g != 0.0 for g in grad_baseline), (
            "the baseline gradient is all zeros; the comparison below is vacuous"
        )
        assert np.allclose(grad_walked, grad_baseline, rtol=1e-6, atol=1e-9), (
            "a preflight changed the gradients of the step that followed it"
        )
    finally:
        _ffi.opt_free(libs, walked.ctx)
        _ffi.opt_free(libs, baseline.ctx)


def test_a_preflight_can_be_repeated_and_still_leaves_the_state_alone(
    tiny_q4_k, tmp_path, load_model, libs: _ffi.Libraries
) -> None:
    """The walk is idempotent, and the second one is the one that used to matter.

    Every call builds its own throwaway forward-only ggml-opt context and frees it, so there is no
    accumulating state for a second walk to trip over — and, before this was true, a walk run
    between two training steps behaved differently from one run before the first (the shared
    context's static buffer was already allocated by then, which took a different early return).
    Calling it in both positions pins that the answer no longer depends on where it lands.
    """
    model, _keep = _prepared(libs, tiny_q4_k, tmp_path, load_model, "repeat")

    try:
        tokens = [7, 11, 13, 17] * (SEQ_LEN // 4)

        before = preflight(libs, model.ctx, tokens)
        _step(libs, model, SEQ_LEN, train=True)
        between = preflight(libs, model.ctx, tokens)
        _step(libs, model, SEQ_LEN, train=True)
        after = preflight(libs, model.ctx, tokens)

        assert before.n_blocked == between.n_blocked == after.n_blocked == 0
        assert before.findings == between.findings == after.findings
    finally:
        _ffi.opt_free(libs, model.ctx)


def test_the_preflight_reports_instead_of_aborting_when_the_backward_could_not_be_built(
    tiny_q4_k, tmp_path, load_model, libs: _ffi.Libraries
) -> None:
    """The preflight must never take the abort it exists to replace — and it used to.

    Sharing the training optimizer context meant ``ggml_opt_build`` ran to completion even at
    ``train=false``, so the walk reached ``ggml_build_backward_expand``. On the models this function
    is *for* — one with a non-differentiable op on the gradient path — that ends in
    ``GGML_ABORT("unsupported ggml op for backward pass")``: the preflight died holding the report
    it had just computed. No fixture here has such an architecture (the suite trains all of them),
    and one that did would stop testing this the moment that op got a backward, which is the trap
    this module's docstring calls out.

    So the same failure is reached the other way, through the *other* assertion in that same
    function: an adapter that was flagged but never ATTACHED puts no PARAM node in the forward
    graph, and ``ggml_build_backward_expand`` opens with
    ``GGML_ASSERT(any_params && "no trainable parameters found ...")``. Building a backward at all
    is therefore fatal here, and not building one is the whole fix — with it, the walk gets to
    return what it found, which is the WARN that says every adapter tensor is in no graph node.
    (That is also the detection path ``farm_api.h`` now points at for an unattached adapter, since
    ``ll_opt_init_lora`` cannot check attachment itself.)

    Without the fix this aborts the process rather than failing an assertion.
    """
    adapter_path = tmp_path / "unattached.gguf"
    create_zero_adapter(tiny_q4_k, adapter_path, r=RANK, seed=7)

    model = load_model(tiny_q4_k, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True)

    # Loaded, flagged — and deliberately never handed to llama_set_adapters_lora.
    adapter = libs.llama.llama_adapter_lora_init(model.model, str(adapter_path).encode())
    assert adapter, "the adapter failed to load"

    try:
        params = _ffi.ll_opt_params(alpha=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0)
        n_flagged = _ffi.opt_init_lora(libs, model.ctx, model.model, [adapter], params)
        assert n_flagged > 0

        try:
            report = preflight(libs, model.ctx, [7, 11, 13, 17] * (SEQ_LEN // 4))
        finally:
            _ffi.opt_free(libs, model.ctx)

        assert report.n_blocked == 0, report.summary()
        assert len(report.warnings) == n_flagged, (
            f"every flagged tensor is outside the graph, so every one should warn: got "
            f"{len(report.warnings)} of {n_flagged}\n{report.summary()}"
        )
        assert all(f.status is Status.WARN for f in report.findings)
        assert "no graph node" in report.warnings[0].detail
    finally:
        libs.llama.llama_adapter_lora_free(adapter)
