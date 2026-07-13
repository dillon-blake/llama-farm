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

import pytest

from learning_llamas import _ffi
from learning_llamas.adapter import create_zero_adapter
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
