"""S1-42: the preflight is wired into the training path, and it actually stops a bad run.

S1-11 built the walker and the report; the 2026-07-15 audit found nothing ever *called* them before
training — the gate was decorative, and a user could train an unsupported architecture and hit a raw
``GGML_ABORT`` inside ``ggml_compute_backward`` instead of the clear verdict. These tests pin the
wiring: a blocked model must refuse to start training with a legible error, the escape hatch must
let someone override that, and a real trainable model must sail through untouched.

The blocked case is driven by substituting the walk's verdict rather than by finding an untrainable
architecture. That is deliberate: every fixture in this repo is trainable (the rest of the suite
trains them), and a test that depended on a currently-untrainable arch would stop testing the wiring
the moment that arch's backward landed — the same trap ``test_preflight`` calls out for the walker.
What is under test here is the *seam* — does the Trainer consult the report and act on it — so the
report is what we control.
"""

from __future__ import annotations

import pytest

from learning_llamas import _ffi
from learning_llamas.adapter import create_zero_adapter
from learning_llamas.preflight import Finding, PreflightError, Report, Status
from learning_llamas.train import Batch, TrainConfig, Trainer

RANK = 4
N_CTX = 64
SEQ_LEN = 32

# What the Trainer's gate calls; patched to a fixed verdict so the seam is what is under test.
_GATE = "learning_llamas.train.loop.preflight_adapter"


@pytest.fixture
def model(tiny_q4_k, tmp_path, load_model, libs: _ffi.Libraries):
    """A real, trainable model with a zero-init adapter attached — the same setup training uses."""
    adapter_path = tmp_path / "adapter.gguf"
    create_zero_adapter(tiny_q4_k, adapter_path, r=RANK, seed=7)

    m = load_model(tiny_q4_k, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True)
    m.attach_adapter(adapter_path, scale=1.0)
    return m


def _blocked_report() -> Report:
    """The report the walker produces for an un-differentiable op on the gradient path."""
    return Report(
        findings=(
            Finding(
                node="blk.0.ffn_gate",
                op="TANH",
                status=Status.BLOCKED,
                detail="ggml_compute_backward has no rule for this op; the backward would abort.",
            ),
        ),
        n_blocked=1,
    )


def _a_batch() -> Batch:
    """One fixed-shape batch of the training length, with real loss on it."""
    return Batch(tokens=[7] * SEQ_LEN, targets=[11] * SEQ_LEN, weights=[1.0] * SEQ_LEN)


def test_a_blocked_model_refuses_to_start_training(monkeypatch, model, libs) -> None:
    """The headline: a blocker turns into a clean refusal at construction, not an abort mid-run."""
    monkeypatch.setattr(_GATE, lambda *_: _blocked_report())

    with pytest.raises(PreflightError) as excinfo:
        Trainer(libs, model, TrainConfig(lr=1e-4))

    message = str(excinfo.value)
    assert "TANH" in message, f"the error must name the offending op; got: {message}"
    assert "blk.0.ffn_gate" in message, "the error should name the offending node"
    assert "NOT trainable" in message
    # And the escape hatch is discoverable from the message itself.
    assert "preflight=False" in message

    # The report rode along on the exception, for a caller that wants to inspect it.
    assert excinfo.value.report.n_blocked == 1
    assert excinfo.value.report.blockers[0].op == "TANH"


def test_the_escape_hatch_trains_a_blocked_model_anyway(monkeypatch, model, libs) -> None:
    """preflight=False skips the gate — the run proceeds, aborts and all, on the user's head."""
    monkeypatch.setattr(_GATE, lambda *_: _blocked_report())

    # Constructs without raising, and takes a real step: the gate was the only thing standing in the
    # way, and turning it off got out of the way completely.
    with Trainer(libs, model, TrainConfig(lr=1e-4, preflight=False)) as trainer:
        metrics = trainer.step(_a_batch())

    assert metrics.micro_step == 0
    assert metrics.n_valid == SEQ_LEN


def test_a_refused_construction_leaves_the_context_reusable(monkeypatch, model, libs) -> None:
    """A refusal leaves no optimizer state behind, so the context still trains afterwards.

    The gate runs before the training optimizer state is even created (and the real walk it calls,
    ``preflight_adapter``, tears down its own throwaway state in a ``finally``), so a blocker
    unwinds to nothing left registered — provable by standing a real trainer up on the context next.
    """
    monkeypatch.setattr(_GATE, lambda *_: _blocked_report())
    with pytest.raises(PreflightError):
        Trainer(libs, model, TrainConfig(lr=1e-4))

    assert int(model.ctx) not in _ffi.farm._LIVE_PARAMS, "optimizer params were left registered"

    # And the context genuinely still trains: undo the fake so the real gate runs, and the model —
    # which is fine — initializes and steps.
    monkeypatch.undo()
    with Trainer(libs, model, TrainConfig(lr=1e-4)) as trainer:
        trainer.step(_a_batch())


def test_the_gate_passes_a_real_trainable_model(model, libs: _ffi.Libraries) -> None:
    """No monkeypatch: the real walk runs at construction and lets a real model through.

    This is what stops the gate from being a wall — the two-tier model (S1-11 D5) is that anything
    passing preflight trains, and the fixture had better pass.
    """
    with Trainer(libs, model, TrainConfig(lr=1e-4)) as trainer:
        metrics = trainer.step(_a_batch())

    assert metrics.n_valid == SEQ_LEN


def test_model_preflight_reports_the_fixture_trainable(model) -> None:
    """The Python surface (S1-11 item 7): ``Model.preflight()`` returns the structured report.

    Standalone — it stands up and tears down its own optimizer state — so it answers the question
    before any Trainer exists.
    """
    report = model.preflight()

    assert report.trainable, report.summary()
    assert report.n_blocked == 0
    assert "trainable" in report.summary()


def test_model_preflight_needs_an_adapter(tiny_q4_k, load_model) -> None:
    """Without an adapter there are no trainable tensors to seed the walk, and it says so."""
    bare = load_model(tiny_q4_k, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True)

    with pytest.raises(RuntimeError, match="attach an adapter"):
        bare.preflight()
