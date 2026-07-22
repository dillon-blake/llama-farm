"""S1-09: a resumed run is an uninterrupted run — bit for bit, or it is not a resume.

There is exactly one interesting assertion in this file, and everything else supports it:

    train 8 steps                                  -> W_reference
    train 4 steps, checkpoint, restore, 4 more     -> W_resumed

    W_resumed == W_reference,  element for element, exactly.

Not "close". A correct resume is not an approximation of an uninterrupted run — it *is* one, because
every input to every step is identical. If the two differ at all, something about the optimizer's
state did not survive the round trip, and the difference will grow over a real run rather than stay
where the test found it.

The counterfactual is the other half, and it is what says the sidecar earns its existence: resuming
from the **adapter alone** — which is what you get if you treat the adapter file as a checkpoint —
does *not* reproduce the reference. AdamW's moments and its bias-correction counter are gone, so the
first steps after the restart are much larger than the ones they follow, and the run lands somewhere
else. That is the bug this ticket exists to prevent, and it is silent: the loss simply jumps at
every resume, and only on the long runs — which are the ones you cannot afford to re-do.

There is a second counterfactual, because the mistake runs both ways: restoring the **sidecar
alone** does not reproduce the reference either. The optimizer graph only exists after a step, and
that priming step moves the weights before ``restore_checkpoint`` gets a chance to say anything.
Both halves go back, in that order, or it is not a resume.
"""

import ctypes

import numpy as np
import pytest

from learning_llamas import _ffi
from learning_llamas.adapter import create_zero_adapter, enumerate_targets
from learning_llamas.checkpoint import (
    CHECKPOINT_TYPE,
    Checkpoint,
    TrainingPosition,
    read_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)
from learning_llamas.train import Batch, TrainConfig, Trainer

RANK = 4
N_CTX = 64
SEQ_LEN = 32

# Large enough that AdamW's moments actually carry history. At 1e-3 the weights barely move, the
# gradient barely changes, and a run resumed from the weights alone reproduces the reference anyway
# -- which would make the counterfactual below pass while proving nothing.
LR = 2e-2


def _batch() -> Batch:
    return Batch(
        tokens=[7, 11, 13, 17] * (SEQ_LEN // 4),
        targets=[11, 13, 17, 7] * (SEQ_LEN // 4),
        weights=[1.0] * SEQ_LEN,
    )


def _adapter_state(libs, model, targets) -> dict[str, list[float]]:
    out = {}
    for target in targets:
        for is_b in (False, True):
            n = libs.farm.ll_debug_n_elements(model.ctx, target.name.encode(), is_b)
            buf = (ctypes.c_float * n)()
            libs.farm.ll_debug_get_tensor(model.ctx, target.name.encode(), is_b, buf, n)
            out[f"{target.name}.{'b' if is_b else 'a'}"] = list(buf)
    return out


def _restore_adapter(libs, model, state: dict[str, list[float]]) -> None:
    for key, values in state.items():
        name, _, suffix = key.rpartition(".")
        buf = (ctypes.c_float * len(values))(*values)
        libs.farm.ll_debug_set_tensor(model.ctx, name.encode(), suffix == "b", buf, len(buf))


@pytest.fixture
def fresh(tiny_q4_k, tmp_path, load_model, libs: _ffi.Libraries):
    """A factory for identical, freshly-initialized trainable models."""
    targets = enumerate_targets(tiny_q4_k)
    counter = {"n": 0}

    def make():
        counter["n"] += 1
        adapter_path = tmp_path / f"a{counter['n']}.gguf"
        create_zero_adapter(tiny_q4_k, adapter_path, r=RANK, seed=7)

        model = load_model(tiny_q4_k, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True)
        model.attach_adapter(adapter_path, scale=1.0)
        model.targets = targets
        return model

    return make


def test_a_resumed_run_is_the_uninterrupted_run(fresh, tmp_path, libs: _ffi.Libraries) -> None:
    """The assertion this ticket exists for."""
    batch = _batch()
    sidecar = tmp_path / "opt.gguf"

    # Reference: eight steps, never stopped.
    model = fresh()
    with Trainer(libs, model, TrainConfig(lr=LR)) as trainer:
        for _ in range(8):
            trainer.step(batch)
        reference = _adapter_state(libs, model, model.targets)

    # Interrupted: four steps, then save BOTH halves of the state -- the adapter and the sidecar.
    model = fresh()
    with Trainer(libs, model, TrainConfig(lr=LR)) as trainer:
        for _ in range(4):
            trainer.step(batch)

        halfway_weights = _adapter_state(libs, model, model.targets)
        n_moments = save_checkpoint(
            libs, model.ctx, sidecar, position=TrainingPosition(micro_step=4)
        )

    assert n_moments == 2 * 2 * len(model.targets), n_moments  # m and v, for each A and each B

    # Resumed: a brand-new context, restored, then four more steps.
    model = fresh()
    with Trainer(libs, model, TrainConfig(lr=LR)) as trainer:
        # The moments do not exist until ggml-opt has built an optimizer graph, which happens on the
        # first training step. So: step once, then overwrite everything that step touched. Its own
        # contribution is entirely discarded, which is what makes the resume exact rather than
        # approximately-exact.
        trainer.step(batch)

        _restore_adapter(libs, model, halfway_weights)
        restore_checkpoint(libs, model.ctx, read_checkpoint(sidecar))

        for _ in range(4):
            trainer.step(batch)

        resumed = _adapter_state(libs, model, model.targets)

    assert set(resumed) == set(reference)
    for name in reference:
        assert resumed[name] == reference[name], (
            f"{name} differs between a resumed run and one that never stopped. Something about the "
            f"optimizer's state did not survive the round trip."
        )


def test_resuming_from_the_adapter_alone_does_not_reproduce_it(
    fresh, tmp_path, libs: _ffi.Libraries
) -> None:
    """The counterfactual: this is what the sidecar buys.

    Treat the adapter file as a checkpoint — restore the weights and nothing else — and the run
    lands somewhere different. AdamW's moments are back at zero and its bias correction is back at
    iteration 1, so ``1/(1 - beta^t)`` is enormous and the first steps after the restart are far
    larger than the ones they follow.

    Without this test, the one above would pass just as happily on a build where the sidecar did
    nothing at all.
    """
    batch = _batch()

    model = fresh()
    with Trainer(libs, model, TrainConfig(lr=LR)) as trainer:
        for _ in range(8):
            trainer.step(batch)
        reference = _adapter_state(libs, model, model.targets)

    model = fresh()
    with Trainer(libs, model, TrainConfig(lr=LR)) as trainer:
        for _ in range(4):
            trainer.step(batch)
        halfway_weights = _adapter_state(libs, model, model.targets)

    # The weights, and only the weights.
    model = fresh()
    with Trainer(libs, model, TrainConfig(lr=LR)) as trainer:
        _restore_adapter(libs, model, halfway_weights)
        for _ in range(4):
            trainer.step(batch)
        weights_only = _adapter_state(libs, model, model.targets)

    worst = max(
        abs(a - b)
        for name in reference
        for a, b in zip(reference[name], weights_only[name], strict=True)
    )
    scale = max(abs(x) for name in reference for x in reference[name])

    assert worst > 1e-4 * scale, (
        f"resuming from the adapter alone reproduced the reference run to within {worst:.3g}. "
        f"Either the optimizer state does not matter here — in which case the test above proves "
        f"nothing — or the learning rate is too small for the moments to carry any history."
    )


def test_restoring_the_sidecar_alone_does_not_reproduce_it(
    fresh, tmp_path, libs: _ffi.Libraries
) -> None:
    """The mirror counterfactual: the sidecar is not a checkpoint either, on its own.

    ``restore_checkpoint`` used to document the resume as "one step, restore, carry on". This is
    that recipe, run with every other advantage handed to it: the run STARTS at the step-4 weights
    (so nothing is missing but the sidecar's own contribution), takes the priming step, restores
    the moments and the iteration counter, and takes the remaining four. Four real steps against
    the reference's last four, from the right weights, with the right ``m``, ``v`` and ``iter``.

    It still lands somewhere else, and there is exactly one reason left: the priming step is a real
    AdamW update that MOVES THE WEIGHTS off step 4, and nothing in ``checkpoint.py`` moves them
    back — the restore overwrites ``m``, ``v`` and ``iter``, not the parameters. So what this
    measures is precisely the priming step's update, which is why it is set up this way rather than
    from a fresh adapter: a fresh adapter would diverge because it took 5 steps against 8, and
    would say nothing about priming at all. Make the priming step weight-neutral and this test
    *should* fail — that is the correct signal for a counterfactual, and it is the discrimination
    the fresh-adapter version did not have.

    So the two counterfactuals bracket the protocol: weights alone diverge (no moments), the
    documented public-API resume diverges (nothing puts the primed weights back). What makes it
    exact is writing the step-N weights over the priming step's, which the test at the top of this
    file does through the debug shim.

    MEASURED, tiny-llama-q4_k at LR=2e-2: worst element differs by 0.108 against a weight scale of
    1.96 — a relative 5.5e-2, i.e. 550x the 1e-4 the assertion asks for. The margin is that wide
    because the priming step is AdamW's *first*, where the bias correction ``1/(1 - beta^t)`` is at
    its largest, so it displaces the weights by roughly the full learning rate.
    """
    batch = _batch()
    sidecar = tmp_path / "opt.gguf"

    model = fresh()
    with Trainer(libs, model, TrainConfig(lr=LR)) as trainer:
        for _ in range(8):
            trainer.step(batch)
        reference = _adapter_state(libs, model, model.targets)

    model = fresh()
    with Trainer(libs, model, TrainConfig(lr=LR)) as trainer:
        for _ in range(4):
            trainer.step(batch)
        halfway_weights = _adapter_state(libs, model, model.targets)
        save_checkpoint(libs, model.ctx, sidecar, position=TrainingPosition(micro_step=4))

    # Parts 1 and 2 of the documented resume, and only those: start at the step-4 weights, prime,
    # restore the sidecar. Part 3 -- putting the weights back AFTER the priming step -- is the one
    # deliberately left out, and it is the only difference from the bit-exact run above.
    model = fresh()
    with Trainer(libs, model, TrainConfig(lr=LR)) as trainer:
        _restore_adapter(libs, model, halfway_weights)
        trainer.step(batch)
        restore_checkpoint(libs, model.ctx, read_checkpoint(sidecar))
        for _ in range(4):
            trainer.step(batch)
        moments_only = _adapter_state(libs, model, model.targets)

    worst = max(
        abs(a - b)
        for name in reference
        for a, b in zip(reference[name], moments_only[name], strict=True)
    )
    scale = max(abs(x) for name in reference for x in reference[name])

    assert worst > 1e-4 * scale, (
        f"a resume that restored the sidecar but not the primed-over weights reproduced the "
        f"reference run to within {worst:.3g}. Either the priming step no longer moves the "
        f"weights — in which case part 3 of the documented protocol is unnecessary and "
        f"restore_checkpoint's docstring is wrong — or something other than the explicit "
        f"_restore_adapter above is putting the step-4 weights back."
    )


def test_the_sidecar_is_not_an_adapter(fresh, tmp_path, libs: _ffi.Libraries) -> None:
    """Stock llama.cpp must refuse it, rather than load optimizer moments as LoRA weights."""
    sidecar = tmp_path / "opt.gguf"

    model = fresh()
    with Trainer(libs, model, TrainConfig(lr=LR)) as trainer:
        trainer.step(_batch())
        save_checkpoint(libs, model.ctx, sidecar)

    import gguf

    kind = gguf.GGUFReader(str(sidecar), "r").get_field(gguf.Keys.General.TYPE)
    assert str(kind.contents()) == CHECKPOINT_TYPE

    # And llama.cpp's own loader says no.
    assert not libs.llama.llama_adapter_lora_init(model.model, str(sidecar).encode()), (
        "stock llama.cpp loaded the optimizer sidecar as a LoRA adapter"
    )


def test_checkpointing_before_the_first_step_is_an_error(fresh, tmp_path, libs) -> None:
    """The moments do not exist yet. That is a question with no answer, not an empty checkpoint."""
    model = fresh()

    with Trainer(libs, model, TrainConfig(lr=LR)):
        with pytest.raises(RuntimeError, match="NOT_INITIALIZED"):
            save_checkpoint(libs, model.ctx, tmp_path / "x.gguf")


def test_a_checkpoint_of_another_run_is_refused(fresh, tmp_path, libs) -> None:
    """A rank-4 adapter's moments must not be poured into a rank-8 one."""
    sidecar = tmp_path / "opt.gguf"

    model = fresh()
    with Trainer(libs, model, TrainConfig(lr=LR)) as trainer:
        trainer.step(_batch())
        save_checkpoint(libs, model.ctx, sidecar)

    ckpt = read_checkpoint(sidecar)

    # Drop one moment: now the checkpoint no longer covers the adapter.
    ckpt.moments.pop(next(iter(ckpt.moments)))

    model = fresh()
    with Trainer(libs, model, TrainConfig(lr=LR)) as trainer:
        trainer.step(_batch())
        with pytest.raises(ValueError, match="different run"):
            restore_checkpoint(libs, model.ctx, ckpt)


def test_resuming_into_a_different_accumulation_period_is_refused(fresh, tmp_path, libs) -> None:
    """The moments were accumulated over windows of a different size."""
    sidecar = tmp_path / "opt.gguf"

    model = fresh()
    with Trainer(libs, model, TrainConfig(lr=LR, grad_accum=2)) as trainer:
        trainer.step(_batch())
        trainer.step(_batch())
        save_checkpoint(libs, model.ctx, sidecar, opt_period=2)

    ckpt = read_checkpoint(sidecar)
    assert ckpt.opt_period == 2

    model = fresh()
    with Trainer(libs, model, TrainConfig(lr=LR)) as trainer:
        trainer.step(_batch())
        with pytest.raises(ValueError, match="opt_period"):
            restore_checkpoint(libs, model.ctx, ckpt, opt_period=1)


def test_the_iteration_counter_round_trips(fresh, tmp_path, libs) -> None:
    """Restoring the moments without the counter corrects them for the wrong iteration."""
    sidecar = tmp_path / "opt.gguf"

    model = fresh()
    with Trainer(libs, model, TrainConfig(lr=LR)) as trainer:
        for _ in range(5):
            trainer.step(_batch())

        live = libs.farm.ll_opt_get_iter(model.ctx)
        save_checkpoint(libs, model.ctx, sidecar)

    assert live > 1, "the counter never advanced; this test would prove nothing"

    ckpt = read_checkpoint(sidecar)
    assert ckpt.iter == live

    model = fresh()
    with Trainer(libs, model, TrainConfig(lr=LR)) as trainer:
        trainer.step(_batch())
        assert libs.farm.ll_opt_get_iter(model.ctx) != live  # a fresh run is at a different count

        restore_checkpoint(libs, model.ctx, ckpt)
        assert libs.farm.ll_opt_get_iter(model.ctx) == live


def test_a_foreign_gguf_is_not_read_as_a_checkpoint(tiny_q4_k, tmp_path) -> None:
    """An adapter is not a checkpoint, and saying so beats a confusing shape error later."""
    adapter = tmp_path / "a.gguf"
    create_zero_adapter(tiny_q4_k, adapter, r=RANK, seed=1)

    with pytest.raises(ValueError, match="not a learning-llamas checkpoint"):
        read_checkpoint(adapter)


def test_moments_are_not_all_zero_after_training(fresh, tmp_path, libs) -> None:
    """If they were, every assertion above would hold for the wrong reason."""
    sidecar = tmp_path / "opt.gguf"

    model = fresh()
    with Trainer(libs, model, TrainConfig(lr=LR)) as trainer:
        for _ in range(3):
            trainer.step(_batch())
        save_checkpoint(libs, model.ctx, sidecar)

    ckpt = read_checkpoint(sidecar)

    assert any(np.abs(v).max() > 0 for v in ckpt.moments.values())
    assert all(np.isfinite(v).all() for v in ckpt.moments.values())


def test_an_unknown_checkpoint_version_is_refused(fresh, tmp_path, libs) -> None:
    """A future sidecar must be refused, not silently half-read."""
    import gguf

    sidecar = tmp_path / "opt.gguf"

    model = fresh()
    with Trainer(libs, model, TrainConfig(lr=LR)) as trainer:
        trainer.step(_batch())
        save_checkpoint(libs, model.ctx, sidecar)

    # Rewrite it with a version this build does not know.
    reader = gguf.GGUFReader(str(sidecar), "r")
    future = tmp_path / "future.gguf"

    writer = gguf.GGUFWriter(str(future), arch=CHECKPOINT_TYPE)
    writer.add_type(CHECKPOINT_TYPE)
    writer.add_uint32("farm.checkpoint.version", 999)
    for tensor in reader.tensors:
        writer.add_tensor(tensor.name, np.array(tensor.data, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    with pytest.raises(ValueError, match="version 999"):
        read_checkpoint(future)

    assert Checkpoint().iter == 1, "a fresh optimizer is at iteration 1, not 0"
