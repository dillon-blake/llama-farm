"""The public API: can somebody who only reads ``learning_llamas`` actually train a model?

Until now the answer was no. Every trainer took ``libs: _ffi.Libraries`` — a private package — and
a ``TrainableModel``, which is a Protocol whose only implementation lived in ``tests/conftest.py``
and was labelled "not a public API". The library had 249 passing tests, a working SFT/DPO/GRPO
stack, and no way for a user to load a GGUF.

So the load-bearing test in this file is the one that runs the whole quickstart using only public
names: it imports nothing private and trains an adapter end to end. If it needs a private name to
do its job, the public API is still missing something and the test says so by failing to import.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import textwrap

import pytest

import learning_llamas
from learning_llamas import (
    ChatTemplate,
    MaskedSample,
    Message,
    Model,
    build_masked_sample,
    create_zero_adapter,
    libraries,
    read_adapter,
    save_adapter,
)
from learning_llamas.train import SFTConfig, TrainableModel, train_sft

SEQ_LEN = 32
N_CTX = 64
RANK = 4

# The tiny fixture has no chat template of its own -- it is a random model, not a released one --
# so bring the simplest thing that has a real prompt/completion boundary.
TEMPLATE = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}"


# ---------------------------------------------------------------------------
# The surface itself.
# ---------------------------------------------------------------------------


def test_importing_the_package_gives_you_something_to_work_with() -> None:
    """``import learning_llamas`` used to bind exactly one name: ``__version__``."""
    assert "Model" in learning_llamas.__all__
    assert "libraries" in learning_llamas.__all__
    assert len(learning_llamas.__all__) > 1


def test_every_exported_name_actually_resolves() -> None:
    """An ``__all__`` entry that does not exist is a broken ``from learning_llamas import *``."""
    missing = [name for name in learning_llamas.__all__ if not hasattr(learning_llamas, name)]
    assert missing == []


def test_libraries_is_loaded_once_and_cached() -> None:
    """Every entry point takes ``libs``, so getting it twice must not open the libraries twice."""
    assert libraries() is libraries()


def test_a_model_satisfies_the_trainable_model_protocol(tiny_q4_k, load_model) -> None:
    """The trainers ask for three handles. The public Model is what supplies them."""
    model = load_model(tiny_q4_k, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True)

    for handle in TrainableModel.__annotations__:
        assert isinstance(getattr(model, handle), int), handle


# ---------------------------------------------------------------------------
# Lifecycle. The adapter is the interesting one -- see attach_adapter's docstring.
# ---------------------------------------------------------------------------


def test_close_is_idempotent(tiny_q4_k, libs) -> None:
    model = Model(tiny_q4_k, libs=libs, n_ctx=N_CTX)
    model.close()
    model.close()

    assert model.ctx == 0
    assert model.model == 0


def test_the_context_manager_frees_on_the_way_out(tiny_q4_k, libs) -> None:
    with Model(tiny_q4_k, libs=libs, n_ctx=N_CTX) as model:
        assert model.ctx

    assert model.ctx == 0


def test_attach_adapter_needs_a_path_or_a_loaded_adapter(tiny_q4_k, load_model) -> None:
    model = load_model(tiny_q4_k, n_ctx=N_CTX)

    with pytest.raises(ValueError, match="either a path or an already-loaded adapter"):
        model.attach_adapter()


def test_a_shared_adapter_has_exactly_one_owner(tiny_q4_k, tmp_path, load_model) -> None:
    """GRPO runs two contexts through one adapter. Only the one that loaded it may free it.

    The freeing itself is checked in a subprocess below -- a double free aborts the process, and
    an aborted process is not a test failure you can read.
    """
    adapter_path = tmp_path / "adapter.gguf"
    create_zero_adapter(tiny_q4_k, adapter_path, r=RANK, seed=7)

    owner = load_model(tiny_q4_k, n_ctx=N_CTX)
    sharer = load_model(tiny_q4_k, n_ctx=N_CTX)

    handle = owner.attach_adapter(adapter_path)
    sharer.attach_adapter(adapter=handle)

    # The same A/B tensors, not a second copy that starts out equal and then quietly diverges.
    assert sharer.adapter == handle
    assert owner._owns_adapter
    assert not sharer._owns_adapter


def test_freeing_two_models_that_share_an_adapter_does_not_double_free(tiny_q4_k, tmp_path) -> None:
    """Out of process, because the failure mode is ``free(): double free detected`` and an abort.

    In process that would take the whole suite down with it and tell you nothing about which test
    did it.
    """
    script = textwrap.dedent(f"""
        from learning_llamas import Model, create_zero_adapter, libraries

        libs = libraries()
        base = {str(tiny_q4_k)!r}
        adapter_path = {str(tmp_path / "shared.gguf")!r}
        create_zero_adapter(base, adapter_path, r={RANK}, seed=7)

        owner = Model(base, libs=libs, n_ctx={N_CTX})
        sharer = Model(base, libs=libs, n_ctx={N_CTX})

        handle = owner.attach_adapter(adapter_path)
        sharer.attach_adapter(adapter=handle)

        sharer.close()   # must NOT free the adapter: it does not own it
        owner.close()    # frees it, exactly once
        print("ok")
    """)

    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )

    assert done.returncode == 0, f"stdout={done.stdout!r} stderr={done.stderr[-2000:]!r}"
    assert "ok" in done.stdout


# ---------------------------------------------------------------------------
# llama.cpp's logs.
# ---------------------------------------------------------------------------


def test_llama_cpp_logs_arrive_through_logging_rather_than_stderr(tiny_q4_k, libs, caplog) -> None:
    """Silencing llama.cpp would throw its errors away along with its noise. Route it instead."""
    with caplog.at_level(logging.DEBUG, logger="llama.cpp"):
        Model(tiny_q4_k, libs=libs, n_ctx=N_CTX).close()

    assert [r for r in caplog.records if r.name == "llama.cpp"]


# ---------------------------------------------------------------------------
# The one that matters.
# ---------------------------------------------------------------------------


def test_the_whole_quickstart_runs_using_only_public_names(tiny_q4_k, tmp_path) -> None:
    """``docs/quickstart.md``, executed: zero adapter -> train -> save -> reload.

    Nothing in this test imports a private name. That is the assertion -- if the public API were
    missing a step, this would not be writable at all.
    """
    libs = libraries()

    base = tiny_q4_k
    adapter_path = tmp_path / "adapter.gguf"
    trained_path = tmp_path / "trained.gguf"

    create_zero_adapter(base, adapter_path, r=RANK, seed=7)

    with Model(base, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True) as model:
        model.attach_adapter(adapter_path, scale=1.0)

        template = ChatTemplate(TEMPLATE)
        samples = [
            build_masked_sample(
                [Message("user", f"q{i}"), Message("assistant", f"a{i}")],
                template,
                model.tokenizer,
            )
            for i in range(4)
        ]

        result = train_sft(
            libs,
            model,
            samples,
            SFTConfig(lr=1e-2, seq_len=SEQ_LEN, pad_id=0, epochs=3),
        )

        info = read_adapter(adapter_path)
        n_written = save_adapter(
            libs, model.adapter, trained_path, architecture=info.architecture, alpha=info.alpha
        )

    losses = [step.loss for step in result.steps]
    assert losses[-1] < losses[0], f"the loss did not fall: {losses}"

    assert n_written > 0
    assert trained_path.exists()

    # And what came out is loadable -- by us, and so by stock llama-cli, which is the point.
    with Model(base, n_ctx=N_CTX) as reloaded:
        reloaded.attach_adapter(trained_path, scale=1.0)
        assert reloaded.adapter


def test_a_trained_adapter_actually_changes_the_logits(tiny_q4_k, tmp_path) -> None:
    """A saved adapter that is a no-op would pass every check above and still be worthless."""
    base = tiny_q4_k
    adapter_path = tmp_path / "adapter.gguf"
    trained_path = tmp_path / "trained.gguf"

    create_zero_adapter(base, adapter_path, r=RANK, seed=7)

    prompt = [1, 5, 9, 13]

    with Model(base, n_ctx=N_CTX) as plain:
        before = plain.logits(prompt)

    with Model(base, n_ctx=N_CTX, n_ubatch=SEQ_LEN, training=True) as model:
        model.attach_adapter(adapter_path, scale=1.0)
        samples = [MaskedSample(tokens=[1, 5, 9, 13, 17, 21], weights=[0, 0, 0, 1, 1, 1])] * 4
        train_sft(
            libs := libraries(), model, samples, SFTConfig(lr=1e-1, seq_len=SEQ_LEN, epochs=3)
        )

        info = read_adapter(adapter_path)
        save_adapter(
            libs, model.adapter, trained_path, architecture=info.architecture, alpha=info.alpha
        )

    with Model(base, n_ctx=N_CTX) as tuned:
        tuned.attach_adapter(trained_path, scale=1.0)
        after = tuned.logits(prompt)

    assert before != after
