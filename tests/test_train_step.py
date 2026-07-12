"""S1-02: a training step whose loss can be masked — the thing the stock path cannot do.

``llama_opt_epoch`` hardcodes an unmasked cross-entropy over dense one-hot labels. That is not a
missing convenience; it is the reason it cannot train an instruction-tuned model at all, because
there the prompt tokens must contribute **nothing** to the loss.

``ll_train_step`` computes

    L = -Σ wᵢ · log_softmax(logitsᵢ)[targetᵢ] / Σ wᵢ

so ``wᵢ = 0`` masks a token out exactly, and the denominator counts only the tokens that count.
"""

import ctypes

import pytest

from learning_llamas import _ffi
from learning_llamas.adapter import create_zero_adapter

N_CTX = 64
N_UBATCH = 32
RANK = 8


def _arrays(tokens: list[int], targets: list[int], weights: list[float]):
    n = len(tokens)
    return (
        (ctypes.c_int32 * n)(*tokens),
        (ctypes.c_int32 * n)(*targets),
        (ctypes.c_float * n)(*weights),
    )


def _step(libs, model, tokens, targets, weights, train=True) -> float:
    tok, tgt, wts = _arrays(tokens, targets, weights)
    loss = ctypes.c_float()
    _ffi.check(
        libs.farm.ll_train_step(model.ctx, tok, tgt, wts, len(tokens), train, ctypes.byref(loss)),
        "ll_train_step",
    )
    return loss.value


@pytest.fixture
def trainer(tiny_f32, tmp_path, load_model, libs: _ffi.Libraries):
    """A LoRA-initialized context ready to take training steps."""
    adapter_path = tmp_path / "adapter.gguf"
    create_zero_adapter(tiny_f32, adapter_path, r=RANK)

    model = load_model(tiny_f32, n_ctx=N_CTX, n_ubatch=N_UBATCH, training=True)
    model.attach_adapter(adapter_path, scale=1.0)

    params = _ffi.ll_opt_params(alpha=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0)
    adapters = (ctypes.c_void_p * 1)(model.adapter)
    _ffi.check(
        libs.farm.ll_opt_init_lora(model.ctx, model.model, adapters, 1, ctypes.byref(params)),
        "ll_opt_init_lora",
    )

    yield model, params

    libs.farm.ll_opt_free(model.ctx)


def test_a_step_runs_and_the_loss_falls(trainer, libs: _ffi.Libraries) -> None:
    """The forked loop trains: repeated steps on one batch drive the loss down."""
    model, _ = trainer
    n = 32
    tokens = [7, 11, 13, 17] * (n // 4)
    targets = tokens[1:] + [tokens[0]]
    weights = [1.0] * n

    losses = [_step(libs, model, tokens, targets, weights) for _ in range(8)]

    assert all(x == x for x in losses), f"loss went NaN: {losses}"  # noqa: PLR0124
    assert losses[-1] < losses[0], f"loss did not fall: {losses}"


def test_zero_weights_give_zero_loss(trainer, libs: _ffi.Libraries) -> None:
    """A fully masked batch contributes nothing. Not "nearly nothing" — nothing."""
    model, _ = trainer
    n = 32
    tokens = [7, 11, 13, 17] * (n // 4)
    targets = tokens[1:] + [tokens[0]]

    loss = _step(libs, model, tokens, targets, [0.0] * n, train=False)

    assert loss == 0.0


def test_masking_changes_the_loss_to_the_masked_mean(trainer, libs: _ffi.Libraries) -> None:
    """Masking is real, and it is a *mean over unmasked tokens* — not a scaled full mean.

    Take two token groups with genuinely different losses. Masking one out must give the *other
    group's own* mean, not some blend of the two. A naive implementation that merely zeroes the
    masked terms but still divides by the total token count would fail this: it would report the
    unmasked group's loss diluted by the zeros.
    """
    model, _ = trainer
    n = 32
    half = n // 2

    # First half: a repeating pattern. Second half: a different one.
    tokens = [7, 11] * (half // 2) + [101, 103] * (half // 2)
    targets = tokens[1:] + [tokens[0]]

    loss_first_only = _step(libs, model, tokens, targets, [1.0] * half + [0.0] * half, train=False)
    loss_second_only = _step(libs, model, tokens, targets, [0.0] * half + [1.0] * half, train=False)
    loss_all = _step(libs, model, tokens, targets, [1.0] * n, train=False)

    # The two halves genuinely differ, so masking is doing something.
    assert loss_first_only != pytest.approx(loss_second_only, rel=1e-3)

    # And the full loss is the mean of the two halves' means — which is only true if each masked
    # loss was normalized by ITS OWN token count.
    assert loss_all == pytest.approx((loss_first_only + loss_second_only) / 2, rel=1e-3)


def test_only_unmasked_tokens_drive_the_gradient(trainer, libs: _ffi.Libraries) -> None:
    """A masked token must not move the weights — the whole point of prompt masking.

    Train on a batch where the *second* half is masked out, then evaluate. If masked tokens were
    leaking into the gradient, the loss on the masked half would fall too. It must not.
    """
    model, _ = trainer
    n = 32
    half = n // 2

    tokens = [7, 11] * (half // 2) + [101, 103] * (half // 2)
    targets = tokens[1:] + [tokens[0]]

    train_mask = [1.0] * half + [0.0] * half
    eval_first = [1.0] * half + [0.0] * half
    eval_second = [0.0] * half + [1.0] * half

    before_first = _step(libs, model, tokens, targets, eval_first, train=False)
    before_second = _step(libs, model, tokens, targets, eval_second, train=False)

    for _ in range(10):
        _step(libs, model, tokens, targets, train_mask, train=True)

    after_first = _step(libs, model, tokens, targets, eval_first, train=False)
    after_second = _step(libs, model, tokens, targets, eval_second, train=False)

    # The half we trained on improved.
    assert after_first < before_first, (before_first, after_first)

    # The masked half did not improve *because of masking*. It may drift a little — the two
    # halves share the same adapter weights — but it must not improve like the trained half did.
    improvement_trained = before_first - after_first
    improvement_masked = before_second - after_second
    assert improvement_masked < improvement_trained, (
        f"the masked half improved by {improvement_masked}, nearly as much as the trained half "
        f"({improvement_trained}) — masked tokens are leaking into the gradient"
    )


def test_eval_does_not_change_the_weights(trainer, libs: _ffi.Libraries) -> None:
    """train=False is a pure forward pass: repeated evals return an identical loss."""
    model, _ = trainer
    n = 32
    tokens = [7, 11, 13, 17] * (n // 4)
    targets = tokens[1:] + [tokens[0]]
    weights = [1.0] * n

    losses = [_step(libs, model, tokens, targets, weights, train=False) for _ in range(3)]

    assert losses[0] == losses[1] == losses[2]


def test_step_before_init_is_rejected(tiny_f32, load_model, libs: _ffi.Libraries) -> None:
    model = load_model(tiny_f32, n_ctx=N_CTX, n_ubatch=N_UBATCH, training=True)
    tok, tgt, wts = _arrays([7, 11], [11, 7], [1.0, 1.0])

    result = libs.farm.ll_train_step(model.ctx, tok, tgt, wts, 2, True, None)
    assert result == _ffi.LLError.NOT_INITIALIZED


def test_null_arguments_are_rejected(trainer, libs: _ffi.Libraries) -> None:
    model, _ = trainer
    tok, tgt, wts = _arrays([7, 11], [11, 7], [1.0, 1.0])

    assert libs.farm.ll_train_step(None, tok, tgt, wts, 2, True, None) == _ffi.LLError.INVALID_ARG
    assert (
        libs.farm.ll_train_step(model.ctx, tok, tgt, wts, 0, True, None) == _ffi.LLError.INVALID_ARG
    )
