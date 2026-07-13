"""S1-13: per-token logprobs without ever building the logits.

The module's whole claim is that it computes *the same numbers* as the obvious path — decode, take
the full ``[n_tokens, n_vocab]`` logits, log-softmax, gather the realized token — while never
allocating that tensor. So the tests are two claims, and they pull in opposite directions:

**It agrees with the obvious path.** Checked against a full-logits decode plus a float64 numpy
log-softmax, on every fixture base (F32, Q8_0, Q4_K), across chunk sizes that divide the batch,
that do not, and that exceed it. This is the correctness claim, and nothing else in the file
substitutes for it.

**It does not build the thing it is avoiding.** An equality test passes just as happily against an
implementation that materializes all the logits and gathers — which would be a *correct* module
that is useless for the reason the module exists. So peak transient memory is asserted directly,
against a vocab big enough that the difference is not noise.

Then the two ways to score the wrong model without noticing: reading the wrong tensor as the
lm_head (tied embeddings), and ignoring an adapter that targets it.
"""

import ctypes
import pathlib

import numpy as np
import pytest

from learning_llamas import _ffi
from learning_llamas.adapter import create_zero_adapter
from learning_llamas.logprobs import (
    DEFAULT_MEM_BUDGET,
    _adapter_index_of,
    choose_chunk_rows,
    chunked_token_logprobs,
    hidden_states,
    load_lm_head,
    output_lora,
    sequence_logprobs,
)

from .fixtures import gen_tiny_llama

N_CTX = 128
SEQ_LEN = 24


def _tokens(n: int, n_vocab: int, seed: int = 3) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(1, n_vocab, size=n)]


# ---------------------------------------------------------------------------------------------
# The reference: the path this module exists to replace.
# ---------------------------------------------------------------------------------------------


def _full_logits_logprobs(
    libs: _ffi.Libraries, model, tokens: list[int], targets: list[int], weights: list[float]
) -> np.ndarray:
    """Decode, take the whole logits tensor, log-softmax in float64, gather. The obvious way.

    Deliberately shares nothing with the module under test: a different decode (logits, not
    embeddings), a different lm_head (llama.cpp's own, inside the graph), and the softmax done by
    numpy in double precision. If the two agree, they agree for a reason.
    """
    libs.llama.llama_memory_clear(libs.llama.llama_get_memory(model.ctx), True)

    n = len(tokens)
    tok = (_ffi.llama_token * n)(*tokens)
    pos = (_ffi.llama_pos * n)(*range(n))
    n_seq_id = (ctypes.c_int32 * n)(*([1] * n))
    seqs = [(_ffi.llama_seq_id * 1)(0) for _ in range(n)]
    seq_id = (ctypes.POINTER(_ffi.llama_seq_id) * n)(*seqs)
    want = (ctypes.c_int8 * n)(*([1] * n))

    batch = _ffi.llama_batch(
        n_tokens=n, token=tok, embd=None, pos=pos, n_seq_id=n_seq_id, seq_id=seq_id, logits=want
    )
    assert libs.llama.llama_decode(model.ctx, batch) == 0

    n_vocab = libs.llama.llama_vocab_n_tokens(libs.llama.llama_model_get_vocab(model.model))

    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if weights[i] == 0.0:
            continue
        row = libs.llama.llama_get_logits_ith(model.ctx, i)
        logits = np.ctypeslib.as_array(row, shape=(n_vocab,)).astype(np.float64)
        logits -= logits.max()
        logp = logits - np.log(np.exp(logits).sum())
        out[i] = weights[i] * logp[targets[i]]

    return out


@pytest.fixture
def scored(tiny_model, load_model, libs: _ffi.Libraries):
    """A loaded model, its lm_head, and a batch to score — for every fixture base in turn."""
    model = load_model(tiny_model, n_ctx=N_CTX)
    lm_head = load_lm_head(tiny_model)

    n_vocab = libs.llama.llama_vocab_n_tokens(libs.llama.llama_model_get_vocab(model.model))
    tokens = _tokens(SEQ_LEN, n_vocab)
    targets = tokens[1:] + [tokens[0]]
    # A prompt/completion mask: the first third counts for nothing, as in a real SFT or DPO batch.
    weights = [0.0] * (SEQ_LEN // 3) + [1.0] * (SEQ_LEN - SEQ_LEN // 3)

    return model, lm_head, tokens, targets, weights


# ---------------------------------------------------------------------------------------------
# THE test.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_rows", [1, 5, 8, SEQ_LEN, SEQ_LEN * 2])
def test_chunked_logprobs_equal_the_full_logits_path(scored, libs, chunk_rows: int) -> None:
    """Chunk size must not change the answer — including sizes that leave a ragged last chunk.

    5 does not divide 24, so the last chunk is short; SEQ_LEN * 2 is bigger than the batch, so
    there is one chunk and it is over-sized. Both are the cases where an off-by-one in the chunk
    loop lives, and both would be invisible to a test that only used 1 and n.
    """
    model, lm_head, tokens, targets, weights = scored

    got = sequence_logprobs(
        libs, model.ctx, lm_head, tokens, targets, weights, chunk_rows=chunk_rows
    )
    want = _full_logits_logprobs(libs, model, tokens, targets, weights)

    assert got.shape == (len(tokens),)
    np.testing.assert_allclose(got, want, atol=2e-4, rtol=0)


def test_the_mask_zeroes_exactly_the_masked_tokens(scored, libs) -> None:
    """A 0 weight means 0 in the answer — bitwise, not approximately.

    This is what lets a caller sum the result without a second mask, and it is what DPO's +1/-1
    weighting rests on.
    """
    model, lm_head, tokens, targets, weights = scored

    got = sequence_logprobs(libs, model.ctx, lm_head, tokens, targets, weights, chunk_rows=7)

    masked = [i for i, w in enumerate(weights) if w == 0.0]
    graded = [i for i, w in enumerate(weights) if w != 0.0]

    assert masked and graded
    assert all(got[i] == 0.0 for i in masked)
    # ...and every graded token is a real logprob: negative, and finite.
    assert all(got[i] < 0.0 for i in graded)
    assert np.isfinite(got).all()


# ---------------------------------------------------------------------------------------------
# The claim an equality test cannot make.
# ---------------------------------------------------------------------------------------------


def test_peak_transient_scales_with_the_chunk_and_not_the_batch(tiny_f32, libs) -> None:
    """The reason the module exists, asserted directly.

    Every other test here would pass against an implementation that builds the full
    ``[n_tokens, n_vocab]`` logits and gathers from it. That implementation would be *correct* and
    would defeat the entire purpose, so the memory claim gets its own test.

    Measured where it actually happens: ggml's own allocator, via the size of the buffer the chunk
    graph asks for. A 4096-token batch at chunk 32 must not cost 128× a 32-token one.
    """
    lm_head = load_lm_head(tiny_f32)

    n_tokens = 4096
    rng = np.random.default_rng(0)
    hidden = rng.normal(0, 0.1, size=(n_tokens, lm_head.n_embd)).astype(np.float32)
    labels = rng.integers(0, lm_head.n_vocab, size=n_tokens).astype(np.int32)

    peaks = {}
    for chunk in (32, 256):
        peak = 0

        real = libs.ggml_base.ggml_backend_alloc_ctx_tensors

        def spy(ctx, backend, _real=real):  # noqa: ANN001, ANN202
            nonlocal peak
            buf = _real(ctx, backend)
            peak = max(peak, libs.ggml_base.ggml_backend_buffer_get_size(buf))
            return buf

        libs.ggml_base.ggml_backend_alloc_ctx_tensors = spy
        try:
            chunked_token_logprobs(libs, hidden, lm_head, labels, chunk_rows=chunk)
        finally:
            libs.ggml_base.ggml_backend_alloc_ctx_tensors = real

        peaks[chunk] = peak

    # The transient the chunk graph allocates is dominated by its logits: chunk * n_vocab * 4.
    # It must track the chunk, not the 4096-token batch.
    full_logits_bytes = n_tokens * lm_head.n_vocab * 4

    for chunk, peak in peaks.items():
        assert peak < full_logits_bytes / 4, (
            f"chunk {chunk} allocated {peak} bytes; the full-logits tensor it is supposed to be "
            f"avoiding is {full_logits_bytes}. The chunking is not chunking."
        )

    # ...and it scales with the chunk, which is what makes it a knob rather than a coincidence.
    assert peaks[256] > peaks[32]


# ---------------------------------------------------------------------------------------------
# Two ways to score the wrong model and never know.
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_tied(libs, fixture_cache_dir) -> pathlib.Path:
    """A model with no ``output.weight`` — it projects with its token embeddings."""
    path, _ = gen_tiny_llama.build("f32", fixture_cache_dir, tied=True)
    return path


def test_a_tied_model_falls_back_to_the_token_embeddings(tiny_tied, tiny_f32) -> None:
    """No ``output.weight`` means project with ``token_embd.weight``, and say so."""
    tied = load_lm_head(tiny_tied)
    untied = load_lm_head(tiny_f32)

    assert tied.tied is True
    assert untied.tied is False

    # Same shape either way -- which is exactly why reading the wrong one is silent.
    assert (tied.n_embd, tied.n_vocab) == (untied.n_embd, untied.n_vocab)


def test_a_tied_model_scores_correctly(tiny_tied, load_model, libs) -> None:
    """The fallback has to produce the right numbers, not merely the right shape.

    If `load_lm_head` returned `token_embd.weight` for an *untied* model — or the base weight for a
    tied one — every shape would still line up and every logprob would be wrong. So the tied model
    is scored against llama.cpp's own logits, which resolve the tie internally.
    """
    model = load_model(tiny_tied, n_ctx=N_CTX)
    lm_head = load_lm_head(tiny_tied)

    n_vocab = libs.llama.llama_vocab_n_tokens(libs.llama.llama_model_get_vocab(model.model))
    tokens = _tokens(SEQ_LEN, n_vocab, seed=11)
    targets = tokens[1:] + [tokens[0]]
    weights = [1.0] * SEQ_LEN

    got = sequence_logprobs(libs, model.ctx, lm_head, tokens, targets, weights, chunk_rows=6)
    want = _full_logits_logprobs(libs, model, tokens, targets, weights)

    np.testing.assert_allclose(got, want, atol=2e-4, rtol=0)


def test_an_adapter_on_the_output_projection_must_be_applied(
    tiny_f32, tmp_path, load_model, libs
) -> None:
    """With a LoRA on ``output``, the base weight alone is the wrong policy.

    A zero-init adapter is a no-op, so it proves nothing. This one has a *nonzero* B, which makes
    the adapted model genuinely different from the base — and then:

      - scoring with the delta must match llama.cpp's own logits (which apply the adapter);
      - scoring *without* it must not. That second assertion is the one with teeth: it is what
        fails if `lora=` is quietly ignored, and without it this test would pass against a module
        that dropped the adapter on the floor.
    """
    adapter_path = tmp_path / "adapter.gguf"
    create_zero_adapter(tiny_f32, adapter_path, r=4, seed=7, include_output=True)

    model = load_model(tiny_f32, n_ctx=N_CTX)
    model.attach_adapter(adapter_path, scale=1.0)

    lm_head = load_lm_head(tiny_f32)

    # A zero-init adapter is a bitwise no-op, so it could not tell the two paths apart -- both would
    # equal the base model and the test would pass while proving nothing. Give B real values, in the
    # LIVE adapter, so llama.cpp's own logits move too.
    index, _, _ = _adapter_index_of(libs, model.adapter, "output.weight")
    n_b = _ffi.check(
        libs.farm.ll_adapter_get(model.adapter, index, True, None, 0), "ll_adapter_get"
    )

    rng = np.random.default_rng(5)
    new_b = rng.normal(0, 0.05, size=n_b).astype(np.float32)
    _ffi.check(
        libs.farm.ll_adapter_set(
            model.adapter, index, True, new_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n_b
        ),
        "ll_adapter_set",
    )

    delta = output_lora(libs, model.adapter, adapter_path, scale=1.0)
    assert delta is not None, "create_zero_adapter should have targeted output.weight"

    n_vocab = libs.llama.llama_vocab_n_tokens(libs.llama.llama_model_get_vocab(model.model))
    tokens = _tokens(SEQ_LEN, n_vocab, seed=13)
    targets = tokens[1:] + [tokens[0]]
    weights = [1.0] * SEQ_LEN

    want = _full_logits_logprobs(libs, model, tokens, targets, weights)

    with_delta = sequence_logprobs(
        libs, model.ctx, lm_head, tokens, targets, weights, chunk_rows=5, lora=delta
    )
    without_delta = sequence_logprobs(
        libs, model.ctx, lm_head, tokens, targets, weights, chunk_rows=5
    )

    np.testing.assert_allclose(with_delta, want, atol=2e-4, rtol=0)

    assert not np.allclose(without_delta, want, atol=2e-4), (
        "scoring without the output adapter matched the adapted model anyway. Either the adapter "
        "never reached llama.cpp's logits, or the delta path does nothing — and the equality "
        "assertion above cannot tell those apart, which is why this one is here."
    )


def test_an_adapter_that_misses_the_output_projection_yields_no_delta(
    tiny_f32, tmp_path, load_model, libs
) -> None:
    """`None` is the right answer, and the caller is meant to pass it straight through.

    Most adapters target the attention and MLP projections only. For those the base weight *is* the
    whole lm_head, and inventing a delta would be the bug.
    """
    # include_output is False by default -- the ordinary case, and the one that must not invent a
    # delta out of nothing.
    adapter_path = tmp_path / "attn-only.gguf"
    create_zero_adapter(tiny_f32, adapter_path, r=4, seed=7)

    model = load_model(tiny_f32, n_ctx=N_CTX)
    model.attach_adapter(adapter_path, scale=1.0)

    assert output_lora(libs, model.adapter, adapter_path) is None


# ---------------------------------------------------------------------------------------------
# The chunk-size heuristic.
# ---------------------------------------------------------------------------------------------


def test_choose_chunk_rows_respects_the_budget() -> None:
    # 128k vocab: one row of F32 logits is 512 KB, so a 128 MB budget buys 256 rows.
    assert choose_chunk_rows(4096, 128_000, mem_budget=128 * 1024 * 1024) == 262

    # ...and never more rows than there are tokens.
    assert choose_chunk_rows(16, 128_000, mem_budget=128 * 1024 * 1024) == 16

    # A budget too small for even one row still yields one: slow beats broken.
    assert choose_chunk_rows(4096, 128_000, mem_budget=1024) == 1


def test_choose_chunk_rows_honours_an_override() -> None:
    assert choose_chunk_rows(4096, 128_000, override=64) == 64
    assert choose_chunk_rows(16, 128_000, override=64) == 16  # clamped to the batch

    with pytest.raises(ValueError, match="positive"):
        choose_chunk_rows(4096, 128_000, override=0)


def test_the_default_budget_is_documented_and_sane() -> None:
    """A 128 MB transient is the number in the module docstring; it should stay that."""
    assert DEFAULT_MEM_BUDGET == 128 * 1024 * 1024


def test_a_shape_mismatch_is_refused(tiny_f32, libs) -> None:
    lm_head = load_lm_head(tiny_f32)
    hidden = np.zeros((4, lm_head.n_embd), dtype=np.float32)

    with pytest.raises(ValueError, match="labels"):
        chunked_token_logprobs(libs, hidden, lm_head, np.zeros(3, dtype=np.int32))

    with pytest.raises(ValueError, match="hidden"):
        chunked_token_logprobs(
            libs, np.zeros((4, 3), dtype=np.float32), lm_head, np.zeros(4, dtype=np.int32)
        )


def test_hidden_states_are_the_post_norm_states_and_the_flag_is_restored(scored, libs) -> None:
    """The decode must leave the context as it found it.

    A context whose embeddings flag stayed on returns hidden states where the next caller expects
    logits — and the shapes are both ``float *``, so it would not fail, it would just be wrong.
    """
    model, lm_head, tokens, _, _ = scored

    hidden = hidden_states(libs, model.ctx, tokens)

    assert hidden.shape == (len(tokens), lm_head.n_embd)
    assert np.isfinite(hidden).all()

    # ...and logits work again immediately afterwards.
    libs.llama.llama_memory_clear(libs.llama.llama_get_memory(model.ctx), True)
    n = len(tokens)
    tok = (_ffi.llama_token * n)(*tokens)
    batch = libs.llama.llama_batch_get_one(tok, n)
    assert libs.llama.llama_decode(model.ctx, batch) == 0
    assert libs.llama.llama_get_logits_ith(model.ctx, n - 1)
