"""S1-00: the training graph builds a backward pass instead of aborting.

This is the regression test for the single most load-bearing change in the project, and it is
worth being precise about what it guards.

Causal-arch attention writes K/V into the KV cache with ``ggml_set_rows`` and reads them back
through a *view*. Both the write and the read are views of the cache buffer, so there is **no
autodiff edge** from ``k_cur``/``v_cur`` to the attention output: the gradient of the loss with
respect to ``wk``/``wv`` cannot flow. ``ggml_build_backward_expand`` does not quietly produce a
wrong gradient — it aborts on the SET_ROWS node ("inplace operations are currently not
supported", ``ggml.c:7093``). That is why upstream's own ``llama-finetune`` dies before printing
a loss, and why *nothing* that builds a backward graph could land before S1-00.

S1-00 gives ``llama_cparams`` a ``training`` flag, set by ``llama_opt_init``, which makes
``build_attn`` attend over ``k_cur``/``v_cur`` directly with a no-cache mask.

So: a test that merely calls the optimizer and checks the loss is finite would be enough to
catch a regression here, because **the failure mode is a hard abort, not a bad number.** The test
goes further and asserts the loss actually *falls*, which additionally catches a gradient that is
zero, disconnected, or wrong-signed.
"""

import ctypes

from learning_llamas import _ffi

# opt_init asserts n_ctx_train % n_batch == 0 and n_batch % n_ubatch == 0.
N_CTX = 64
N_UBATCH = 32


def _make_dataset(libs: _ffi.Libraries, tokens: list[int], n_ctx: int):
    """Build a next-token-prediction dataset: labels are the inputs shifted by one.

    Mirrors ``common_opt_dataset_init`` (``common/common.cpp:1936``), which is what
    ``llama-finetune`` uses.

    ``n_ctx`` must be the context's *actual* ``llama_n_ctx()``, not the value you asked for:
    llama.cpp pads it (here, a requested 64 becomes 256). Size the dataset from the request and
    ``ggml_opt_dataset_get_batch_host`` reads past the end of the permutation and aborts, because
    ``opt_epoch`` derives its batch size from the real ``n_ctx``. That is why
    ``common_opt_dataset_init`` reads it back rather than trusting the caller.
    """
    stride = n_ctx // 2
    ndata = (len(tokens) - n_ctx - 1) // stride
    assert ndata > 0, "corpus is too short for this context size"

    dataset = libs.ggml_base.ggml_opt_dataset_init(
        int(_ffi.GGMLType.I32),  # type_data
        int(_ffi.GGMLType.I32),  # type_label
        n_ctx,  # ne_datapoint
        n_ctx,  # ne_label
        ndata,
        1,  # ndata_shard
    )

    # ggml owns the storage; ask it for the pointer rather than mirroring struct ggml_tensor.
    data_ptr = libs.ggml_base.ggml_get_data(libs.ggml_base.ggml_opt_dataset_data(dataset))
    label_ptr = libs.ggml_base.ggml_get_data(libs.ggml_base.ggml_opt_dataset_labels(dataset))

    buf_type = ctypes.c_int32 * (ndata * n_ctx)
    data = buf_type.from_address(data_ptr)
    labels = buf_type.from_address(label_ptr)

    for idata in range(ndata):
        start = idata * stride
        for i in range(n_ctx):
            data[idata * n_ctx + i] = tokens[start + i]
            labels[idata * n_ctx + i] = tokens[start + i + 1]  # shifted by one

    return dataset, ndata


def _corpus(n: int) -> list[int]:
    """A repeating token pattern — learnable enough that the loss must fall in a few epochs."""
    pattern = [7, 11, 13, 17, 19, 23, 29, 31]
    return [pattern[i % len(pattern)] for i in range(n)]


def test_training_graph_builds_a_backward_pass_and_the_loss_falls(
    libs: _ffi.Libraries, tiny_f32, load_model
) -> None:
    """The whole of S1-00, in one assertion pair.

    Before S1-00 this aborts inside ``llama_opt_epoch`` with
    ``ggml.c:7093: GGML_ASSERT(!node->view_src || ...)`` — the process dies, so the test does not
    fail, it *crashes*. Reaching the assertions at all is most of the proof.
    """
    model = load_model(tiny_f32, n_ctx=N_CTX, n_ubatch=N_UBATCH, training=True)

    # The context's REAL n_ctx, not the one we asked for: llama.cpp pads it.
    n_ctx = libs.llama.llama_n_ctx(model.ctx)

    dataset, ndata = _make_dataset(libs, _corpus(n_ctx * 8), n_ctx)

    opt_params = _ffi.OptimizerParams(libs)
    opt_params.adamw.alpha = 1e-3  # a real learning rate: we want the loss to move
    get_opt_pars, get_opt_pars_ud = opt_params.callback()

    lopt = _ffi.llama_opt_params()
    lopt.n_ctx_train = 0  # use the context's n_ctx
    lopt.param_filter = ctypes.cast(
        libs.llama.llama_opt_param_filter_all, _ffi.llama_opt_param_filter
    )
    lopt.param_filter_ud = None
    lopt.get_opt_pars = get_opt_pars
    lopt.get_opt_pars_ud = get_opt_pars_ud
    lopt.optimizer_type = int(_ffi.OptimizerType.ADAMW)

    # This is the call that flips cparams.training, and therefore the call that decides whether
    # the graph that follows can be differentiated at all.
    libs.llama.llama_opt_init(model.ctx, model.model, lopt)

    idata_split = int(ndata * 0.8)
    result_train = libs.ggml_base.ggml_opt_result_init()
    result_eval = libs.ggml_base.ggml_opt_result_init()

    losses = []
    try:
        for _ in range(3):
            libs.ggml_base.ggml_opt_result_reset(result_train)
            libs.ggml_base.ggml_opt_result_reset(result_eval)

            # <-- Pre-S1-00, this aborts. No exception, no failure: SIGABRT.
            libs.llama.llama_opt_epoch(
                model.ctx, dataset, result_train, result_eval, idata_split, None, None
            )

            loss = ctypes.c_double()
            unc = ctypes.c_double()
            libs.ggml_base.ggml_opt_result_loss(result_train, ctypes.byref(loss), ctypes.byref(unc))
            losses.append(loss.value)
    finally:
        libs.ggml_base.ggml_opt_result_free(result_train)
        libs.ggml_base.ggml_opt_result_free(result_eval)
        libs.ggml_base.ggml_opt_dataset_free(dataset)

    # The backward graph built and ran: we have finite losses.
    assert all(loss == loss for loss in losses), f"loss went NaN: {losses}"  # noqa: PLR0124
    assert all(loss > 0.0 for loss in losses), f"loss is not positive: {losses}"

    # ...and the gradients are real. A disconnected, zeroed, or wrong-signed gradient would leave
    # the loss flat or push it up.
    assert losses[-1] < losses[0], f"loss did not fall over 3 epochs: {losses}"
