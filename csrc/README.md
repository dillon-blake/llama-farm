# `csrc/` — the C shim (`liblearningllamas`)

Layer 1 of the architecture (BLUEPRINT §3). A thin C++ library compiled **against
llama.cpp's private `src/` internals**, exposing everything Python needs through a flat C
ABI in `farm_api.h`.

The shim is not avoidable. The per-ubatch training loop learning-llamas needs
(`llama_context::opt_epoch_iter`, declared at `vendor/llama.cpp/src/llama-context.h:207`)
depends on private C++ types — `graph_params`, `llm_graph_result`, `balloc`, `memory` — and
the public `llama_opt_epoch` hardcodes the wrong loss with no masking support.

| Ticket | What it adds |
|---|---|
| S0-03 | The build, `farm_api.h`, and the `ll_version` / `ll_probe` probe symbols |
| S1-01 | `ll_opt_init_lora` — `ggml_set_param` on the adapter A/B tensors |
| S1-02 | `ll_train_step` — the forked training loop with a pluggable loss |
