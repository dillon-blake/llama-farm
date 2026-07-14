"""Build the HF-transformers twin of the GGUF fixture — the same model, in the other convention.

**This file exists because of one permutation, and getting it wrong is invisible.**

GGUF's llama arch is ``LLAMA_ROPE_TYPE_NORM`` (ggml mode 0), which rotates **interleaved adjacent**
pairs ``(x[2k], x[2k+1])``. HF's ``LlamaAttention`` uses ``rotate_half``, which rotates
**split-half** pairs ``(x[k], x[k + d/2])``. These are different functions of the same weights.
``convert_hf_to_gguf.py`` reconciles them by permuting the Q and K *weight rows* at conversion
time; this module applies the inverse, because it is going the other way.

Write the twin without the permutation and everything still works: the model loads, the loss falls,
PEFT trains happily. It is simply a **different model** from the one llama.cpp is running, so the
recorded curve is a reference to nothing, and the only symptom is a "convergence gate" that drifts
for reasons nobody can find.

The permutation, derived rather than copied
-------------------------------------------
Let ``sigma`` map the interleaved layout to the split-half one: ``sigma(x)[k] = x[2k]`` and
``sigma(x)[k + d/2] = x[2k+1]``. Then sigma carries ggml's pair ``(2k, 2k+1)`` onto HF's pair
``(k, k + d/2)``, and both rotate that pair by the same ``theta_k``, so::

    sigma(rope_ggml(q)) == rope_hf(sigma(q))

Setting ``W_hf = sigma(W_gguf)`` (a permutation of the *output rows*, per head) therefore gives
``q_hf = sigma(q_gguf)``, and the rotated vectors correspond under sigma. Attention scores are
``q . k`` over the head dimension, and permuting **both** q and k by the same sigma leaves every
dot product unchanged — so the two models compute the same attention, and hence the same logits.

That last step is also why the **LoRA tensors need no permutation at all**:

* ``A`` acts on the input side (``n_embd``), which sigma does not touch, and ``dA`` comes out
  identical on both sides (``dy_hf . B_hf == sigma(dy) . sigma(B)``, since sigma is orthogonal).
* ``B`` produces the permuted output rows, so ``B_gguf = sigma(B_hf)`` — but ``B`` is **zero** at
  init, and ``sigma(0) = 0``. Its gradient is sigma-related thereafter, and AdamW is elementwise,
  so the two trajectories stay sigma-related forever.

The two runs therefore have **identical losses** while their ``B`` tensors differ by a row
permutation. Only the base weights need permuting — and :func:`verify` checks that they were.
"""

from __future__ import annotations

import numpy as np

from ..fixtures.gen_tiny_llama import HPARAMS, TinyLlamaHParams, _model_tensors


def to_hf_rope_convention(w: np.ndarray, n_head: int) -> np.ndarray:
    """Permute a Q or K projection's output rows from ggml's interleaved layout to HF's split-half.

    ``hf[a*d/2 + b] = gguf[2b + a]`` for ``a`` in ``{0, 1}`` — exactly the inverse of
    ``convert_hf_to_gguf.py``'s ``LlamaModel.permute``.

    Args:
        w: ``(n_head * head_dim, n_in)``, in the GGUF's row order.
        n_head: The number of heads this projection has. **For K this is ``n_head_kv``**, not
            ``n_head`` — llama.cpp's own converter special-cases it, and under grouped-query
            attention using the wrong one silently scrambles the key heads.
    """
    n_out, n_in = w.shape
    d = n_out // n_head
    return w.reshape(n_head, d // 2, 2, n_in).swapaxes(1, 2).reshape(n_out, n_in)


def build(hp: TinyLlamaHParams = HPARAMS, seed: int = 20260712):  # noqa: ANN201 — torch types
    """Return an HF ``LlamaForCausalLM`` holding the same model as the GGUF fixture.

    The weights come from ``gen_tiny_llama._model_tensors`` — the *same function* the GGUF fixture
    is written from, with the same seed — so the two cannot drift. Only the Q/K RoPE convention
    differs, and that is what :func:`to_hf_rope_convention` fixes.

    Requires torch and transformers, which are **not** dependencies of this project. Only the
    one-time recorder imports this module.
    """
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    tensors = _model_tensors(hp, seed)

    cfg = LlamaConfig(
        vocab_size=hp.n_vocab,
        hidden_size=hp.n_embd,
        intermediate_size=hp.n_ff,
        num_hidden_layers=hp.n_layer,
        num_attention_heads=hp.n_head,
        num_key_value_heads=hp.n_head_kv,
        max_position_embeddings=hp.n_ctx_train,
        rms_norm_eps=hp.rms_eps,
        rope_theta=hp.rope_freq_base,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(cfg)

    sd: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.tensor(tensors["token_embd.weight"]),
        "model.norm.weight": torch.tensor(tensors["output_norm.weight"]),
        "lm_head.weight": torch.tensor(tensors["output.weight"]),
    }
    for il in range(hp.n_layer):
        p, h = f"blk.{il}.", f"model.layers.{il}."
        sd[h + "input_layernorm.weight"] = torch.tensor(tensors[p + "attn_norm.weight"])
        sd[h + "post_attention_layernorm.weight"] = torch.tensor(tensors[p + "ffn_norm.weight"])

        # The two that must be permuted, and nothing else.
        sd[h + "self_attn.q_proj.weight"] = torch.tensor(
            to_hf_rope_convention(tensors[p + "attn_q.weight"], hp.n_head)
        )
        sd[h + "self_attn.k_proj.weight"] = torch.tensor(
            to_hf_rope_convention(tensors[p + "attn_k.weight"], hp.n_head_kv)
        )

        # V is never roped, and O consumes the attention output rather than producing a roped one,
        # so neither is touched.
        sd[h + "self_attn.v_proj.weight"] = torch.tensor(tensors[p + "attn_v.weight"])
        sd[h + "self_attn.o_proj.weight"] = torch.tensor(tensors[p + "attn_output.weight"])

        sd[h + "mlp.gate_proj.weight"] = torch.tensor(tensors[p + "ffn_gate.weight"])
        sd[h + "mlp.up_proj.weight"] = torch.tensor(tensors[p + "ffn_up.weight"])
        sd[h + "mlp.down_proj.weight"] = torch.tensor(tensors[p + "ffn_down.weight"])

    missing, unexpected = model.load_state_dict(sd, strict=False)
    unexpected = [k for k in unexpected if "rotary_emb" not in k]
    missing = [k for k in missing if "rotary_emb" not in k]
    if missing or unexpected:
        raise RuntimeError(
            f"HF twin state dict mismatch: missing={missing} unexpected={unexpected}"
        )

    return model.to(torch.float32).eval()


def verify(model, tokens: np.ndarray, ref_logits: np.ndarray) -> float:  # noqa: ANN001
    """Assert the twin is the same model, by comparing its logits to the float64 reference.

    This is the check that makes the whole PEFT comparison mean anything. The float64 reference has
    already been pinned against ggml's training path to ~1e-6 (``tests/test_convergence.py``), so
    if the twin agrees with the reference, it agrees with llama.cpp — and the RoPE permutation,
    which is otherwise unobservable, is confirmed to be right.

    Returns:
        The max absolute logit deviation. Recorded into ``reference_curve.json`` as evidence.

    Raises:
        RuntimeError: If the twin is not the same model. A wrong (or missing) RoPE permutation
            lands in the ones, not the thousandths.
    """
    import torch

    with torch.no_grad():
        got = model(torch.tensor(tokens[None, :], dtype=torch.long)).logits[0].numpy()

    dev = float(np.abs(got.astype(np.float64) - ref_logits).max())
    spread = float(ref_logits.max() - ref_logits.min())

    if dev / spread > 1e-3:
        raise RuntimeError(
            f"the HF twin is NOT the same model as the GGUF fixture: max logit deviation {dev:.3e} "
            f"against a logit spread of {spread:.3e}. The overwhelmingly likely cause is the Q/K "
            "RoPE row permutation (see this module's docstring) — GGUF rotates interleaved pairs, "
            "HF rotates split-half pairs, and without the permutation these are different models."
        )
    return dev
