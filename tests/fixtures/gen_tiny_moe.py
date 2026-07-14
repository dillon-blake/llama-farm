"""A tiny Mixtral-shaped MoE fixture (S1-28).

The MoE kernels (S1-25/S1-26/S1-27) are checked op-by-op: against finite differences, against a
double-precision reference, across thread counts, across strides. All of that says the arithmetic
is right. None of it says a MoE model *trains* — that needs the router, the top-k, the expert
gather, the LoRA-on-a-3D-expert-stack pattern and the optimizer all working together, in a real
graph, on a real GGUF.

So this builds one. Same `llama` architecture as the dense fixture — Mixtral is a llama with
`expert_count` set — with the dense FFN replaced by a router plus three 3D expert stacks.

The tensor that matters is `ffn_gate_inp` (the router) and the `*_exps` stacks: those are what
`build_moe_ffn` feeds to `ggml_mul_mat_id`, and therefore what makes `build_lora_mm_id` put the
trainable LoRA A/B tensors **inside** the expert operand. That is the discovery S1-25 turns on, and
this fixture is the only thing in the repo that exercises it end to end.
"""

from __future__ import annotations

import dataclasses
import hashlib
import pathlib
from importlib.metadata import version

import gguf
import numpy as np

from .gen_tiny_llama import (
    _FILE_TYPE,
    _QUANT_TYPE,
    CHAT_TEMPLATE,
    VARIANTS,
    _load_vocab,
    _quantize,
)


@dataclasses.dataclass
class TinyMoEHParams:
    """Hyperparameters of the MoE fixture.

    Attributes:
        n_expert: How many experts each layer holds. Two would technically route, but four means a
            token's chosen pair is a genuine *subset* — so an expert that no token picked in a
            given step is reachable, and OUT_PROD_ID_GRP's zero-fill of unused experts is exercised
            rather than assumed.
        n_expert_used: Top-k. 2 is Mixtral's.
    """

    n_layer: int = 2
    n_embd: int = 256
    n_head: int = 4
    n_head_kv: int = 2
    n_ff: int = 256
    n_vocab: int = 512
    n_ctx_train: int = 512
    n_expert: int = 4
    n_expert_used: int = 2
    rms_eps: float = 1e-5
    rope_freq_base: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def n_embd_gqa(self) -> int:
        return self.head_dim * self.n_head_kv

    def validate(self) -> None:
        """K-quants work in blocks of 256, so every quantized row must be a multiple of it."""
        if self.n_expert_used > self.n_expert:
            raise ValueError(f"n_expert_used {self.n_expert_used} > n_expert {self.n_expert}")
        for name, n in (("n_embd", self.n_embd), ("n_ff", self.n_ff), ("n_vocab", self.n_vocab)):
            if n % 256 != 0:
                raise ValueError(f"{name}={n} is not a multiple of 256; K-quants need that")


HPARAMS = TinyMoEHParams()


def _model_tensors(hp: TinyMoEHParams, seed: int) -> dict[str, np.ndarray]:
    """Every tensor as F32 numpy, in GGUF's reversed-shape convention.

    A numpy array of shape ``(n_out, n_in)`` is written with ``ne = [n_in, n_out]``, so an expert
    stack of numpy shape ``(n_expert, n_out, n_in)`` is written with ``ne = [n_in, n_out,
    n_expert]`` — which is exactly the 3D operand ``ggml_mul_mat_id`` wants.
    """
    rng = np.random.default_rng(seed)

    def weights(*shape: int) -> np.ndarray:
        return rng.normal(0.0, 0.02, size=shape).astype(np.float32)

    def norm(n: int) -> np.ndarray:
        return np.ones(n, dtype=np.float32)

    tensors: dict[str, np.ndarray] = {
        "token_embd.weight": weights(hp.n_vocab, hp.n_embd),
        "output_norm.weight": norm(hp.n_embd),
        "output.weight": weights(hp.n_vocab, hp.n_embd),
    }

    n_embd_head = hp.n_head * hp.head_dim
    for il in range(hp.n_layer):
        tensors[f"blk.{il}.attn_norm.weight"] = norm(hp.n_embd)
        tensors[f"blk.{il}.attn_q.weight"] = weights(n_embd_head, hp.n_embd)
        tensors[f"blk.{il}.attn_k.weight"] = weights(hp.n_embd_gqa, hp.n_embd)
        tensors[f"blk.{il}.attn_v.weight"] = weights(hp.n_embd_gqa, hp.n_embd)
        tensors[f"blk.{il}.attn_output.weight"] = weights(hp.n_embd, n_embd_head)
        tensors[f"blk.{il}.ffn_norm.weight"] = norm(hp.n_embd)

        # The router. ne = [n_embd, n_expert]. Its output goes through softmax + argsort_top_k, and
        # the top-k indices are I32 -- which is what keeps the router off the gradient path while
        # the top-k *weights* stay on it (S1-25's "router needs no new kernels" analysis).
        tensors[f"blk.{il}.ffn_gate_inp.weight"] = weights(hp.n_expert, hp.n_embd)

        # The three expert stacks. 3D, and that is the whole point.
        tensors[f"blk.{il}.ffn_gate_exps.weight"] = weights(hp.n_expert, hp.n_ff, hp.n_embd)
        tensors[f"blk.{il}.ffn_up_exps.weight"] = weights(hp.n_expert, hp.n_ff, hp.n_embd)
        tensors[f"blk.{il}.ffn_down_exps.weight"] = weights(hp.n_expert, hp.n_embd, hp.n_ff)

    return tensors


def _write(path: pathlib.Path, hp: TinyMoEHParams, variant: str, seed: int) -> None:
    tokens, scores, types = _load_vocab(hp.n_vocab)
    tensors = _model_tensors(hp, seed)

    writer = gguf.GGUFWriter(str(path), arch="llama")

    writer.add_name("tiny-moe-fixture")
    writer.add_context_length(hp.n_ctx_train)
    writer.add_embedding_length(hp.n_embd)
    writer.add_block_count(hp.n_layer)
    writer.add_feed_forward_length(hp.n_ff)
    writer.add_head_count(hp.n_head)
    writer.add_head_count_kv(hp.n_head_kv)
    writer.add_layer_norm_rms_eps(hp.rms_eps)
    writer.add_rope_dimension_count(hp.head_dim)
    writer.add_rope_freq_base(hp.rope_freq_base)
    writer.add_vocab_size(hp.n_vocab)
    writer.add_file_type(_FILE_TYPE[variant])

    # These two keys are the entire difference between a llama and a Mixtral, as far as llama.cpp
    # is concerned: set them and llm_build_llama routes the FFN through build_moe_ffn.
    writer.add_expert_count(hp.n_expert)
    writer.add_expert_used_count(hp.n_expert_used)

    writer.add_chat_template(CHAT_TEMPLATE)
    writer.add_tokenizer_model("llama")
    writer.add_tokenizer_pre("default")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(types)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_unk_token_id(0)
    writer.add_add_bos_token(True)
    writer.add_add_eos_token(False)

    qtype = _QUANT_TYPE.get(variant)
    for name, data in tensors.items():
        # 1-D norms stay F32. So does the router: it is [n_embd, n_expert] with n_expert = 4, and a
        # 4-wide row cannot be K-quantized (blocks are 256). Real Mixtral GGUFs keep it F32 too.
        if qtype is None or data.ndim == 1 or name.endswith("ffn_gate_inp.weight"):
            writer.add_tensor(name, data)
        elif data.ndim == 3:
            # An expert stack quantizes ROW BY ROW, exactly like a 2-D weight — the expert axis is
            # just more rows. Flatten it, quantize, and hand the writer the element shape back,
            # because a uint8 byte array cannot tell it what the expert axis was.
            #
            # The byte array keeps the expert axis: GGUFWriter recovers the element shape by
            # reversing the block packing on the LAST axis only, so it must see
            # (n_expert, n_out, row_bytes) rather than a flattened (n_expert*n_out, row_bytes).
            n_expert, n_out, n_in = data.shape
            packed = _quantize(data.reshape(n_expert * n_out, n_in), qtype)
            writer.add_tensor(name, packed.reshape(n_expert, n_out, -1), raw_dtype=qtype)
        else:
            writer.add_tensor(name, _quantize(data, qtype), raw_dtype=qtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def cache_key(hp: TinyMoEHParams = HPARAMS) -> str:
    """A content hash of this generator, its hyperparameters, and gguf-py's version.

    Newlines normalized, for the reason spelled out in
    :func:`tests.fixtures.gen_tiny_llama.cache_key`: hashing raw bytes makes the key depend on the
    checkout's line endings, and git gives Windows CRLF.
    """
    source = pathlib.Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    payload = b"".join(
        [
            source.encode(),
            repr(hp).encode(),
            version("gguf").encode(),
        ]
    )
    return hashlib.sha256(payload).hexdigest()[:16]


def build(
    variant: str,
    cache_dir: pathlib.Path,
    hp: TinyMoEHParams = HPARAMS,
    seed: int = 20260714,
) -> tuple[pathlib.Path, bool]:
    """Return the path to a MoE fixture, generating it only if it is not already cached.

    Returns:
        A ``(path, generated)`` pair. ``generated`` is False on a cache hit.

    Raises:
        ValueError: If the variant is unknown or the dimensions cannot be K-quantized.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")

    hp.validate()

    out_dir = cache_dir / cache_key(hp)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"tiny-moe-{variant}.gguf"

    if path.exists():
        return path, False

    _write(path, hp, variant, seed)
    return path, True
