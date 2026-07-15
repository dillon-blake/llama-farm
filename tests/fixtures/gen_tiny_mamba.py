"""A tiny Mamba-1 fixture (S1-47).

The SSM backward kernels (S1-29b/S1-30/S1-31) are checked op-by-op: ``SSM_CONV_BACK`` and
``SSM_SCAN_BACK`` each have MODE_GRAD cases against a finite difference of their own forward. All of
that says the two kernels' arithmetic is right in isolation. None of it says a Mamba model *trains*
-- that needs the ``ssm_in``/``ssm_x``/``ssm_dt``/``ssm_out`` projections carrying LoRA, the conv,
the selective scan, the D skip and the SiLU gate all composed in one real graph, on a real GGUF,
with the optimizer stepping the adapter. The MoE side has exactly this (``gen_tiny_moe.py`` +
``test_moe_gradients.py``); the SSM side never did, and that gap is the audit's ``moe-ssm`` major.

So this builds one. Architecture ``mamba`` (llama.cpp's Mamba-1), the smallest dims that still
exercise every SSM-specific op honestly:

* ``d_conv = 4`` -- a real causal conv window, not a degenerate 1-tap;
* ``d_state = 16`` -- Mamba's own default, so the scan's inner reduction is not a single term;
* per-state decay ``A`` of shape ``{d_state, d_inner}`` -- the Mamba-1 branch of ``ssm_scan``
  (``A->ne[0] != 1``), which is what the fork's backward runs today. Mamba-2's scalar-``A`` /
  ``n_group>1`` path is a separate boundary; see the S1-47 ticket.

The one hard constraint llama.cpp imposes: ``2 * n_embd == d_inner`` (``models/mamba.cpp:43``).
Everything else here is free because the fixture is F32 -- there is no quantized variant, and that
is not laziness: ``ssm_conv1d.weight`` has ``ne[0] = d_conv = 4``, a four-wide row that no K-quant
block (256 elements) can hold, so a "Q4_K Mamba" is not a thing that exists at these dims. F32 is
the honest and only variant.
"""

from __future__ import annotations

import dataclasses
import hashlib
import pathlib
from importlib.metadata import version

import gguf
import numpy as np

from .gen_tiny_llama import (
    CHAT_TEMPLATE,
    _load_vocab,
)

# F32 only. See the module docstring: the conv weight's four-wide rows cannot be K-quantized, so a
# quantized Mamba fixture at these dims is impossible rather than merely skipped.
VARIANTS = ("f32",)


@dataclasses.dataclass
class TinyMambaHParams:
    """Hyperparameters of the Mamba-1 fixture.

    Attributes:
        n_embd: Residual stream width. ``d_inner`` is pinned to ``2 * n_embd`` by llama.cpp's Mamba
            loader, so this is the one knob that moves the SSM width.
        d_conv: Causal conv-1d kernel width. Mamba's default is 4; anything less stops exercising
            the sliding window the ``SSM_CONV_BACK`` kernel reverses.
        d_state: SSM state dimension (the ``d_state`` reduction inside the scan). Mamba's default.
        dt_rank: Rank of the ``dt`` (time-step) low-rank projection out of ``ssm_x``.
        n_vocab: Vocabulary size. Sliced from the reference SPM vocab, same as the other fixtures.
    """

    n_layer: int = 2
    n_embd: int = 64
    d_conv: int = 4
    d_state: int = 16
    dt_rank: int = 16
    n_vocab: int = 512
    n_ctx_train: int = 512
    rms_eps: float = 1e-5

    @property
    def d_inner(self) -> int:
        """Mamba-1's inner width. llama.cpp supports only the expansion factor of 2."""
        return 2 * self.n_embd

    def validate(self) -> None:
        """Fail loudly on a config the fork cannot build or train.

        The only structural constraint at F32 is the expansion factor, which the loader enforces
        with a ``runtime_error`` -- catching it here names the reason instead.
        """
        if self.d_inner != 2 * self.n_embd:
            raise ValueError("llama.cpp's Mamba supports only d_inner == 2 * n_embd")


HPARAMS = TinyMambaHParams()


def _model_tensors(hp: TinyMambaHParams, seed: int) -> dict[str, np.ndarray]:
    """Every tensor as F32 numpy, in GGUF's reversed-shape convention.

    A numpy array of shape ``(n_out, n_in)`` is written with ``ne = [n_in, n_out]``. The frozen
    SSM tensors -- ``ssm_a`` (decay) and ``ssm_d`` (skip) -- carry NO ``.weight`` suffix, exactly
    as ``models/mamba.cpp`` names them; that also keeps them out of :func:`enumerate_targets`,
    which only matches ``*.weight``.
    """
    rng = np.random.default_rng(seed)
    d_inner, d_state, d_conv, dt_rank = hp.d_inner, hp.d_state, hp.d_conv, hp.dt_rank

    def weights(*shape: int) -> np.ndarray:
        # RMSNorm rescales the residual stream to unit magnitude, so 0.02-scale projections leave
        # the SSM activations O(1) -- the same reasoning as the llama fixture.
        return rng.normal(0.0, 0.02, size=shape).astype(np.float32)

    def norm(n: int) -> np.ndarray:
        return np.ones(n, dtype=np.float32)

    tensors: dict[str, np.ndarray] = {
        "token_embd.weight": weights(hp.n_vocab, hp.n_embd),
        "output_norm.weight": norm(hp.n_embd),
        "output.weight": weights(hp.n_vocab, hp.n_embd),
    }

    # S4D-real decay, the standard Mamba init: A[i0, h] = -(i0 + 1). Stored already-negated, as
    # llama.cpp expects (the scan computes exp(dt_softplus * A), so A < 0 is the decay). Making it
    # deterministic and state-dependent means the per-state (Mamba-1) branch of ssm_scan is
    # genuinely exercised -- a constant A would hide an index bug in the d_state reduction.
    a_row = -(np.arange(d_state, dtype=np.float32) + 1.0)  # (d_state,)
    ssm_a = np.broadcast_to(a_row[None, :], (d_inner, d_state)).copy()  # numpy (d_inner, d_state)

    for il in range(hp.n_layer):
        p = f"blk.{il}."
        tensors[p + "attn_norm.weight"] = norm(hp.n_embd)
        tensors[p + "ssm_in.weight"] = weights(2 * d_inner, hp.n_embd)
        tensors[p + "ssm_conv1d.weight"] = weights(d_inner, d_conv)
        tensors[p + "ssm_conv1d.bias"] = weights(d_inner)
        tensors[p + "ssm_x.weight"] = weights(dt_rank + 2 * d_state, d_inner)
        tensors[p + "ssm_dt.weight"] = weights(d_inner, dt_rank)
        # dt bias small: dt_softplus = softplus(dt_proj + dt_bias) then lands near softplus(0) ~=
        # 0.69, a healthy step size -- neither a vanished nor a saturated recurrence.
        tensors[p + "ssm_dt.bias"] = weights(d_inner)
        tensors[p + "ssm_a"] = ssm_a
        tensors[p + "ssm_d"] = norm(d_inner)  # unit skip, Mamba's D init
        tensors[p + "ssm_out.weight"] = weights(hp.n_embd, d_inner)

    return tensors


def _write(path: pathlib.Path, hp: TinyMambaHParams, variant: str, seed: int) -> None:
    tokens, scores, types = _load_vocab(hp.n_vocab)
    tensors = _model_tensors(hp, seed)

    writer = gguf.GGUFWriter(str(path), arch="mamba")

    writer.add_name("tiny-mamba-fixture")
    writer.add_context_length(hp.n_ctx_train)
    writer.add_embedding_length(hp.n_embd)
    writer.add_block_count(hp.n_layer)
    writer.add_layer_norm_rms_eps(hp.rms_eps)
    writer.add_vocab_size(hp.n_vocab)
    writer.add_file_type(gguf.LlamaFileType.ALL_F32)

    # The four keys that define a Mamba. No rope, no head_count: a recurrent arch has neither, and
    # llama.cpp loads both as optional (defaulting to 0) for exactly this family.
    writer.add_ssm_conv_kernel(hp.d_conv)
    writer.add_ssm_inner_size(hp.d_inner)
    writer.add_ssm_state_size(hp.d_state)
    writer.add_ssm_time_step_rank(hp.dt_rank)

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

    for name, data in tensors.items():
        writer.add_tensor(name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def cache_key(hp: TinyMambaHParams = HPARAMS) -> str:
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
    hp: TinyMambaHParams = HPARAMS,
    seed: int = 20260715,
) -> tuple[pathlib.Path, bool]:
    """Return the path to a Mamba fixture, generating it only if it is not already cached.

    Returns:
        A ``(path, generated)`` pair. ``generated`` is False on a cache hit.

    Raises:
        ValueError: If the variant is unknown (only ``f32`` exists; see the module docstring).
    """
    if variant not in VARIANTS:
        raise ValueError(
            f"unknown variant {variant!r}, expected one of {VARIANTS} "
            "(Mamba's four-wide conv rows cannot be K-quantized, so F32 is the only variant)"
        )

    hp.validate()

    out_dir = cache_dir / cache_key(hp)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"tiny-mamba-{variant}.gguf"

    if path.exists():
        return path, False

    tmp = path.with_suffix(".gguf.partial")
    _write(tmp, hp, variant, seed)
    tmp.rename(path)

    return path, True
