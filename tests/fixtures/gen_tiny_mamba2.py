"""A tiny Mamba-2 fixture (B-10).

The Mamba-1 fixture (:mod:`tests.fixtures.gen_tiny_mamba`) builds ``n_group == 1`` -- llama.cpp's
Mamba-1 has no other. The ``SSM_SCAN`` backward's group-index routing (``g = h/(nh/ng)``, folding
every head of a group into one ``dB``/``dC`` slab) therefore never runs on a real training graph
there, and S1-47 refused ``n_group > 1`` for want of an oracle. This fixture is the ``n_group > 1``
training graph: arch ``mamba2``, ``ssm.group_count = 2``, so B/C are shared across pairs of heads
and the fold is genuinely exercised.

Mamba-2 differs from Mamba-1 in the ways the reference (``reference_mamba2.py``) and this generator
both have to honour:

* **One projection in, not three.** ``ssm_in`` emits ``[z | xBC | dt]`` in one matmul
  (``d_in_proj = 2*d_inner + 2*n_group*d_state + n_head``). There is no ``ssm_x``/``ssm_dt``
  projection, so the only LoRA targets are ``ssm_in`` and ``ssm_out``.
* **The conv covers x, B and C together.** ``ssm_conv1d`` is ``d_inner + 2*n_group*d_state`` wide;
  B and C pass through the causal conv and SiLU before the scan, then split off.
* **Heads.** ``n_head = ssm.time_step_rank`` and ``head_dim = d_inner / n_head``. ``A`` and ``D``
  are one scalar per head (the ``ssm_scan`` scalar-``A`` branch), ``dt`` is one value per head.
* **A grouped gate norm.** After ``silu(z) * y`` an RMSNorm (``ssm_norm``) runs per group over
  ``d_inner / n_group`` channels, before ``ssm_out``.

F32 only, for the same reason as the Mamba-1 fixture: the conv weight's ``d_conv``-wide rows cannot
be K-quantized. The dims are the smallest that keep every axis non-degenerate -- ``head_dim`` and
``n_group`` both > 1, and ``n_head / n_group == 4`` (8 heads over 2 groups) so a group's ``dB``/
``dC`` really does fold several heads rather than passing one straight through.
"""

from __future__ import annotations

import dataclasses
import hashlib
import pathlib
from importlib.metadata import version

import gguf
import numpy as np

from . import llama_source
from .gen_tiny_llama import (
    CHAT_TEMPLATE,
    _load_vocab,
)

# F32 only. See the module docstring and the Mamba-1 fixture: the conv weight's narrow rows cannot
# be K-quantized, so a quantized Mamba-2 at these dims is impossible rather than merely skipped.
VARIANTS = ("f32",)


@dataclasses.dataclass
class TinyMamba2HParams:
    """Hyperparameters of the Mamba-2 fixture.

    Attributes:
        n_embd: Residual stream width. Unlike Mamba-1, llama.cpp's Mamba-2 does not pin
            ``d_inner`` to ``2 * n_embd``, so the two move independently.
        d_inner: SSM inner width. Split into ``n_head`` heads of ``head_dim`` each.
        n_head: Number of SSM heads (``ssm.time_step_rank`` in the GGUF -- Mamba-2 reuses that key
            for the head count). ``A``, ``D`` and ``dt`` are one value per head.
        n_group: Number of B/C groups. Heads ``0..n_head/n_group-1`` share group 0, and so on -- the
            routing the ``SSM_SCAN`` backward folds over.
        d_state: SSM state dimension (the inner reduction of the scan).
        d_conv: Causal conv-1d kernel width.
        n_vocab: Vocabulary size, sliced from the reference SPM vocab.
    """

    n_layer: int = 2
    n_embd: int = 32
    d_inner: int = 32
    n_head: int = 8
    n_group: int = 2
    d_state: int = 8
    d_conv: int = 4
    n_vocab: int = 512
    n_ctx_train: int = 512
    rms_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.d_inner // self.n_head

    @property
    def conv_dim(self) -> int:
        """The conv (and its x/B/C split) span x plus both grouped B and C."""
        return self.d_inner + 2 * self.n_group * self.d_state

    @property
    def d_in_proj(self) -> int:
        """``ssm_in`` emits ``[z | xBC | dt]`` in one matmul."""
        return self.d_inner + self.conv_dim + self.n_head

    def validate(self) -> None:
        """Fail loudly on a config the Mamba-2 graph cannot build."""
        if self.d_inner % self.n_head != 0:
            raise ValueError("d_inner must be a multiple of n_head")
        if self.d_inner % self.n_group != 0:
            raise ValueError("d_inner must be a multiple of n_group")
        if self.n_head % self.n_group != 0:
            raise ValueError("n_head must be a multiple of n_group (the SSM_SCAN group fold)")


HPARAMS = TinyMamba2HParams()

# The weight-RNG seed, hoisted out of ``build``'s signature so :func:`cache_key` can hash it. A seed
# that is not in the key is a seed that does not name the fixture: ask for a different one and a
# warm cache hands back the default-seed model, because the directory it lives in is the same.
SEED = 20260716


def _model_tensors(hp: TinyMamba2HParams, seed: int) -> dict[str, np.ndarray]:
    """Every tensor as F32 numpy, in GGUF's reversed-shape convention.

    A numpy array of shape ``(n_out, n_in)`` is written with ``ne = [n_in, n_out]``. The frozen
    per-head ``ssm_a`` (decay) and ``ssm_d`` (skip) carry NO ``.weight`` suffix, exactly as
    ``models/mamba2.cpp`` names them, which also keeps them out of :func:`enumerate_targets`.
    """
    rng = np.random.default_rng(seed)
    d_inner, d_conv = hp.d_inner, hp.d_conv
    n_head, n_group = hp.n_head, hp.n_group

    def weights(*shape: int) -> np.ndarray:
        return rng.normal(0.0, 0.02, size=shape).astype(np.float32)

    def norm(*shape: int) -> np.ndarray:
        return np.ones(shape, dtype=np.float32)

    tensors: dict[str, np.ndarray] = {
        "token_embd.weight": weights(hp.n_vocab, hp.n_embd),
        "output_norm.weight": norm(hp.n_embd),
        "output.weight": weights(hp.n_vocab, hp.n_embd),
    }

    # Scalar decay per head, stored already-negated (the scan computes exp(dt_softplus * A), so
    # A < 0 is the decay). Modest magnitudes keep the exp recurrence -- and a finite difference of
    # it -- well conditioned. Head-dependent so an index bug in the scalar-A branch cannot hide.
    a_col = -0.5 * (np.arange(n_head, dtype=np.float32) + 1.0)  # (n_head,)
    ssm_a = a_col[:, None].copy()  # numpy (n_head, 1) -> ne {1, n_head}

    for il in range(hp.n_layer):
        p = f"blk.{il}."
        tensors[p + "attn_norm.weight"] = norm(hp.n_embd)
        tensors[p + "ssm_in.weight"] = weights(hp.d_in_proj, hp.n_embd)
        tensors[p + "ssm_conv1d.weight"] = weights(hp.conv_dim, d_conv)
        tensors[p + "ssm_conv1d.bias"] = weights(hp.conv_dim)
        # dt bias small so softplus(dt) lands near softplus(0) ~= 0.69: a healthy step size.
        tensors[p + "ssm_dt.bias"] = weights(n_head)
        tensors[p + "ssm_a"] = ssm_a
        tensors[p + "ssm_d"] = norm(n_head, 1)  # unit skip per head; ne {1, n_head}
        tensors[p + "ssm_norm.weight"] = norm(n_group, d_inner // n_group)
        tensors[p + "ssm_out.weight"] = weights(hp.n_embd, d_inner)

    return tensors


def _write(path: pathlib.Path, hp: TinyMamba2HParams, variant: str, seed: int) -> None:
    tokens, scores, types = _load_vocab(hp.n_vocab)
    tensors = _model_tensors(hp, seed)

    writer = gguf.GGUFWriter(str(path), arch="mamba2")

    writer.add_name("tiny-mamba2-fixture")
    writer.add_context_length(hp.n_ctx_train)
    writer.add_embedding_length(hp.n_embd)
    writer.add_block_count(hp.n_layer)
    writer.add_layer_norm_rms_eps(hp.rms_eps)
    writer.add_vocab_size(hp.n_vocab)
    writer.add_file_type(gguf.LlamaFileType.ALL_F32)

    # Mamba-2 reuses ssm.time_step_rank as the head count, and adds ssm.group_count.
    writer.add_ssm_conv_kernel(hp.d_conv)
    writer.add_ssm_inner_size(hp.d_inner)
    writer.add_ssm_state_size(hp.d_state)
    writer.add_ssm_time_step_rank(hp.n_head)
    writer.add_ssm_group_count(hp.n_group)

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


def cache_key(hp: TinyMamba2HParams = HPARAMS, seed: int = SEED) -> str:
    """A content hash of everything the fixture bytes depend on.

    This generator, **the llama generator it borrows ``CHAT_TEMPLATE`` and ``_load_vocab`` from**,
    the hyperparameters, the weight seed and gguf-py's version -- matching
    :func:`tests.fixtures.gen_tiny_mamba.cache_key`, including the newline normalization that keeps
    a CRLF checkout from producing a different key for the same model.
    """
    source = pathlib.Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    payload = b"".join(
        [
            source.encode(),
            llama_source().encode(),
            repr(hp).encode(),
            repr(seed).encode(),
            version("gguf").encode(),
        ]
    )
    return hashlib.sha256(payload).hexdigest()[:16]


def build(
    variant: str,
    cache_dir: pathlib.Path,
    hp: TinyMamba2HParams = HPARAMS,
    seed: int = SEED,
) -> tuple[pathlib.Path, bool]:
    """Return the path to a Mamba-2 fixture, generating it only if not already cached.

    Returns:
        A ``(path, generated)`` pair. ``generated`` is False on a cache hit.

    Raises:
        ValueError: If the variant is unknown (only ``f32`` exists; see the module docstring).
    """
    if variant not in VARIANTS:
        raise ValueError(
            f"unknown variant {variant!r}, expected one of {VARIANTS} "
            "(Mamba's narrow conv rows cannot be K-quantized, so F32 is the only variant)"
        )

    hp.validate()

    out_dir = cache_dir / cache_key(hp, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"tiny-mamba2-{variant}.gguf"

    if path.exists():
        return path, False

    tmp = path.with_suffix(".gguf.partial")
    _write(tmp, hp, variant, seed)
    tmp.rename(path)

    return path, True
