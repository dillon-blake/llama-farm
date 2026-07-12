"""Synthesize the tiny llama-arch fixture models the test suite trains and evaluates on.

Three variants, all generated (never committed): **F32**, **Q8_0**, and **Q4_K**. Quantized
bases are covered from day one on purpose — BLUEPRINT §10 (risk 2) notes that backward through
quantized weights is engine-supported but lightly exercised, and asks for validation per quant
type. A fixture set that is F32-only would hide exactly the class of bug the project is most
exposed to.

Dimensions
----------
The model is deliberately tiny but its dimensions are **not** arbitrary:

    n_layer=2  n_embd=256  n_head=4 (head_dim=64)  n_head_kv=2  n_ff=512  n_vocab=512

``n_embd`` and ``n_ff`` are multiples of 256 because **Q4_K's superblock is 256 elements**, so
every quantized row must be a multiple of 256. The ticket's suggested ``n_embd=64`` cannot be
Q4_K-quantized at all — ``ggml_quantize_chunk`` would either abort or silently need a fallback
type, and the "Q4_K fixture" would not actually be Q4_K. The rows that must divide are
``ne[0]``: ``n_embd`` for q/k/v/gate/up, ``n_head*head_dim`` for attn_output, and ``n_ff`` for
ffn_down.

The vocabulary is the first 512 entries of llama.cpp's own reference SPM vocab
(``vendor/llama.cpp/models/ggml-vocab-llama-spm.gguf``), which covers the three special tokens
plus the **complete 256-token byte-fallback range**. A truncated vocab that cut the byte range
short would load but not tokenize, and S1-06's data layer needs it to tokenize.

Quantization
------------
Q8_0 could be written from numpy, but Q4_K cannot: gguf-py's quantizer does not implement
K-quants. So both go through ``ggml_quantize_chunk`` over ctypes — llama.cpp's own in-repo
pattern for driving libggml from Python (``gguf-py/tests/test_quants.py``). Using one path for
both means the Q8_0 and Q4_K fixtures are produced by exactly the code that produces them at
runtime.

Norm tensors stay F32: they are 1-D, and quantization tooling leaves them alone.
"""

from __future__ import annotations

import ctypes
import hashlib
import pathlib
from dataclasses import dataclass
from importlib.metadata import version

import gguf
import numpy as np

from learning_llamas import _ffi

# The reference vocab we slice. Vendored and pinned, so this is reproducible.
REFERENCE_VOCAB_GGUF = (
    pathlib.Path(__file__).resolve().parents[2]
    / "vendor/llama.cpp/models/ggml-vocab-llama-spm.gguf"
)

VARIANTS = ("f32", "q8_0", "q4_k")

_QUANT_TYPE = {
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
    "q4_k": gguf.GGMLQuantizationType.Q4_K,
}

# gguf.LlamaFileType, for the general.file_type KV. Informational, but stock tooling reads it.
_FILE_TYPE = {
    "f32": gguf.LlamaFileType.ALL_F32,
    "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
    "q4_k": gguf.LlamaFileType.MOSTLY_Q4_K_M,
}


@dataclass(frozen=True)
class TinyLlamaHParams:
    """Hyperparameters of the fixture model."""

    n_layer: int = 2
    n_embd: int = 256
    n_head: int = 4
    n_head_kv: int = 2
    n_ff: int = 512
    n_vocab: int = 512
    n_ctx_train: int = 512
    rms_eps: float = 1e-5
    rope_freq_base: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def n_embd_gqa(self) -> int:
        """The K/V projection width under grouped-query attention."""
        return self.n_head_kv * self.head_dim

    def validate(self) -> None:
        """Fail loudly on dimensions that cannot be K-quantized.

        Q4_K's superblock is 256 elements, so a row that is not a multiple of 256 cannot be
        quantized to it. Catching that here beats discovering it as a ggml abort.
        """
        block = 256
        rows = {
            "n_embd (q/k/v/gate/up rows)": self.n_embd,
            "n_head*head_dim (attn_output rows)": self.n_head * self.head_dim,
            "n_ff (ffn_down rows)": self.n_ff,
        }
        bad = {name: value for name, value in rows.items() if value % block != 0}
        if bad:
            raise ValueError(
                f"these row lengths are not multiples of {block} and cannot be Q4_K-quantized: "
                f"{bad}"
            )
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")


HPARAMS = TinyLlamaHParams()


def _load_vocab(n_vocab: int) -> tuple[list[bytes], list[float], list[int]]:
    """Slice the first ``n_vocab`` entries out of llama.cpp's reference SPM vocab."""
    if not REFERENCE_VOCAB_GGUF.exists():
        raise FileNotFoundError(
            f"{REFERENCE_VOCAB_GGUF} is missing — run: git submodule update --init --recursive"
        )

    reader = gguf.GGUFReader(str(REFERENCE_VOCAB_GGUF), "r")

    def _field(key: str) -> gguf.ReaderField:
        field = reader.get_field(key)
        if field is None:
            raise ValueError(f"{REFERENCE_VOCAB_GGUF} has no {key}")
        return field

    tokens_field = _field("tokenizer.ggml.tokens")
    scores_field = _field("tokenizer.ggml.scores")
    types_field = _field("tokenizer.ggml.token_type")

    tokens = [bytes(tokens_field.parts[idx]) for idx in tokens_field.data[:n_vocab]]
    scores = [float(scores_field.parts[idx][0]) for idx in scores_field.data[:n_vocab]]
    types = [int(types_field.parts[idx][0]) for idx in types_field.data[:n_vocab]]

    if len(tokens) < n_vocab:
        raise ValueError(f"reference vocab has only {len(tokens)} tokens, need {n_vocab}")

    return tokens, scores, types


def _model_tensors(hp: TinyLlamaHParams, seed: int) -> dict[str, np.ndarray]:
    """Build every model tensor as F32 numpy, in GGUF's reversed-shape convention.

    A numpy array of shape ``(n_out, n_in)`` is written with ``ne = [n_in, n_out]``.
    """
    rng = np.random.default_rng(seed)

    def weights(*shape: int) -> np.ndarray:
        # Small values keep the untrained model's logits in a sane range, which matters for the
        # exactness of the no-op comparison (no inf/NaN to compare).
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
        tensors[f"blk.{il}.ffn_gate.weight"] = weights(hp.n_ff, hp.n_embd)
        tensors[f"blk.{il}.ffn_up.weight"] = weights(hp.n_ff, hp.n_embd)
        tensors[f"blk.{il}.ffn_down.weight"] = weights(hp.n_embd, hp.n_ff)

    return tensors


def _quantize(data: np.ndarray, qtype: gguf.GGMLQuantizationType) -> np.ndarray:
    """Quantize a 2-D F32 array with ggml_quantize_chunk, returning the raw block bytes.

    gguf-py can write Q8_0 from numpy but not Q4_K, so both types go through libggml. That also
    means the fixtures are quantized by exactly the code that dequantizes them at runtime.

    Returns:
        A ``uint8`` array shaped ``(nrows, row_bytes)`` — the *byte* shape, which is what
        ``GGUFWriter`` wants for a quantized tensor. It reverses the block packing itself to
        recover the element shape.
    """
    libs = _ffi.load()
    src = np.ascontiguousarray(data, dtype=np.float32)
    nrows, n_per_row = src.shape

    row_bytes = libs.ggml_base.ggml_row_size(int(qtype), n_per_row)
    dst = ctypes.create_string_buffer(row_bytes * nrows)

    written = libs.ggml_base.ggml_quantize_chunk(
        int(qtype),
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(dst, ctypes.c_void_p),
        0,
        nrows,
        n_per_row,
        None,  # no importance matrix
    )
    if written != row_bytes * nrows:
        raise RuntimeError(
            f"ggml_quantize_chunk wrote {written} bytes, expected {row_bytes * nrows}"
        )

    return np.frombuffer(dst.raw, dtype=np.uint8).reshape(nrows, row_bytes)


def _write(path: pathlib.Path, hp: TinyLlamaHParams, variant: str, seed: int) -> None:
    tokens, scores, types = _load_vocab(hp.n_vocab)
    tensors = _model_tensors(hp, seed)

    writer = gguf.GGUFWriter(str(path), arch="llama")

    writer.add_name("tiny-llama-fixture")
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
        # 1-D norms stay F32, exactly as the quantization tooling leaves them.
        if qtype is None or data.ndim == 1:
            writer.add_tensor(name, data)
        else:
            writer.add_tensor(name, _quantize(data, qtype), raw_dtype=qtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def cache_key(hp: TinyLlamaHParams = HPARAMS) -> str:
    """A content hash of the generator, its hyperparameters, and the gguf-py version.

    Any change to how a fixture is built invalidates the cache, so a stale fixture can never
    silently survive an edit to this file.
    """
    payload = b"".join(
        [
            pathlib.Path(__file__).read_bytes(),
            repr(hp).encode(),
            version("gguf").encode(),
        ]
    )
    return hashlib.sha256(payload).hexdigest()[:16]


def build(
    variant: str,
    cache_dir: pathlib.Path,
    hp: TinyLlamaHParams = HPARAMS,
    seed: int = 20260712,
) -> tuple[pathlib.Path, bool]:
    """Return the path to a fixture model, generating it only if it is not already cached.

    Args:
        variant: One of ``VARIANTS``.
        cache_dir: Directory to cache generated fixtures in. Gitignored; never committed.
        hp: The model's hyperparameters.
        seed: Seed for the weight RNG. Fixed, so fixtures are deterministic.

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
    path = out_dir / f"tiny-llama-{variant}.gguf"

    if path.exists():
        return path, False

    # Write to a temporary name and rename, so a crash mid-write cannot leave a truncated file
    # that the next run mistakes for a cache hit.
    tmp = path.with_suffix(".gguf.partial")
    _write(tmp, hp, variant, seed)
    tmp.rename(path)

    return path, True
