"""Train LoRA adapters on frozen, quantized GGUF models using llama.cpp's ggml backend.

The base model's weights stay quantized and memory-mapped; only the F32 LoRA A/B tensors
receive gradients, and the whole training step — forward, backward, and the AdamW update —
runs inside ggml, GPU-resident wherever the backend supports it.

The shape of a run is always the same four steps::

    from learning_llamas import Model, create_zero_adapter, libraries, save_adapter
    from learning_llamas.train import SFTConfig, train_sft

    create_zero_adapter("base.gguf", "adapter.gguf", r=16)          # 1. a no-op adapter
    with Model("base.gguf", n_ctx=512, n_ubatch=256, training=True) as model:
        model.attach_adapter("adapter.gguf")                        # 2. train through it
        train_sft(libraries(), model, samples, SFTConfig(seq_len=256))
        save_adapter(libraries(), model.ctx, "trained.gguf")        # 3. write it back

    # 4. llama-cli -m base.gguf --lora trained.gguf

The adapter GGUF that comes out is loadable by stock llama.cpp — no merge step required, and
nothing in this library ever writes to the base model. See ``docs/quickstart.md``.
"""

from .adapter import (
    AdapterInfo,
    LoraTarget,
    create_zero_adapter,
    enumerate_targets,
    read_adapter,
    save_adapter,
)
from .checkpoint import Checkpoint, read_checkpoint, restore_checkpoint, save_checkpoint
from .data import ChatTemplate, MaskedSample, Message, Tokenizer, build_masked_sample
from .export import merge
from .logprobs import LmHead, load_lm_head, sequence_logprobs
from .model import Model, libraries
from .preflight import Report, Status, preflight
from .verify import SelfVerified

__version__ = "0.0.1"

__all__ = [
    "AdapterInfo",
    "ChatTemplate",
    "Checkpoint",
    "LmHead",
    "LoraTarget",
    "MaskedSample",
    "Message",
    "Model",
    "Report",
    "SelfVerified",
    "Status",
    "Tokenizer",
    "__version__",
    "build_masked_sample",
    "create_zero_adapter",
    "enumerate_targets",
    "libraries",
    "load_lm_head",
    "merge",
    "preflight",
    "read_adapter",
    "read_checkpoint",
    "restore_checkpoint",
    "save_adapter",
    "save_checkpoint",
    "sequence_logprobs",
]
