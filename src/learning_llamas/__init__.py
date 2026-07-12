"""Train LoRA adapters on frozen, quantized GGUF models using llama.cpp's ggml backend.

The base model's weights stay quantized and memory-mapped; only the F32 LoRA A/B tensors
receive gradients, and the whole training step — forward, backward, and the AdamW update —
runs inside ggml, GPU-resident wherever the backend supports it.
"""

__version__ = "0.0.1"

__all__ = ["__version__"]
