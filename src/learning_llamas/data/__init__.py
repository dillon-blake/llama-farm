"""The data layer: chat templating, tokenization, and loss masking.

The loss mask is the part that matters. See :mod:`learning_llamas.data.mask` for why masking by
token count is wrong, and what this does instead.
"""

from .mask import BoundaryMergeError, MaskedSample, build_masked_sample, completion_text
from .template import ChatTemplate, Message
from .tokenize import Tokenizer

__all__ = [
    "BoundaryMergeError",
    "ChatTemplate",
    "MaskedSample",
    "Message",
    "Tokenizer",
    "build_masked_sample",
    "completion_text",
]
