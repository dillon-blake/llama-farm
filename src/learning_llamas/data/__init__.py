"""The data layer: chat templating, tokenization, and loss masking.

The loss mask is the part that matters. See :mod:`learning_llamas.data.mask` for why masking by
token count is wrong, and what this does instead.
"""

from .mask import BoundaryMergeError, MaskedSample, build_masked_sample, completion_text
from .template import (
    ChatTemplate,
    Message,
    UnknownSpecialTokenError,
    guard_special_tokens,
)
from .tokenize import Tokenizer

__all__ = [
    "BoundaryMergeError",
    "ChatTemplate",
    "MaskedSample",
    "Message",
    "Tokenizer",
    "UnknownSpecialTokenError",
    "build_masked_sample",
    "completion_text",
    "guard_special_tokens",
]
