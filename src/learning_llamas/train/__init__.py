"""Training: the step loop, the learning-rate schedules, and the SFT trainer.

The loop is shared. SFT (S1-05), DPO (S1-14) and GRPO (S1-16) differ in how a batch is built and
what the loss means; they do not differ in how a batch is pushed through the graph, how gradients
accumulate, or how the learning rate moves. That common part lives in :mod:`.loop`.
"""

from .loop import Batch, Hooks, StepMetrics, TrainableModel, TrainConfig, Trainer, run
from .schedule import constant, warmup_cosine
from .sft import SFTConfig, SFTResult, collate, to_batch, train_sft

__all__ = [
    "Batch",
    "Hooks",
    "SFTConfig",
    "SFTResult",
    "StepMetrics",
    "TrainConfig",
    "Trainer",
    "TrainableModel",
    "collate",
    "constant",
    "run",
    "to_batch",
    "train_sft",
    "warmup_cosine",
]
