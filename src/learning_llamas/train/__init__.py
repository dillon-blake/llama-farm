"""Training: the step loop, the learning-rate schedules, and the SFT trainer.

The loop is shared. SFT (S1-05), DPO (S1-14) and GRPO (S1-16) differ in how a batch is built and
what the loss means; they do not differ in how a batch is pushed through the graph, how gradients
accumulate, or how the learning rate moves. That common part lives in :mod:`.loop`.
"""

from .dpo import DPOConfig, DPOResult, DPOTrainer, Preference, reference_logratios, train_dpo
from .grpo import (
    GRPOBatch,
    GRPOConfig,
    GRPOMetrics,
    GRPOResult,
    GRPOTrainer,
    train_grpo,
)
from .loop import Batch, Hooks, StepMetrics, TrainableModel, TrainConfig, Trainer, run
from .packing import n_sequences, pack
from .rollout import (
    Rollout,
    RolloutBatch,
    RolloutEngine,
    RolloutStats,
    SamplerConfig,
    group_advantages,
    length_reward,
    substring_reward,
    token_reward,
)
from .schedule import constant, warmup_cosine
from .sft import SFTConfig, SFTResult, collate, to_batch, train_sft

__all__ = [
    "Batch",
    "DPOConfig",
    "DPOResult",
    "Rollout",
    "DPOTrainer",
    "Preference",
    "Hooks",
    "SFTConfig",
    "SFTResult",
    "StepMetrics",
    "TrainConfig",
    "Trainer",
    "TrainableModel",
    "collate",
    "reference_logratios",
    "train_dpo",
    "constant",
    "n_sequences",
    "pack",
    "run",
    "to_batch",
    "train_sft",
    "warmup_cosine",
    "Rollout",
    "RolloutBatch",
    "RolloutEngine",
    "RolloutStats",
    "SamplerConfig",
    "group_advantages",
    "length_reward",
    "substring_reward",
    "token_reward",
    "GRPOBatch",
    "GRPOConfig",
    "GRPOMetrics",
    "GRPOResult",
    "GRPOTrainer",
    "train_grpo",
]
