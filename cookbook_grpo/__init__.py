"""Cookbook-based GRPO training for TTT-Discover subproblems."""

from cookbook_grpo.parser import extract_boxed_answer, normalize_answer
from cookbook_grpo.rewards import compute_reward
from cookbook_grpo.dataset import load_subproblems, SubproblemDataset, SubproblemDatasetBuilder
from cookbook_grpo.env import SubproblemEnv
from cookbook_grpo.grpo_overrides import (
    grpo_compute_advantages,
    apply_sequence_normalization,
    patch_grpo_advantages,
)

__all__ = [
    "extract_boxed_answer",
    "normalize_answer",
    "compute_reward",
    "load_subproblems",
    "SubproblemDataset",
    "SubproblemDatasetBuilder",
    "SubproblemEnv",
    "grpo_compute_advantages",
    "apply_sequence_normalization",
    "patch_grpo_advantages",
]
