"""Monkey-patches for tinker_cookbook to match the original DeepSeek GRPO algorithm.

The original GRPO (DeepSeekMath, arXiv:2402.03300) differs from the cookbook defaults:
  1. Advantages are std-normalized: A_i = (r_i - mean) / std  (cookbook only mean-centers)
  2. Loss uses sequence-level normalization: divide by |o_i| per trajectory
     (cookbook applies a flat scalar to all action tokens without length division)

This module provides:
  - `grpo_compute_advantages`: drop-in replacement for cookbook's `compute_advantages`
    that adds std normalization
  - `apply_sequence_normalization`: post-processes datums to divide each datum's
    advantages by its action token count (mask sum)
  - `patch_grpo_advantages`: applies the monkey-patch at import time
"""

from __future__ import annotations

import torch

from tinker_cookbook.rl.data_processing import (
    assemble_training_data as _orig_assemble,
)
from tinker_cookbook.rl.types import TrajectoryGroup

import tinker


def grpo_compute_advantages(trajectory_groups_P: list[TrajectoryGroup]) -> list[torch.Tensor]:
    """Compute advantages with full GRPO normalization: (r - mean) / std.

    Groups where all rewards are identical get zero advantages (the caller
    should filter these out via remove_constant_reward_groups=True).
    """
    advantages_P: list[torch.Tensor] = []

    for traj_group in trajectory_groups_P:
        rewards_G = torch.tensor(traj_group.get_total_rewards(), dtype=torch.float64)
        mean = rewards_G.mean()
        std = rewards_G.std(correction=0)
        if std < 1e-8:
            advantages_P.append(torch.zeros(len(rewards_G)))
        else:
            advantages_P.append(((rewards_G - mean) / std).float())

    return advantages_P


def apply_sequence_normalization(data_D: list[tinker.Datum]) -> list[tinker.Datum]:
    """Divide each datum's advantages by its action token count (mask sum).

    This implements the original GRPO's 1/|o_i| per-sequence normalization.
    The importance_sampling loss sums (ratio * advantage) across tokens, so
    dividing advantage by token count makes the effective loss per trajectory
    equal to: (1/|o_i|) * sum_t(ratio_t * A_i), matching the original paper.
    """
    for datum in data_D:
        mask = datum.loss_fn_inputs["mask"].to_torch()
        action_token_count = mask.sum()
        if action_token_count > 0:
            advantages = datum.loss_fn_inputs["advantages"].to_torch()
            normalized = advantages / action_token_count
            datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(normalized)
    return data_D


def grpo_assemble_training_data(
    trajectory_groups_P: list[TrajectoryGroup],
    advantages_P: list[torch.Tensor],
) -> tuple[list[tinker.Datum], list[dict[str, int]]]:
    """Assemble training data with sequence-level normalization applied."""
    data_D, metadata_D = _orig_assemble(trajectory_groups_P, advantages_P)
    apply_sequence_normalization(data_D)
    return data_D, metadata_D


def patch_grpo_advantages():
    """Monkey-patch the cookbook to use original GRPO advantage computation and normalization.

    Call this before invoking `train.main()`.
    """
    import tinker_cookbook.rl.data_processing as dp
    import tinker_cookbook.rl.train as train_module

    dp.compute_advantages = grpo_compute_advantages
    train_module.compute_advantages = grpo_compute_advantages

    dp.assemble_training_data = grpo_assemble_training_data
    train_module.assemble_training_data = grpo_assemble_training_data
