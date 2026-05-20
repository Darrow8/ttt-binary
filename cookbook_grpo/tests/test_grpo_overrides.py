"""Tests for the GRPO override module (std normalization + sequence-level loss)."""

import torch
import tinker
from tinker.types.tensor_data import TensorData

from cookbook_grpo.grpo_overrides import (
    grpo_compute_advantages,
    apply_sequence_normalization,
)


class FakeTrajectoryGroup:
    """Minimal stand-in for TrajectoryGroup with get_total_rewards."""

    def __init__(self, rewards: list[float]):
        self._rewards = rewards

    def get_total_rewards(self) -> list[float]:
        return self._rewards


def test_grpo_advantages_std_normalized():
    """Advantages should be (r - mean) / std."""
    group = FakeTrajectoryGroup([1.0, 0.0, 0.0, 1.0])
    advantages = grpo_compute_advantages([group])
    assert len(advantages) == 1
    adv = advantages[0]

    # mean = 0.5, std = 0.5 -> advantages are [1, -1, -1, 1]
    expected = torch.tensor([1.0, -1.0, -1.0, 1.0])
    assert torch.allclose(adv, expected, atol=1e-5)


def test_grpo_advantages_uniform_rewards():
    """Uniform rewards should produce zero advantages."""
    group = FakeTrajectoryGroup([0.5, 0.5, 0.5])
    advantages = grpo_compute_advantages([group])
    assert torch.allclose(advantages[0], torch.zeros(3), atol=1e-8)


def test_grpo_advantages_varied():
    """Test with non-symmetric reward distribution."""
    group = FakeTrajectoryGroup([0.0, 0.01, 1.0])
    advantages = grpo_compute_advantages([group])
    adv = advantages[0]

    rewards = torch.tensor([0.0, 0.01, 1.0], dtype=torch.float64)
    mean = rewards.mean()
    std = rewards.std(correction=0)
    expected = ((rewards - mean) / std).float()
    assert torch.allclose(adv, expected, atol=1e-5)


def test_sequence_normalization():
    """Advantages should be divided by action token count (mask sum)."""
    mask = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
    advantages = torch.tensor([0.0, 0.0, 0.6, 0.6, 0.6])

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2, 3, 4, 5]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor([2, 3, 4, 5, 6])),
            "logprobs": TensorData.from_torch(torch.zeros(5)),
            "advantages": TensorData.from_torch(advantages),
            "mask": TensorData.from_torch(mask),
        },
    )

    apply_sequence_normalization([datum])

    result = datum.loss_fn_inputs["advantages"].to_torch()
    # 3 action tokens, so divide by 3: 0.6/3 = 0.2
    expected = torch.tensor([0.0, 0.0, 0.2, 0.2, 0.2])
    assert torch.allclose(result, expected, atol=1e-6)


def test_sequence_normalization_no_action_tokens():
    """Edge case: if mask is all zeros, advantages should remain zero."""
    mask = torch.tensor([0.0, 0.0, 0.0])
    advantages = torch.tensor([0.0, 0.0, 0.0])

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor([2, 3, 4])),
            "logprobs": TensorData.from_torch(torch.zeros(3)),
            "advantages": TensorData.from_torch(advantages),
            "mask": TensorData.from_torch(mask),
        },
    )

    apply_sequence_normalization([datum])

    result = datum.loss_fn_inputs["advantages"].to_torch()
    assert torch.allclose(result, torch.zeros(3))
