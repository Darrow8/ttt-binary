"""Tests for the GRPO override module (std normalization + sequence-level loss)."""

import asyncio
from unittest.mock import AsyncMock, patch

import torch
import tinker
from tinker.types.tensor_data import TensorData

from cookbook_grpo.grpo_overrides import (
    grpo_compute_advantages,
    apply_sequence_normalization,
    grpo_incorporate_kl_penalty,
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


def test_kl_penalty_normalization():
    """KL penalty terms should be divided by action token count, same as advantages."""
    mask = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])  # 4 action tokens
    # Pre-KL advantages already normalized by 1/4
    pre_advantages = torch.tensor([0.0, 0.0, 0.25, 0.25, 0.25, 0.25])

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2, 3, 4, 5, 6]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor([2, 3, 4, 5, 6, 7])),
            "logprobs": TensorData.from_torch(torch.zeros(6)),
            "advantages": TensorData.from_torch(pre_advantages.clone()),
            "mask": TensorData.from_torch(mask),
        },
    )

    # Simulate the KL penalty delta that incorporate_kl_penalty would add
    kl_delta = torch.tensor([0.0, 0.0, 0.4, 0.4, 0.4, 0.4])

    async def fake_incorporate_kl(data_D, client, coef, discount):
        for d in data_D:
            adv = d.loss_fn_inputs["advantages"].to_torch()
            d.loss_fn_inputs["advantages"] = TensorData.from_torch(adv + kl_delta)
        return {"kl_policy_base": 0.1}

    from cookbook_grpo.grpo_overrides import grpo_incorporate_kl_penalty

    # Patch the import inside the function
    with patch("tinker_cookbook.rl.metrics.incorporate_kl_penalty", new=fake_incorporate_kl):
        result = asyncio.run(grpo_incorporate_kl_penalty(
            [datum], None, 0.04, 0.0
        ))

    final_advantages = datum.loss_fn_inputs["advantages"].to_torch()
    # KL delta of 0.4 per token should be divided by 4 action tokens = 0.1 per token
    # Final = pre_advantages + kl_delta/4 = 0.25 + 0.1 = 0.35 for action tokens
    expected = torch.tensor([0.0, 0.0, 0.35, 0.35, 0.35, 0.35])
    assert torch.allclose(final_advantages, expected, atol=1e-6)
