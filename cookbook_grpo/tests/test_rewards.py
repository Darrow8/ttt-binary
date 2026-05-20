"""Tests for the reward function."""

import pytest

from cookbook_grpo.rewards import compute_reward


class TestComputeReward:
    def test_correct_answer(self):
        response = "After careful analysis, the answer is \\boxed{866}."
        assert compute_reward(response, "866") == 1.0

    def test_correct_with_normalization(self):
        response = "\\boxed{3264.0000}"
        assert compute_reward(response, "3264") == 1.0

    def test_wrong_boxed_answer(self):
        response = "I think it's \\boxed{42}."
        assert compute_reward(response, "866") == 0.0

    def test_no_boxed_answer(self):
        response = "I'm not sure, maybe 866."
        assert compute_reward(response, "866") == 0.0

    def test_custom_reward_values(self):
        response = "\\boxed{866}"
        assert compute_reward(response, "866", reward_correct=2.0) == 2.0

        response2 = "\\boxed{42}"
        assert compute_reward(response2, "866", reward_wrong=0.05) == 0.05

        response3 = "no answer"
        assert compute_reward(response3, "866", reward_none=-0.1) == -0.1

    def test_empty_response(self):
        assert compute_reward("", "866") == 0.0

    def test_multiple_boxes_uses_last(self):
        response = "First try \\boxed{42}, actually \\boxed{866}"
        assert compute_reward(response, "866") == 1.0
