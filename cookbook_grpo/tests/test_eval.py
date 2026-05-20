"""Tests for evaluation logic."""

import pytest

from cookbook_grpo.parser import extract_boxed_answer, normalize_answer, answers_match
from cookbook_grpo.rewards import compute_reward


class TestEvalAccuracy:
    """Test that evaluation counting logic is correct."""

    def test_accuracy_counting(self):
        responses = [
            "\\boxed{866}",     # correct
            "\\boxed{42}",      # wrong
            "\\boxed{866}",     # correct
            "no answer",        # no boxed
            "\\boxed{866.0}",   # correct (normalization)
        ]
        reference = "866"

        correct = sum(1 for r in responses if compute_reward(r, reference) >= 1.0)
        assert correct == 3

        accuracy = correct / len(responses)
        assert accuracy == pytest.approx(0.6)

    def test_answer_frequency_table(self):
        from collections import Counter

        responses = [
            "\\boxed{866}",
            "\\boxed{42}",
            "\\boxed{866}",
            "\\boxed{100}",
            "\\boxed{866}",
            "no answer",
        ]

        answers = []
        for r in responses:
            pred = extract_boxed_answer(r)
            if pred is not None:
                answers.append(normalize_answer(pred))

        freq = Counter(answers)
        assert freq["866"] == 3
        assert freq["42"] == 1
        assert freq["100"] == 1
        assert len(answers) == 5  # one had no answer

    def test_reward_distribution(self):
        responses = [
            "\\boxed{866}",     # 1.0
            "\\boxed{42}",      # 0.01
            "no boxed",         # 0.0
            "\\boxed{866}",     # 1.0
        ]
        reference = "866"
        rewards = [compute_reward(r, reference) for r in responses]

        assert rewards == [1.0, 0.01, 0.0, 1.0]
        assert sum(1 for r in rewards if r == 1.0) == 2
        assert sum(1 for r in rewards if r == 0.01) == 1
        assert sum(1 for r in rewards if r == 0.0) == 1
