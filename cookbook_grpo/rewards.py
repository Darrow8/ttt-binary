"""Reward function for TTT-Discover subproblem training."""

from __future__ import annotations

from cookbook_grpo.parser import extract_boxed_answer, answers_match


# Default reward values matching the paper
REWARD_CORRECT = 1.0
REWARD_WRONG_ANSWER = 0.01
REWARD_NO_ANSWER = 0.0


def compute_reward(
    response: str,
    reference: str,
    *,
    reward_correct: float = REWARD_CORRECT,
    reward_wrong: float = REWARD_WRONG_ANSWER,
    reward_none: float = REWARD_NO_ANSWER,
) -> float:
    """Compute reward for a model response against a reference answer.

    Returns:
        reward_correct (1.0) if extracted answer matches reference
        reward_wrong (0.01) if a boxed answer exists but doesn't match
        reward_none (0.0) if no boxed answer can be extracted
    """
    predicted = extract_boxed_answer(response)
    if predicted is None:
        return reward_none
    if answers_match(predicted, reference):
        return reward_correct
    return reward_wrong
