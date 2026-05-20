"""ProblemEnv implementation for TTT-Discover subproblems."""

from __future__ import annotations

from functools import partial

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    StepResult,
)
from tinker_cookbook.utils import logtree
from tinker_cookbook.utils.logtree_formatters import ConversationFormatter

from cookbook_grpo.parser import extract_boxed_answer, answers_match
from cookbook_grpo.rewards import compute_reward, REWARD_CORRECT, REWARD_WRONG_ANSWER, REWARD_NO_ANSWER


class SubproblemEnv(ProblemEnv):
    """Environment for a single TTT-Discover subproblem.

    Uses binary reward: 1.0 (correct) or 0.0 (wrong/no answer).
    """

    def __init__(
        self,
        problem: str,
        answer: str,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        prompt_suffix: str = " Put your final answer inside \\boxed{}.",
        reward_correct: float = REWARD_CORRECT,
        reward_wrong: float = REWARD_WRONG_ANSWER,
        reward_none: float = REWARD_NO_ANSWER,
    ):
        # format_coef=0 because we handle format reward inside the total reward
        super().__init__(renderer, convo_prefix, format_coef=0.0)
        self.problem = problem
        self.answer = answer
        self.prompt_suffix = prompt_suffix
        self.reward_correct = reward_correct
        self.reward_wrong = reward_wrong
        self.reward_none = reward_none

    def get_question(self) -> str:
        return self.problem + self.prompt_suffix

    def check_format(self, sample_str: str) -> bool:
        return extract_boxed_answer(sample_str) is not None

    def check_answer(self, sample_str: str) -> bool:
        predicted = extract_boxed_answer(sample_str)
        if predicted is None:
            return False
        return answers_match(predicted, self.answer)

    def get_reference_answer(self) -> str:
        return self.answer

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        """Custom step that uses the three-tier reward scheme."""
        convo = (self.convo_prefix or []) + [{"role": "user", "content": self.get_question()}]
        message, _parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)

        reward = compute_reward(
            content,
            self.answer,
            reward_correct=self.reward_correct,
            reward_wrong=self.reward_wrong,
            reward_none=self.reward_none,
        )

        predicted = extract_boxed_answer(content)
        correct = reward >= self.reward_correct
        has_boxed = predicted is not None

        with logtree.scope_header("Prompt"):
            logtree.log_formatter(ConversationFormatter(messages=convo))
        with logtree.scope_header("Policy Response"):
            logtree.log_formatter(ConversationFormatter(messages=[message]))
        with logtree.scope_header("Reward"):
            logtree.table_from_dict(
                {
                    "reference_answer": self.answer,
                    "extracted_answer": predicted or "(none)",
                    "has_boxed": has_boxed,
                    "correct": correct,
                    "reward": f"{reward:.3f}",
                },
                caption="Reward components",
            )

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "correct": float(correct),
                "format": float(has_boxed),
            },
        )


def make_subproblem_group_builder(
    problem: str,
    answer: str,
    renderer: renderers.Renderer,
    group_size: int,
    convo_prefix: list[renderers.Message] | None = None,
    prompt_suffix: str = " Put your final answer inside \\boxed{}.",
    reward_correct: float = REWARD_CORRECT,
    reward_wrong: float = REWARD_WRONG_ANSWER,
    reward_none: float = REWARD_NO_ANSWER,
) -> ProblemGroupBuilder:
    """Create a ProblemGroupBuilder for one subproblem."""
    return ProblemGroupBuilder(
        env_thunk=partial(
            SubproblemEnv,
            problem,
            answer,
            renderer,
            convo_prefix=convo_prefix,
            prompt_suffix=prompt_suffix,
            reward_correct=reward_correct,
            reward_wrong=reward_wrong,
            reward_none=reward_none,
        ),
        num_envs=group_size,
        dataset_name="subproblems",
    )
