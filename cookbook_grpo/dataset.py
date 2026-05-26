"""Dataset loading and RLDataset implementation for subproblems."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chz

from tinker_cookbook import model_info, renderers
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from cookbook_grpo.env import make_subproblem_group_builder
from cookbook_grpo.rewards import REWARD_CORRECT, REWARD_WRONG_ANSWER, REWARD_NO_ANSWER


@dataclass
class SubproblemRecord:
    """A single subproblem with its pseudo-label."""

    problem: str
    answer: str
    id: str | None = None
    agreement_rate: float | None = None
    metadata: dict[str, Any] | None = None
    reward_weight: float = 1.0  # Multiplier on reward_correct; set by inverse-frequency weighting


def load_subproblems(path: str | Path) -> list[SubproblemRecord]:
    """Load subproblems from a JSONL file.

    Each line must have at minimum "prompt" and "reference" fields
    (or "problem" and "answer" as alternatives).

    Raises ValueError with line number for malformed rows.
    """
    path = Path(path)
    records: list[SubproblemRecord] = []

    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Malformed JSON at {path}:{line_num}: {e}"
                ) from e

            problem = row.get("prompt") or row.get("problem")
            answer = row.get("reference") or row.get("answer")

            if not problem:
                raise ValueError(
                    f"Missing 'prompt' or 'problem' field at {path}:{line_num}"
                )
            if not answer:
                raise ValueError(
                    f"Missing 'reference' or 'answer' field at {path}:{line_num}"
                )

            records.append(SubproblemRecord(
                problem=problem,
                answer=str(answer),
                id=row.get("id"),
                agreement_rate=row.get("agreement_rate"),
                metadata=row.get("metadata"),
            ))

    return records


class SubproblemDataset(RLDataset):
    """RLDataset that cycles through subproblems for multiple epochs."""

    def __init__(
        self,
        records: list[SubproblemRecord],
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        num_epochs: int = 50,
        convo_prefix: list[renderers.Message] | None = None,
        prompt_suffix: str = " Put your final answer inside \\boxed{}.",
        reward_correct: float = REWARD_CORRECT,
        reward_wrong: float = REWARD_WRONG_ANSWER,
        reward_none: float = REWARD_NO_ANSWER,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.records = records
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.num_epochs = num_epochs
        self.convo_prefix = convo_prefix
        self.prompt_suffix = prompt_suffix
        self.reward_correct = reward_correct
        self.reward_wrong = reward_wrong
        self.reward_none = reward_none

        # Build the full epoch-expanded list of records
        rng = random.Random(seed)
        self._expanded: list[SubproblemRecord] = []
        for _ in range(num_epochs):
            epoch_records = list(records)
            if shuffle:
                rng.shuffle(epoch_records)
            self._expanded.extend(epoch_records)

        self._num_batches = math.ceil(len(self._expanded) / batch_size)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self._expanded))
        return [
            make_subproblem_group_builder(
                problem=rec.problem,
                answer=rec.answer,
                renderer=self.renderer,
                group_size=self.group_size,
                convo_prefix=self.convo_prefix,
                prompt_suffix=self.prompt_suffix,
                reward_correct=self.reward_correct * rec.reward_weight,
                reward_wrong=self.reward_wrong,
                reward_none=self.reward_none,
            )
            for rec in self._expanded[start:end]
        ]

    def __len__(self) -> int:
        return self._num_batches


@chz.chz
class SubproblemDatasetBuilder(RLDatasetBuilder):
    """Builder for subproblem dataset, compatible with tinker_cookbook.rl.train.Config."""

    data_path: str
    batch_size: int = 25
    group_size: int = 16
    num_epochs: int = 50
    model_name: str = "openai/gpt-oss-120b"
    renderer_name: str | None = None
    prompt_suffix: str = " Put your final answer inside \\boxed{}."
    system_prompt: str | None = None
    reward_correct: float = REWARD_CORRECT
    reward_wrong: float = REWARD_WRONG_ANSWER
    reward_none: float = REWARD_NO_ANSWER
    shuffle: bool = True
    seed: int = 42

    # Optional eval path — if provided, builds a test dataset from a separate file
    eval_data_path: str | None = None

    # If True, multiply reward_correct per-problem by the class-balanced inverse-frequency
    # weight of its reference answer. Discourages reward-hacking by always emitting the
    # most common answer in the dataset (e.g. "32" appears 18/50 times in conics-reproduce-50).
    # Formula: weight[ans] = N / (K * count[ans]) where N = total records, K = unique answers.
    weight_by_inverse_frequency: bool = False

    async def __call__(self) -> tuple[SubproblemDataset, SubproblemDataset | None]:
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name
        )
        tokenizer = get_tokenizer(self.model_name)
        renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

        convo_prefix: list[renderers.Message] | None = None
        if self.system_prompt:
            convo_prefix = [{"role": "system", "content": self.system_prompt}]

        records = load_subproblems(self.data_path)

        if self.weight_by_inverse_frequency:
            from collections import Counter
            answer_counts = Counter(rec.answer for rec in records)
            n_records = len(records)
            n_classes = len(answer_counts)
            for rec in records:
                rec.reward_weight = n_records / (n_classes * answer_counts[rec.answer])
            import logging
            logging.getLogger(__name__).info(
                "Frequency-weighted rewards enabled: %d unique answers across %d records. "
                "Most-common='%s' (%d×, weight=%.3f), least-common='%s' (%d×, weight=%.3f).",
                n_classes, n_records,
                answer_counts.most_common(1)[0][0], answer_counts.most_common(1)[0][1],
                n_records / (n_classes * answer_counts.most_common(1)[0][1]),
                answer_counts.most_common()[-1][0], answer_counts.most_common()[-1][1],
                n_records / (n_classes * answer_counts.most_common()[-1][1]),
            )

        train_dataset = SubproblemDataset(
            records=records,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            num_epochs=self.num_epochs,
            convo_prefix=convo_prefix,
            prompt_suffix=self.prompt_suffix,
            reward_correct=self.reward_correct,
            reward_wrong=self.reward_wrong,
            reward_none=self.reward_none,
            shuffle=self.shuffle,
            seed=self.seed,
        )

        test_dataset = None
        if self.eval_data_path:
            eval_records = load_subproblems(self.eval_data_path)
            test_dataset = SubproblemDataset(
                records=eval_records,
                batch_size=len(eval_records),
                group_size=1,
                renderer=renderer,
                num_epochs=1,
                convo_prefix=convo_prefix,
                prompt_suffix=self.prompt_suffix,
                reward_correct=self.reward_correct,
                reward_wrong=self.reward_wrong,
                reward_none=self.reward_none,
                shuffle=False,
                seed=self.seed,
            )

        return train_dataset, test_dataset
