"""
Problem loading from JSONL, CSV, HuggingFace datasets, or plain Python lists.

Each problem is a dict with at minimum a ``prompt`` key. An optional
``reference`` key holds the ground-truth answer used by reward functions.
Any extra keys are preserved and forwarded to the reward function.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


@dataclass
class Problem:
    prompt: str
    reference: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    # Convenience: allow ``problem["key"]`` access
    def __getitem__(self, key: str) -> Any:
        if key == "prompt":
            return self.prompt
        if key == "reference":
            return self.reference
        return self.meta[key]


def load_problems(
    source: str | Path | list[dict],
    *,
    prompt_field: str = "prompt",
    reference_field: str = "reference",
    split: str = "train",
    limit: int | None = None,
) -> list[Problem]:
    """Load problems from various sources.

    Args:
        source: One of:
            - path to a ``.jsonl`` file (one JSON object per line)
            - path to a ``.csv`` file (header row required)
            - a HuggingFace dataset identifier like ``"openai/gsm8k"``
            - a plain Python list of dicts
        prompt_field: Key that holds the prompt/question text.
        reference_field: Key that holds the reference answer.
        split: HF dataset split (only used for HF datasets).
        limit: Cap the number of problems loaded.

    Returns:
        A list of :class:`Problem` instances.
    """
    if isinstance(source, list):
        rows = source
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix == ".jsonl":
            rows = _load_jsonl(path)
        elif path.suffix == ".csv":
            rows = _load_csv(path)
        elif path.exists():
            raise ValueError(f"Unsupported file type: {path.suffix}")
        else:
            rows = _load_hf(str(source), split)
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    problems: list[Problem] = []
    for row in rows:
        prompt = row.get(prompt_field, "")
        if not prompt:
            continue
        reference = str(row.get(reference_field, ""))
        meta = {k: v for k, v in row.items() if k not in (prompt_field, reference_field)}
        problems.append(Problem(prompt=prompt, reference=reference, meta=meta))
        if limit and len(problems) >= limit:
            break

    logger.info("Loaded %d problems from %s", len(problems), source if not isinstance(source, list) else "list")
    return problems


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _load_hf(name: str, split: str) -> list[dict]:
    import datasets as hf_datasets
    ds = hf_datasets.load_dataset(name, split=split)
    return [dict(row) for row in ds]
