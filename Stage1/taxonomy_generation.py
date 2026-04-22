"""Taxonomy-first subproblem generation.

Stage 1 variant that:
  1. Decomposes a hard target problem into 10 distinct reasoning skills.
  2. Generates 10 agreement-window-passing subproblems per skill.

Coexists with Stage1/distinct_llm_prompting.py; does not replace it.

Non-negotiable constraints (see design spec):
- Model hardcoded to openai/gpt-oss-120b-maas for all three call types.
- No max_tokens on any completion call.
- No client-side timeouts.
- All problem text + reasoning traces in LaTeX.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

# Ensure repo root is on sys.path so this script can import sibling packages
# (Stage1 and downstream stages) regardless of how it was invoked.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Non-negotiable constants — do NOT parameterize on CLI.
# ---------------------------------------------------------------------------

GENERATOR_MODEL = "openai/gpt-oss-120b-maas"
TEMPERATURE = 0.7

# ---------------------------------------------------------------------------
# Defaults for CLI-tunable knobs.
# ---------------------------------------------------------------------------

N_SKILLS_DEFAULT = 10
PROBLEMS_PER_SKILL_DEFAULT = 10
MAX_CANDIDATES_PER_SKILL_DEFAULT = 100
N_SAMPLES_DEFAULT = 10
AGREE_LOW_DEFAULT = 0.60
AGREE_HIGH_DEFAULT = 0.80
GEN_WORKERS_DEFAULT = 4
MAX_WORKERS_DEFAULT = 16


@dataclass
class Skill:
    name: str
    description: str
    example_problem_hint: str


# ---------------------------------------------------------------------------
# Phase 1 — decomposition
# ---------------------------------------------------------------------------

DECOMPOSE_PROMPT = """\
You are designing a curriculum to help a student learn to solve a hard
target problem by mastering its component reasoning skills first.

Target problem:
{target}

Decompose this target into EXACTLY {n_skills} distinct reasoning skills. Each skill:
- Must be a component of the target -- a specific reasoning step or tool,
  not a rephrasing of the whole problem.
- Must be DISTINCT from the others: no two skills should test the same
  underlying reasoning.
- Must be testable in a self-contained subproblem that can be solved
  without requiring the other skills.
- Should fall roughly in difficulty order, prerequisite to advanced.

Respond with JSON only, no prose, exactly this shape:
{{
  "skills": [
    {{
      "name": "Short skill name (3-10 words)",
      "description": "1-2 sentences explaining what the skill is.",
      "example_problem_hint": "One sentence sketching what a problem testing this skill looks like."
    }}
  ]
}}

There must be exactly {n_skills} entries in the skills array.
"""


def _extract_json_block(text: str) -> str:
    """Return the outermost JSON object found in `text`, or raise ValueError."""
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found in response")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError("unbalanced braces in response")


def decompose_target(
    client,
    target: str,
    *,
    n_skills: int = N_SKILLS_DEFAULT,
    max_retries: int = 3,
) -> list[Skill]:
    """Call the generator model once to decompose `target` into `n_skills` skills.

    Retries up to `max_retries` times on parse failure or wrong skill count.
    Raises ValueError after retries exhausted.
    """
    prompt = DECOMPOSE_PROMPT.format(target=target, n_skills=n_skills)

    last_err: Exception | None = None
    for attempt in range(max_retries):
        resp = client.chat.completions.create(
            model=GENERATOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
        )
        raw = (resp.choices[0].message.content or "").strip()
        try:
            block = _extract_json_block(raw)
            parsed = json.loads(block)
            skills_list = parsed.get("skills")
            if not isinstance(skills_list, list):
                raise ValueError("'skills' key missing or not a list")
            if len(skills_list) != n_skills:
                raise ValueError(f"expected {n_skills} skills, got {len(skills_list)}")
            _required = ("name", "description", "example_problem_hint")
            for i, entry in enumerate(skills_list):
                if not isinstance(entry, dict):
                    raise ValueError(f"skill[{i}] is not an object")
                for k in _required:
                    v = entry.get(k)
                    if not isinstance(v, str) or not v.strip():
                        raise ValueError(f"skill[{i}].{k} must be a non-empty string")
            return [
                Skill(
                    name=entry["name"],
                    description=entry["description"],
                    example_problem_hint=entry["example_problem_hint"],
                )
                for entry in skills_list
            ]
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
            last_err = e
            continue

    raise ValueError(f"failed to parse skills after {max_retries} attempts: {last_err}")


def save_skills(
    path: str,
    skills: list[Skill],
    *,
    target_path: str,
    target_hash: str,
    model: str,
) -> None:
    """Write skills + provenance to `path` atomically."""
    data = {
        "target_problem_path": target_path,
        "target_problem_hash": target_hash,
        "generator_model": model,
        "decomposed_at": datetime.now(timezone.utc).isoformat(),
        "skills": [asdict(s) for s in skills],
    }
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_skills(path: str) -> list[Skill] | None:
    """Load skills from `path`, or return None if the file doesn't exist or is malformed."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return [Skill(**entry) for entry in data.get("skills", [])]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def target_text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
