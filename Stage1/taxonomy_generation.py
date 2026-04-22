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
import re
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


# ---------------------------------------------------------------------------
# Phase 2 — per-skill generation
# ---------------------------------------------------------------------------

# Pure answer-checking helpers (duplicated from distinct_llm_prompting so this
# module has no hard dependency on the openai package at import time).

_NUMERIC_ANSWER_RE = re.compile(r"^[+-]?\d+([.,]\d+)?(/\d+)?$")


def _is_numeric_answer(answer: str) -> bool:
    return bool(_NUMERIC_ANSWER_RE.match(answer))


def normalize_answer(answer: str) -> str:
    a = answer.strip().lower()
    a = re.sub(r"[\\${}]", "", a)
    a = re.sub(r"\s+", "", a)
    a = a.replace(",", ".")
    try:
        from fractions import Fraction

        if "/" in a and a.replace("/", "").replace("-", "").replace(".", "").isdigit():
            val = float(Fraction(a))
            a = f"{val:.10g}"
        elif a.replace(".", "").replace("-", "").isdigit():
            val = float(a)
            a = f"{val:.10g}"
    except (ValueError, ZeroDivisionError):
        pass
    return a


# solve_and_check_agreement is imported lazily to avoid pulling in the openai
# package at module load time (the installed version may be incompatible).
# At runtime the real function is loaded on first call; in tests it is
# replaced via monkeypatch before generate_for_skill is invoked.
solve_and_check_agreement = None  # type: ignore[assignment]


def _get_solve_fn():
    """Return solve_and_check_agreement, importing it lazily if needed."""
    global solve_and_check_agreement
    if solve_and_check_agreement is None:
        from Stage1.distinct_llm_prompting import (  # noqa: E402
            solve_and_check_agreement as _sca,
        )
        solve_and_check_agreement = _sca
    return solve_and_check_agreement


GENERATE_PROMPT = """\
You are designing one subproblem to help a student practice a specific
reasoning skill.

The end goal is mastery of this hard target problem (for context only --
do NOT generate a variant of the target):

{target}

Skill to test:
Name: {skill_name}
Description: {skill_description}
Example hint: {skill_hint}

Requirements:
- The subproblem tests THIS SKILL SPECIFICALLY, in isolation.
- A student who has mastered only this skill should be able to solve it.
  The problem must not rely on the other skills from the taxonomy.
- The answer MUST be a single number (integer or decimal). If a decimal,
  ask the solver to round to 4 decimal places.
- State the problem in 3-10 sentences.
- ALL math must be written in LaTeX using \\(...\\) for inline and
  \\[...\\] for display. No ASCII math ("x^2", "sqrt(5)", "sum from i=1 to n",
  etc.) -- use proper LaTeX.
- The problem statement MUST end with this exact sentence:
  "Put your final answer inside \\boxed{{}}."

Output format:
Begin your response with <problem> on its own line, then the full problem
statement, then </problem> on its own line. No other text before, between,
or after the tags.
"""


_PROBLEM_TAG_RE = re.compile(r"<problem>(.*?)</problem>", re.DOTALL)


def _parse_problem(raw: str) -> str:
    """Extract the <problem>...</problem> content. Return '' if tags missing."""
    m = _PROBLEM_TAG_RE.search(raw)
    if not m:
        return ""
    return m.group(1).strip()


def _generate_one_candidate(client, target: str, skill: Skill, _temperature: float = TEMPERATURE) -> str:
    """Call the generator once, return the raw problem text (possibly empty)."""
    prompt = GENERATE_PROMPT.format(
        target=target,
        skill_name=skill.name,
        skill_description=skill.description,
        skill_hint=skill.example_problem_hint,
    )
    resp = client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=_temperature,
    )
    raw = (resp.choices[0].message.content or "")
    return _parse_problem(raw)


def generate_for_skill(
    *,
    client,
    target: str,
    skill: Skill,
    n_target: int,
    n_samples: int,
    max_candidates: int,
    agree_low: float,
    agree_high: float,
) -> tuple[list[dict], list[dict], dict]:
    """Generate candidates for a single skill until n_target keeps or max_candidates attempts.

    Returns:
        (keeps, skips, stats) where stats is
        {"name": skill.name, "n_target", "n_passed", "n_attempted", "status"}.
    """
    keeps: list[dict] = []
    skips: list[dict] = []
    attempted = 0

    while len(keeps) < n_target and attempted < max_candidates:
        attempted += 1
        problem_text = _generate_one_candidate(client, target, skill)
        if not problem_text:
            skips.append({
                "skill": skill.name,
                "problem": "",
                "reason": "generator_no_tags_or_empty",
            })
            continue

        agreement, majority_ans, all_answers, all_solutions = _get_solve_fn()(
            client, GENERATOR_MODEL, problem_text, n_samples=n_samples,
        )

        numeric = _is_numeric_answer(normalize_answer(majority_ans))
        in_range = agree_low <= agreement <= agree_high
        kept = bool(majority_ans) and numeric and in_range

        record = {
            "skill": skill.name,
            "problem": problem_text,
            "ground_truth_answer": majority_ans,
            "agreement_rate": agreement,
            "all_answers": all_answers,
            "all_solutions": all_solutions,
            "n_samples": n_samples,
        }
        if kept:
            keeps.append(record)
        else:
            reason = (
                "empty_answer" if not majority_ans
                else "non_numeric" if not numeric
                else "out_of_window"
            )
            skips.append({**record, "reason": reason})

    status = "ok" if len(keeps) >= n_target else "capped"
    stats = {
        "name": skill.name,
        "n_target": n_target,
        "n_passed": len(keeps),
        "n_attempted": attempted,
        "status": status,
    }
    return keeps, skips, stats
