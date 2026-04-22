# Taxonomy-First Subproblem Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new Stage 1 variant script, `Stage1/taxonomy_generation.py`, that decomposes a hard target problem into 10 reasoning skills and generates 10 subproblems per skill (target: 100 subproblems). Existing `Stage1/distinct_llm_prompting.py` is untouched; the two pipelines coexist for ablation.

**Architecture:** Two-phase. Phase 1 = one LLM call → JSON list of 10 skills cached to `runs/<id>/skills.json`. Phase 2 = per-skill generate-until-10-pass loop, reusing `solve_and_check_agreement`, `extract_answer`, `_save_atomic`, and the Vertex client builder from `distinct_llm_prompting.py`. Outputs land in `runs/<id>/stage1_taxonomy/<ts>/{keeps,skips,per_skill_stats}.json`.

**Tech Stack:** Python stdlib + `openai` + `tenacity` (already in `requirements.txt`). No new deps.

**Reference:** spec at `docs/superpowers/specs/2026-04-21-taxonomy-generation-design.md`.

---

## File Structure

**Create:**
- `Stage1/__init__.py` — empty, makes `Stage1` importable as a package so the test file can `from Stage1.distinct_llm_prompting import ...`.
- `Stage1/taxonomy_generation.py` — the new script. ~300 lines.
- `tests/__init__.py` — empty (if not present already).
- `tests/test_taxonomy_generation.py` — unit tests with mocked LLM client. ~200 lines.

**Modify:** none. The existing `Stage1/distinct_llm_prompting.py` is imported from, not changed.

**Non-negotiable constants** (in `taxonomy_generation.py`, enforced by tests):
```python
GENERATOR_MODEL = "openai/gpt-oss-120b-maas"   # no CLI override
TEMPERATURE = 0.7                               # same as distinct_llm_prompting
N_SKILLS_DEFAULT = 10
PROBLEMS_PER_SKILL_DEFAULT = 10
MAX_CANDIDATES_PER_SKILL_DEFAULT = 100
N_SAMPLES_DEFAULT = 10
```

Chat completion calls pass `model`, `messages`, `temperature` only — no `max_tokens`, no `timeout`.

---

### Task 1: Scaffold the module and `Skill` dataclass

**Files:**
- Create: `Stage1/__init__.py` (empty)
- Create: `tests/__init__.py` (empty if missing)
- Create: `Stage1/taxonomy_generation.py` (header + `Skill` dataclass only for now)
- Create: `tests/test_taxonomy_generation.py` (imports + first test)

- [ ] **Step 1.1: Verify pytest is available**

Run: `python -c "import pytest; print(pytest.__version__)"`

If it fails, add `pytest>=7` to `requirements.txt` and `pip install pytest`.

- [ ] **Step 1.2: Create `Stage1/__init__.py` and `tests/__init__.py`**

Both are empty files. They make `Stage1` and `tests` importable as packages so pytest collection works from the repo root.

- [ ] **Step 1.3: Write a failing smoke test**

Create `tests/test_taxonomy_generation.py`:

```python
"""Unit tests for Stage1.taxonomy_generation."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from Stage1.taxonomy_generation import (
    GENERATOR_MODEL,
    Skill,
)


def test_generator_model_hardcoded():
    """The model constant must be the 120b Vertex MaaS model, not overridable."""
    assert GENERATOR_MODEL == "openai/gpt-oss-120b-maas"


class TestSkill:
    def test_fields(self):
        s = Skill(
            name="Bezout intersection count",
            description="Counting degrees of intersection on projective varieties.",
            example_problem_hint="Compute the intersection number of two plane curves.",
        )
        assert s.name == "Bezout intersection count"
        assert asdict(s) == {
            "name": "Bezout intersection count",
            "description": "Counting degrees of intersection on projective varieties.",
            "example_problem_hint": "Compute the intersection number of two plane curves.",
        }
```

- [ ] **Step 1.4: Run — expect fail**

Run: `pytest tests/test_taxonomy_generation.py -v`

Expected: `ModuleNotFoundError: No module named 'Stage1.taxonomy_generation'`. If any test passes, stop and investigate.

- [ ] **Step 1.5: Create `Stage1/taxonomy_generation.py` with header + `Skill`**

```python
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
```

- [ ] **Step 1.6: Verify the smoke tests pass**

Run: `pytest tests/test_taxonomy_generation.py -v`

Expected: both tests pass.

- [ ] **Step 1.7: Commit**

```bash
git add Stage1/__init__.py tests/__init__.py Stage1/taxonomy_generation.py tests/test_taxonomy_generation.py
git commit -m "scaffold Stage1/taxonomy_generation.py with Skill dataclass

Skeleton only: module header, non-negotiable constants, Skill
dataclass. Decomposition and per-skill generation added in subsequent
tasks. Design: docs/superpowers/specs/2026-04-21-taxonomy-generation-design.md"
```

---

### Task 2: Phase 1 — decompose target into 10 skills

**Files:**
- Modify: `Stage1/taxonomy_generation.py` (add `DECOMPOSE_PROMPT`, `decompose_target`, `save_skills`, `load_skills`)
- Modify: `tests/test_taxonomy_generation.py` (add `TestDecompose` class)

- [ ] **Step 2.1: Write failing tests**

Append to `tests/test_taxonomy_generation.py`:

```python
import json
from unittest.mock import MagicMock

from Stage1.taxonomy_generation import (
    DECOMPOSE_PROMPT,
    decompose_target,
    load_skills,
    save_skills,
)


def _mock_client_returning(content: str) -> MagicMock:
    """Build a mock OpenAI client whose chat.completions.create returns `content`."""
    client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    client.chat.completions.create.return_value = resp
    return client


_VALID_SKILLS_JSON = json.dumps({
    "skills": [
        {"name": f"Skill {i}",
         "description": f"Description for skill {i}.",
         "example_problem_hint": f"Hint for skill {i}."}
        for i in range(10)
    ]
})


class TestDecompose:
    def test_valid_json_returns_ten_skills(self):
        client = _mock_client_returning(_VALID_SKILLS_JSON)
        skills = decompose_target(client, "the target problem text")
        assert len(skills) == 10
        assert all(isinstance(s, Skill) for s in skills)
        assert skills[0].name == "Skill 0"

    def test_uses_hardcoded_model(self):
        client = _mock_client_returning(_VALID_SKILLS_JSON)
        decompose_target(client, "target")
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == GENERATOR_MODEL

    def test_no_max_tokens_no_timeout(self):
        client = _mock_client_returning(_VALID_SKILLS_JSON)
        decompose_target(client, "target")
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert "max_tokens" not in call_kwargs
        assert "timeout" not in call_kwargs

    def test_includes_target_in_prompt(self):
        client = _mock_client_returning(_VALID_SKILLS_JSON)
        decompose_target(client, "UNIQUE TARGET STRING")
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        assert "UNIQUE TARGET STRING" in user_msg

    def test_parse_failure_retries_then_raises(self):
        client = _mock_client_returning("not json at all")
        with pytest.raises(ValueError, match="parse"):
            decompose_target(client, "target", max_retries=3)
        # 3 retries = 3 calls total
        assert client.chat.completions.create.call_count == 3

    def test_wrong_skill_count_raises(self):
        bad = json.dumps({"skills": [{"name": "only one", "description": "x", "example_problem_hint": "y"}]})
        client = _mock_client_returning(bad)
        with pytest.raises(ValueError, match="expected 10 skills"):
            decompose_target(client, "target", max_retries=1)


class TestSkillsPersistence:
    def test_round_trip(self, tmp_path):
        skills = [
            Skill(name=f"S{i}", description=f"D{i}", example_problem_hint=f"H{i}")
            for i in range(10)
        ]
        path = tmp_path / "skills.json"
        save_skills(str(path), skills, target_path="target.txt",
                    target_hash="abc123", model=GENERATOR_MODEL)
        loaded = load_skills(str(path))
        assert loaded is not None
        assert len(loaded) == 10
        assert loaded[0].name == "S0"

    def test_load_missing_returns_none(self, tmp_path):
        assert load_skills(str(tmp_path / "nope.json")) is None
```

- [ ] **Step 2.2: Run — expect fail**

Run: `pytest tests/test_taxonomy_generation.py -v`

Expected: `ImportError` on `decompose_target`, `save_skills`, etc.

- [ ] **Step 2.3: Implement Phase 1 in `Stage1/taxonomy_generation.py`**

Append:

```python
import hashlib
import re

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
    # Common case: the model obeys and emits only JSON.
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    # Fallback: find the first { and the matching closing }.
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
            return [
                Skill(
                    name=entry["name"],
                    description=entry["description"],
                    example_problem_hint=entry["example_problem_hint"],
                )
                for entry in skills_list
            ]
        except (ValueError, KeyError, json.JSONDecodeError) as e:
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
```

- [ ] **Step 2.4: Run tests — expect all green**

Run: `pytest tests/test_taxonomy_generation.py -v`

Expected: every test in `TestDecompose` and `TestSkillsPersistence` passes, plus the Task 1 tests.

- [ ] **Step 2.5: Commit**

```bash
git add Stage1/taxonomy_generation.py tests/test_taxonomy_generation.py
git commit -m "taxonomy-gen: Phase 1 decomposition (cached to skills.json)

decompose_target() makes one LLM call, parses the JSON skills list,
retries on parse failure, and returns Skill objects. save_skills /
load_skills persist the decomposition with target hash and model
metadata for reproducibility."
```

---

### Task 3: Phase 2 — per-skill generate-until loop

**Files:**
- Modify: `Stage1/taxonomy_generation.py` (add `GENERATE_PROMPT`, `_parse_problem`, `generate_for_skill`)
- Modify: `tests/test_taxonomy_generation.py` (add `TestPerSkillGeneration` class)

- [ ] **Step 3.1: Write failing tests**

Append to `tests/test_taxonomy_generation.py`:

```python
from Stage1.taxonomy_generation import (
    GENERATE_PROMPT,
    _parse_problem,
    generate_for_skill,
)


class TestParseProblem:
    def test_extracts_between_tags(self):
        raw = "<problem>\nFind the value of \\(x\\). Put your final answer inside \\boxed{}.\n</problem>"
        assert _parse_problem(raw) == "Find the value of \\(x\\). Put your final answer inside \\boxed{}."

    def test_returns_empty_when_no_tags(self):
        assert _parse_problem("no tags here") == ""

    def test_strips_surrounding_whitespace(self):
        raw = "\n\n<problem>   hi   </problem>\n"
        assert _parse_problem(raw) == "hi"


class TestGenerateForSkill:
    def test_stops_at_n_target_keeps(self, monkeypatch):
        # Mock: generator always returns a valid problem; solver always agrees at 0.70.
        from Stage1 import taxonomy_generation as tg

        gen_calls = {"n": 0}

        def fake_gen_candidate(client, target, skill, _temperature=TEMPERATURE):
            gen_calls["n"] += 1
            return f"Find n. Put your final answer inside \\boxed{{}}. Candidate {gen_calls['n']}."

        def fake_solve(client, model, problem_text, n_samples, pool=None):
            # Return (agreement, majority, all_answers, all_solutions)
            return (0.70, "42", ["42"] * n_samples, ["reasoning"] * n_samples)

        monkeypatch.setattr(tg, "_generate_one_candidate", fake_gen_candidate)
        monkeypatch.setattr(tg, "solve_and_check_agreement", fake_solve)

        skill = Skill("s", "d", "h")
        keeps, skips, stats = generate_for_skill(
            client=MagicMock(),
            target="target",
            skill=skill,
            n_target=3,
            n_samples=5,
            max_candidates=50,
            agree_low=0.60,
            agree_high=0.80,
        )
        assert len(keeps) == 3
        assert stats["n_passed"] == 3
        assert stats["n_attempted"] == 3  # every candidate was a keep
        assert stats["status"] == "ok"

    def test_caps_at_max_candidates(self, monkeypatch):
        from Stage1 import taxonomy_generation as tg

        def fake_gen(client, target, skill, _temperature=TEMPERATURE):
            return "Find n. \\boxed{}"

        def fake_solve(client, model, problem_text, n_samples, pool=None):
            # All candidates fail the agreement window (too high -> skip)
            return (0.95, "42", ["42"] * n_samples, ["r"] * n_samples)

        monkeypatch.setattr(tg, "_generate_one_candidate", fake_gen)
        monkeypatch.setattr(tg, "solve_and_check_agreement", fake_solve)

        skill = Skill("s", "d", "h")
        keeps, skips, stats = generate_for_skill(
            client=MagicMock(),
            target="target",
            skill=skill,
            n_target=10,
            n_samples=5,
            max_candidates=7,
            agree_low=0.60,
            agree_high=0.80,
        )
        assert len(keeps) == 0
        assert stats["n_attempted"] == 7
        assert stats["status"] == "capped"

    def test_rejects_non_numeric(self, monkeypatch):
        from Stage1 import taxonomy_generation as tg

        def fake_gen(client, target, skill, _temperature=TEMPERATURE):
            return "Find n. \\boxed{}"

        def fake_solve(client, model, problem_text, n_samples, pool=None):
            return (0.70, "does not exist", ["does not exist"] * n_samples, ["r"] * n_samples)

        monkeypatch.setattr(tg, "_generate_one_candidate", fake_gen)
        monkeypatch.setattr(tg, "solve_and_check_agreement", fake_solve)

        skill = Skill("s", "d", "h")
        keeps, skips, stats = generate_for_skill(
            client=MagicMock(),
            target="target",
            skill=skill,
            n_target=5,
            n_samples=5,
            max_candidates=3,
            agree_low=0.60,
            agree_high=0.80,
        )
        assert len(keeps) == 0
        assert stats["status"] == "capped"
```

- [ ] **Step 3.2: Run — expect failing imports**

Run: `pytest tests/test_taxonomy_generation.py -v`

Expected: `ImportError` on `GENERATE_PROMPT`, `_parse_problem`, `generate_for_skill`.

- [ ] **Step 3.3: Implement Phase 2 in `taxonomy_generation.py`**

Append:

```python
# ---------------------------------------------------------------------------
# Phase 2 — per-skill generation
# ---------------------------------------------------------------------------

from Stage1.distinct_llm_prompting import (  # noqa: E402
    _is_numeric_answer,
    normalize_answer,
    solve_and_check_agreement,
)


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

        agreement, majority_ans, all_answers, all_solutions = solve_and_check_agreement(
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
```

- [ ] **Step 3.4: Run tests — expect all green**

Run: `pytest tests/test_taxonomy_generation.py -v`

Expected: all `TestParseProblem` and `TestGenerateForSkill` tests pass, plus earlier tests.

- [ ] **Step 3.5: Commit**

```bash
git add Stage1/taxonomy_generation.py tests/test_taxonomy_generation.py
git commit -m "taxonomy-gen: Phase 2 per-skill generate-until loop

generate_for_skill() pipelines candidates for one skill: generate,
solve n_samples times, check agreement window + numeric. Stops at
n_target keeps or max_candidates attempts. Reuses the solver and
answer-extraction infrastructure from distinct_llm_prompting.py."
```

---

### Task 4: Orchestration, output writing, CLI

**Files:**
- Modify: `Stage1/taxonomy_generation.py` (add `build_taxonomy_dataset`, `main`)
- Modify: `tests/test_taxonomy_generation.py` (add `TestBuildTaxonomyDataset` class)

- [ ] **Step 4.1: Write failing tests**

Append:

```python
class TestBuildTaxonomyDataset:
    def test_end_to_end_writes_expected_files(self, tmp_path, monkeypatch):
        from Stage1 import taxonomy_generation as tg

        skills_payload = json.dumps({
            "skills": [
                {"name": f"Skill {i}", "description": f"desc {i}", "example_problem_hint": f"hint {i}"}
                for i in range(10)
            ]
        })

        def fake_decompose(client, target, *, n_skills=10, max_retries=3):
            data = json.loads(skills_payload)
            return [Skill(**e) for e in data["skills"]]

        def fake_generate_for_skill(*, client, target, skill, n_target, n_samples, max_candidates, agree_low, agree_high):
            keeps = [
                {
                    "skill": skill.name,
                    "problem": f"Problem {i} for {skill.name}. \\boxed{{}}",
                    "ground_truth_answer": str(i),
                    "agreement_rate": 0.70,
                    "all_answers": [str(i)] * n_samples,
                    "all_solutions": [f"reasoning {i}"] * n_samples,
                    "n_samples": n_samples,
                }
                for i in range(n_target)
            ]
            stats = {
                "name": skill.name,
                "n_target": n_target,
                "n_passed": len(keeps),
                "n_attempted": n_target,
                "status": "ok",
            }
            return keeps, [], stats

        monkeypatch.setattr(tg, "decompose_target", fake_decompose)
        monkeypatch.setattr(tg, "generate_for_skill", fake_generate_for_skill)
        monkeypatch.setattr(tg, "get_client", lambda: (MagicMock(), GENERATOR_MODEL))

        out_dir = tmp_path / "run1"
        skills_path = tmp_path / "skills.json"

        tg.build_taxonomy_dataset(
            target_text="TARGET PROBLEM BODY",
            target_path="data/target-problems/fake.txt",
            out_dir=str(out_dir),
            skills_path=str(skills_path),
            n_skills=10,
            problems_per_skill=3,
            max_candidates_per_skill=20,
            n_samples=5,
            agree_low=0.60,
            agree_high=0.80,
        )

        # Files exist
        assert (out_dir / "keeps.json").exists()
        assert (out_dir / "skips.json").exists()
        assert (out_dir / "per_skill_stats.json").exists()
        assert skills_path.exists()

        keeps = json.load(open(out_dir / "keeps.json"))
        assert keeps["n_problems"] == 30
        assert keeps["generator_model"] == GENERATOR_MODEL
        assert keeps["solve_model"] == GENERATOR_MODEL
        assert all("skill" in p for p in keeps["problems"])

        stats = json.load(open(out_dir / "per_skill_stats.json"))
        assert stats["total_passed"] == 30
        assert stats["total_target"] == 30
        assert len(stats["skills"]) == 10

    def test_reuses_cached_skills(self, tmp_path, monkeypatch):
        """If skills.json exists, decompose_target is NOT called."""
        from Stage1 import taxonomy_generation as tg

        # Pre-seed skills.json
        skills = [Skill(f"S{i}", f"d{i}", f"h{i}") for i in range(10)]
        skills_path = tmp_path / "skills.json"
        tg.save_skills(str(skills_path), skills,
                       target_path="data/target-problems/fake.txt",
                       target_hash=tg.target_text_hash("TARGET"),
                       model=GENERATOR_MODEL)

        decompose_was_called = {"flag": False}

        def fake_decompose(*a, **kw):
            decompose_was_called["flag"] = True
            raise AssertionError("should not have been called")

        def fake_generate_for_skill(*, skill, n_target, **_kw):
            return ([], [], {"name": skill.name, "n_target": n_target,
                             "n_passed": 0, "n_attempted": 0, "status": "capped"})

        monkeypatch.setattr(tg, "decompose_target", fake_decompose)
        monkeypatch.setattr(tg, "generate_for_skill", fake_generate_for_skill)
        monkeypatch.setattr(tg, "get_client", lambda: (MagicMock(), GENERATOR_MODEL))

        tg.build_taxonomy_dataset(
            target_text="TARGET",
            target_path="data/target-problems/fake.txt",
            out_dir=str(tmp_path / "run1"),
            skills_path=str(skills_path),
            n_skills=10,
            problems_per_skill=1,
            max_candidates_per_skill=1,
            n_samples=1,
            agree_low=0.60,
            agree_high=0.80,
        )

        assert decompose_was_called["flag"] is False
```

- [ ] **Step 4.2: Run — expect failing imports**

Run: `pytest tests/test_taxonomy_generation.py -v`

Expected: `ImportError` on `build_taxonomy_dataset`.

- [ ] **Step 4.3: Implement orchestration in `taxonomy_generation.py`**

Append:

```python
from Stage1.distinct_llm_prompting import get_client, load_problem_from_txt  # noqa: E402


def build_taxonomy_dataset(
    *,
    target_text: str,
    target_path: str,
    out_dir: str,
    skills_path: str,
    n_skills: int,
    problems_per_skill: int,
    max_candidates_per_skill: int,
    n_samples: int,
    agree_low: float,
    agree_high: float,
) -> None:
    """Orchestrate Phase 1 + Phase 2. Write keeps/skips/stats to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    client, _ = get_client()

    # Phase 1 — decomposition (cached on disk).
    skills = load_skills(skills_path)
    if skills is None or len(skills) != n_skills:
        print(f"=== Taxonomy decomposition ===")
        print(f"Model:   {GENERATOR_MODEL}")
        print(f"Target:  {target_path} ({len(target_text)} chars)")
        print(f"Decomposing into {n_skills} skills...")
        skills = decompose_target(client, target_text, n_skills=n_skills)
        save_skills(
            skills_path, skills,
            target_path=target_path,
            target_hash=target_text_hash(target_text),
            model=GENERATOR_MODEL,
        )
        print(f"  {len(skills)} skills written to {skills_path}")
    else:
        print(f"Reusing cached skills from {skills_path}")

    # Phase 2 — per-skill generate-until.
    all_keeps: list[dict] = []
    all_skips: list[dict] = []
    all_stats: list[dict] = []

    keeps_path = os.path.join(out_dir, "keeps.json")
    skips_path = os.path.join(out_dir, "skips.json")
    stats_path = os.path.join(out_dir, "per_skill_stats.json")

    print(f"\n=== Per-skill generation ===")
    for i, skill in enumerate(skills, start=1):
        print(f"\n[{i}/{len(skills)}] {skill.name}")
        keeps, skips, stats = generate_for_skill(
            client=client,
            target=target_text,
            skill=skill,
            n_target=problems_per_skill,
            n_samples=n_samples,
            max_candidates=max_candidates_per_skill,
            agree_low=agree_low,
            agree_high=agree_high,
        )
        all_keeps.extend(keeps)
        all_skips.extend(skips)
        all_stats.append(stats)
        print(f"  done: {stats['n_passed']}/{stats['n_target']} passed "
              f"after {stats['n_attempted']} attempts ({stats['status']})")

        # Persist incrementally after each skill so a crash doesn't lose progress.
        _write_outputs(out_dir, target_text, agree_low, agree_high,
                       all_keeps, all_skips, all_stats,
                       problems_per_skill * n_skills)

    total_passed = sum(s["n_passed"] for s in all_stats)
    total_attempted = sum(s["n_attempted"] for s in all_stats)
    ok_count = sum(1 for s in all_stats if s["status"] == "ok")
    target_total = problems_per_skill * n_skills
    print(f"\n{'=' * 70}")
    print(f"  Taxonomy dataset complete: {total_passed}/{target_total}")
    print(f"  Skills ok: {ok_count}/{len(all_stats)}")
    print(f"  Total attempted: {total_attempted}")
    print(f"{'=' * 70}")


def _write_outputs(
    out_dir: str,
    target_text: str,
    agree_low: float,
    agree_high: float,
    keeps: list[dict],
    skips: list[dict],
    stats: list[dict],
    target_total: int,
) -> None:
    keeps_payload = {
        "source_problem": target_text,
        "target_agreement_low": agree_low,
        "target_agreement_high": agree_high,
        "n_problems": len(keeps),
        "generator_model": GENERATOR_MODEL,
        "solve_model": GENERATOR_MODEL,
        "problems": keeps,
    }
    skips_payload = {
        "source_problem": target_text,
        "target_agreement_low": agree_low,
        "target_agreement_high": agree_high,
        "n_problems": len(skips),
        "problems": skips,
    }
    stats_payload = {
        "skills": stats,
        "total_passed": sum(s["n_passed"] for s in stats),
        "total_attempted": sum(s["n_attempted"] for s in stats),
        "total_target": target_total,
    }
    _save_json_atomic(os.path.join(out_dir, "keeps.json"), keeps_payload)
    _save_json_atomic(os.path.join(out_dir, "skips.json"), skips_payload)
    _save_json_atomic(os.path.join(out_dir, "per_skill_stats.json"), stats_payload)


def _save_json_atomic(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(
        description="Taxonomy-first subproblem generation (Stage 1 variant)."
    )
    parser.add_argument("--problem-path", type=str, required=True,
                        help="Path to .txt target problem.")
    parser.add_argument("--runs-subdir", type=str, default=None,
                        help="Run id (default: problem stem).")
    parser.add_argument("--n-skills", type=int, default=N_SKILLS_DEFAULT)
    parser.add_argument("--problems-per-skill", type=int, default=PROBLEMS_PER_SKILL_DEFAULT)
    parser.add_argument("--max-candidates-per-skill", type=int, default=MAX_CANDIDATES_PER_SKILL_DEFAULT)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES_DEFAULT)
    parser.add_argument("--agree-low", type=float, default=AGREE_LOW_DEFAULT)
    parser.add_argument("--agree-high", type=float, default=AGREE_HIGH_DEFAULT)
    parser.add_argument("--gen-workers", type=int, default=GEN_WORKERS_DEFAULT,
                        help="(reserved for v2; skills run sequentially in v1)")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS_DEFAULT)
    parser.add_argument("--output", type=str, default=None,
                        help="Override run-directory path.")
    parser.add_argument("--failed-solutions", type=str, default=None,
                        help="Accepted for CLI symmetry; unused in v1.")
    args = parser.parse_args()

    target_text = load_problem_from_txt(args.problem_path)
    problem_stem = os.path.splitext(os.path.basename(os.path.abspath(args.problem_path)))[0]
    runs_subdir = (args.runs_subdir or "").strip() or problem_stem

    repo_root = str(_REPO_ROOT)
    runs_root = os.path.join(repo_root, "runs", runs_subdir)
    skills_path = os.path.join(runs_root, "skills.json")

    if args.output:
        out_dir = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
        out_dir = os.path.join(runs_root, "stage1_taxonomy", ts)

    print(f"Loaded problem from {args.problem_path} "
          f"(runs/{runs_subdir}/, {len(target_text)} chars)\n")

    build_taxonomy_dataset(
        target_text=target_text,
        target_path=args.problem_path,
        out_dir=out_dir,
        skills_path=skills_path,
        n_skills=args.n_skills,
        problems_per_skill=args.problems_per_skill,
        max_candidates_per_skill=args.max_candidates_per_skill,
        n_samples=args.n_samples,
        agree_low=args.agree_low,
        agree_high=args.agree_high,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.4: Run tests — expect all green**

Run: `pytest tests/test_taxonomy_generation.py -v`

Expected: all tests pass including `TestBuildTaxonomyDataset`.

- [ ] **Step 4.5: Verify CLI parses**

Run: `python Stage1/taxonomy_generation.py --help`

Expected: help text lists all flags; no traceback.

- [ ] **Step 4.6: Commit**

```bash
git add Stage1/taxonomy_generation.py tests/test_taxonomy_generation.py
git commit -m "taxonomy-gen: orchestration, output writing, CLI

build_taxonomy_dataset() runs Phase 1 (with skills.json caching) then
iterates Phase 2 per skill, persisting keeps/skips/per_skill_stats
after each skill so a crash mid-run doesn't lose progress. CLI matches
the existing Stage 1 shape but hardcodes the model and accepts
--failed-solutions for symmetry (unused in v1)."
```

---

### Task 5: End-to-end manual smoke test

**Files:** none modified.

- [ ] **Step 5.1: Confirm all tests pass**

Run: `pytest tests/ -v`

Expected: every test green.

- [ ] **Step 5.2: Tiny live run (requires API credentials)**

Symlink the worktree's `runs/` and `.env` the same way the `dedupe` branch smoke test did, then:

```bash
cd /Users/andrewsung/ttt-binary-taxonomy
python Stage1/taxonomy_generation.py \
    --problem-path data/target-problems/conics.txt \
    --runs-subdir conics-tangent-5 \
    --n-skills 3 \
    --problems-per-skill 2 \
    --max-candidates-per-skill 6 \
    --n-samples 3
```

This targets just 3 skills × 2 problems × 3 solve samples — cheap. Watch for:

- `runs/conics-tangent-5/skills.json` created with 3 skills.
- `runs/conics-tangent-5/stage1_taxonomy/<ts>/keeps.json` has `generator_model`, `solve_model`, `skill` tags.
- Every stored problem text contains `\boxed{}` and `\(...\)` or `\[...\]`.

- [ ] **Step 5.3: Re-run to verify `skills.json` reuse**

Run the same command again. Expected output starts with:

```
Reusing cached skills from runs/conics-tangent-5/skills.json
```

No second decomposition call.

- [ ] **Step 5.4: If smoke test surfaced issues, fix and commit**

Otherwise no commit needed.

---

## Self-Review Notes

**Spec coverage:**
- `GENERATOR_MODEL` hardcoded → enforced by `test_generator_model_hardcoded` and `test_uses_hardcoded_model`. No CLI flag exposes it. ✓
- No `max_tokens`, no `timeout` → enforced by `test_no_max_tokens_no_timeout`. ✓
- Two-phase flow with `skills.json` caching → Task 2 + Task 4 (Step 4.1 `test_reuses_cached_skills`). ✓
- Per-skill generate-until with 100-attempt cap → Task 3 (`test_caps_at_max_candidates`). ✓
- `keeps.json` has `skill` field + model metadata → Task 4 (`test_end_to_end_writes_expected_files`). ✓
- `per_skill_stats.json` schema → Task 4. ✓
- `\boxed{}` in problem text + LaTeX requirement → Task 3 generator prompt. ✓
- No failed-solutions in generator prompt → Task 3 generator prompt omits failed solutions entirely. `--failed-solutions` CLI flag exists but passes through to nothing. ✓
- Coexists with `distinct_llm_prompting.py` → no modifications to that file. ✓

**Placeholder scan:** none. All code steps have actual code.

**Type / name consistency:** `Skill` fields (`name`, `description`, `example_problem_hint`) consistent across dataclass, JSON schema, prompts, and tests. `generate_for_skill` return tuple `(keeps, skips, stats)` consistent at every call site. `build_taxonomy_dataset` kwargs consistent with `main()` argparse. `GENERATOR_MODEL` used everywhere (no string literals).
