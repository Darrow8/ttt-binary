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
from dataclasses import dataclass, asdict, field
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
    # Added by the 6-step pipeline (spec in subproblem_generation_pipeline.md).
    # Optional for backwards compatibility with legacy skills.json files that
    # only carry name + description.
    serves_interpretations: list[str] = field(default_factory=list)
    addresses_features: list[str] = field(default_factory=list)
    role: str = ""  # prerequisite_fact | computational_technique | named_theorem | domain_identity


# ---------------------------------------------------------------------------
# Phase 1 — decomposition
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Six-step pipeline (see subproblem_generation_pipeline.md):
#   1. Structural feature extraction  -> FEATURES_PROMPT
#   2. Interpretation enumeration     -> INTERPRETATIONS_PROMPT
#   3. Skill decomposition            -> DECOMPOSE_PROMPT
#   4. Adversarial coverage critic    -> CRITIC_PROMPT
#   5. Subproblem generation          -> GENERATE_PROMPT (below)
#   6. Dedupe                         -> Stage1/dedupe_subproblems.py
# ---------------------------------------------------------------------------


# Step 1 -- spec §1 verbatim, plus a final JSON formatting clause so the output
# is machine-parseable downstream.
FEATURES_PROMPT = """\
You will be given a problem with a definite correct answer (numerical, categorical, or otherwise checkable). Do NOT attempt to solve the problem. Your task is to analyze the problem statement itself.

Answer the following questions. For each, quote the specific words or phrases from the problem statement that support your answer.

1. WHAT IS THE ANSWER, AS AN OBJECT?
Describe the nature of what the answer must be — a count, a probability, a rate, a ratio, a physical quantity with specific units, a yes/no, a parameter value, a dimensionless number, etc. Do not guess the value; characterize the type.

2. WHAT OPERATIONS ARE APPLIED TO PRODUCE THE ANSWER?
The problem may request a raw quantity, or it may apply transformations: an average, a limit, a specific rounded form, a ratio to a reference, an asymptotic leading coefficient, a difference from a baseline, a probability of a specific event. List every transformation the problem statement applies to the underlying quantity of interest.

3. WHAT AGGREGATION OR INDEXING IS PRESENT?
Is the answer about a single object, or is it aggregated over a family of objects? If aggregated, over what family, and with what weighting (uniform, measure-weighted, conditional, limiting)? If there is no aggregation, note that explicitly.

4. WHAT REGIME, LIMIT, OR BOUNDARY IS SPECIFIED?
The problem may specify an asymptotic regime, a range of parameters, boundary conditions, an idealization (e.g., "assume infinite population," "in the large-N limit," "for sufficiently large p"), or a specific scale of interest. What constraints on the regime does the problem impose? What is being held fixed and what is being varied?

5. WHAT NAMED OBJECTS, SYSTEMS, OR MODELS APPEAR?
If the problem names a specific theorem, equation, system, model, organism, process, or construction, each named object brings a body of theory with it. List each named object and briefly note what theory or assumptions it invokes.

6. WHAT IS THE SIMPLEST RELATED PROBLEM, AND HOW DOES THIS ONE DEVIATE FROM IT?
Write down the simplest, most direct question someone could ask about the same subject matter. Then identify how the actual problem statement deviates from that simple phrasing. Every deviation is intentional; each one is a signal about what techniques the problem requires.

7. WHAT FEATURES OF THE PROBLEM STATEMENT SEEM SURPRISING OR EASY TO OVERLOOK?
Identify phrases, conditions, or qualifiers that a quick reader might gloss over but that constrain the answer. Unusual word choices, precise quantifiers, specific phrasings — anything that would not appear in a generic version of the problem.

For each answer, quote the relevant portion of the problem statement verbatim.

Problem: {problem}

Respond with a single JSON object, no prose before or after, with exactly these keys:
{{
  "q1": "<answer to question 1, including verbatim quotes>",
  "q2": "<answer to question 2, including verbatim quotes>",
  "q3": "<answer to question 3, including verbatim quotes>",
  "q4": "<answer to question 4, including verbatim quotes>",
  "q5": "<answer to question 5, including verbatim quotes>",
  "q6": "<answer to question 6, including verbatim quotes>",
  "q7": "<answer to question 7, including verbatim quotes>"
}}
"""


# Step 2 -- spec §2 verbatim (already specifies JSON output format).
INTERPRETATIONS_PROMPT = """\
You will be given a problem and a structural analysis of that problem's statement. Do NOT attempt to solve the problem yet. Your task is to produce THREE distinct interpretations of what the problem is literally asking for.

Each interpretation must include:
- A precise statement of what the answer IS, as a specific object produced by a specific procedure
- The body of theory, technique, or knowledge that would be used to compute that object
- A justification grounded in specific features of the problem statement, citing clauses from the structural analysis

Constraints on your interpretations:
- They must differ in what body of theory or technique they invoke. If two interpretations would be solved by the same technique, they are the same interpretation and should be merged.
- One interpretation should be the most obvious surface reading — the interpretation that pattern-matching on the problem's subject matter would suggest.
- At least one of the other interpretations must take seriously the features identified in question 6 and question 7 of the structural analysis (the deviation from the simplest phrasing, and the easily-overlooked details). These features are there for a reason; a correct interpretation should explain why they are there.
- Do not hedge. Each interpretation should commit to a definite answer type and technique. If you are uncertain which interpretation is right, that uncertainty is captured by having multiple interpretations, not by hedging within each.

Problem: {problem}

Structural analysis: {structural_analysis}

Output a JSON list of three interpretations, each with:
- answer_type: what the answer IS under this interpretation
- technique: the body of theory that would compute it
- justification: which features of the problem statement support this interpretation
- ignored_features: which features of the problem statement this interpretation does not fully account for (be honest; if an interpretation perfectly accounts for everything, say so)

Respond with a single JSON object, no prose before or after, with this exact shape:
{{
  "interpretations": [
    {{"answer_type": "...", "technique": "...", "justification": "...", "ignored_features": "..."}},
    {{"answer_type": "...", "technique": "...", "justification": "...", "ignored_features": "..."}},
    {{"answer_type": "...", "technique": "...", "justification": "...", "ignored_features": "..."}}
  ]
}}
"""


# Step 3 -- spec §3 verbatim (already specifies JSON output).
# `n_skills` is a soft target; the spec prefers coverage over a fixed count.
# `critic_feedback` is optional context appended on revision loops.
DECOMPOSE_PROMPT = """\
You will be given a problem, a structural analysis of it, and several interpretations of what it is asking. Your task is to produce a union skill set — a list of atomic skills such that a solver who has mastered all of them would be equipped to solve the problem, regardless of which interpretation turns out to be correct.

Requirements:

1. For EACH interpretation, list the skills a solver would need to execute that interpretation. Label each skill with which interpretation(s) it serves.

2. For EACH feature identified in the structural analysis (especially from questions 4, 5, 6, and 7), verify that at least one skill addresses that feature. If no skill addresses it, add a skill that does.

3. Skills must be ATOMIC: each skill should be a specific, nameable technique, theorem, identity, or body of knowledge. Prefer "knowledge of [specific theorem or technique]" over "algebraic geometry" or "population genetics."

4. Skills must be LOAD-BEARING: if removing the skill would prevent the solver from completing the interpretation, it belongs. If the interpretation would work without it, it does not.

5. Skills should be stated as PREREQUISITES, not as problem-specific observations. "Knowledge of the harmonic sum identity for expected cycle counts in random permutations" is a skill. "The answer involves a sum of reciprocals" is not.

6. Do not bias toward any one interpretation. The skill set should cover all three interpretations, even if you suspect one is more likely correct than the others. The purpose is to hedge against the model's own pattern-matching.

Soft target: aim for approximately {n_skills} skills. You may output slightly more or slightly fewer if coverage strictly requires it, but do not pad and do not strip essential skills.

Problem: {problem}
Structural analysis: {structural_analysis}
Interpretations: {interpretations}
{critic_feedback}

Output a JSON list of skills, each with:
- name: short identifier
- description: 1-2 sentences
- serves_interpretations: which interpretations this skill supports (list of interpretation indices as strings, e.g. ["0", "2"], or interpretation names; see the Interpretations list above for identifiers)
- addresses_features: which structural features this skill addresses (if any; list of question identifiers like ["q4", "q6"])
- role: one of "prerequisite_fact", "computational_technique", "named_theorem", "domain_identity"

Respond with a single JSON object, no prose before or after, with this exact shape:
{{
  "skills": [
    {{
      "name": "...",
      "description": "...",
      "serves_interpretations": ["..."],
      "addresses_features": ["..."],
      "role": "..."
    }}
  ]
}}
"""


# Step 4 -- spec §4 verbatim, plus JSON output clause for machine parsing.
CRITIC_PROMPT = """\
Below is a problem and a decomposition of it into atomic skills. Your task is to assess whether the skill set is adequate, by checking coverage.

You will NOT attempt to solve the problem. You will only check whether the skill set addresses every part of the problem statement.

Procedure:

1. Read the problem statement and segment it into clauses (sentences or meaningful sub-sentences).

2. For each clause, identify whether any skill in the list addresses the techniques required by that clause. Quote the clause and either cite the covering skill(s) or mark it as UNCOVERED.

3. For each skill, identify which clause(s) of the problem statement it addresses. If a skill addresses no clause, mark it as UNUSED.

4. Perform the stripping test: mentally remove every clause of the problem that is covered by some skill. What remains? If what remains still contains essential content from the original problem (a transformation, a limit, an aggregation, a qualifier), then the skill set has a gap. Describe what is left over.

5. Perform the inverse stripping test: if you had only the skills listed, with no other knowledge, could you in principle reconstruct what kind of problem these skills are meant to solve? If the skills suggest a different problem than the one given, the skill set is misaligned with the actual problem.

Problem: {problem}
Skills: {skills}

Output:
- uncovered_clauses: problem clauses with no covering skill
- unused_skills: skills addressing no clause
- stripping_test_residue: what remains of the problem after removing covered clauses
- alignment_check: does the skill set suggest a different problem than the one given?
- recommendation: "skills adequate" or "skills need revision, see [specific gaps]"

Respond with a single JSON object, no prose before or after, with this exact shape:
{{
  "uncovered_clauses": ["..."],
  "unused_skills": ["..."],
  "stripping_test_residue": "...",
  "alignment_check": "...",
  "recommendation": "skills adequate" or "skills need revision: ..."
}}
"""


def _robust_completion(client, prompt: str, *, temperature: float = TEMPERATURE,
                       max_retries: int = 8) -> str:
    """Call client.chat.completions.create and return the content string.

    Guards against two Vertex-MaaS quirks that the lower-level
    distinct_llm_prompting.call_llm also handles (but with a 180s
    timeout baked in, which we don't want here):
      1. Occasional raw-string responses (returns a `str` instead of
         a ChatCompletion object).
      2. Empty `choices` or empty `content`.
    Retries with exponential backoff up to `max_retries` attempts.
    No max_tokens. No timeout.
    Returns "" if all retries fail (matching call_llm's failure mode).
    """
    import random

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=GENERATOR_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            if isinstance(resp, str):
                raise ValueError(f"vertex returned raw string: {resp[:200]!r}")
            if not getattr(resp, "choices", None):
                raise ValueError("response has no choices")
            content = resp.choices[0].message.content or ""
            if not content:
                raise ValueError("empty response content")
            return content
        except Exception as e:
            last_err = e
            # Auth-token refresh: long-running jobs outlive the initial ADC
            # token. On 401 we re-read the token from ADC and update the
            # client in place so the next retry uses fresh credentials.
            from Stage1.distinct_llm_prompting import (
                _is_auth_error, _get_vertex_access_token,
            )
            if _is_auth_error(e):
                try:
                    client.api_key = _get_vertex_access_token()
                    print("  [info] refreshed Vertex ADC token", flush=True)
                except Exception as refresh_err:
                    print(
                        f"  [warn] token refresh failed: "
                        f"{type(refresh_err).__name__}: {str(refresh_err)[:120]}",
                        flush=True,
                    )
            # Jittered exponential backoff: 1s, 2s, 4s, ..., capped at 60s
            delay = min(2 ** attempt, 60) + random.uniform(0, 1)
            print(
                f"  [warn] LLM call attempt {attempt + 1}/{max_retries} "
                f"failed ({type(e).__name__}: {str(e)[:120]}); retrying in "
                f"{delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)

    print(
        f"  [warn] LLM call failed after {max_retries} attempts: {last_err}",
        flush=True,
    )
    return ""


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


def _parse_json_object(raw: str) -> dict:
    """Shared JSON-object parser used by steps 1, 2, 3, 4."""
    if not raw:
        raise ValueError("empty completion")
    block = _extract_json_block(raw)
    parsed = json.loads(block)
    if not isinstance(parsed, dict):
        raise ValueError("expected JSON object at top level")
    return parsed


def extract_features(
    client,
    target: str,
    *,
    max_retries: int = 3,
) -> dict:
    """Step 1: structural feature extraction (spec §1).

    Returns dict with keys q1..q7.
    """
    prompt = FEATURES_PROMPT.format(problem=target)
    last_err: Exception | None = None
    for _ in range(max_retries):
        raw = _robust_completion(client, prompt).strip()
        try:
            parsed = _parse_json_object(raw)
            # Require all seven keys; tolerate extra keys.
            missing = [k for k in ("q1", "q2", "q3", "q4", "q5", "q6", "q7")
                       if not isinstance(parsed.get(k), str) or not parsed[k].strip()]
            if missing:
                raise ValueError(f"features missing keys: {missing}")
            return {k: parsed[k] for k in ("q1", "q2", "q3", "q4", "q5", "q6", "q7")}
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
            last_err = e
            continue
    raise ValueError(f"failed to parse features after {max_retries} attempts: {last_err}")


def enumerate_interpretations(
    client,
    target: str,
    features: dict,
    *,
    max_retries: int = 3,
) -> list[dict]:
    """Step 2: interpretation enumeration (spec §2).

    Returns list of >= 3 interpretation dicts, each with keys
    answer_type, technique, justification, ignored_features.
    """
    prompt = INTERPRETATIONS_PROMPT.format(
        problem=target,
        structural_analysis=json.dumps(features, indent=2, ensure_ascii=False),
    )
    last_err: Exception | None = None
    required_keys = ("answer_type", "technique", "justification", "ignored_features")
    for _ in range(max_retries):
        raw = _robust_completion(client, prompt).strip()
        try:
            parsed = _parse_json_object(raw)
            interps = parsed.get("interpretations")
            if not isinstance(interps, list) or len(interps) < 3:
                raise ValueError(f"expected >=3 interpretations, got {type(interps).__name__} "
                                 f"len={len(interps) if isinstance(interps, list) else 'n/a'}")
            clean = []
            for i, entry in enumerate(interps):
                if not isinstance(entry, dict):
                    raise ValueError(f"interpretation[{i}] not an object")
                miss = [k for k in required_keys if not isinstance(entry.get(k), str)]
                if miss:
                    raise ValueError(f"interpretation[{i}] missing keys: {miss}")
                clean.append({k: entry[k] for k in required_keys})
            return clean
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
            last_err = e
            continue
    raise ValueError(f"failed to parse interpretations after {max_retries} attempts: {last_err}")


def decompose_target(
    client,
    target: str,
    *,
    features: dict,
    interpretations: list[dict],
    n_skills: int = N_SKILLS_DEFAULT,
    critic_feedback: dict | None = None,
    max_retries: int = 3,
) -> list[Skill]:
    """Step 3: skill decomposition (spec §3).

    Produces a union skill set that covers all interpretations and all
    structural features. `critic_feedback` is optional context appended on
    revision rounds driven by Step 4.
    """
    critic_block = ""
    if critic_feedback:
        critic_block = (
            "\nCritic feedback on the previous skill set (address these gaps):\n"
            + json.dumps(critic_feedback, indent=2, ensure_ascii=False)
        )
    prompt = DECOMPOSE_PROMPT.format(
        problem=target,
        structural_analysis=json.dumps(features, indent=2, ensure_ascii=False),
        interpretations=json.dumps(interpretations, indent=2, ensure_ascii=False),
        n_skills=n_skills,
        critic_feedback=critic_block,
    )

    last_err: Exception | None = None
    for _ in range(max_retries):
        raw = _robust_completion(client, prompt).strip()
        try:
            parsed = _parse_json_object(raw)
            skills_list = parsed.get("skills")
            if not isinstance(skills_list, list) or not skills_list:
                raise ValueError("'skills' key missing, not a list, or empty")
            out: list[Skill] = []
            for i, entry in enumerate(skills_list):
                if not isinstance(entry, dict):
                    raise ValueError(f"skill[{i}] is not an object")
                for k in ("name", "description"):
                    v = entry.get(k)
                    if not isinstance(v, str) or not v.strip():
                        raise ValueError(f"skill[{i}].{k} must be a non-empty string")
                serves = entry.get("serves_interpretations", [])
                if not isinstance(serves, list):
                    serves = []
                serves = [str(x) for x in serves]
                addresses = entry.get("addresses_features", [])
                if not isinstance(addresses, list):
                    addresses = []
                addresses = [str(x) for x in addresses]
                role = entry.get("role", "")
                if not isinstance(role, str):
                    role = ""
                out.append(Skill(
                    name=entry["name"],
                    description=entry["description"],
                    serves_interpretations=serves,
                    addresses_features=addresses,
                    role=role,
                ))
            return out
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
            last_err = e
            continue
    raise ValueError(f"failed to parse skills after {max_retries} attempts: {last_err}")


def run_critic(
    client,
    target: str,
    skills: list[Skill],
    *,
    max_retries: int = 3,
) -> dict:
    """Step 4: adversarial coverage critic (spec §4).

    Returns the critic's JSON verdict. The caller reads `recommendation` to
    decide whether to loop back to step 3.
    """
    skills_payload = [asdict(s) for s in skills]
    prompt = CRITIC_PROMPT.format(
        problem=target,
        skills=json.dumps(skills_payload, indent=2, ensure_ascii=False),
    )
    last_err: Exception | None = None
    for _ in range(max_retries):
        raw = _robust_completion(client, prompt).strip()
        try:
            parsed = _parse_json_object(raw)
            if "recommendation" not in parsed:
                raise ValueError("critic output missing 'recommendation'")
            return parsed
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
            last_err = e
            continue
    # If the critic itself is malformed, treat as adequate (don't block).
    print(f"  [warn] critic parse failed after {max_retries} attempts: {last_err}; "
          "treating as 'skills adequate'", flush=True)
    return {
        "uncovered_clauses": [],
        "unused_skills": [],
        "stripping_test_residue": "",
        "alignment_check": "",
        "recommendation": "skills adequate",
        "_critic_error": str(last_err),
    }


def _critic_says_revise(critic: dict) -> bool:
    rec = str(critic.get("recommendation", "")).lower().strip()
    # Spec recommends exactly "skills adequate" or "skills need revision, see ...".
    return rec.startswith("skills need revision")


def decompose_with_critic(
    client,
    target: str,
    *,
    features: dict,
    interpretations: list[dict],
    n_skills: int = N_SKILLS_DEFAULT,
    max_revisions: int = 2,
) -> tuple[list[Skill], list[dict]]:
    """Run step 3 + step 4, looping up to `max_revisions` times on revision.

    Returns (final_skills, critic_history) where critic_history is the list
    of every critic verdict produced during the loop (useful for audit).
    """
    critic_history: list[dict] = []
    skills = decompose_target(
        client, target,
        features=features, interpretations=interpretations,
        n_skills=n_skills,
    )
    critic = run_critic(client, target, skills)
    critic_history.append(critic)
    revisions = 0
    while _critic_says_revise(critic) and revisions < max_revisions:
        revisions += 1
        print(f"  [info] critic flagged gaps; revision round {revisions}/{max_revisions}",
              flush=True)
        skills = decompose_target(
            client, target,
            features=features, interpretations=interpretations,
            n_skills=n_skills,
            critic_feedback=critic,
        )
        critic = run_critic(client, target, skills)
        critic_history.append(critic)
    if _critic_says_revise(critic):
        print(f"  [warn] critic still flags gaps after {max_revisions} revisions; "
              "proceeding with current skill set", flush=True)
    else:
        print(f"  [info] critic accepted skill set after {revisions} revision(s)",
              flush=True)
    return skills, critic_history


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
    """Load skills from `path`, or return None if the file doesn't exist or is malformed.

    Tolerates extra keys. Reads the six-step pipeline fields
    (serves_interpretations, addresses_features, role) if present; falls back
    to empty defaults on legacy files that only have name + description.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        skills = []
        for entry in data.get("skills", []):
            serves = entry.get("serves_interpretations", []) or []
            if not isinstance(serves, list):
                serves = []
            serves = [str(x) for x in serves]
            addresses = entry.get("addresses_features", []) or []
            if not isinstance(addresses, list):
                addresses = []
            addresses = [str(x) for x in addresses]
            role = entry.get("role", "") or ""
            if not isinstance(role, str):
                role = ""
            skills.append(Skill(
                name=entry["name"],
                description=entry["description"],
                serves_interpretations=serves,
                addresses_features=addresses,
                role=role,
            ))
        return skills
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


# Step 5 -- spec §5 verbatim for CORE constraints (relevance / isolation /
# checkability / non-triviality / honesty about difficulty), followed by an
# ADDITIONAL CONSTRAINTS appendix preserving the hard-earned rules from prior
# runs: no answer leakage, unambiguous answer, per-skill memory of prior
# attempts, structural diversity, target-leakage guard, output format.
GENERATE_PROMPT = """\
You are generating a subproblem to test mastery of one specific skill. The subproblem will be used to verify whether a solver has the skill in hand.

You have access to:
- The parent problem
- The skill to be tested
- Which interpretation(s) of the parent problem use this skill
- Which feature(s) of the parent problem this skill addresses

Constraints on the subproblem:

1. RELEVANCE: The subproblem should test this skill in a way that mirrors how the parent problem uses it. If the parent problem applies the skill in a specific regime, at a specific scale, or to a specific kind of object, the subproblem should do the same. Do not generate subproblems that test the skill in scenarios the parent problem does not touch.

2. ISOLATION: The subproblem should test this skill and ONLY this skill. A solver who has mastered this skill but lacks the other skills needed for the parent problem should still be able to solve the subproblem.

3. CHECKABILITY: The subproblem must have a definite, short-form answer that can be verified automatically (a number, a yes/no, a specific named object, a closed-form expression).

4. NON-TRIVIALITY: The subproblem should require actual application of the skill, not just recall of a definition. If the skill is "knowledge of theorem X," the subproblem should require using theorem X to compute something, not just stating theorem X.

5. HONESTY ABOUT DIFFICULTY: The subproblem should be at a difficulty where a solver who knows the skill would solve it reliably but a solver who doesn't would fail reliably. Avoid subproblems where the answer can be guessed, pattern-matched, or brute-forced without the skill.

Parent problem: {target}
Skill: {skill_name} — {skill_description}
Interpretations using this skill: {interpretations_for_skill}
Features addressed: {features_for_skill}

Output fields (conceptual, not format — format is specified at the end):
- problem_statement: the subproblem as it would be presented to a solver
- answer: the correct answer in a checkable form
- solution_sketch: a brief explanation of how the skill is applied
- why_relevant: a sentence explaining how this subproblem mirrors the parent problem's use of this skill
- failure_mode: what mistake a solver who lacks this skill would make

---

ADDITIONAL CONSTRAINTS (enforced IN ADDITION to the five above; learned from prior runs on this codebase):

A. Other skills in the taxonomy -- your subproblem must NOT require any of these to solve (this is the ISOLATION constraint in concrete form):
{other_skill_names_bulleted}

   If you cannot construct a subproblem isolating this skill without invoking the others, set `problem_statement` to the literal string "UNISOLATABLE: <one sentence reason>". This signals the skill is not cleanly separable and the decomposition needs revision -- do not fake an isolated problem.

B. No answer leakage. The `problem_statement` must not state, imply, or walk the solver up to the numerical answer. In particular:
   - Do NOT include phrases like "which gives X", "therefore ... = X", "this equals X", "so the class is X * H", "note that the degree is X". These turn the problem into reading comprehension, not reasoning.
   - Do NOT include the final numerical answer -- as a digit, as a word, or as a closed-form expression that trivially evaluates to it -- anywhere in the setup.
   - Present only the givens; the solver must derive every computed quantity, including any intermediate quantity that determines the final answer.

C. Unambiguous answer. Exactly one conventional reading of the problem statement must yield exactly one answer.
   - If a term has multiple standard conventions in the field (e.g., "dimension of a projective space" is n-1 vs n depending on whether you mean projective dim or vector-space dim; "degree of a map" can mean topological vs algebraic; indexing starting from 0 vs 1), either disambiguate explicitly in-line or avoid the term.
   - The expected answer must NOT depend on which convention a competent solver adopts. If you cannot ensure this without leaking the answer, pick a different concrete instantiation of the skill.

D. No target leakage. Do not reproduce the parent problem's setup, notation, or specific numerical parameters. The subproblem must be a fresh concrete instance so that solving it doesn't amount to partially solving the parent.

E. Prior attempts for THIS skill (same-skill memory). You MUST produce a subproblem that is neither a paraphrase of any of these NOR has the same numerical answer as any of them. If the skill admits a parametric family (varying degree, dimension, number of conditions, configuration, etc.), pick a DIFFERENT point in that family than any shown below. Do not default to the simplest textbook case if it has already been used.
{prior_attempts_bulleted}

F. Structural diversity (load-bearing). Varying only a numerical parameter while keeping the same sentence template is NOT enough. If your problem reads as a find-and-replace of a previous problem (same phrasing, same sentence order, same objects, just a different number), you have failed. Instead, vary the STRUCTURE across attempts: recast the same computation in a different object, use a different algebraic incarnation, swap the direction of the question (given X, find Y vs given Y, find X), change the ambient setting, or combine the skill with a small non-leaky twist that doesn't cross into another skill. At minimum the sentence pattern of the new problem must NOT be a substring rewrite of any prior one.

G. Answer format. The `answer` must be a single number (integer or decimal). If a decimal, the `problem_statement` must instruct the solver to round to 4 decimal places. State the `problem_statement` in 3-10 sentences. ALL math must be written in LaTeX using \\(...\\) for inline and \\[...\\] for display. The `problem_statement` MUST end with this exact sentence: "Put your final answer inside \\boxed{{}}."

Self-audit before output (do this internally, do not print):
(i) Answer-leak grep. Mentally solve the problem and note the final numerical answer N. Scan the problem statement. Does N appear anywhere -- as digit, word, or trivially-evaluatable expression? If yes, rewrite to remove it.
(ii) Convention check. For every named quantity the solver must compute, is there a single standard convention in the field? If a term has multiple conventions, disambiguate or reformulate.
(iii) Intermediate-leak grep. Does the statement hand the solver any computed quantity that determines N via routine arithmetic (e.g., giving the degree of each factor when the problem asks for the product of degrees)? If yes, remove or rephrase.
(iv) Novelty check. Is the answer N equal to any of the prior answers listed above? Is the problem a paraphrase of any prior problem? If either, pick a different instance of the skill (different parameter, different configuration) and redo.
(v) Relevance check. Does `why_relevant` articulate a specific connection between this subproblem and the parent problem's use of the skill? Post-hoc rationalization ("this tests the skill in a general way") is a failure mode -- anchor on the specific regime/scale/object the parent uses.
Revise internally until all five pass, then emit the final output.

Output format (STRICT):
Begin your response with <problem> on its own line, then the full `problem_statement` text (or the UNISOLATABLE sentinel), then </problem> on its own line. Immediately after, emit the metadata fields as a single JSON object between <metadata> and </metadata> tags, with keys "answer", "solution_sketch", "why_relevant", "failure_mode". No other text before, between, or after the tags.
"""


_PROBLEM_TAG_RE = re.compile(r"<problem>(.*?)</problem>", re.DOTALL)
_METADATA_TAG_RE = re.compile(r"<metadata>(.*?)</metadata>", re.DOTALL)


def _parse_problem(raw: str) -> str:
    """Extract the <problem>...</problem> content. Return '' if tags missing."""
    m = _PROBLEM_TAG_RE.search(raw)
    if not m:
        return ""
    return m.group(1).strip()


def _parse_metadata(raw: str) -> dict:
    """Extract the <metadata>JSON</metadata> block (spec §5 output fields).

    Returns {} if missing or malformed -- the caller should not fail generation
    on a missing metadata block; solver consensus is the authoritative answer.
    The metadata is stored as provenance only.
    """
    m = _METADATA_TAG_RE.search(raw)
    if not m:
        return {}
    try:
        parsed = json.loads(m.group(1).strip())
        if not isinstance(parsed, dict):
            return {}
        return parsed
    except json.JSONDecodeError:
        return {}


# How many prior attempts (problem + answer) to feed back into the
# generator as "avoid these." Bumped from 12 -> 50 after observing cycling:
# once the window scrolled past 12 the generator happily regenerated
# shapes it had already made. At 50 entries x ~250 tokens each the memory
# block is ~12K tokens, well within context.
_PRIOR_ATTEMPTS_MEMORY = 50
# Truncation for each prior problem text in the memory bullet (chars).
# 220 is enough to capture the sentence template so the generator can
# detect "I'm just swapping a parameter and re-using the same shape."
_PRIOR_PROBLEM_SNIPPET_CHARS = 220


def _format_prior_attempts(
    prior_problems: list[str],
    prior_answers: list[str],
) -> str:
    """Render the memory block for GENERATE_PROMPT.

    Pairs prior_problems[i] with prior_answers[i]. Keeps only the last
    _PRIOR_ATTEMPTS_MEMORY entries so the prompt stays bounded.
    Each problem is truncated to _PRIOR_PROBLEM_SNIPPET_CHARS.
    """
    if not prior_problems:
        return "(none yet -- this is the first attempt for this skill.)"
    probs = prior_problems[-_PRIOR_ATTEMPTS_MEMORY:]
    answers = prior_answers[-_PRIOR_ATTEMPTS_MEMORY:]
    lines = []
    for i, (p, a) in enumerate(zip(probs, answers), start=1):
        snippet = (p or "").strip().replace("\n", " ")
        if len(snippet) > _PRIOR_PROBLEM_SNIPPET_CHARS:
            snippet = snippet[:_PRIOR_PROBLEM_SNIPPET_CHARS] + "..."
        ans = a if a else "(no answer)"
        lines.append(f"- [prior #{i}] answer={ans!r}: {snippet}")
    return "\n".join(lines)


def _interpretations_for_skill(
    skill: Skill,
    all_interpretations: list[dict],
) -> str:
    """Render the interpretations this skill serves into a prompt-ready string.

    Matches by index first (spec recommends "0", "1", "2" as identifiers),
    falls back to matching by technique name fragment. If no match,
    returns a stringified view of ALL interpretations so the generator at
    least has the context.
    """
    if not all_interpretations:
        return "(no interpretations available)"
    picked: list[dict] = []
    for ident in skill.serves_interpretations:
        if ident.isdigit() and 0 <= int(ident) < len(all_interpretations):
            picked.append(all_interpretations[int(ident)])
            continue
        ident_low = ident.lower()
        for interp in all_interpretations:
            if ident_low in interp.get("technique", "").lower():
                picked.append(interp)
                break
    if not picked:
        picked = all_interpretations  # graceful fallback: show them all
    return json.dumps(picked, indent=2, ensure_ascii=False)


def _features_for_skill(
    skill: Skill,
    all_features: dict,
) -> str:
    """Render the features this skill addresses into a prompt-ready string."""
    if not all_features:
        return "(no features available)"
    picked: dict = {}
    for key in skill.addresses_features:
        if key in all_features:
            picked[key] = all_features[key]
    if not picked:
        picked = all_features
    return json.dumps(picked, indent=2, ensure_ascii=False)


def _generate_one_candidate(
    client,
    target: str,
    skill: Skill,
    *,
    other_skill_names: list[str],
    all_interpretations: list[dict],
    all_features: dict,
    prior_problems: list[str] | None = None,
    prior_answers: list[str] | None = None,
    _temperature: float = TEMPERATURE,
) -> tuple[str, dict]:
    """Call the generator once. Returns (problem_text, metadata_dict).

    `problem_text` may be empty (tags missing) or the UNISOLATABLE sentinel.
    `metadata_dict` holds the spec's answer/solution_sketch/why_relevant/
    failure_mode fields when present; empty dict when absent or malformed.
    The metadata is provenance only -- solver consensus is still the
    authoritative ground truth for filtering.
    """
    other_bulleted = "\n".join(f"- {name}" for name in other_skill_names) or "(none)"
    prior_bulleted = _format_prior_attempts(
        prior_problems or [], prior_answers or [],
    )
    prompt = GENERATE_PROMPT.format(
        target=target,
        skill_name=skill.name,
        skill_description=skill.description,
        interpretations_for_skill=_interpretations_for_skill(skill, all_interpretations),
        features_for_skill=_features_for_skill(skill, all_features),
        other_skill_names_bulleted=other_bulleted,
        prior_attempts_bulleted=prior_bulleted,
    )
    raw = _robust_completion(client, prompt, temperature=_temperature)
    return _parse_problem(raw), _parse_metadata(raw)


def generate_for_skill(
    *,
    client,
    target: str,
    skill: Skill,
    other_skill_names: list[str],
    all_interpretations: list[dict],
    all_features: dict,
    n_target: int,
    n_samples: int,
    max_candidates: int,
    agree_low: float,
    agree_high: float,
    solve_pool=None,
    on_keep=None,
    on_skip=None,
    seed_prior_problems: list[str] | None = None,
    seed_prior_answers: list[str] | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Generate candidates for a single skill until n_target keeps or max_candidates attempts.

    Args:
        other_skill_names: names of the OTHER skills, used in the prompt
            to enforce that the generated subproblem doesn't require any
            of them.
        all_interpretations: step-2 output. The prompt filters to the ones
            this skill serves via ``skill.serves_interpretations``.
        all_features: step-1 output (keys q1..q7). The prompt filters to
            the ones this skill addresses via ``skill.addresses_features``.
        solve_pool: concurrent.futures.ThreadPoolExecutor used by the
            real solve_and_check_agreement to fan out n_samples solve
            calls. Required at runtime (the real function unconditionally
            calls pool.submit); left None for tests that monkeypatch the
            solve function.
        seed_prior_problems / seed_prior_answers: optional memory to
            pre-populate (used by top-up mode to carry over what the
            previous pass already tried for this skill).

    Returns:
        (keeps, skips, stats) where stats is
        {"name": skill.name, "n_target", "n_passed", "n_attempted", "status"}.
    """
    keeps: list[dict] = []
    skips: list[dict] = []
    attempted = 0
    # Memory of past candidates for THIS skill (problem text + majority answer)
    # fed back into the generator so it actively diversifies. Optionally
    # pre-seeded by the top-up loop.
    prior_problems: list[str] = list(seed_prior_problems or [])
    prior_answers: list[str] = list(seed_prior_answers or [])

    while len(keeps) < n_target and attempted < max_candidates:
        attempted += 1
        problem_text, metadata = _generate_one_candidate(
            client, target, skill,
            other_skill_names=other_skill_names,
            all_interpretations=all_interpretations,
            all_features=all_features,
            prior_problems=prior_problems,
            prior_answers=prior_answers,
        )
        if not problem_text:
            skip_record = {
                "skill": skill.name,
                "problem": "",
                "reason": "generator_no_tags_or_empty",
            }
            skips.append(skip_record)
            if on_skip is not None:
                try:
                    on_skip(skip_record)
                except Exception as e:
                    print(f"  [warn] on_skip callback raised: {e!r}", flush=True)
            print(
                f"  attempt {attempted}: skip (no_tags)  "
                f"[kept {len(keeps)}/{n_target}]",
                flush=True,
            )
            continue

        # UNISOLATABLE sentinel: the generator reports the skill cannot
        # be separated from the others. Don't burn solve compute on it.
        if problem_text.startswith("UNISOLATABLE"):
            reason_text = problem_text[len("UNISOLATABLE"):].lstrip(": ").strip()
            skip_record = {
                "skill": skill.name,
                "problem": problem_text,
                "reason": "unisolatable",
                "reason_detail": reason_text,
            }
            skips.append(skip_record)
            if on_skip is not None:
                try:
                    on_skip(skip_record)
                except Exception as e:
                    print(f"  [warn] on_skip callback raised: {e!r}", flush=True)
            print(
                f"  attempt {attempted}: skip (unisolatable)  {reason_text[:80]!r}  "
                f"[kept {len(keeps)}/{n_target}]",
                flush=True,
            )
            continue

        agreement, majority_ans, all_answers, all_solutions = _get_solve_fn()(
            client, GENERATOR_MODEL, problem_text,
            n_samples=n_samples, pool=solve_pool,
        )

        numeric = _is_numeric_answer(normalize_answer(majority_ans))
        in_range = agree_low <= agreement <= agree_high
        kept = bool(majority_ans) and numeric and in_range

        # Feed every scored candidate back into the memory the next
        # generator call sees -- especially the too-easy ones, since
        # those are the duplicates the generator needs to escape.
        prior_problems.append(problem_text)
        prior_answers.append(str(majority_ans) if majority_ans else "")

        record = {
            "skill": skill.name,
            "problem": problem_text,
            "ground_truth_answer": majority_ans,
            "agreement_rate": agreement,
            "all_answers": all_answers,
            "all_solutions": all_solutions,
            "n_samples": n_samples,
            # Spec §5 provenance from the generator; NOT authoritative --
            # solver consensus above is. Stored for audit / debugging.
            "generator_metadata": metadata,
        }
        if kept:
            keeps.append(record)
            if on_keep is not None:
                try:
                    on_keep(record)
                except Exception as e:
                    print(f"  [warn] on_keep callback raised: {e!r}", flush=True)
            print(
                f"  attempt {attempted}: KEEP  agreement={agreement:.2f} "
                f"answer={str(majority_ans)[:30]!r}  "
                f"[kept {len(keeps)}/{n_target}]",
                flush=True,
            )
        else:
            if not majority_ans:
                reason = "empty_answer"
            elif not numeric:
                reason = "non_numeric"
            elif agreement < agree_low:
                reason = "too_hard"
            else:
                reason = "too_easy"
            skip_record = {**record, "reason": reason}
            skips.append(skip_record)
            if on_skip is not None:
                try:
                    on_skip(skip_record)
                except Exception as e:
                    print(f"  [warn] on_skip callback raised: {e!r}", flush=True)
            print(
                f"  attempt {attempted}: skip ({reason})  agreement={agreement:.2f} "
                f"answer={str(majority_ans)[:30]!r}  "
                f"[kept {len(keeps)}/{n_target}]",
                flush=True,
            )

    status = "ok" if len(keeps) >= n_target else "capped"
    stats = {
        "name": skill.name,
        "n_target": n_target,
        "n_passed": len(keeps),
        "n_attempted": attempted,
        "status": status,
    }
    return keeps, skips, stats


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

# Lazy-imported from distinct_llm_prompting (same rationale as solve_and_check_agreement).
get_client = None  # type: ignore[assignment]
load_problem_from_txt = None  # type: ignore[assignment]


def _lazy_import_get_client():
    global get_client
    if get_client is None:
        from Stage1.distinct_llm_prompting import get_client as _gc  # noqa: E402
        get_client = _gc


def _lazy_import_load_problem_from_txt():
    global load_problem_from_txt
    if load_problem_from_txt is None:
        from Stage1.distinct_llm_prompting import load_problem_from_txt as _lpft  # noqa: E402
        load_problem_from_txt = _lpft


def _lazy_import_distinct_helpers():
    """Import both helpers. Kept for main() which needs both."""
    _lazy_import_get_client()
    _lazy_import_load_problem_from_txt()


def _maybe_load_cached_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_json_side(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


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
    max_workers: int = MAX_WORKERS_DEFAULT,
    start_from_skill: int = 1,
    min_per_skill: int = 0,
    max_topup_rounds: int = 6,
) -> None:
    """Run the six-step pipeline end to end.

    Steps 1-4 (features, interpretations, decompose, critic) execute once,
    producing features.json, interpretations.json, skills.json, and
    critic.json next to ``skills_path``. These are cached and reused on
    subsequent runs for the same target. Steps 5 (per-skill subproblem
    generation) and 6 (dedupe, external) follow.

    If ``min_per_skill`` > 0, after the main generation loop any skill with
    fewer than ``min_per_skill`` keeps is retried in additional rounds
    (bounded by ``max_topup_rounds``) with its prior attempts carried over
    as memory. A skill that produces zero new keeps across two consecutive
    top-up rounds is declared stalled and skipped.
    """
    import concurrent.futures

    _lazy_import_get_client()
    os.makedirs(out_dir, exist_ok=True)
    client, _ = get_client()

    runs_root = os.path.dirname(skills_path)
    features_path = os.path.join(runs_root, "features.json")
    interpretations_path = os.path.join(runs_root, "interpretations.json")
    critic_path = os.path.join(runs_root, "critic.json")

    # ---- Steps 1-4: structural analysis, interpretations, decomposition,
    #      coverage critic. All four are cached on disk to avoid re-running
    #      on repeat invocations for the same target.
    cached_features = _maybe_load_cached_json(features_path)
    cached_interps = _maybe_load_cached_json(interpretations_path)
    cached_skills = load_skills(skills_path)

    need_pipeline_head = (
        cached_features is None
        or cached_interps is None
        or cached_skills is None
    )
    if need_pipeline_head:
        print(f"=== Steps 1-4: structural analysis & skill decomposition ===")
        print(f"Model:   {GENERATOR_MODEL}")
        print(f"Target:  {target_path} ({len(target_text)} chars)")

        if cached_features is None:
            print("  [1/4] extracting structural features (spec §1)...")
            features = extract_features(client, target_text)
            _save_json_side(features_path, {
                "target_problem_path": target_path,
                "target_problem_hash": target_text_hash(target_text),
                "generator_model": GENERATOR_MODEL,
                "extracted_at": datetime.now(timezone.utc).isoformat(),
                "features": features,
            })
            print(f"        features written to {features_path}")
        else:
            features = cached_features.get("features", cached_features)
            print(f"  [1/4] reusing cached features from {features_path}")

        if cached_interps is None:
            print("  [2/4] enumerating interpretations (spec §2)...")
            interpretations = enumerate_interpretations(client, target_text, features)
            _save_json_side(interpretations_path, {
                "target_problem_path": target_path,
                "target_problem_hash": target_text_hash(target_text),
                "generator_model": GENERATOR_MODEL,
                "enumerated_at": datetime.now(timezone.utc).isoformat(),
                "interpretations": interpretations,
            })
            print(f"        {len(interpretations)} interpretations written to "
                  f"{interpretations_path}")
        else:
            interpretations = cached_interps.get("interpretations", [])
            print(f"  [2/4] reusing cached interpretations from {interpretations_path} "
                  f"({len(interpretations)} interps)")

        if cached_skills is None:
            print("  [3-4/4] decomposing with critic loop (spec §3, §4)...")
            skills, critic_history = decompose_with_critic(
                client, target_text,
                features=features,
                interpretations=interpretations,
                n_skills=n_skills,
            )
            save_skills(
                skills_path, skills,
                target_path=target_path,
                target_hash=target_text_hash(target_text),
                model=GENERATOR_MODEL,
            )
            _save_json_side(critic_path, {
                "target_problem_path": target_path,
                "target_problem_hash": target_text_hash(target_text),
                "generator_model": GENERATOR_MODEL,
                "critiqued_at": datetime.now(timezone.utc).isoformat(),
                "critic_history": critic_history,
                "n_revision_rounds": len(critic_history) - 1,
            })
            print(f"        {len(skills)} skills written to {skills_path}")
            print(f"        critic history written to {critic_path}")
        else:
            skills = cached_skills
            print(f"  [3-4/4] reusing cached skills from {skills_path} "
                  f"({len(skills)} skills)")
    else:
        skills = cached_skills
        features = cached_features.get("features", cached_features)
        interpretations = cached_interps.get("interpretations", [])
        print(f"Reusing all cached pipeline outputs (features, interpretations, skills)")

    # ---- Step 5: per-skill subproblem generation
    all_keeps: list[dict] = []
    all_skips: list[dict] = []
    all_stats: list[dict] = []
    target_total = problems_per_skill * len(skills)

    # Incremental-write callbacks: every time generate_for_skill accepts or
    # rejects a candidate, append it and re-flush keeps.json / skips.json.
    def _on_keep(record: dict) -> None:
        all_keeps.append(record)
        _write_outputs(out_dir, target_text, agree_low, agree_high,
                       all_keeps, all_skips, all_stats, target_total)

    def _on_skip(record: dict) -> None:
        all_skips.append(record)
        _write_outputs(out_dir, target_text, agree_low, agree_high,
                       all_keeps, all_skips, all_stats, target_total)

    skill_names = [s.name for s in skills]
    if start_from_skill > 1:
        print(f"\n(resume mode: skipping skills 1..{start_from_skill - 1}, "
              f"starting at skill {start_from_skill})")

    # Per-skill memory of all attempted problems + answers. Lives for the
    # duration of build_taxonomy_dataset so the top-up loop can pick up
    # where the main loop left off without forgetting what was already tried.
    skill_memory: dict[str, tuple[list[str], list[str]]] = {
        s.name: ([], []) for s in skills
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as solve_pool:
        print(f"\n=== Step 5: per-skill subproblem generation ===")
        for i, skill in enumerate(skills, start=1):
            if i < start_from_skill:
                print(f"\n[{i}/{len(skills)}] {skill.name}  -- skipped (resume)")
                continue
            print(f"\n[{i}/{len(skills)}] {skill.name}")
            others = [name for j, name in enumerate(skill_names, start=1) if j != i]
            _keeps, _skips, stats = generate_for_skill(
                client=client,
                target=target_text,
                skill=skill,
                other_skill_names=others,
                all_interpretations=interpretations,
                all_features=features,
                n_target=problems_per_skill,
                n_samples=n_samples,
                max_candidates=max_candidates_per_skill,
                agree_low=agree_low,
                agree_high=agree_high,
                solve_pool=solve_pool,
                on_keep=_on_keep,
                on_skip=_on_skip,
            )
            del _keeps, _skips
            all_stats.append(stats)
            # Track memory for potential top-up round: collect what was
            # tried for this skill across all its attempts.
            skill_memory[skill.name] = _collect_prior_for_skill(
                all_keeps, all_skips, skill.name,
            )
            print(f"  done: {stats['n_passed']}/{stats['n_target']} passed "
                  f"after {stats['n_attempted']} attempts ({stats['status']})")
            _write_outputs(out_dir, target_text, agree_low, agree_high,
                           all_keeps, all_skips, all_stats, target_total)

        # ---- Top-up loop: re-run any skill that fell below min_per_skill
        #      until it reaches the threshold OR stalls out (no progress
        #      across two consecutive rounds).
        if min_per_skill > 0:
            print(f"\n=== Top-up: targeting min {min_per_skill} keeps per skill ===")
            stalled: set[str] = set()
            last_counts: dict[str, int] = {
                s.name: _count_keeps_for(all_keeps, s.name) for s in skills
            }
            no_progress_streak: dict[str, int] = {s.name: 0 for s in skills}
            for round_num in range(1, max_topup_rounds + 1):
                below = [
                    s for s in skills
                    if _count_keeps_for(all_keeps, s.name) < min_per_skill
                    and s.name not in stalled
                ]
                if not below:
                    print(f"  all skills >= {min_per_skill} keeps; top-up done")
                    break
                print(f"\n  --- top-up round {round_num} "
                      f"({len(below)} skills below threshold) ---")
                for skill in below:
                    cur = _count_keeps_for(all_keeps, skill.name)
                    need = min_per_skill - cur
                    others = [s.name for s in skills if s.name != skill.name]
                    seed_p, seed_a = skill_memory.get(skill.name, ([], []))
                    print(f"\n  [top-up r{round_num}] {skill.name}: "
                          f"{cur}/{min_per_skill}, need {need} more")
                    _k, _s, stats = generate_for_skill(
                        client=client,
                        target=target_text,
                        skill=skill,
                        other_skill_names=others,
                        all_interpretations=interpretations,
                        all_features=features,
                        n_target=need,  # generate only what's missing this round
                        n_samples=n_samples,
                        max_candidates=max_candidates_per_skill,
                        agree_low=agree_low,
                        agree_high=agree_high,
                        solve_pool=solve_pool,
                        on_keep=_on_keep,
                        on_skip=_on_skip,
                        seed_prior_problems=seed_p,
                        seed_prior_answers=seed_a,
                    )
                    del _k, _s
                    # Merge the top-up stats into all_stats (append a
                    # top-up entry rather than mutate the original so the
                    # audit trail shows per-round progress).
                    stats["name"] = f"{skill.name} (top-up r{round_num})"
                    all_stats.append(stats)
                    # Refresh memory + stall tracking.
                    skill_memory[skill.name] = _collect_prior_for_skill(
                        all_keeps, all_skips, skill.name,
                    )
                    new_count = _count_keeps_for(all_keeps, skill.name)
                    if new_count <= last_counts[skill.name]:
                        no_progress_streak[skill.name] += 1
                    else:
                        no_progress_streak[skill.name] = 0
                    last_counts[skill.name] = new_count
                    if no_progress_streak[skill.name] >= 2:
                        print(f"  [top-up] {skill.name}: no progress in "
                              f"2 consecutive rounds -- stalled, skipping")
                        stalled.add(skill.name)
                    _write_outputs(out_dir, target_text, agree_low, agree_high,
                                   all_keeps, all_skips, all_stats, target_total)

    # ---- Final report
    total_attempted = sum(s["n_attempted"] for s in all_stats)
    per_skill_final = {s.name: _count_keeps_for(all_keeps, s.name) for s in skills}
    print(f"\n{'=' * 70}")
    print(f"  Taxonomy dataset complete")
    print(f"  Total attempts (incl. top-up): {total_attempted}")
    for name, n in per_skill_final.items():
        mark = "✓" if (min_per_skill == 0 or n >= min_per_skill) else "✗"
        print(f"    {mark} {n:3d}  {name}")
    if min_per_skill > 0:
        below = [n for n in per_skill_final.values() if n < min_per_skill]
        if below:
            print(f"  WARNING: {len(below)} skill(s) below min_per_skill={min_per_skill}")
        else:
            print(f"  All {len(per_skill_final)} skills reached >= {min_per_skill}")
    print(f"{'=' * 70}")


def _count_keeps_for(all_keeps: list[dict], skill_name: str) -> int:
    return sum(1 for r in all_keeps if r.get("skill") == skill_name)


def _collect_prior_for_skill(
    all_keeps: list[dict],
    all_skips: list[dict],
    skill_name: str,
) -> tuple[list[str], list[str]]:
    """Gather every problem text + answer tried for a skill across keeps+skips.

    The order (keeps first, skips in generation order) is approximate but
    good enough for memory-feedback diversity; the generator sees recent
    attempts which is what matters.
    """
    probs: list[str] = []
    ans: list[str] = []
    for r in all_keeps:
        if r.get("skill") == skill_name and r.get("problem"):
            probs.append(r["problem"])
            ans.append(str(r.get("ground_truth_answer", "")))
    for r in all_skips:
        if r.get("skill") == skill_name and r.get("problem"):
            probs.append(r["problem"])
            ans.append(str(r.get("ground_truth_answer", "")))
    return probs, ans


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
    _lazy_import_distinct_helpers()

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
    parser.add_argument("--start-from-skill", type=int, default=1,
                        help="1-based index of the first skill to generate for. "
                             "Earlier skills are skipped (useful to resume mid-run "
                             "after a crash). Default 1 = generate all.")
    parser.add_argument("--min-per-skill", type=int, default=0,
                        help="If > 0, after the main generation loop any skill "
                             "with fewer keeps than this is retried in top-up "
                             "rounds (with prior-attempts memory carried over) "
                             "until the threshold is met or the skill stalls. "
                             "Default 0 = disabled.")
    parser.add_argument("--max-topup-rounds", type=int, default=6,
                        help="Outer cap on top-up rounds (default 6). A stalled "
                             "skill (no progress across 2 consecutive rounds) is "
                             "skipped automatically regardless of this cap.")
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
        max_workers=args.max_workers,
        start_from_skill=args.start_from_skill,
        min_per_skill=args.min_per_skill,
        max_topup_rounds=args.max_topup_rounds,
    )


if __name__ == "__main__":
    main()
