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


# ---------------------------------------------------------------------------
# Phase 1 — decomposition
# ---------------------------------------------------------------------------

DECOMPOSE_PROMPT = """\
You are designing a curriculum to help a student learn to solve a hard
target problem by mastering its component reasoning skills first. Each
skill will be used to generate subproblems with a single NUMERICAL
answer (integer or decimal) so downstream agreement checking is
mechanical. Skills whose natural subproblems don't admit a single
numerical answer are not usable here -- do not include them.

Target problem:
{target}

Decompose this target into EXACTLY {n_skills} distinct reasoning skills.

Quality requirements (apply ALL):

1. Orthogonality. For every pair of skills (A, B), a student could
   plausibly be strong at A but weak at B, or vice versa. If not, the
   pair is too coupled -- merge and pick a different skill to fill
   the slot.

2. No restatement. A skill is INVALID if solving a subproblem that
   tests it requires the same insight as the target problem itself.
   Skills are *components used in service of* the target, not *the
   target restated at smaller scale*. If a skill's subproblem would
   essentially be the target with smaller numbers or a simpler case,
   it is a restatement, not a component -- replace it.

3. Coverage of the hard part. Before finalizing, identify THE SINGLE
   HARDEST INSIGHT required to solve the target. At least 2 of the
   {n_skills} skills must build directly toward that insight. Skills
   that only cover setup, formalism, or routine machinery while
   ignoring the hard insight produce a curriculum that does not
   actually teach the target.

4. Name the technique (load-bearing). For the hardest 2-3 skills
   (the ones covering the hard insight from requirement 3), the
   name AND description must name a SPECIFIC technique, named
   theorem, or named mathematical object -- not a goal or a
   paraphrase. Examples of OK: "Chebotarev density theorem",
   "Veronese embedding", "Chasles' characteristic numbers (mu, nu)
   on the space of complete conics", "Lang-Weil estimate". Examples
   of NOT OK: "compute the correction term", "find the count",
   "handle the excess locus". A skill described only by its goal
   without naming the tool is hand-waving and will produce
   subproblems that cannot actually isolate the skill.

5. Anti-padding. If you find yourself listing the same technique
   applied to n=2, n=3, n=4, ... (or any other parametric sweep of
   the same underlying reasoning), that is ONE skill, not several.
   Difficulty progression must come from genuinely DIFFERENT
   reasoning, not from incrementing a parameter. Near-duplicates
   like "codimension of tangency-to-X" and "degree of the
   tangency-to-X divisor" are also one skill, not two.

6. No numbers in skill text. Skill names and descriptions must NOT
   contain specific numerical answers, dimensions, degrees, or
   counts (e.g. "= 6", "2^5", "dimension 5", "degree 3"). Describe
   the TECHNIQUE or OBJECT the skill teaches, not the numerical
   output. The subproblem-generation step instantiates concrete
   numbers; the skill itself must be the transferable method.
     Bad:  "Degree of the line-tangency divisor (= 2)"
     Good: "Computing the divisor class of a tangency-to-fixed-
            curve condition via contact-order analysis"
     Bad:  "Naive Bezout count 6^5 for five tangency conditions"
     Good: "Naive intersection number of repeated tangency
            divisors via Bezout, prior to excess-intersection
            correction"

6b. Parametric family (load-bearing). Each skill MUST admit a
    parametric family of subproblems -- i.e. the description must
    be broad enough that the subproblem-generator can pick multiple
    DIFFERENT concrete instances that all test the same technique
    but produce DIFFERENT numerical answers. A skill pinned to one
    specific instance (one degree, one dimension, one configuration)
    produces identical subproblems with identical answers and is
    useless downstream. Every description must either (a) name an
    explicit parameter the generator can vary (e.g. "degree d",
    "dimension n", "number of conditions k"), or (b) refer to a
    technique that naturally applies to many configurations.
      Bad:  "Compute the dimension of the space of homogeneous
             degree-two polynomials in three variables" -- pinned
             to one instance, answer is always 5.
      Good: "Compute the dimension of the linear system of plane
             curves of degree d (equivalently, of hypersurfaces of
             degree d in P^n) using the binomial coefficient
             count." -- d and n are free, the generator can pick
             many pairs.
      Bad:  "Degree of the discriminant hypersurface of singular
             plane conics" -- one number.
      Good: "Degree of the discriminant hypersurface of singular
             degree-d plane curves (or hypersurfaces) as a
             function of d and the ambient dimension."

7. No arithmetic-only skills. A skill whose only content is
   mechanical arithmetic on a quantity produced by another skill
   ("compute floor(100 * L)", "subtract the correction from the
   naive count") is not a skill -- it's a postprocessing step.
   Exclude it; reuse the slot for a real reasoning component.

8. Difficulty spread, not flat. The skills must span a real range:
   skill 1 should be doable by a strong undergraduate in the relevant
   field; skill {n_skills} should be doable only by someone close to
   mastering the target. If skills in the middle all feel like the
   same difficulty, the decomposition is too flat.

9. Numerical answerability. Each skill's natural subproblem must admit
   a single-number answer. Exclude skills that are about formulating,
   proving, classifying, or constructing -- those do not fit the
   pipeline.

Self-audit before output:
(a) Pairwise orthogonality: are any two skills testing the same
    reasoning? If yes, fix.
(b) Restatement: does any skill's subproblem require solving the
    target? If yes, replace it.
(c) Coverage: is THE hardest insight represented in at least 2
    skills? If not, add them.
(d) Named techniques on hard skills: do the 2-3 hardest skills each
    name a specific technique / theorem / object? If not, rewrite
    them -- goal-only descriptions are disallowed.
(e) Anti-padding: is any pair of skills the same technique at
    different parameter values (n=2 vs n=3) or paired facts about
    the same object? Collapse them into one, fill the slot with
    a different skill.
(f) No numbers in skill text: does any skill name or description
    contain a specific numerical answer / dimension / degree?
    If yes, rewrite to describe the technique, not the value.
(f2) Parametric family: for each skill, could a subproblem-generator
    produce 10+ NON-IDENTICAL subproblems with DIFFERENT numerical
    answers that all test this skill? If not (the skill is pinned
    to one instance), broaden the description to name at least one
    parameter that can vary.
(g) No arithmetic-only skills: is any skill merely postprocessing
    (e.g. "compute floor of...", "subtract correction from...")?
    If yes, drop it and pick a real reasoning component.
(h) Difficulty spread: is there real progression from skill 1 to
    skill {n_skills} in terms of REASONING, not just parameter
    size? If not, widen it.
(i) Numerical answerability: does every skill admit a numeric
    subproblem? If not, drop it.
Revise internally until all nine pass, then output the JSON.

Respond with JSON only, no prose, exactly this shape:
{{
  "skills": [
    {{
      "name": "Short skill name (3-10 words)",
      "description": "1-2 sentences explaining what the skill is and why it is relevant to the target."
    }}
  ]
}}

There must be exactly {n_skills} entries in the skills array, ordered
by difficulty (easiest first, hardest last).
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
        raw = _robust_completion(client, prompt).strip()
        if not raw:
            last_err = ValueError("decompose got empty completion after retries")
            continue
        try:
            block = _extract_json_block(raw)
            parsed = json.loads(block)
            skills_list = parsed.get("skills")
            if not isinstance(skills_list, list):
                raise ValueError("'skills' key missing or not a list")
            if len(skills_list) != n_skills:
                raise ValueError(f"expected {n_skills} skills, got {len(skills_list)}")
            _required = ("name", "description")
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
    """Load skills from `path`, or return None if the file doesn't exist or is malformed.

    Tolerates extra keys (e.g. legacy `example_problem_hint` from old runs)
    by extracting only the fields the current Skill dataclass defines.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        skills = []
        for entry in data.get("skills", []):
            skills.append(Skill(
                name=entry["name"],
                description=entry["description"],
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


GENERATE_PROMPT = """\
You are designing one subproblem to help a student practice a specific
reasoning skill.

The end goal is mastery of this hard target problem (for context only --
do NOT generate a variant of the target):

{target}

Skill to test (#{skill_index} of {n_skills}, ordered by difficulty):
Name: {skill_name}
Description: {skill_description}

Other skills in the taxonomy -- your subproblem must NOT require any
of these to solve:
{other_skill_names_bulleted}

Previously generated subproblems for THIS skill and their final
numerical answers -- you MUST produce a subproblem that is neither a
paraphrase of any of these NOR has the same numerical answer as any
of them. If the skill admits a parametric family (varying degree,
dimension, number of conditions, configuration, etc.), pick a
DIFFERENT point in that family than any shown below. Do not default
to the simplest textbook case if it has already been used.

Structural diversity (load-bearing). Varying only a numerical
parameter while keeping the same sentence template is NOT enough.
If your problem reads as a find-and-replace of a previous problem
(same phrasing, same sentence order, same objects, just a different
number), you have failed. Instead, vary the STRUCTURE across
attempts. For a given skill, cycle through its equivalent
formulations, alternative framings, and distinct concrete
instantiations. Examples of structural variation (applied generally
to any skill, not just geometry): recast the same computation in a
different geometric object (curves -> hypersurfaces -> complete
intersections), use a different algebraic incarnation (polynomial
coefficient count vs cohomology vs Hilbert polynomial value), swap
the direction of the question (given degree, find dimension vs
given dimension, find degree), change the ambient setting (affine
vs projective vs weighted projective), or combine the skill with a
small non-leaky twist (a nonstandard basis, a non-obvious
identification) that doesn't cross into another skill. At minimum
the sentence pattern of the new problem must NOT be a substring
rewrite of any prior one.
{prior_attempts_bulleted}

Requirements:

1. Isolation. The subproblem tests the named skill ALONE. It must NOT
   require any of the other skills listed above. If you cannot
   construct a subproblem isolating this skill without invoking the
   others, output the literal string
   "UNISOLATABLE: <one sentence reason>"
   inside the <problem> tags instead of a problem statement. This
   signals the skill is not cleanly separable and the decomposition
   needs revision -- do not fake an isolated problem in that case.

2. No answer leakage. The problem statement must not state, imply, or
   walk the solver up to the numerical answer. In particular:
   - Do NOT include phrases like "which gives X", "therefore ... = X",
     "this equals X", "so the class is X * H", "note that the degree is X".
     These turn the problem into reading comprehension, not reasoning.
   - Do NOT include the final numerical answer -- as a digit, as a
     word, or as a closed-form expression that trivially evaluates to
     it -- anywhere in the setup.
   - Present only the givens; the solver must derive every computed
     quantity, including any intermediate quantity that determines the
     final answer.

3. Unambiguous answer. Exactly one conventional reading of the problem
   statement must yield exactly one answer. In particular:
   - If a term has multiple standard conventions in the field
     (e.g., "dimension of a projective space" is 11 vs 12 depending on
     whether you mean projective dim or vector-space dim; "degree of a
     map" can mean topological vs algebraic; indexing starting from 0
     vs 1), either disambiguate explicitly in-line or avoid the term.
   - The expected answer must NOT depend on which convention a competent
     solver adopts. If you cannot ensure this without leaking the
     answer, pick a different concrete instantiation of the skill.

4. No target leakage. Do not reproduce the target problem's setup,
   notation, or specific numerical parameters. The subproblem must be
   a fresh concrete instance so that solving it doesn't amount to
   partially solving the target.

5. Difficulty calibrated to position. This is skill #{skill_index}
   of {n_skills}. Skill #1 should be doable in 5-10 minutes by
   someone with the relevant background; skill #{n_skills} can
   require 30+ minutes of nontrivial work. Calibrate accordingly --
   do not make every subproblem the same difficulty.

6. The answer MUST be a single number (integer or decimal). If a
   decimal, ask the solver to round to 4 decimal places.

7. State the problem in 3-10 sentences.

8. ALL math must be written in LaTeX using \\(...\\) for inline and
   \\[...\\] for display. No ASCII math ("x^2", "sqrt(5)",
   "sum from i=1 to n", etc.) -- use proper LaTeX.

9. The problem statement MUST end with this exact sentence:
   "Put your final answer inside \\boxed{{}}."

Self-audit before output (do this internally, do not print):
(a) Answer-leak grep. Mentally solve the problem and note the final
    numerical answer N. Scan the problem statement. Does N appear
    anywhere -- as digit, word, or trivially-evaluatable expression?
    If yes, rewrite to remove it.
(b) Convention check. For every named quantity the solver must
    compute, is there a single standard convention in the field?
    If a term has multiple conventions, disambiguate or reformulate.
(c) Intermediate-leak grep. Does the statement hand the solver any
    computed quantity that determines N via routine arithmetic
    (e.g., giving the degree of each factor when the problem asks
    for the product of degrees)? If yes, remove or rephrase.
(d) Novelty check. Is the answer N equal to any of the prior answers
    listed above? Is the problem a paraphrase of any prior problem?
    If either, pick a different instance of the skill (different
    parameter, different configuration) and redo.
Revise internally until all four pass, then emit the problem.

Output format:
Begin your response with <problem> on its own line, then the full
problem statement (or the UNISOLATABLE sentinel), then </problem> on
its own line. No other text before, between, or after the tags.
"""


_PROBLEM_TAG_RE = re.compile(r"<problem>(.*?)</problem>", re.DOTALL)


def _parse_problem(raw: str) -> str:
    """Extract the <problem>...</problem> content. Return '' if tags missing."""
    m = _PROBLEM_TAG_RE.search(raw)
    if not m:
        return ""
    return m.group(1).strip()


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


def _generate_one_candidate(
    client,
    target: str,
    skill: Skill,
    *,
    skill_index: int,
    n_skills: int,
    other_skill_names: list[str],
    prior_problems: list[str] | None = None,
    prior_answers: list[str] | None = None,
    _temperature: float = TEMPERATURE,
) -> str:
    """Call the generator once, return the raw problem text (possibly empty).

    May also return the literal string "UNISOLATABLE: <reason>" if the
    model cannot construct a subproblem isolating this skill from the
    others -- this is a useful signal that the decomposition is
    miscalibrated, not an error.
    """
    other_bulleted = "\n".join(f"- {name}" for name in other_skill_names) or "(none)"
    prior_bulleted = _format_prior_attempts(
        prior_problems or [], prior_answers or [],
    )
    prompt = GENERATE_PROMPT.format(
        target=target,
        skill_index=skill_index,
        n_skills=n_skills,
        skill_name=skill.name,
        skill_description=skill.description,
        other_skill_names_bulleted=other_bulleted,
        prior_attempts_bulleted=prior_bulleted,
    )
    raw = _robust_completion(client, prompt, temperature=_temperature)
    return _parse_problem(raw)


def generate_for_skill(
    *,
    client,
    target: str,
    skill: Skill,
    skill_index: int,
    n_skills: int,
    other_skill_names: list[str],
    n_target: int,
    n_samples: int,
    max_candidates: int,
    agree_low: float,
    agree_high: float,
    solve_pool=None,
    on_keep=None,
    on_skip=None,
) -> tuple[list[dict], list[dict], dict]:
    """Generate candidates for a single skill until n_target keeps or max_candidates attempts.

    Args:
        skill_index: 1-based position of this skill in the difficulty
            ordering (for the calibrate-difficulty-by-position prompt).
        n_skills: total skills in the taxonomy.
        other_skill_names: names of the OTHER skills, used in the prompt
            to enforce that the generated subproblem doesn't require any
            of them.
        solve_pool: concurrent.futures.ThreadPoolExecutor used by the
            real solve_and_check_agreement to fan out n_samples solve
            calls. Required at runtime (the real function unconditionally
            calls pool.submit); left None for tests that monkeypatch the
            solve function.

    Returns:
        (keeps, skips, stats) where stats is
        {"name": skill.name, "n_target", "n_passed", "n_attempted", "status"}.
    """
    keeps: list[dict] = []
    skips: list[dict] = []
    attempted = 0
    # Memory of past candidates for THIS skill (problem text + majority answer)
    # fed back into the generator so it actively diversifies.
    prior_problems: list[str] = []
    prior_answers: list[str] = []

    while len(keeps) < n_target and attempted < max_candidates:
        attempted += 1
        problem_text = _generate_one_candidate(
            client, target, skill,
            skill_index=skill_index,
            n_skills=n_skills,
            other_skill_names=other_skill_names,
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
) -> None:
    """Orchestrate Phase 1 + Phase 2. Write keeps/skips/stats to out_dir.

    `max_workers` sizes the ThreadPoolExecutor used to fan out the
    n_samples solve calls per candidate. Required at runtime; tests
    that monkeypatch the solve function bypass the pool entirely.
    """
    import concurrent.futures

    _lazy_import_get_client()
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
    target_total = problems_per_skill * n_skills

    # Incremental-write callbacks: every time generate_for_skill accepts or
    # rejects a candidate, append it to the appropriate list and re-flush
    # keeps.json / skips.json so both files grow one entry at a time while
    # the run is in progress.
    def _on_keep(record: dict) -> None:
        all_keeps.append(record)
        _write_outputs(out_dir, target_text, agree_low, agree_high,
                       all_keeps, all_skips, all_stats, target_total)

    def _on_skip(record: dict) -> None:
        all_skips.append(record)
        _write_outputs(out_dir, target_text, agree_low, agree_high,
                       all_keeps, all_skips, all_stats, target_total)

    # One pool shared across all skills; it fans out each candidate's
    # n_samples solve calls. solve_and_check_agreement requires pool=
    # (it unconditionally calls pool.submit).
    skill_names = [s.name for s in skills]
    if start_from_skill > 1:
        print(f"\n(resume mode: skipping skills 1..{start_from_skill - 1}, "
              f"starting at skill {start_from_skill})")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as solve_pool:
        print(f"\n=== Per-skill generation ===")
        for i, skill in enumerate(skills, start=1):
            if i < start_from_skill:
                print(f"\n[{i}/{len(skills)}] {skill.name}  -- skipped (resume)")
                continue
            print(f"\n[{i}/{len(skills)}] {skill.name}")
            others = [name for j, name in enumerate(skill_names, start=1) if j != i]
            # on_keep / on_skip populate all_keeps / all_skips as each
            # candidate lands, so the returned lists are intentionally
            # ignored to avoid double-appending.
            _keeps, _skips, stats = generate_for_skill(
                client=client,
                target=target_text,
                skill=skill,
                skill_index=i,
                n_skills=len(skills),
                other_skill_names=others,
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
            print(f"  done: {stats['n_passed']}/{stats['n_target']} passed "
                  f"after {stats['n_attempted']} attempts ({stats['status']})")

            # End-of-skill flush: captures the updated stats + any skips
            # accumulated during this skill (skips aren't incrementally
            # written; keeps.json is the real-time view).
            _write_outputs(out_dir, target_text, agree_low, agree_high,
                           all_keeps, all_skips, all_stats, target_total)

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
    )


if __name__ == "__main__":
    main()
