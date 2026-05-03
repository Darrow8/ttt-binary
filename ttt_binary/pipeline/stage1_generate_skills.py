"""Stage 1 — generate X candidate skills for a target problem.

Input:  problem statement (string).
Output: data/skills/<problem_id>.json with a list of skill objects:
        {name, description, preconditions, postconditions, example}

The skill generator sees ONLY the problem statement — never a gold solution.

LEAKAGE POLICY:
We forbid skills that name the final answer or describe the full solution chain.
We DO NOT forbid skills from referencing classical results that produce
intermediate constants (degrees, dimensions, group orders, invariants) — these
are pieces of mathematical knowledge a strong solver brings to the problem,
not solution leakage.

PROMPT-LEAKAGE POLICY:
The prompt itself must not contain specific numbers, theorem names, or
mathematical objects that were chosen because they relate to the target
problem. All concrete examples in the prompt come from problems that are
deliberately unrelated to the target.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ttt_binary.llm import call_anthropic, parse_json_loose


SYSTEM_PROMPT = """\
You decompose a math problem into reusable reasoning skills. The problem is hard \
and you do NOT need to solve it. You will be given the problem statement only — \
no solution. Your job is to enumerate primitive techniques that a solver of this \
problem might compose. Treat each skill as a small, transferable tool a \
mathematician would invoke in many different problems, not as a step bound to \
this particular problem. Skills should reflect the actual mathematical \
machinery the problem requires — including classical results and named \
theorems — not vague generic descriptions of "doing math"."""


PROMPT_TEMPLATE = """\
Decompose the following problem into exactly {n_skills} distinct reasoning \
skills. Each skill is a PRIMITIVE TECHNIQUE that a strong mathematician might \
invoke as one step inside a longer solution. Skills must be REUSABLE across \
many problems in the same domain — they describe HOW to do something, not WHAT \
the final answer is.

WHAT MAKES A SKILL GOOD:
- It names a real mathematical move at the granularity of "one conceptual \
  step" — roughly one paragraph to explain. Not too coarse ("solve the \
  problem") and not too fine ("multiply two integers").
- It has clear input-output structure: you can describe what data it consumes \
  (preconditions) and what it produces (postconditions).
- It is reusable across many problems, not a one-shot trick specific to the \
  target.
- A skill MAY reference classical results by name and may indicate that those \
  classical results produce specific constants (degrees, dimensions, group \
  orders, etc.) that a solver looks up or invokes. This is technique, not \
  leakage.

LEAKAGE RULES — these are the only things that count as leaking:
- Skill names must NOT be the literal final answer to the target problem. \
  Names that bake in a specific numerical result (e.g. of the form \
  "evaluate_to_<N>", "answer_is_<N>", or a name that is just a number) leak \
  the answer. Names that describe the technique that produces such a result \
  are fine.
- Skill descriptions must NOT state the final answer to the target problem.
- Skill descriptions must NOT walk through the full solution chain — i.e. \
  do not describe a multi-step pipeline that, if followed, produces the \
  answer. Each skill is ONE move.
- Skill examples must use parameters drawn from a problem DIFFERENT from the \
  target. The example illustrates the technique on an unrelated setup.

WHAT IS NOT LEAKAGE (do not avoid these):
- Referencing classical theorems by name when they are standard tools in \
  the relevant field.
- Mentioning that a classical result produces a specific constant for a \
  specific input — as long as the skill is about HOW to invoke such a \
  result, not WHICH specific result solves the target.
- Naming standard mathematical objects (groups, rings, fields, sequences, \
  combinatorial constants) when they are part of the technique vocabulary.

TECHNIQUE-FAMILY BREADTH:
The {n_skills} skills should span the major mathematical families relevant \
to the target. First identify which kinds of machinery a strong solver \
would need by reading the problem carefully — different problems pull from \
different combinations of fields (algebra, analysis, combinatorics, \
geometry, number theory, probability, etc.). Aim for breadth: better to \
have one skill from each of several families than several skills from one \
family.

STRUCTURAL CONSTRAINTS:
- Skill names: lowercase snake_case identifiers describing the technique. \
  Names should describe what the skill DOES, not the answer it produces.
- Each skill must be PRIMITIVE — describable in 1-2 sentences. Do not lump \
  together a multi-step pipeline as one skill.
- The {n_skills} skills must be GENUINELY DISTINCT: no two should be \
  reformulations of each other.

For each skill emit:
- name: short snake_case identifier
- description: 1-2 sentences naming the TECHNIQUE and when it applies
- preconditions: what must be true to apply it
- postconditions: what is true after applying it (what the skill produces)
- example: a minimal worked example IN ISOLATION, drawn from a problem \
  different from the target

EXAMPLE OF A WELL-FORMED SKILL (from an unrelated problem about graph \
colorings — included only to calibrate format and abstraction level):
{{
  "name": "apply_deletion_contraction_recurrence",
  "description": "For a graph invariant satisfying a deletion-contraction \
identity (such as the chromatic polynomial), reduce computation on a graph \
G to computation on G with one edge deleted and G with that edge contracted, \
then recurse until reaching base cases.",
  "preconditions": "A graph G and an invariant P that satisfies \
P(G) = P(G \\\\ e) - P(G / e) for every edge e.",
  "postconditions": "The value of P on G is expressed in terms of P on \
strictly smaller graphs, leading to a base case (edgeless graph or single \
vertex) where the value is known.",
  "example": "To compute the chromatic polynomial of a triangle K_3 in k \
colors, pick any edge e and apply the recurrence: P(K_3, k) = P(P_3, k) \
- P(K_2, k) = k(k-1)^2 - k(k-1) = k(k-1)(k-2)."
}}

Notice this example skill: it names a classical technique (deletion- \
contraction), references specific objects (chromatic polynomial, K_3, P_3), \
and gives a concrete computation. None of this is leakage because the \
skill is reusable and the example is from a problem unrelated to the target.

Respond as a single JSON object with this exact shape (no prose around it):
{{
  "skills": [
    {{
      "name": "...",
      "description": "...",
      "preconditions": "...",
      "postconditions": "...",
      "example": "..."
    }}
  ]
}}

Problem:
{problem}
"""


# Patterns that suggest a skill name is the literal final answer rather than
# a technique. We reject names that are nothing but a number, or that contain
# answer-shaped phrases like "evaluate_to_<N>" or "answer_is_<N>".
_ANSWER_NAME_PATTERNS = [
    re.compile(r"^\d+$"),                             # pure number
    re.compile(r"evaluate[_\s]?to[_\s]?\d", re.I),    # evaluate_to_<digits>
    re.compile(r"answer[_\s]?is[_\s]?\d", re.I),      # answer_is_<digits>
    re.compile(r"final[_\s]?answer", re.I),
    re.compile(r"the[_\s]?answer", re.I),
]


def _looks_like_answer_name(name: str) -> bool:
    """True iff the skill name looks like a literal answer rather than a technique."""
    n = name.strip().lower()
    return any(pat.search(n) for pat in _ANSWER_NAME_PATTERNS)


def _validate_skills(skills: list[dict], n_skills: int) -> tuple[bool, str]:
    """Cheap structural + anti-leak validation. Returns (ok, reason)."""
    if len(skills) != n_skills:
        return False, f"expected {n_skills} skills, got {len(skills)}"
    seen = set()
    for s in skills:
        for f in ("name", "description", "preconditions", "postconditions", "example"):
            if not s.get(f) or not isinstance(s.get(f), str):
                return False, f"skill {s.get('name')!r} missing/empty field {f!r}"
        name = s["name"].strip()
        if not name:
            return False, "empty skill name"
        if _looks_like_answer_name(name):
            return False, (
                f"skill name {name!r} looks like a literal answer rather than "
                "a technique (names of the form 'evaluate_to_<N>', "
                "'answer_is_<N>', or pure numbers are not allowed)"
            )
        # Names should be snake_case-ish: no whitespace, alphanumeric+underscore
        if not re.match(r"^[a-z][a-z0-9_]*$", name):
            return False, (
                f"skill name {name!r} is not snake_case "
                "(must be lowercase, start with a letter, contain only [a-z0-9_])"
            )
        key = name.lower()
        if key in seen:
            return False, f"duplicate skill name: {name!r}"
        seen.add(key)
    return True, ""


def generate_skills(
    problem: str,
    *,
    n_skills: int = 10,
    model: str = "openai/gpt-oss-120b-maas",
    temperature: float = 0.7,
    max_retries: int = 4,
) -> list[dict]:
    """Generate n_skills, retrying if the generator violates structural or anti-leak rules."""
    prompt = PROMPT_TEMPLATE.format(n_skills=n_skills, problem=problem)
    last_err = ""
    for attempt in range(max_retries):
        retry_prompt = prompt
        if attempt > 0 and last_err:
            retry_prompt = (
                prompt
                + f"\n\nIMPORTANT: your previous attempt was invalid — {last_err}. "
                "Re-emit the JSON object with exactly the requested number of "
                "skills, all fields populated, and valid snake_case names that "
                "describe techniques rather than literal answers."
            )
        text = call_anthropic(
            retry_prompt, model=model, system=SYSTEM_PROMPT, temperature=temperature,
        )
        try:
            obj = parse_json_loose(text)
        except Exception as e:
            last_err = f"JSON parse failed: {e}"
            print(f"  [stage1 retry {attempt+1}] {last_err}", flush=True)
            continue
        skills = obj.get("skills", obj if isinstance(obj, list) else [])
        if not isinstance(skills, list):
            last_err = f"top-level not a list: {type(skills).__name__}"
            print(f"  [stage1 retry {attempt+1}] {last_err}", flush=True)
            continue
        ok, reason = _validate_skills(skills, n_skills)
        if ok:
            return skills
        last_err = reason
        print(f"  [stage1 retry {attempt+1}] {reason}", flush=True)
    raise RuntimeError(
        f"stage1 skill generation failed after {max_retries} attempts: {last_err}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem-id", required=True)
    ap.add_argument("--problem-file", required=True,
                    help="Path to a text file containing the problem statement")
    ap.add_argument("--n-skills", type=int, default=10)
    ap.add_argument("--model", default="openai/gpt-oss-120b-maas")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out-dir", default="ttt_binary/data/skills")
    args = ap.parse_args()

    problem = Path(args.problem_file).read_text().strip()
    skills = generate_skills(
        problem,
        n_skills=args.n_skills,
        model=args.model,
        temperature=args.temperature,
    )
    out_path = Path(args.out_dir) / f"{args.problem_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "problem_id": args.problem_id,
        "n_skills": len(skills),
        "model": args.model,
        "skills": skills,
    }, indent=2))
    print(f"wrote {out_path} ({len(skills)} skills)")


if __name__ == "__main__":
    main()
