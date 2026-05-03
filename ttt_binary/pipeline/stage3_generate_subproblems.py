"""Stage 3 — multi-part chained subproblem generation with consensus calibration.

For each unordered M-tuple of skills:
  1. Generator produces an M-part subproblem with cumulative dependency:
     part (a) requires skill 1; part (b) starts with "Let x = your answer
     to part (a)" and uses x as input to skill 2; part (c) starts with
     "Let y = your answer to part (b)" and uses skill 3. This makes the
     chain load-bearing by structure — you cannot solve later parts
     without earlier parts' answers.
  2. K critic solves on the full multi-part problem; each response ends with
     a JSON ANSWERS line giving one numerical answer per part.
  3. Per-part clustering yields per-part consensus answers and per-part
     solve rates s_i. Aggregate r_bar = mean(s_i) is the continuous reward
     expectation (also the future training reward = k/m).
  4. Accept iff every part is well-posed (p2_i < ambiguity_threshold,
     unparseable_i <= max_unparseable) AND r_bar ∈ band.
  5. Out-of-band → regenerate with directional + (future) shortcut-aware
     feedback. Cap at max_regen.

Output: data/subproblems/<problem_id>.json (multi-part schema)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import sys
import threading
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ttt_binary.answer_extract import extract_answers_multipart
from ttt_binary.cluster import (
    MultipartDecision,
    UNPARSEABLE,
    cluster_answers,
    decide_multipart,
    regen_feedback_multipart,
)
from ttt_binary.llm import call_anthropic, call_openai, parse_json_loose


GEN_SYSTEM = """\
You write multi-part math subproblems that exercise a specified set of \
reasoning skills in a strict chain. Each part requires exactly one of the \
listed skills. CRITICALLY, each part beyond the first must take its INPUT \
DATA from the previous part's ANSWER — making the chain load-bearing by \
structure, not just by topical association.

Each part has a single numerical final answer rounded to 4 decimal places. \
Do NOT solve the problem yourself or report any answers — reference answers \
come from majority consensus over independent solver attempts. The chain \
must be MINIMAL: solving the problem must genuinely require all {m} skills, \
and the question is wrong if any |S| < {m} subset of the skills suffices."""


GEN_PROMPT_TEMPLATE = """\
Produce ONE multi-part subproblem with exactly {m} parts labelled \
({part_labels}). Each part requires exactly one of the {m} skills below, in \
the listed order. The chain is enforced by ANSWER-LEVEL DEPENDENCY: each \
part beyond the first begins by binding a variable to the previous part's \
answer, and uses that variable as input.

HARD CONSTRAINTS:
- Each part has a single, fully-determined NUMERICAL final answer (a real \
  number) reportable inside \\boxed{} rounded to 4 decimal places.
- Part ({first_label}) is a well-posed standalone question requiring skill 1.
- For every subsequent part, its text MUST begin with the literal phrase \
  "Let <var> = your answer to part (<previous_label>)." and then use <var> \
  as input to that part's skill. (Use <var> = x for part b, y for part c, z \
  for part d, etc.) The dependency is structural, not optional.
- Each part's text must include "Round your final answer to 4 decimal places \
  and place it inside \\boxed{}."
- Do NOT mention skill names, the word "skill", or any meta-language inside \
  any part's text.
- Do NOT include or hint at any part's answer. Do NOT solve the problem.
- The full chain must be MINIMAL: removing any one skill must make the \
  overall problem unsolvable.
- Vary numerical parameters across drafts so different combinations produce \
  different per-part answers.
{difficulty_hint}

Skills (use ALL of them, in this order):
{skills_block}

Respond as a single JSON object (no prose around it) with this exact shape:
{
  "parts": [
    {
      "label": "{first_label}",
      "skill": "<skill_1_name>",
      "text": "the part's question, including the rounding-and-boxing instruction. NO reference to a previous part."
    },
    {
      "label": "<next_label>",
      "skill": "<skill_2_name>",
      "text": "Let x = your answer to part ({first_label}). ... uses x as input to skill 2 ... include the rounding-and-boxing instruction."
    }
    /* ... continue for all {m} parts in order ... */
  ],
  "skill_chain_rationale": "1-3 sentences describing the dependency order: which part's output feeds which part's input.",
  "per_skill_role": {
    "<skill_1_name>": "<one sentence: how part {first_label} invokes skill 1, what its input is, what its output is>",
    "<skill_2_name>": "..."
    /* one entry per skill */
  }
}

Verify before responding: (a) every one of the {m} skill names appears as a \
key in per_skill_role; (b) the parts list has exactly {m} entries with \
labels in order ({part_labels}); (c) every part beyond ({first_label}) \
literally begins with "Let <var> = your answer to part (...)"; (d) skipping \
any one skill makes the overall problem unsolvable.
"""


SOLVE_SYSTEM = """\
You are a careful and rigorous math student. The problem has multiple \
labelled parts; solve them in order. Each part has a single numerical final \
answer rounded to 4 decimal places.

For each part, work the solution step-by-step and end that part with \
\\boxed{X.XXXX} (e.g. \\boxed{866.0000}, \\boxed{0.5000}, \\boxed{-2.3457}). \
Always include trailing zeros to fill all 4 decimals.

After ALL parts, end your full response with a single line of the form:

ANSWERS: {"<label_1>": "X.XXXX", "<label_2>": "Y.YYYY", ...}

The ANSWERS line is mandatory and must contain EXACTLY one entry per part \
in the same labels as the problem."""


def _format_skill(s: dict) -> str:
    return (
        f"- {s['name']}\n"
        f"    description:    {s['description']}\n"
        f"    preconditions:  {s['preconditions']}\n"
        f"    postconditions: {s['postconditions']}\n"
        f"    example:        {s['example']}"
    )


PART_LABELS = "abcdefghijklmnopqrstuvwxyz"


def _labels_for(m: int) -> list[str]:
    if m > len(PART_LABELS):
        raise ValueError(f"chain length {m} exceeds supported labels (max {len(PART_LABELS)})")
    return list(PART_LABELS[:m])


def _hint_from_decision(prev: MultipartDecision | None,
                        shortcut_hint: str | None = None) -> str:
    """Build the difficulty hint for the next regeneration attempt.

    Combines the standard band-based feedback with an optional
    shortcut-aware hint produced by the audit (step 2 — currently a no-op
    stub; see _observe_shortcuts below).
    """
    parts: list[str] = []
    if prev is None:
        parts.append(
            "- Aim for r_bar (mean per-part solve rate) near 0.5: each part "
            "should be hard enough that a strong solver gets it right roughly "
            "half the time. Vary per-part difficulty so the chain has both "
            "tractable and challenging steps."
        )
    else:
        parts.append(regen_feedback_multipart(prev))
    if shortcut_hint:
        parts.append(shortcut_hint)
    return "\n".join(parts)


DIAGNOSE_JUDGE_SYSTEM = """\
You diagnose why a multi-part math problem fell outside its target difficulty \
band, given the problem, the m intended skills, and a sample of solver traces.

You produce a CONCRETE one-sentence diagnosis and a CONCRETE one-sentence regen \
instruction the generator can use to fix the next attempt. Generic advice \
("simplify it", "make it harder") is not useful — always name WHICH part is the \
bottleneck and WHAT specific structural change addresses it. If the diagnosis \
isn't clear from the traces, respond with nulls so the generator gets generic \
band-based feedback instead.
"""


# --- TOO_EASY: identify the shortcut across correct traces ----------------
SHORTCUT_PROMPT = """\
A multi-part math problem with intended {m}-skill chain is below. {n_traces} \
critic traces all reached the CORRECT answers, but r_bar={r_bar:.2f} (well above \
the 0.6 ceiling), suggesting they bypassed the chain via a shortcut.

Identify the DOMINANT SHORTCUT across these traces.

Problem:
---
{rendered_problem}
---

Intended skills:
{skills_block}

Per-part solve rates: {per_part_rates}

Correct solution traces:
{traces_block}

Respond as a JSON object:
{
  "diagnosis":          "<one-sentence: what shortcut do these traces share that bypasses the intended chain — name the alternative technique>",
  "responsible_parts":  ["<labels of parts the shortcut bypasses>"],
  "regen_instruction":  "<one-sentence concrete fix: e.g. 'In part (c), replace the n-th root cover with a quotient by a non-cyclic group so the fiber size cannot be read off the equation.'>"
}
If no clear common shortcut, respond with all three keys null.
"""


# --- TOO_HARD: identify why critics fail ----------------------------------
DIFFICULTY_PROMPT = """\
A multi-part math problem with intended {m}-skill chain is below. {n_traces} \
critic traces are shown. The problem was REJECT_TOO_HARD_OR_AMBIGUOUS: \
r_bar={r_bar:.2f} (below 0.4) or some part had too many unparseable answers.

Identify the DOMINANT cause of failure. Common categories:
  (1) ARITHMETIC OVERLOAD: solvers know the technique but the numbers are too \
      large / too many decimal places / too many cases.
  (2) WORDING AMBIGUITY: the problem is interpretable in multiple ways and \
      different solvers go down different paths.
  (3) MISSING CONSTRAINT: the answer is underdetermined (multiple valid roots, \
      free parameters, conventions).
  (4) SKILL MISFIT: solvers don't know which technique to use, so they thrash.
  (5) BAD COMPOSITION: an earlier part's answer makes a later part \
      ill-defined (e.g. division by zero, log of negative).

Problem:
---
{rendered_problem}
---

Intended skills:
{skills_block}

Per-part solve rates: {per_part_rates}
Per-part unparseable counts: {per_part_unparseable}

Solver traces (mixed correctness):
{traces_block}

Respond as a JSON object:
{
  "diagnosis":          "<one-sentence: which category, which part, what specifically>",
  "responsible_parts":  ["<labels of the bottleneck parts>"],
  "regen_instruction":  "<one-sentence concrete simplification: name parameters to shrink, an ambiguity to disambiguate, a constraint to add, etc.>"
}
If no clear cause, respond with all three keys null.
"""


# --- AMBIGUOUS: a specific part has two competing answers -----------------
AMBIGUITY_PROMPT = """\
A multi-part math problem with intended {m}-skill chain is below. {n_traces} \
solver traces are shown. The problem was REJECT_AMBIGUOUS: at least one part \
has two competing answer clusters near 50/50. Per-part stats: \
{per_part_summary}.

Identify WHY the ambiguous part admits two valid answers. Common causes:
  (a) TWO REAL ROOTS: an equation has multiple solutions and the problem \
      doesn't specify which to take.
  (b) CONVENTION: e.g. principal vs. signed value, branch of complex log, \
      orientation, ordering.
  (c) UNCLEAR INPUT: the previous part's answer is itself ambiguous, \
      cascading into this part.
  (d) WORDING: the problem statement is interpretable in two senses.

Problem:
---
{rendered_problem}
---

Intended skills:
{skills_block}

Solver traces (showing the disagreement):
{traces_block}

Respond as a JSON object:
{
  "diagnosis":          "<one-sentence: which part, which two answers, which cause>",
  "responsible_parts":  ["<labels of ambiguous parts>"],
  "regen_instruction":  "<one-sentence concrete disambiguation: pin the convention, restrict to one root, add a uniqueness clause, etc.>"
}
If unclear, respond with all three keys null.
"""


def _format_skills_for_judge(parts_objs: list[dict], per_skill_role: dict) -> str:
    lines: list[str] = []
    for p in parts_objs:
        sk = p["skill"]
        role = per_skill_role.get(sk, "(no role provided)")
        lines.append(f"- part ({p['label']}): {sk}\n    role: {role}")
    return "\n".join(lines)


def _format_per_part_rates(decision: MultipartDecision) -> str:
    return ", ".join(f"{p.label}={p.p1:.2f}" for p in decision.per_part)


def _format_per_part_unparseable(decision: MultipartDecision) -> str:
    return ", ".join(f"{p.label}={p.n_unparseable}" for p in decision.per_part)


def _format_per_part_summary(decision: MultipartDecision) -> str:
    return ", ".join(
        f"{p.label}: p1={p.p1:.2f}, p2={p.p2:.2f}, "
        f"top2={list(p.clusters.keys())[:2]}"
        for p in decision.per_part
    )


def _sample_fully_correct_traces(cal_attempts: list[dict],
                                 decision: MultipartDecision) -> list[str]:
    """Traces where every per-part answer matches its consensus."""
    from ttt_binary.cluster import _canonicalize
    consensus_canon = {p.label: p.consensus_answer for p in decision.per_part
                       if p.consensus_answer is not None}
    out = []
    for a in cal_attempts:
        if not a.get("ok") or not a.get("text"):
            continue
        pp = a.get("predicted_parts") or {}
        ok = True
        for label, cons in consensus_canon.items():
            pred = pp.get(label)
            if pred is None or _canonicalize(str(pred)) != cons:
                ok = False
                break
        if ok:
            out.append(a["text"])
    return out


def _sample_failed_traces(cal_attempts: list[dict],
                          decision: MultipartDecision) -> list[str]:
    """Traces that did NOT fully match consensus — the diverse/wrong ones.
    Includes traces with some parseable answers (informative) but excludes
    fully-empty failures."""
    from ttt_binary.cluster import _canonicalize
    consensus_canon = {p.label: p.consensus_answer for p in decision.per_part
                       if p.consensus_answer is not None}
    out = []
    for a in cal_attempts:
        if not a.get("ok") or not a.get("text"):
            continue
        pp = a.get("predicted_parts") or {}
        # Trace counts as "failed" if at least one part doesn't match consensus
        # OR consensus didn't exist for some part (which is itself the failure
        # signal).
        any_wrong = False
        for label in [p.label for p in decision.per_part]:
            pred = pp.get(label)
            cons = consensus_canon.get(label)
            if cons is None:
                any_wrong = True
                break
            if pred is None or _canonicalize(str(pred)) != cons:
                any_wrong = True
                break
        if any_wrong:
            out.append(a["text"])
    return out


def _sample_competing_cluster_traces(cal_attempts: list[dict],
                                     decision: MultipartDecision,
                                     ambiguous_label: str) -> list[str]:
    """For an ambiguous part, sample traces from BOTH top clusters so the judge
    can see the disagreement directly."""
    from ttt_binary.cluster import _canonicalize, UNPARSEABLE
    # Find the ambiguous part's top two cluster keys.
    target = next((p for p in decision.per_part if p.label == ambiguous_label), None)
    if target is None:
        return []
    parseable = [(k, v) for k, v in target.clusters.items() if k != UNPARSEABLE]
    parseable.sort(key=lambda kv: -kv[1])
    if len(parseable) < 2:
        return []
    cluster_a, cluster_b = parseable[0][0], parseable[1][0]
    in_a, in_b = [], []
    for a in cal_attempts:
        if not a.get("ok") or not a.get("text"):
            continue
        pp = a.get("predicted_parts") or {}
        pred = pp.get(ambiguous_label)
        if pred is None:
            continue
        canon = _canonicalize(str(pred))
        if canon == cluster_a and len(in_a) < 3:
            in_a.append(a["text"])
        elif canon == cluster_b and len(in_b) < 3:
            in_b.append(a["text"])
    return in_a + in_b


def _observe_shortcuts(
    parts_objs: list[dict],
    per_skill_role: dict,
    cal_attempts: list[dict],
    decision: MultipartDecision,
    *,
    judge_model: str,
    max_traces: int = 5,
) -> str | None:
    """Data-driven feedback loop for ALL three rejection modes (symmetric).

    Routes by decision.kind:
      - REJECT_TOO_EASY            -> identify the shortcut (correct traces)
      - REJECT_TOO_HARD_OR_AMBIGUOUS -> identify the difficulty cause (failed traces)
      - REJECT_AMBIGUOUS           -> identify why the ambiguous part splits
                                       (traces from both competing clusters)
      - ACCEPT or other            -> None

    Returns a one-sentence regen hint or None if signal is too weak / judge
    call fails. The name is preserved for backwards compatibility but the
    function now handles all failure modes, not just shortcuts.
    """
    if decision.kind == "ACCEPT" or not decision.per_part:
        return None

    rendered = _render_full_problem(parts_objs)
    skills_block = _format_skills_for_judge(parts_objs, per_skill_role or {})
    per_part_rates = _format_per_part_rates(decision)
    m = len(parts_objs)

    if decision.kind == "REJECT_TOO_EASY":
        traces = _sample_fully_correct_traces(cal_attempts, decision)[:max_traces]
        if len(traces) < 2:
            return None
        traces_block = "\n\n--- TRACE ---\n".join(t[:3500] for t in traces)
        prompt = (
            SHORTCUT_PROMPT
            .replace("{n_traces}", str(len(traces)))
            .replace("{m}", str(m))
            .replace("{r_bar:.2f}", f"{decision.r_bar:.2f}")
            .replace("{rendered_problem}", rendered)
            .replace("{skills_block}", skills_block)
            .replace("{per_part_rates}", per_part_rates)
            .replace("{traces_block}", traces_block)
        )
        tag = "OBSERVED SHORTCUT"
    elif decision.kind == "REJECT_TOO_HARD_OR_AMBIGUOUS":
        traces = _sample_failed_traces(cal_attempts, decision)[:max_traces]
        if len(traces) < 2:
            return None
        traces_block = "\n\n--- TRACE ---\n".join(t[:3500] for t in traces)
        prompt = (
            DIFFICULTY_PROMPT
            .replace("{n_traces}", str(len(traces)))
            .replace("{m}", str(m))
            .replace("{r_bar:.2f}", f"{decision.r_bar:.2f}")
            .replace("{rendered_problem}", rendered)
            .replace("{skills_block}", skills_block)
            .replace("{per_part_rates}", per_part_rates)
            .replace("{per_part_unparseable}", _format_per_part_unparseable(decision))
            .replace("{traces_block}", traces_block)
        )
        tag = "OBSERVED DIFFICULTY ISSUE"
    elif decision.kind == "REJECT_AMBIGUOUS":
        # Find the ambiguous part (largest p2).
        ambig = max(decision.per_part, key=lambda p: p.p2)
        traces = _sample_competing_cluster_traces(cal_attempts, decision, ambig.label)[:max_traces]
        if len(traces) < 2:
            return None
        traces_block = "\n\n--- TRACE ---\n".join(t[:3500] for t in traces)
        prompt = (
            AMBIGUITY_PROMPT
            .replace("{n_traces}", str(len(traces)))
            .replace("{m}", str(m))
            .replace("{rendered_problem}", rendered)
            .replace("{skills_block}", skills_block)
            .replace("{per_part_summary}", _format_per_part_summary(decision))
            .replace("{traces_block}", traces_block)
        )
        tag = "OBSERVED AMBIGUITY"
    else:
        return None

    try:
        response = call_openai(
            prompt,
            model=judge_model,
            system=DIAGNOSE_JUDGE_SYSTEM,
            temperature=0.0,
        )
        obj = parse_json_loose(response)
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None
    diagnosis = obj.get("diagnosis")
    responsible = obj.get("responsible_parts") or []
    instruction = obj.get("regen_instruction")
    if not diagnosis or not instruction:
        return None

    parts_str = ", ".join(responsible) if responsible else "(unspecified)"
    return (
        f"- {tag} in the previous attempt (parts {parts_str}): {diagnosis} "
        f"For the next attempt: {instruction}"
    )


def _generate_one(
    skills_in_combo: list[dict],
    *,
    prev_decision: MultipartDecision | None,
    shortcut_hint: str | None,
    generator_model: str,
    temperature: float,
) -> dict:
    skills_block = "\n".join(_format_skill(s) for s in skills_in_combo)
    m = len(skills_in_combo)
    labels = _labels_for(m)
    # Use plain .replace() rather than .format() because the template body
    # contains literal LaTeX braces (\boxed{X.XXXX}, etc) that would otherwise
    # be misinterpreted as positional placeholders.
    prompt = (
        GEN_PROMPT_TEMPLATE
        .replace("{m}", str(m))
        .replace("{first_label}", labels[0])
        .replace("{part_labels}", ", ".join(labels))
        .replace("{difficulty_hint}",
                 _hint_from_decision(prev_decision, shortcut_hint))
        .replace("{skills_block}", skills_block)
    )
    text = call_anthropic(
        prompt,
        model=generator_model,
        system=GEN_SYSTEM.replace("{m}", str(m)),
        temperature=temperature,
    )
    obj = parse_json_loose(text)
    for k in ("parts", "skill_chain_rationale", "per_skill_role"):
        if k not in obj:
            raise ValueError(f"generator missing field {k}: keys={list(obj)}")

    parts = obj["parts"]
    if not isinstance(parts, list) or len(parts) != m:
        raise ValueError(
            f"parts must be a list of length {m}, got "
            f"{type(parts).__name__} of length "
            f"{len(parts) if isinstance(parts, list) else 'N/A'}"
        )
    seen_labels = []
    for i, part in enumerate(parts):
        if not isinstance(part, dict):
            raise ValueError(f"parts[{i}] must be a dict, got {type(part).__name__}")
        for k in ("label", "skill", "text"):
            if k not in part:
                raise ValueError(f"parts[{i}] missing field {k}: keys={list(part)}")
        if part["label"] != labels[i]:
            raise ValueError(
                f"parts[{i}].label = {part['label']!r}, expected {labels[i]!r}"
            )
        seen_labels.append(part["label"])
        # Cumulative-dependency check for parts beyond the first.
        if i > 0:
            text_lower = part["text"].lstrip().lower()
            prev_label = labels[i - 1]
            if not text_lower.startswith(("let ",)) or f"part ({prev_label})" not in part["text"].lower():
                raise ValueError(
                    f"parts[{i}].text must begin with 'Let <var> = your answer "
                    f"to part ({prev_label}).' — got prefix "
                    f"{part['text'][:80]!r}"
                )

    psr = obj["per_skill_role"]
    if not isinstance(psr, dict):
        raise ValueError(f"per_skill_role must be a dict, got {type(psr).__name__}")
    expected_names = {s["name"] for s in skills_in_combo}
    role_names = {str(k) for k in psr.keys()}
    missing = expected_names - role_names
    if missing:
        raise ValueError(
            f"per_skill_role missing entries for: {sorted(missing)}; "
            f"got keys={sorted(role_names)} (decorative-skill failure)"
        )
    # Sanity: each part's listed skill must be one of the combo skills.
    for i, part in enumerate(parts):
        if part["skill"] not in expected_names:
            raise ValueError(
                f"parts[{i}].skill = {part['skill']!r} not in combo "
                f"{sorted(expected_names)}"
            )
    return obj


def _render_full_problem(parts: list[dict]) -> str:
    """Render the multi-part problem as a single text block to send to the critic."""
    lines: list[str] = []
    for p in parts:
        lines.append(f"Part ({p['label']}). {p['text']}")
    lines.append(
        "\nAfter solving every part, end your response with a single line of "
        "the form: ANSWERS: {\"a\": \"X.XXXX\", \"b\": \"Y.YYYY\", ...} with one "
        "entry per part."
    )
    return "\n\n".join(lines)


def _calibrate(
    parts: list[dict],
    *,
    k: int,
    critic_model: str,
    parallel: int = 8,
) -> list[dict]:
    """Run K critic solves on a multi-part problem at temperature 0.7 in parallel.

    Each attempt receives the full rendered multi-part problem and is asked
    to produce one boxed answer per part plus a final ANSWERS: {...} JSON
    line. The line is parsed into a dict keyed by part label.

    Returns per-attempt records:
        {ok: bool, predicted_parts: dict[label, str|None], text: str, error?: str}
    """
    labels = [p["label"] for p in parts]
    rendered = _render_full_problem(parts)

    def one_attempt(_i: int) -> dict:
        try:
            text = call_openai(
                rendered,
                model=critic_model,
                system=SOLVE_SYSTEM,
                temperature=0.7,
            )
        except Exception as e:
            return {
                "ok": False,
                "error": str(e)[:200],
                "predicted_parts": {label: None for label in labels},
                "text": "",
            }
        predicted_parts = extract_answers_multipart(text, labels)
        return {"ok": True, "text": text, "predicted_parts": predicted_parts}

    attempts: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=parallel) as pool:
        for r in pool.map(one_attempt, range(k)):
            attempts.append(r)
    return attempts


def _process_combo(
    combo_idx: int,
    skills_in_combo: list[dict],
    *,
    band: tuple[float, float],
    ambiguity_threshold: float,
    k_calibrate: int,
    max_regen: int,
    generator_model: str,
    critic_model: str,
    judge_model: str,
    temperature: float,
    max_unparseable: int = 3,
    write_attempt=None,  # callable(record: dict, accepted: bool) -> None
) -> dict:
    """Generate→calibrate→per-part-cluster→multipart-decide→regen loop.

    Output record schema (multi-part, consensus-based):
        parts (list of {label, skill, text, consensus_answer, p1, p2,
                        n_unparseable, clusters}),
        r_bar (mean of per-part p1),
        per_skill_role, skill_chain_rationale,
        per_iteration log, calibration_attempts.

    Status values: "accepted" / "REJECT_*" / "errored_out".
    """
    skill_names = [s["name"] for s in skills_in_combo]
    last_decision: MultipartDecision | None = None
    last_record: dict | None = None
    per_iteration: list[dict] = []
    n_transient_errors = 0
    attempt = 0
    # Cumulative shortcut log: every observed shortcut across all regen
    # iterations is passed forward, so the generator sees the full list of
    # bypass techniques to block on the next attempt.
    shortcut_history: list[str] = []

    while attempt <= max_regen:
        # Concatenate ALL prior shortcut hints into one block so the generator
        # cannot reintroduce a previously-blocked shortcut.
        shortcut_hint = "\n".join(shortcut_history) if shortcut_history else None
        try:
            gen = _generate_one(
                skills_in_combo,
                prev_decision=last_decision,
                shortcut_hint=shortcut_hint,
                generator_model=generator_model,
                temperature=temperature,
            )
            cal_attempts = _calibrate(
                gen["parts"],
                k=k_calibrate,
                critic_model=critic_model,
            )
        except Exception as e:
            n_transient_errors += 1
            print(f"  [combo {combo_idx}] error attempt {attempt}: {e}", flush=True)
            per_iteration.append({
                "attempt": attempt,
                "kind": "ERROR",
                "reason": f"{type(e).__name__}: {e}"[:300],
            })
            time.sleep(1)
            if n_transient_errors >= 3:
                rec = {
                    "combo_idx": combo_idx,
                    "skills_used": skill_names,
                    "status": "errored_out",
                    "regeneration_attempts": attempt,
                    "n_transient_errors": n_transient_errors,
                    "per_iteration": per_iteration,
                    "error": f"{type(e).__name__}: {e}"[:300],
                }
                if write_attempt is not None:
                    write_attempt(rec, accepted=False)
                return rec
            continue

        # Per-part clustering on the K critic responses.
        gen_parts = gen["parts"]
        labels = [p["label"] for p in gen_parts]
        per_part_clusters: list[tuple[str, dict[str, int]]] = []
        for label in labels:
            preds = [a.get("predicted_parts", {}).get(label) for a in cal_attempts]
            per_part_clusters.append((label, cluster_answers(preds)))

        decision = decide_multipart(
            per_part_clusters,
            k_calibrate=k_calibrate,
            band=band,
            ambiguity_threshold=ambiguity_threshold,
            max_unparseable=max_unparseable,
        )
        last_decision = decision

        # Step 2: when the problem was too easy, ask the judge what shortcut
        # the critic used and append that to the cumulative shortcut log so
        # subsequent regens block all previously-observed bypass techniques.
        new_shortcut = _observe_shortcuts(
            gen_parts,
            gen.get("per_skill_role") or {},
            cal_attempts,
            decision,
            judge_model=judge_model,
        )
        if new_shortcut:
            shortcut_history.append(new_shortcut)

        per_iteration.append({
            "attempt": attempt,
            "kind": decision.kind,
            "r_bar": decision.r_bar,
            "per_part": [
                {
                    "label": p.label,
                    "consensus": p.consensus_answer,
                    "p1": p.p1,
                    "p2": p.p2,
                    "n_unparseable": p.n_unparseable,
                }
                for p in decision.per_part
            ],
            "reason": decision.reason,
        })

        record = {
            "combo_idx": combo_idx,
            "skills_used": skill_names,
            "per_skill_role": gen.get("per_skill_role"),
            "skill_chain_rationale": gen.get("skill_chain_rationale"),
            "parts": [
                {
                    "label": gp["label"],
                    "skill": gp["skill"],
                    "text": gp["text"],
                    "consensus_answer": pp.consensus_answer,
                    "p1": pp.p1,
                    "p2": pp.p2,
                    "n_unparseable": pp.n_unparseable,
                    "clusters": dict(pp.clusters),
                }
                for gp, pp in zip(gen_parts, decision.per_part)
            ],
            "r_bar": decision.r_bar,
            "k_calibrate": k_calibrate,
            "regeneration_attempts": attempt,
            "per_iteration": list(per_iteration),
            "calibration_attempts": [
                {
                    "ok": a.get("ok"),
                    "predicted_parts": a.get("predicted_parts"),
                    "error": a.get("error"),
                    "text": a.get("text"),
                }
                for a in cal_attempts
            ],
        }
        last_record = record

        per_part_summary = ", ".join(f"{p.label}={p.p1:.2f}" for p in decision.per_part)
        if decision.kind == "ACCEPT":
            record["status"] = "accepted"
            print(
                f"  [combo {combo_idx}] accepted "
                f"(r_bar={decision.r_bar:.2f}, parts: {per_part_summary}, "
                f"attempts={attempt+1})",
                flush=True,
            )
            if write_attempt is not None:
                write_attempt(record, accepted=True)
            return record

        # Rejected — log immediately, then regen.
        print(
            f"  [combo {combo_idx}] {decision.kind} "
            f"(r_bar={decision.r_bar:.2f}, parts: {per_part_summary}); "
            f"regen {attempt+1}/{max_regen}",
            flush=True,
        )
        if write_attempt is not None:
            rec_for_skip = dict(record, status=decision.kind)
            write_attempt(rec_for_skip, accepted=False)
        attempt += 1

    if last_record is None:
        rec = {
            "combo_idx": combo_idx,
            "skills_used": skill_names,
            "status": "errored_out",
            "regeneration_attempts": attempt,
            "per_iteration": per_iteration,
            "cap_out": True,
        }
        if write_attempt is not None:
            write_attempt(rec, accepted=False)
        return rec
    last_record["status"] = last_decision.kind if last_decision else "errored_out"
    last_record["cap_out"] = True
    if write_attempt is not None:
        write_attempt(last_record, accepted=False)
    return last_record


def _aggregate_stats(results: list[dict]) -> dict:
    """Run-end summary stats over multi-part records."""
    n = len(results)
    accepted = [r for r in results if r.get("status") == "accepted"]
    fail_reasons = Counter(
        r.get("status") for r in results if r.get("status") != "accepted"
    )
    r_bar_values = [
        r.get("r_bar") for r in accepted
        if isinstance(r.get("r_bar"), (int, float))
    ]
    r_bar_hist_bins = [0.40, 0.45, 0.50, 0.55, 0.60]
    r_bar_hist = {f"<= {b:.2f}": 0 for b in r_bar_hist_bins}
    for v in r_bar_values:
        for b in r_bar_hist_bins:
            if v <= b:
                r_bar_hist[f"<= {b:.2f}"] += 1
                break
    r_bar_mean = (sum(r_bar_values) / len(r_bar_values)) if r_bar_values else None

    # Per-part-rate stats across accepted records (s_i values flattened).
    s_values: list[float] = []
    for r in accepted:
        for p in r.get("parts", []) or []:
            v = p.get("p1")
            if isinstance(v, (int, float)):
                s_values.append(v)
    s_mean = (sum(s_values) / len(s_values)) if s_values else None

    return {
        "n_total": n,
        "n_accepted": len(accepted),
        "pct_accepted": (len(accepted) / n) if n else 0.0,
        "fail_counts": dict(fail_reasons),
        "fail_pcts": {k: v / n for k, v in fail_reasons.items()} if n else {},
        "r_bar_mean_accepted": r_bar_mean,
        "r_bar_histogram_accepted": r_bar_hist,
        "per_part_solve_rate_mean": s_mean,
        "n_per_part_observations": len(s_values),
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path or not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def generate_subproblems(
    problem_id: str,
    skills: list[dict],
    *,
    m: int,
    band: tuple[float, float],
    ambiguity_threshold: float,
    k_calibrate: int,
    max_regen: int,
    generator_model: str,
    critic_model: str,
    temperature: float,
    workers: int,
    out_path: Path,
    judge_model: str | None = None,   # step 2: model for shortcut-detection judge
    keeps_path: Path | None = None,
    skips_path: Path | None = None,
    max_unparseable: int = 3,
    max_combos: int | None = None,
) -> list[dict]:
    """For each C(X, M) combination: generate -> calibrate -> decide -> regen.

    Per-attempt writes:
        keeps_path  -- one JSONL line per ACCEPTED combo (final accepted record)
        skips_path  -- one JSONL line per REJECTED attempt (mid-run AND cap-out)

    Both files are appended to as work happens, so partial progress survives a
    crash. The final consolidated <id>.json is still written at the end.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if keeps_path is None:
        keeps_path = out_path.parent / f"{problem_id}.keeps.jsonl"
    if skips_path is None:
        skips_path = out_path.parent / f"{problem_id}.skips.jsonl"

    # Shuffle combinations deterministically by problem_id so:
    #   (a) `--max-combos N` smoke runs sample across the skill pool, not just
    #       the first N lexicographic tuples (which all share skills 0 and 1);
    #   (b) re-running with the same problem_id yields the same ordering, so
    #       resume by combo_idx works correctly across runs;
    #   (c) different problems get different orderings.
    n_skills = len(skills)
    full_combos = list(combinations(range(n_skills), m))
    rng = random.Random(f"stage3:{problem_id}:{n_skills}:{m}")
    rng.shuffle(full_combos)
    combos = full_combos
    if max_combos is not None and max_combos < len(combos):
        combos = combos[:max_combos]
        print(f"smoke mode: capped at {max_combos} of "
              f"C({n_skills},{m}) shuffled combinations", flush=True)
    print(f"enumerating {len(combos)} combinations of {m} skills from {n_skills}",
          flush=True)

    # Resume support: a combo is "done" if it has a keep, OR a skip record
    # with cap_out=True, OR a skip record with errored_out status.
    keep_records = _read_jsonl(keeps_path)
    skip_records = _read_jsonl(skips_path)
    accepted_idx: dict[int, dict] = {r["combo_idx"]: r for r in keep_records
                                     if "combo_idx" in r}
    capped_idx: dict[int, dict] = {}
    for r in skip_records:
        if "combo_idx" not in r:
            continue
        if r.get("cap_out") is True or r.get("status") == "errored_out":
            capped_idx[r["combo_idx"]] = r
    done_ids = set(accepted_idx) | set(capped_idx)
    if done_ids:
        print(
            f"  resume: {len(accepted_idx)} accepted in {keeps_path.name}, "
            f"{len(capped_idx)} cap-outs in {skips_path.name} "
            f"-> skipping {len(done_ids)} combos",
            flush=True,
        )

    todo = [(i, c) for i, c in enumerate(combos) if i not in done_ids]

    write_lock = threading.Lock()

    def write_attempt(rec: dict, *, accepted: bool) -> None:
        path = keeps_path if accepted else skips_path
        line = json.dumps(rec) + "\n"
        with write_lock:
            with path.open("a") as f:
                f.write(line)
                f.flush()

    # Default judge_model to the critic model so a single-model setup just
    # works without extra config; users can override for cross-model judging.
    judge_model_resolved = judge_model or critic_model

    def task(item):
        i, idx_tuple = item
        return _process_combo(
            i,
            [skills[j] for j in idx_tuple],
            band=band,
            ambiguity_threshold=ambiguity_threshold,
            k_calibrate=k_calibrate,
            max_regen=max_regen,
            generator_model=generator_model,
            critic_model=critic_model,
            judge_model=judge_model_resolved,
            temperature=temperature,
            max_unparseable=max_unparseable,
            write_attempt=write_attempt,
        )

    new_results: list[dict] = []
    if workers <= 1:
        for item in todo:
            new_results.append(task(item))
    else:
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(task, item) for item in todo]
            for fut in cf.as_completed(futs):
                new_results.append(fut.result())

    # Combine resumed + new results for the rolled-up summary.
    all_results: list[dict] = []
    all_results.extend(accepted_idx.values())
    # For capped-out, drop ones we just re-ran (shouldn't happen since they
    # were in done_ids) and prefer the new record otherwise.
    new_idx = {r.get("combo_idx") for r in new_results}
    for cidx, rec in capped_idx.items():
        if cidx not in new_idx and cidx not in accepted_idx:
            all_results.append(rec)
    all_results.extend(new_results)

    all_results.sort(key=lambda r: r.get("combo_idx", -1))
    accepted = [r for r in all_results if r.get("status") == "accepted"]
    failed = [r for r in all_results if r.get("status") != "accepted"]
    stats = _aggregate_stats(all_results)

    out_path.write_text(json.dumps({
        "problem_id": problem_id,
        "m": m,
        "n_skills": len(skills),
        "n_combinations": len(combos),
        "n_accepted": len(accepted),
        "n_failed": len(failed),
        "band": band,
        "ambiguity_threshold": ambiguity_threshold,
        "k_calibrate": k_calibrate,
        "max_regen": max_regen,
        "keeps_file": str(keeps_path),
        "skips_file": str(skips_path),
        "stats": stats,
        "subproblems": accepted,
        "failed": failed,
    }, indent=2))
    print(f"wrote {out_path} (accepted={len(accepted)}, failed={len(failed)})")
    print(f"  per-generation logs: {keeps_path}, {skips_path}")
    print("aggregate stats:", json.dumps(stats, indent=2))
    return accepted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem-id", required=True)
    ap.add_argument("--skills-file", required=True,
                    help="Path to skills JSON from Stage 1")
    ap.add_argument("--m", type=int, default=3, help="chain length")
    ap.add_argument("--band-lo", type=float, default=0.4)
    ap.add_argument("--band-hi", type=float, default=0.6)
    ap.add_argument("--ambiguity-threshold", type=float, default=0.2,
                    help="max allowed second-cluster fraction p2 (REVISIONS.md)")
    ap.add_argument("--k-calibrate", type=int, default=10)
    ap.add_argument("--max-regen", type=int, default=5)
    ap.add_argument("--max-unparseable", type=int, default=3,
                    help="reject if more than this many critic attempts return no parseable answer")
    ap.add_argument("--max-combos", type=int, default=None,
                    help="process at most N of the C(X,M) combinations (smoke-test mode)")
    ap.add_argument("--generator-model", default="openai/gpt-oss-120b-maas")
    ap.add_argument("--critic-model", default="openai/gpt-oss-120b-maas")
    ap.add_argument("--judge-model", default=None,
                    help="Step 2 shortcut-detection judge model (default: same as critic)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default="ttt_binary/data/subproblems")
    ap.add_argument("--keeps-file", default=None,
                    help="JSONL path appended on each ACCEPTED combo (default: <out-dir>/<id>.keeps.jsonl)")
    ap.add_argument("--skips-file", default=None,
                    help="JSONL path appended on each REJECTED attempt (default: <out-dir>/<id>.skips.jsonl)")
    args = ap.parse_args()

    skills_obj = json.loads(Path(args.skills_file).read_text())
    skills = skills_obj["skills"]
    out_path = Path(args.out_dir) / f"{args.problem_id}.json"
    keeps = Path(args.keeps_file) if args.keeps_file else None
    skips = Path(args.skips_file) if args.skips_file else None
    generate_subproblems(
        problem_id=args.problem_id,
        skills=skills,
        m=args.m,
        band=(args.band_lo, args.band_hi),
        ambiguity_threshold=args.ambiguity_threshold,
        k_calibrate=args.k_calibrate,
        max_regen=args.max_regen,
        generator_model=args.generator_model,
        critic_model=args.critic_model,
        judge_model=args.judge_model,
        temperature=args.temperature,
        workers=args.workers,
        out_path=out_path,
        keeps_path=keeps,
        skips_path=skips,
        max_unparseable=args.max_unparseable,
        max_combos=args.max_combos,
    )


if __name__ == "__main__":
    main()
