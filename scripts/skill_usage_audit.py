"""Skill-usage audit for chain-composition subproblems.

Question: when the critic produces a CORRECT solution to a chain-of-m problem,
does its solution actually invoke each of the m intended skills, or does it
shortcut via a different path? If most correct traces bypass the chain, the
chain is decorative — and the generator needs harder constraints.

Input: a JSONL file produced by stage3 (keeps or skips); each record has
fields skills_used, per_skill_role, skill_chain_rationale, problem_text,
consensus_answer, and calibration_attempts (each with predicted+text).

For each (problem, correct_trace) pair we ask a judge LLM:
  - did this trace USE / PARTIALLY_USE / NOT_USE skill X (one judgment per skill)?
  - did the trace follow the intended chain order?

Output: <out>.jsonl with one record per (problem, trace, skill) judgment, plus
a printed summary aggregating per-skill usage rates and per-problem chain
fidelity.

Usage:
  python scripts/skill_usage_audit.py \\
      --input ttt_binary/data/subproblems/conics-v3-smoke5b.skips.jsonl \\
      --out  ttt_binary/data/audits/conics-v3-smoke5b.skill_usage.jsonl \\
      --max-problems 10 --traces-per-problem 3
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
load_dotenv()

from ttt_binary.llm import call_openai, parse_json_loose


JUDGE_SYSTEM = """\
You are a careful evaluator of mathematical reasoning. You read a problem, a \
list of named techniques (skills) that the problem was designed to require, \
and a candidate solution trace. Your job is to determine, for each named \
skill, whether the trace actually invokes that skill in solving the problem. \
Do not be charitable: if the trace does not explicitly use a skill, mark it \
NOT_USED, even if the skill could in principle have been useful."""


JUDGE_PROMPT = """\
A problem is below, along with the {m} named skills the problem author \
intended a solver to chain in order. A solution trace from a strong LLM is \
also below. The trace produced the correct final answer.

Your task: for EACH skill listed, judge whether the trace's reasoning \
actually invokes that skill. Use one of:
  - USED:           the trace explicitly uses this skill (or a textbook \
synonym) as a load-bearing step in its argument.
  - PARTIALLY_USED: the trace gestures at the skill but does not rely on it \
(e.g., mentions the concept without using its conclusion).
  - NOT_USED:       the trace reaches the answer without invoking this skill \
at all (i.e., it took a shortcut or used different machinery).

Also judge whether the trace follows the intended chain ORDER (the skill \
chain rationale describes which skill's output should feed which skill's \
input). Use:
  - FOLLOWS_CHAIN:    the trace's argument structure matches the intended \
dependency order.
  - DIFFERENT_PATH:   the trace solves the problem with a substantially \
different argument structure, even if it uses some of the same skills.
  - SHORTCUT:         the trace bypasses the chain entirely (e.g., recalls \
a memorized result, or uses a single high-level theorem that obviates the \
intended decomposition).

Respond as a single JSON object (no prose around it):
{
  "skill_usage": {
    "<skill_name_1>": "USED" | "PARTIALLY_USED" | "NOT_USED",
    "<skill_name_2>": "...",
    ...
  },
  "chain_fidelity": "FOLLOWS_CHAIN" | "DIFFERENT_PATH" | "SHORTCUT",
  "rationale": "1-2 sentences justifying the judgments above."
}

Problem:
---
{problem_text}
---

Intended skills (the author's stated chain):
{skills_block}

Intended chain rationale:
{chain_rationale}

Solution trace (which produced the correct answer {consensus_answer}):
---
{trace_text}
---
"""


def _format_skills_block(skills_used: list[str], per_skill_role: dict) -> str:
    lines = []
    for name in skills_used:
        role = per_skill_role.get(name, "(no role provided)")
        lines.append(f"- {name}\n    role: {role}")
    return "\n".join(lines)


def _judge_one(record: dict, trace_idx: int, *, judge_model: str) -> dict | None:
    """Run one judge call for (record, trace_idx). Returns the judge JSON augmented
    with bookkeeping fields, or None on parse/network failure."""
    cal = record.get("calibration_attempts", [])
    if trace_idx >= len(cal):
        return None
    a = cal[trace_idx]
    trace_text = a.get("text") or ""
    if not trace_text:
        return None

    skills_used = record.get("skills_used") or []
    per_skill_role = record.get("per_skill_role") or {}
    prompt = (
        JUDGE_PROMPT
        .replace("{m}", str(len(skills_used)))
        .replace("{problem_text}", record["problem_text"])
        .replace("{skills_block}", _format_skills_block(skills_used, per_skill_role))
        .replace("{chain_rationale}", record.get("skill_chain_rationale") or "(none)")
        .replace("{consensus_answer}", str(record.get("consensus_answer", "")))
        .replace("{trace_text}", trace_text[:8000])  # trim very long traces
    )
    try:
        text = call_openai(
            prompt,
            model=judge_model,
            system=JUDGE_SYSTEM,
            temperature=0.0,
        )
        obj = parse_json_loose(text)
    except Exception as e:
        return {
            "combo_idx": record.get("combo_idx"),
            "trace_idx": trace_idx,
            "skills_used": skills_used,
            "error": f"{type(e).__name__}: {e}"[:300],
        }
    return {
        "combo_idx": record.get("combo_idx"),
        "trace_idx": trace_idx,
        "skills_used": skills_used,
        "judge": obj,
    }


def _print_summary(judgments: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("SKILL USAGE AUDIT — summary")
    print("=" * 70)
    n = len(judgments)
    n_err = sum(1 for j in judgments if "error" in j)
    n_ok = n - n_err
    print(f"total judgments:  {n}  (ok={n_ok}, errors={n_err})")
    if n_ok == 0:
        return

    chain_counts: Counter = Counter()
    per_skill_counts: dict[str, Counter] = defaultdict(Counter)
    for j in judgments:
        if "error" in j or not j.get("judge"):
            continue
        judge = j["judge"]
        chain_counts[judge.get("chain_fidelity", "UNKNOWN")] += 1
        for skill, verdict in (judge.get("skill_usage") or {}).items():
            per_skill_counts[skill][verdict] += 1

    print("\nChain fidelity (fraction of correct traces by argument structure):")
    for k, v in chain_counts.most_common():
        print(f"  {k:20s} {v:4d}  ({v/n_ok:.1%})")

    print("\nPer-skill usage rate (across judgments where each skill appeared):")
    print(f"  {'skill':50s}  {'USED':>6s} {'PART':>6s} {'NONE':>6s}  {'%USED':>6s}")
    for skill, counts in sorted(per_skill_counts.items()):
        used = counts.get("USED", 0)
        partial = counts.get("PARTIALLY_USED", 0)
        none = counts.get("NOT_USED", 0)
        total = used + partial + none
        pct = used / total if total else 0.0
        print(f"  {skill[:50]:50s}  {used:6d} {partial:6d} {none:6d}  {pct:6.1%}")

    if chain_counts.get("SHORTCUT", 0) / n_ok > 0.3:
        print("\n>>> WARNING: >30% of correct traces SHORTCUT the chain. Generator")
        print("    is producing problems whose intended chain is bypassed by 120b.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="Path to a stage3 keeps/skips JSONL file")
    ap.add_argument("--out", required=True,
                    help="JSONL path for per-judgment output")
    ap.add_argument("--judge-model", default="openai/gpt-oss-120b-maas")
    ap.add_argument("--max-problems", type=int, default=20,
                    help="Max distinct (combo_idx, regen) records to audit")
    ap.add_argument("--traces-per-problem", type=int, default=3,
                    help="How many correct traces to judge per problem (sampled)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"missing input: {src}")
    records = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    # Only audit records with a parseable consensus answer (skips with errors etc don't count).
    records = [r for r in records if r.get("consensus_answer") and r.get("calibration_attempts")]
    records = records[: args.max_problems]
    print(f"auditing {len(records)} records from {src.name}")

    # Build (record, trace_idx) tasks: pick the first K traces per record where
    # predicted == consensus_answer (i.e., correct critic solutions).
    tasks: list[tuple[dict, int]] = []
    for rec in records:
        consensus = str(rec.get("consensus_answer"))
        correct_idxs = [
            i for i, a in enumerate(rec.get("calibration_attempts", []))
            if a.get("text") and str(a.get("predicted")) == consensus
        ]
        for i in correct_idxs[: args.traces_per_problem]:
            tasks.append((rec, i))
    print(f"  -> {len(tasks)} (problem, correct-trace) judgments to run")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    judgments: list[dict] = []
    with out_path.open("w") as fout, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [
            pool.submit(_judge_one, rec, i, judge_model=args.judge_model)
            for rec, i in tasks
        ]
        for fut in as_completed(futs):
            r = fut.result()
            if r is None:
                continue
            judgments.append(r)
            fout.write(json.dumps(r) + "\n")
            fout.flush()

    print(f"\nwrote {len(judgments)} judgments to {out_path}")
    _print_summary(judgments)


if __name__ == "__main__":
    main()
