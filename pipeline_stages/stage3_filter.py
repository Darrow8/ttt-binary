"""
Stage 3 — filter aggregated subproblems for quality.

Two filters applied in order:

  3a. Rounding-only disagreement removal
      Drops problems whose disagreement is purely a rounding artifact
      (all answers parse to floats within REL_TOL). Ports
      Stage1/build_filtered_keeps.py:36 unchanged at REL_TOL=1e-3.

  3b. Multi-step / bad-premise LLM judge
      Asks a small judge model to reject problems with subparts, multiple
      numerical answers, or contradictory premises. Majority vote over
      JUDGE_TRIES=3 calls. Reject on >=2/3 reject verdicts.

Reads:  runs/<id>/aggregated_keeps.json
Writes: runs/<id>/filtered_keeps.json   (atomic)
        runs/<id>/filter_judge.json     (per-problem judge verdicts)
        runs/<id>/filter_stats.json     (counts at each stage)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "Stage1"))
import distinct_llm_prompting as s1  # noqa: E402

REL_TOL = 1e-3
ABS_TOL = 1e-6

JUDGE_TRIES = 3
JUDGE_REJECT_THRESHOLD = 2  # >=2 of JUDGE_TRIES votes reject -> reject
DEFAULT_JUDGE_MODEL = "openai/gpt-oss-20b-maas"
DEFAULT_WORKERS = 16


def _save_atomic(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ── Filter 3a: rounding-only ────────────────────────────────────────────────

def _parse_numeric(s: str) -> float | None:
    s = s.strip().replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def _all_close(values: list[float], rel_tol: float, abs_tol: float) -> bool:
    if not values:
        return True
    ref = values[0]
    for v in values[1:]:
        diff = abs(v - ref)
        scale = max(abs(ref), abs(v), 1e-15)
        if diff > max(rel_tol * scale, abs_tol):
            return False
    return True


def is_rounding_problem(problem: dict) -> bool:
    answers = problem.get("all_answers", [])
    if not answers:
        return False
    if len(set(answers)) == 1:
        return False
    parsed = []
    for a in answers:
        v = _parse_numeric(a)
        if v is None:
            return False
        parsed.append(v)
    if any(not math.isfinite(v) for v in parsed):
        return False
    return _all_close(parsed, REL_TOL, ABS_TOL)


# ── Filter 3b: multi-step / bad-premise LLM judge ────────────────────────────

JUDGE_PROMPT = """\
You are auditing a math problem for use in a training dataset.

Reject the problem if ANY of the following are true:
  1. It has subparts (e.g. "(a)", "(b)", "Part 1", "First find X, then find Y").
  2. It asks for more than one distinct numerical answer.
  3. The premise is contradictory or ill-defined.

Otherwise accept it.

## Problem

{problem}

## Output

Return JSON only, no other text. Format:
{{"verdict": "accept", "reason": "..."}}
or
{{"verdict": "reject", "reason": "..."}}
"""


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_verdict(raw: str) -> tuple[str, str]:
    """Return (verdict, reason). verdict is 'accept' | 'reject' | 'parse_error'."""
    m = _JSON_RE.search(raw)
    if not m:
        return "parse_error", raw[:200]
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return "parse_error", raw[:200]
    v = str(obj.get("verdict", "")).strip().lower()
    if v not in ("accept", "reject"):
        return "parse_error", raw[:200]
    return v, str(obj.get("reason", ""))[:300]


def _judge_one(client, model: str, problem_text: str) -> dict:
    prompt = JUDGE_PROMPT.format(problem=problem_text)
    votes: list[tuple[str, str]] = []
    for _ in range(JUDGE_TRIES):
        raw = s1.call_llm(client, model, prompt, temperature=0.3)
        votes.append(_parse_verdict(raw or ""))
    rejects = sum(1 for v, _ in votes if v == "reject")
    accepts = sum(1 for v, _ in votes if v == "accept")
    parse_errs = sum(1 for v, _ in votes if v == "parse_error")
    final_reject = rejects >= JUDGE_REJECT_THRESHOLD
    return {
        "verdict": "reject" if final_reject else "accept",
        "rejects": rejects,
        "accepts": accepts,
        "parse_errors": parse_errs,
        "votes": [{"verdict": v, "reason": r} for v, r in votes],
    }


# ── Pipeline ────────────────────────────────────────────────────────────────

def filter_one(
    problem_id: str,
    *,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    workers: int = DEFAULT_WORKERS,
    skip_judge: bool = False,
) -> dict:
    runs_root = REPO_ROOT / "runs" / problem_id
    aggregated_path = runs_root / "aggregated_keeps.json"
    if not aggregated_path.exists():
        raise FileNotFoundError(
            f"Missing {aggregated_path}. Run Stage 2 (aggregate) first."
        )

    with open(aggregated_path) as f:
        agg = json.load(f)
    problems: list[dict] = agg.get("problems", [])
    n_total = len(problems)

    print(f"\n{'='*70}")
    print(f"  Stage 3: filter id={problem_id}")
    print(f"  Input:  {aggregated_path}  ({n_total} problems)")
    print(f"  Judge:  {judge_model}  (skip={skip_judge})")
    print(f"{'='*70}\n")

    # ── 3a rounding ─────────────────────────────────────────────────────────
    after_3a: list[dict] = []
    removed_3a: list[dict] = []
    for p in problems:
        if is_rounding_problem(p):
            removed_3a.append(p)
        else:
            after_3a.append(p)
    print(f"  3a rounding-only:   removed {len(removed_3a)}, kept {len(after_3a)}")

    # ── 3b LLM judge ────────────────────────────────────────────────────────
    after_3b: list[dict] = []
    removed_3b: list[dict] = []
    judge_records: list[dict] = []

    if skip_judge or not after_3a:
        after_3b = after_3a
        print(f"  3b judge:           SKIPPED")
    else:
        client, default_model = s1.get_client()
        use_model = judge_model or default_model

        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_judge_one, client, use_model, p["problem"]): i
                for i, p in enumerate(after_3a)
            }
            results: dict[int, dict] = {}
            done_count = 0
            for fut in concurrent.futures.as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    print(f"  [warn] judge raised on idx {i}: {e}")
                    results[i] = {"verdict": "accept", "rejects": 0, "accepts": 0, "parse_errors": 0, "votes": []}
                done_count += 1
                if done_count % 10 == 0 or done_count == len(after_3a):
                    print(f"    judged {done_count}/{len(after_3a)}  ({time.time()-t0:.1f}s)")

        for i, p in enumerate(after_3a):
            r = results[i]
            judge_records.append({
                "index": i,
                "problem_snippet": p.get("problem", "")[:160],
                **r,
            })
            if r["verdict"] == "reject":
                removed_3b.append(p)
            else:
                after_3b.append(p)
        print(f"  3b judge:           removed {len(removed_3b)}, kept {len(after_3b)}")

    # ── write outputs ───────────────────────────────────────────────────────
    filtered_path = runs_root / "filtered_keeps.json"
    _save_atomic(filtered_path, {
        "id": problem_id,
        "source_problem": agg.get("source_problem"),
        "n_problems": len(after_3b),
        "n_total_before_filter": n_total,
        "n_removed_rounding": len(removed_3a),
        "n_removed_judge": len(removed_3b),
        "problems": after_3b,
    })
    print(f"\nWrote {filtered_path}  ({len(after_3b)} problems)")

    judge_path = runs_root / "filter_judge.json"
    _save_atomic(judge_path, {
        "id": problem_id,
        "judge_model": judge_model if not skip_judge else None,
        "tries_per_problem": JUDGE_TRIES,
        "reject_threshold": JUDGE_REJECT_THRESHOLD,
        "records": judge_records,
    })

    stats_path = runs_root / "filter_stats.json"
    stats = {
        "id": problem_id,
        "n_total_before_filter": n_total,
        "n_after_3a_rounding": len(after_3a),
        "n_after_3b_judge": len(after_3b),
        "n_removed_3a_rounding": len(removed_3a),
        "n_removed_3b_judge": len(removed_3b),
        "kept_fraction": (len(after_3b) / n_total) if n_total else 0.0,
    }
    _save_atomic(stats_path, stats)
    print(f"Stats: {stats}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Stage 3: filter aggregated subproblems")
    parser.add_argument("--id", type=str, required=True)
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--skip-judge", action="store_true",
                        help="Run only Filter 3a (rounding). Useful for quick iteration.")
    args = parser.parse_args()
    filter_one(
        args.id,
        judge_model=args.judge_model,
        workers=args.workers,
        skip_judge=args.skip_judge,
    )


if __name__ == "__main__":
    main()
