#!/usr/bin/env python3
"""
Scan all keeps.json files and identify problems where answer disagreement
is caused by rounding errors rather than genuinely different solutions.

A "rounding problem" is one where:
  - The agreement_rate < 1.0 (i.e. not all answers are identical strings)
  - But all answers, when parsed as floats, are within a small relative
    tolerance of each other (i.e. they represent the same number, just
    rounded/truncated differently).

This also catches cases like "8.1898" vs "8.19" vs "8.1899".
"""

import json
import math
import re
import sys
from pathlib import Path

RUNS_DIR = Path(__file__).parent / "runs"

# Two answers are "the same up to rounding" if their float values differ
# by less than this *relative* tolerance (0.1% by default).
REL_TOL = 1e-3
# Also allow a small absolute tolerance for values near zero.
ABS_TOL = 1e-6


def parse_numeric(s: str) -> float | None:
    """Try to parse a string as a number. Returns None if it fails."""
    s = s.strip()
    # Remove commas (e.g. "1,234")
    s = s.replace(",", "")
    # Remove trailing periods
    s = s.rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def all_close(values: list[float], rel_tol: float, abs_tol: float) -> bool:
    """Check if all float values are within tolerance of each other."""
    if not values:
        return True
    ref = values[0]
    for v in values[1:]:
        diff = abs(v - ref)
        # Use the larger of the two as the scale
        scale = max(abs(ref), abs(v), 1e-15)
        if diff > max(rel_tol * scale, abs_tol):
            return False
    return True


def analyze_problem(problem: dict) -> dict | None:
    """
    If the problem's disagreement is caused by rounding, return a summary dict.
    Otherwise return None.
    """
    answers = problem.get("all_answers", [])
    if not answers:
        return None

    unique_strs = set(answers)
    # If all answers are identical strings, there's no disagreement at all
    if len(unique_strs) == 1:
        return None

    # Try to parse all answers as numbers
    parsed = []
    for a in answers:
        v = parse_numeric(a)
        if v is None:
            # Non-numeric answer — can't be a rounding issue
            return None
        parsed.append(v)

    # Reject if any value is non-finite (inf, nan)
    if any(not math.isfinite(v) for v in parsed):
        return None

    # Check if all parsed values are close
    if not all_close(parsed, REL_TOL, ABS_TOL):
        return None

    # This IS a rounding problem
    unique_vals = sorted(set(parsed))
    spread = max(parsed) - min(parsed)
    mean_val = sum(parsed) / len(parsed)

    return {
        "problem_snippet": problem.get("problem", "")[:120] + "...",
        "ground_truth": problem.get("ground_truth_answer"),
        "agreement_rate": problem.get("agreement_rate"),
        "unique_answer_strings": sorted(unique_strs),
        "unique_float_values": unique_vals,
        "spread": spread,
        "relative_spread": spread / abs(mean_val) if mean_val != 0 else float("inf"),
        "n_samples": problem.get("n_samples", len(answers)),
    }


def main():
    rounding_problems = []
    total_problems = 0

    for run_dir in sorted(RUNS_DIR.iterdir()):
        keeps_path = run_dir / "keeps.json"
        if not keeps_path.is_file():
            continue
        with open(keeps_path) as f:
            data = json.load(f)

        for i, prob in enumerate(data.get("problems", [])):
            total_problems += 1
            result = analyze_problem(prob)
            if result is not None:
                result["run"] = run_dir.name
                result["index"] = i
                rounding_problems.append(result)

    # ── Report ──────────────────────────────────────────────────────────
    print(f"Scanned {total_problems} total kept problems across all runs.\n")
    print(f"Found {len(rounding_problems)} problems where disagreement is "
          f"likely caused by rounding errors:\n")
    print("=" * 80)

    for j, rp in enumerate(rounding_problems, 1):
        print(f"\n[{j}] Run: {rp['run']}  (problem index {rp['index']})")
        print(f"    Problem: {rp['problem_snippet']}")
        print(f"    Ground truth: {rp['ground_truth']}")
        print(f"    Agreement rate: {rp['agreement_rate']}")
        print(f"    Unique answer strings: {rp['unique_answer_strings']}")
        print(f"    Unique float values:   {rp['unique_float_values']}")
        print(f"    Spread: {rp['spread']:.6g}  "
              f"(relative: {rp['relative_spread']:.2e})")
        print()

    # ── Write JSON ──────────────────────────────────────────────────────
    out_path = Path(__file__).parent / "rounding_problems.json"
    with open(out_path, "w") as f:
        json.dump({
            "total_problems_scanned": total_problems,
            "n_rounding_problems": len(rounding_problems),
            "problems": rounding_problems,
        }, f, indent=2)
    print(f"Wrote {len(rounding_problems)} rounding problems to {out_path}")

    # ── Summary ─────────────────────────────────────────────────────────
    if rounding_problems:
        print("=" * 80)
        print(f"\nSummary: {len(rounding_problems)}/{total_problems} kept "
              f"problems ({100*len(rounding_problems)/total_problems:.1f}%) "
              f"have rounding-only disagreement.\n")
        print("These problems should be removed or re-evaluated: the model "
              "is solving them correctly but the string-match agreement check "
              "penalises minor formatting/rounding differences.")
    else:
        print("No rounding-error problems found.")


if __name__ == "__main__":
    main()
