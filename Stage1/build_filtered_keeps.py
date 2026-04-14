#!/usr/bin/env python3
"""
Collect all kept problems across runs, excluding those whose disagreement
is caused by rounding errors. Outputs filtered_keeps.json.
"""

import json
import math
from pathlib import Path

RUNS_DIR = Path(__file__).parent / "runs"
REL_TOL = 1e-3
ABS_TOL = 1e-6


def parse_numeric(s: str) -> float | None:
    s = s.strip().replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def all_close(values: list[float], rel_tol: float, abs_tol: float) -> bool:
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
        v = parse_numeric(a)
        if v is None:
            return False
        parsed.append(v)
    if any(not math.isfinite(v) for v in parsed):
        return False
    return all_close(parsed, REL_TOL, ABS_TOL)


def main():
    kept = []
    removed = []
    total = 0

    for run_dir in sorted(RUNS_DIR.iterdir()):
        keeps_path = run_dir / "keeps.json"
        if not keeps_path.is_file():
            continue
        with open(keeps_path) as f:
            data = json.load(f)

        for prob in data.get("problems", []):
            total += 1
            if is_rounding_problem(prob):
                removed.append(prob)
            else:
                kept.append(prob)

    out_path = Path(__file__).parent / "filtered_keeps.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_problems": len(kept),
            "n_removed_rounding": len(removed),
            "n_total_before_filter": total,
            "problems": kept,
        }, f, indent=2)

    print(f"Total problems scanned:    {total}")
    print(f"Removed (rounding errors): {len(removed)}")
    print(f"Kept after filtering:      {len(kept)}")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
