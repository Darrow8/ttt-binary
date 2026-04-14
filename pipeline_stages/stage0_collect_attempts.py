"""
Stage 0 — collect base-model attempts for one hard problem.

Per the paper (template.tex:194):
  1. Sample N=100 solutions from the base model on the target problem
     at temperature 0.7.
  2. Extract the final numerical answer from each.
  3. Identify the K most common answers.
  4. Save one randomly-chosen reasoning trace per group as the
     failed-attempts context for Stage 1.

Output: runs/<id>/base_attempts.json with shape:
  {
    "id": "<id>",
    "model": "...",
    "n_samples": 100,
    "n_with_answer": 87,
    "answer_counts": {"100": 41, "0": 12, "866": 0, ...},
    "top_attempts": [
      {"answer": "100", "reasoning": "...", "count": 41},
      {"answer": "0", "reasoning": "...", "count": 12},
      {"answer": "...", "reasoning": "...", "count": ...}
    ]
  }
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "Stage1"))

import distinct_llm_prompting as s1  # noqa: E402

DEFAULT_N = 100
DEFAULT_K_TOP = 3
DEFAULT_WORKERS = 16


def _save_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def collect_attempts(
    *,
    problem_id: str,
    problem_set: Path,
    output_path: Path,
    n: int = DEFAULT_N,
    k_top: int = DEFAULT_K_TOP,
    workers: int = DEFAULT_WORKERS,
    model: str | None = None,
    seed: int = 0,
) -> dict:
    statement, row = s1.load_hard_problem(str(problem_set), problem_id)
    expected = row.get("expected_answer")

    client, default_model = s1.get_client()
    use_model = model or default_model

    print(f"\n{'='*70}")
    print(f"  Stage 0: base attempts for id={problem_id}")
    print(f"  Model:   {use_model}")
    print(f"  N:       {n}")
    print(f"  Workers: {workers}")
    print(f"  Out:     {output_path}")
    print(f"{'='*70}\n")

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(s1._solve_one, client, use_model, statement) for _ in range(n)]
        results: list[tuple[str, str]] = []
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"  [warn] sample {i} raised: {e}")
                results.append(("", ""))
            if i % 10 == 0 or i == n:
                print(f"  collected {i}/{n}  ({time.time()-t0:.1f}s elapsed)")

    answers = [a for a, _ in results]
    n_with_answer = sum(1 for a in answers if a)

    # Group reasoning traces by their extracted answer.
    by_answer: dict[str, list[str]] = {}
    for a, soln in results:
        if not a:
            continue
        by_answer.setdefault(a, []).append(soln)

    counts = Counter(answers)
    counts.pop("", None)

    rng = random.Random(seed)
    top_groups = counts.most_common(k_top)
    top_attempts = [
        {
            "answer": ans,
            "count": cnt,
            "reasoning": rng.choice(by_answer[ans]),
        }
        for ans, cnt in top_groups
    ]

    pass_rate = None
    if expected is not None:
        n_correct = sum(1 for a in answers if a and a.strip() == str(expected).strip())
        pass_rate = n_correct / n if n else 0.0

    summary = {
        "id": problem_id,
        "model": use_model,
        "n_samples": n,
        "n_with_answer": n_with_answer,
        "k_top": k_top,
        "answer_counts": dict(counts),
        "top_attempts": top_attempts,
        "expected_answer": expected,
        "baseline_pass_rate_n_small": pass_rate,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    _save_atomic(output_path, summary)
    print(f"\nSaved {output_path}")
    print(
        f"  n_with_answer={n_with_answer}/{n}  "
        f"unique_answers={len(counts)}  "
        f"top_groups={[(a, c) for a, c in top_groups]}"
    )
    if pass_rate is not None:
        print(f"  baseline_pass_rate (small N={n}): {pass_rate:.2%}  "
              f"(use Stage 7 with --adapter base for the N=500 number)")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage 0: base-model attempt collection")
    parser.add_argument(
        "--problem-set", type=Path,
        default=REPO_ROOT / "problems" / "hard_problems.jsonl",
    )
    parser.add_argument("--id", type=str, required=True)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--k-top", type=int, default=DEFAULT_K_TOP)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None,
                        help="Defaults to runs/<id>/base_attempts.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_path = args.output or (REPO_ROOT / "runs" / args.id / "base_attempts.json")
    collect_attempts(
        problem_id=args.id,
        problem_set=args.problem_set,
        output_path=output_path,
        n=args.n,
        k_top=args.k_top,
        workers=args.workers,
        model=args.model,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
