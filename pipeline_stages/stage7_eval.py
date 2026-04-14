"""
Stage 7 — evaluate one (hard_problem, adapter) pair at N=500.

Per the paper (template.tex:182): the headline metric is the 500-sample
solve rate against the verified expected answer.

Reads:  problems/hard_problems.jsonl                           (statement + expected_answer)
        runs/<adapter>/grpo/...  (when --adapter != base)      (tinker checkpoint)
Writes: runs/<id>/eval_<adapter>.json                          {pass_rate, n, all_answers, expected}

Adapter modes:
  --adapter base                  Vertex base model (no tinker).
  --adapter <other-id> | shared   Load latest tinker checkpoint for that adapter.
                                  Pass --adapter-checkpoint to override.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "Stage1"))
import distinct_llm_prompting as s1  # noqa: E402

DEFAULT_N = 500
DEFAULT_WORKERS = 16
DEFAULT_TEMPERATURE = 0.7


def _save_atomic(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _normalize(answer: str) -> str:
    return s1.normalize_answer(answer or "")


def _is_correct(extracted: str, expected: str) -> bool:
    if not extracted or not expected:
        return False
    return _normalize(extracted) == _normalize(expected)


def evaluate(
    problem_id: str,
    *,
    adapter: str,
    n: int = DEFAULT_N,
    workers: int = DEFAULT_WORKERS,
    adapter_checkpoint: str | None = None,
    save_solutions: bool = False,
) -> dict:
    statement, row = s1.load_hard_problem(
        str(REPO_ROOT / "problems" / "hard_problems.jsonl"), problem_id
    )
    expected = row.get("expected_answer")
    if expected is None:
        raise ValueError(
            f"id={problem_id!r} has expected_answer=null in hard_problems.jsonl. "
            "Fill it in before running Stage 7."
        )

    if adapter == "base":
        client, default_model = s1.get_client()
        model = default_model
        client_label = f"vertex:{model}"
    else:
        ckpt = adapter_checkpoint or os.environ.get("TINKER_CHECKPOINT")
        if not ckpt:
            raise SystemExit(
                f"--adapter {adapter} requires --adapter-checkpoint or TINKER_CHECKPOINT. "
                "Pass the tinker:// path of the trained adapter checkpoint."
            )
        client = s1.TinkerClient(ckpt)
        model = s1.TINKER_BASE_MODEL
        client_label = f"tinker:{ckpt}"

    print(f"\n{'='*70}")
    print(f"  Stage 7: eval id={problem_id}  adapter={adapter}")
    print(f"  Client:   {client_label}")
    print(f"  N:        {n}")
    print(f"  Workers:  {workers}")
    print(f"  Expected: {expected}")
    print(f"{'='*70}\n")

    t0 = time.time()
    answers: list[str] = []
    solutions: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(s1._solve_one, client, model, statement) for _ in range(n)]
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                ans, soln = f.result()
            except Exception as e:
                print(f"  [warn] sample {i} raised: {e}")
                ans, soln = "", ""
            answers.append(ans)
            solutions.append(soln)
            if i % 25 == 0 or i == n:
                running_correct = sum(1 for a in answers if _is_correct(a, str(expected)))
                print(
                    f"  {i}/{n}  ({time.time()-t0:.0f}s)  "
                    f"running pass rate {running_correct}/{i} = {running_correct/i:.2%}"
                )

    n_correct = sum(1 for a in answers if _is_correct(a, str(expected)))
    n_with_answer = sum(1 for a in answers if a)
    pass_rate = n_correct / n if n else 0.0

    counts = Counter(answers)
    counts.pop("", None)
    most_common = counts.most_common(10)

    out_path = REPO_ROOT / "runs" / problem_id / f"eval_{adapter}.json"
    payload: dict = {
        "id": problem_id,
        "adapter": adapter,
        "client": client_label,
        "expected_answer": str(expected),
        "n_samples": n,
        "n_with_answer": n_with_answer,
        "n_correct": n_correct,
        "pass_rate": pass_rate,
        "most_common_answers": [{"answer": a, "count": c} for a, c in most_common],
        "elapsed_seconds": round(time.time() - t0, 1),
        "all_answers": answers,
    }
    if save_solutions:
        payload["all_solutions"] = solutions
    _save_atomic(out_path, payload)

    print(f"\nWrote {out_path}")
    print(f"  pass_rate: {n_correct}/{n} = {pass_rate:.2%}")
    print(f"  most common: {most_common[:5]}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Stage 7: N=500 hard-problem eval")
    parser.add_argument("--id", type=str, required=True,
                        help="Hard-problem id to evaluate against")
    parser.add_argument("--adapter", type=str, required=True,
                        help="'base' for Vertex base model, otherwise a per-id or 'shared' adapter name")
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--adapter-checkpoint", type=str, default=None,
                        help="Tinker path for non-base adapters (or set TINKER_CHECKPOINT)")
    parser.add_argument("--save-solutions", action="store_true",
                        help="Also persist all_solutions in the eval JSON (large)")
    args = parser.parse_args()

    evaluate(
        problem_id=args.id,
        adapter=args.adapter,
        n=args.n,
        workers=args.workers,
        adapter_checkpoint=args.adapter_checkpoint,
        save_solutions=args.save_solutions,
    )


if __name__ == "__main__":
    main()
