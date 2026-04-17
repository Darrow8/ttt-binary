"""
Stage 0 — collect base-model attempts for one hard problem.

Samples N=500 solutions from gpt-oss-120b-maas (the model we are working with)
and saves ALL reasoning traces — including the hidden chain-of-thought captured
via `reasoning_content` — to runs/<id>/base_attempts.json. Output format
matches inference/infer.py's results.json so it can be inspected directly:

  {
    "mode": "remote",
    "model": "openai/gpt-oss-120b-maas",
    "n_samples": 500,
    "problem": "...",
    "id": "<problem id>",
    "expected_answer": "...",
    "started_at": "...",
    "completed": true,
    "results": [
      {"sample_idx": 0, "answer": "...", "reasoning": "...",
       "elapsed_s": 41.42, "has_hidden_reasoning": true,
       "reasoning_content_length": 19647},
      ...
    ],
    "finished_at": "...",
    "total_time_s": 1758.2,
    "summary": {"majority_answer": "...", "agreement_rate": 0.646,
                "answer_distribution": {...}, "n_valid_answers": 500,
                "n_empty": 0, "n_correct": 0, "baseline_pass_rate": 0.0}
  }

Stage 1 reads this same file and extracts the top-K most-common-answer
reasoning traces as the "failed solutions" context for subproblem generation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "Stage1"))
import distinct_llm_prompting as s1  # noqa: E402

DEFAULT_N = 500
DEFAULT_WORKERS = 32
TEMPERATURE = 0.7
MODEL = "openai/gpt-oss-120b-maas"
MAX_TOKENS = 16384
PER_CALL_TIMEOUT = 180.0
MAX_RETRIES = 3

SOLVE_PROMPT = """\
## Problem

{problem}

## Instructions

Solve this problem step by step. You MUST show all of your reasoning, \
calculations, and intermediate steps IN YOUR RESPONSE — do not skip ahead \
to the answer. Think carefully and work through the math explicitly. \
Write out every key derivation.

Round your answer to 4 decimal places if necessary. \
Your answer must be a number, not an expression. \
Put your final answer inside \\boxed{{}}.

"""


def _save_atomic(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _solve_once(client, problem: str, sample_idx: int) -> dict:
    """Sample one solution. Captures both visible content and (if present)
    the hidden chain-of-thought via `reasoning_content`."""
    prompt = SOLVE_PROMPT.format(problem=problem)
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                timeout=PER_CALL_TIMEOUT,
            )
            elapsed = time.time() - t0
            if isinstance(resp, str):
                # Vertex sometimes returns a raw string instead of a parsed
                # ChatCompletion under transient overload. Treat as failure.
                print(
                    f"  [sample {sample_idx+1}] vertex returned raw string "
                    f"(attempt {attempt+1}/{MAX_RETRIES}): {resp[:200]!r}"
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                continue
            msg = resp.choices[0].message
            content = msg.content or ""
            reasoning_content = getattr(msg, "reasoning_content", None) or ""
            if reasoning_content:
                solution = reasoning_content + "\n\n" + content
            else:
                solution = content
            if not solution:
                print(
                    f"  [sample {sample_idx+1}] empty response "
                    f"(attempt {attempt+1}/{MAX_RETRIES})"
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                continue
            answer = s1.extract_answer(solution)
            result = {
                "sample_idx": sample_idx,
                "answer": answer,
                "reasoning": solution,
                "elapsed_s": round(elapsed, 2),
            }
            if reasoning_content:
                result["has_hidden_reasoning"] = True
                result["reasoning_content_length"] = len(reasoning_content)
            return result
        except Exception as e:
            print(
                f"  [sample {sample_idx+1}] error "
                f"(attempt {attempt+1}/{MAX_RETRIES}): {e}"
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    return {
        "sample_idx": sample_idx,
        "answer": "",
        "reasoning": "",
        "elapsed_s": 0,
        "error": "all retries exhausted",
    }


def _finalize_summary(
    output_data: dict,
    expected: str | None,
    n: int,
) -> None:
    """Compute the summary block from the current results and attach it."""
    results = output_data.get("results", [])
    answers = [r.get("answer", "") for r in results if r.get("answer")]
    if not answers:
        return
    counter = Counter(answers)
    majority_answer, majority_count = counter.most_common(1)[0]
    summary = {
        "majority_answer": majority_answer,
        "agreement_rate": round(majority_count / len(answers), 3),
        "answer_distribution": dict(counter.most_common()),
        "n_valid_answers": len(answers),
        "n_empty": n - len(answers),
    }
    if expected is not None:
        n_correct = sum(
            1 for a in answers if a.strip() == str(expected).strip()
        )
        summary["n_correct"] = n_correct
        summary["baseline_pass_rate"] = round(n_correct / n, 4)
    output_data["summary"] = summary


def collect_attempts(
    *,
    problem_id: str,
    problem_set: Path,
    output_path: Path,
    n: int = DEFAULT_N,
    workers: int = DEFAULT_WORKERS,
    resume: bool = False,
    force: bool = False,
) -> dict:
    statement, row = s1.load_hard_problem(str(problem_set), problem_id)
    expected = row.get("expected_answer")

    # ── Safety: refuse to clobber a non-empty file without --resume or --force ──
    if output_path.exists() and not resume and not force:
        try:
            with open(output_path) as f:
                existing = json.load(f)
            existing_count = sum(
                1 for r in existing.get("results", [])
                if isinstance(r, dict) and r.get("answer")
            )
        except Exception:
            existing_count = 0
        if existing_count > 0:
            raise SystemExit(
                f"\n{output_path} already has {existing_count} samples with answers.\n"
                f"Refusing to overwrite.\n\n"
                f"  - To continue from where it left off (fill empty + missing slots): "
                f"pass --resume\n"
                f"  - To start over and discard the existing data: pass --force\n"
            )

    # ── Resume: load existing results if requested ─────────────────────────
    existing_results: list[dict] = []
    existing_indices: set[int] = set()
    original_started_at: str | None = None
    previous_total_time: float = 0.0

    if resume and output_path.exists():
        try:
            with open(output_path) as f:
                prev = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  [warn] --resume: existing file unreadable ({e}), starting fresh")
            prev = {}

        # Count samples with non-empty answers as "done". Samples that
        # exhausted their retries and returned empty — OR samples that
        # were never attempted — both get re-sampled. The `completed`
        # flag on the previous run is ignored deliberately: a run can
        # be "complete" in the sense of "we stopped trying" but still
        # have failed slots that a resume pass should fill.
        total_in_prev = 0
        empty_in_prev = 0
        for r in prev.get("results", []):
            if not isinstance(r, dict) or "sample_idx" not in r:
                continue
            total_in_prev += 1
            if r.get("answer"):
                existing_results.append(r)
                existing_indices.add(int(r["sample_idx"]))
            else:
                empty_in_prev += 1
        original_started_at = prev.get("started_at")
        try:
            previous_total_time = float(prev.get("total_time_s") or 0)
        except (TypeError, ValueError):
            previous_total_time = 0.0
        if total_in_prev:
            print(
                f"  --resume: previous file had {total_in_prev} samples "
                f"({len(existing_indices)} with answers, {empty_in_prev} empty); "
                f"will re-sample empty + missing slots"
            )

    missing_indices = [i for i in range(n) if i not in existing_indices]

    # ── Banner ──────────────────────────────────────────────────────────────
    client, _ = s1.get_client()

    print(f"\n{'='*70}")
    print(f"  Stage 0: base attempts for id={problem_id}")
    print(f"  Model:    {MODEL}")
    print(f"  N:        {n}")
    print(f"  Workers:  {workers}")
    print(f"  Out:      {output_path}")
    print(f"  Expected: {expected}")
    if resume:
        print(
            f"  Resume:   {len(existing_indices)}/{n} existing, "
            f"{len(missing_indices)} to sample"
        )
    print(f"{'='*70}\n")

    # ── Shared state ────────────────────────────────────────────────────────
    results: list[dict] = list(existing_results)
    lock = threading.Lock()

    output_data: dict = {
        "mode": "remote",
        "model": MODEL,
        "n_samples": n,
        "problem": statement.strip(),
        "id": problem_id,
        "expected_answer": expected,
        "started_at": original_started_at or datetime.now().isoformat(),
        "completed": False,
        "results": sorted(results, key=lambda r: r.get("sample_idx", 0)),
    }
    _save_atomic(output_path, output_data)

    # ── Short-circuit: nothing missing, finalize and exit ──────────────────
    if not missing_indices:
        _finalize_summary(output_data, expected, n)
        output_data["completed"] = True
        output_data["finished_at"] = datetime.now().isoformat()
        output_data["total_time_s"] = round(previous_total_time, 2)
        _save_atomic(output_path, output_data)
        print(
            f"\nAll {n} samples already present. Wrote finalized file to {output_path}"
        )
        return output_data

    # ── Sampling loop (fresh or resumed) ────────────────────────────────────
    def _on_complete(future: concurrent.futures.Future) -> None:
        try:
            result = future.result()
        except Exception as e:
            result = {"error": str(e)}
        with lock:
            results.append(result)
            output_data["results"] = sorted(
                results, key=lambda r: r.get("sample_idx", 0)
            )
            _save_atomic(output_path, output_data)
        idx = result.get("sample_idx", "?")
        ans = result.get("answer", "")
        t = result.get("elapsed_s", 0)
        print(
            f"  [sample {idx+1}/{n}] answer={ans!r}  "
            f"({t:.1f}s)  [{len(results)}/{n} done]"
        )

    t_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for i in missing_indices:
            fut = pool.submit(_solve_once, client, statement, i)
            fut.add_done_callback(_on_complete)
    session_time = time.time() - t_start
    total_time = previous_total_time + session_time

    output_data["completed"] = True
    output_data["finished_at"] = datetime.now().isoformat()
    output_data["total_time_s"] = round(total_time, 2)

    _finalize_summary(output_data, expected, n)
    summary = output_data.get("summary", {})

    if summary:
        print(f"\n{'='*60}")
        print(f"  Majority answer: {summary['majority_answer']}")
        n_valid = summary["n_valid_answers"]
        print(
            f"  Agreement: {summary['agreement_rate']*n_valid:.0f}/{n_valid} "
            f"({summary['agreement_rate']:.0%})"
        )
        if expected is not None:
            print(
                f"  Baseline pass rate: {summary['n_correct']}/{n} = "
                f"{summary['baseline_pass_rate']:.2%}"
            )
        print(
            f"  Session time: {session_time:.1f}s  ({session_time/60:.1f} min)"
        )
        print(
            f"  Total time (all sessions): {total_time:.1f}s  "
            f"({total_time/60:.1f} min)"
        )
        print(f"{'='*60}")

    _save_atomic(output_path, output_data)
    print(f"\nResults saved to {output_path}")
    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Stage 0: base-model attempts (full per-sample reasoning)"
    )
    parser.add_argument(
        "--problem-set",
        type=Path,
        default=REPO_ROOT / "problems" / "hard_problems.jsonl",
    )
    parser.add_argument("--id", type=str, required=True)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to runs/<id>/base_attempts.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "If runs/<id>/base_attempts.json already exists, load its "
            "`results` array, skip already-sampled sample_idx values, and "
            "only sample the missing ones. Safe to ^C mid-run and re-invoke."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite an existing base_attempts.json even if it has "
            "samples with answers. Mutually exclusive with --resume."
        ),
    )
    args = parser.parse_args()

    if args.resume and args.force:
        parser.error("--resume and --force are mutually exclusive")

    output_path = args.output or (
        REPO_ROOT / "runs" / args.id / "base_attempts.json"
    )
    collect_attempts(
        problem_id=args.id,
        problem_set=args.problem_set,
        output_path=output_path,
        n=args.n,
        workers=args.workers,
        resume=args.resume,
        force=args.force,
    )


if __name__ == "__main__":
    main()
