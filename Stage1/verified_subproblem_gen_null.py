"""
TTT-Discover + Verify (NULL-aware): subproblem generation with NULL abstention.

Variant of verified_subproblem_gen.py where:
  1. The solve prompt tells the model it MAY return \\boxed{NULL} if it is
     genuinely unsure of the answer.
  2. Self-consistency is computed ONLY over non-NULL answers.
  3. If fewer than MIN_NON_NULL_ANSWERS (default 8) non-NULL answers are
     returned for a candidate, the candidate is skipped regardless of the
     agreement rate of the non-null subset.

This produces a cleaner ground-truth signal: the majority answer is computed
over confident responses only, and problems where the model is systematically
confused (too many NULLs) are discarded rather than given a noisy label.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import Stage1.distinct_llm_prompting as s1  # noqa: E402
import Stage1.verified_subproblem_gen as vsg  # noqa: E402
from pipeline_stages.dedupe import DedupeIndex  # noqa: E402

GeneratedProblem = s1.GeneratedProblem
Dataset = s1.Dataset

# ---------------------------------------------------------------------------
# NULL-aware constants
# ---------------------------------------------------------------------------
MIN_NON_NULL_ANSWERS = 8  # skip if fewer non-NULL answers than this

# Solve prompt variant that allows NULL abstention
NULL_SOLVE_PROMPT = """\
## Problem

{problem}

Solve the problem step by step.

When you are done, write your final answer on the very last line in exactly
this format:

\\boxed{{<answer>}}

For example: \\boxed{{0.6079}} or \\boxed{{42}}

Your answer MUST be a single decimal number (e.g. 0.6079, 42, 3.1416).
Do NOT write symbolic expressions like 1/pi^2, 6/pi^2, sqrt(2), or ln(2).
If the answer is irrational or a fraction, round to 4 decimal places.

IMPORTANT: If you are genuinely unsure of the answer after working through the
problem, you may write \\boxed{{NULL}} instead of guessing.  Only do this when
you truly cannot arrive at a confident numerical answer.

"""


# ---------------------------------------------------------------------------
# NULL-aware solve helpers
# ---------------------------------------------------------------------------
_NULL_TOKENS = frozenset({"null", "none", "n/a", "na", "unknown"})

_NULL_ANSWER_RE = re.compile(r"^null$", re.IGNORECASE)


def _is_null_answer(answer: str) -> bool:
    return answer.strip().lower() in _NULL_TOKENS


def _solve_one_null(
    client,
    model: str,
    problem: str,
) -> tuple[str, str]:
    """Solve once with the NULL-aware prompt. Returns (answer, solution)."""
    prompt = NULL_SOLVE_PROMPT.format(problem=problem)
    for attempt in range(s1._SOLVE_MAX_RETRIES):
        solution = s1.call_llm(client, model, prompt, temperature=0.7)
        if not solution:
            continue
        answer = s1.extract_answer(solution)
        if answer:
            return answer, solution
        if attempt < s1._SOLVE_MAX_RETRIES - 1:
            print("r", end="", flush=True)
    answer = s1.extract_answer(solution) if solution else ""
    return answer, solution


def solve_and_check_agreement_null(
    client,
    model: str,
    problem: str,
    n_samples: int = 10,
    pool: concurrent.futures.ThreadPoolExecutor | None = None,
) -> tuple[float, str, list[str], list[str], int, int]:
    """
    Like s1.solve_and_check_agreement but:
    - Uses the NULL-aware solve prompt.
    - Returns agreement computed over non-NULL answers only.
    - Also returns (n_non_null, n_null) counts.

    Returns (agreement_rate, majority_answer, all_answers, all_solutions,
             n_non_null, n_null).
    If n_non_null < MIN_NON_NULL_ANSWERS, agreement_rate is returned as -1.0
    to signal that this candidate should be skipped.
    """
    futures = [
        pool.submit(_solve_one_null, client, model, problem)
        for _ in range(n_samples)
    ]
    results = [f.result() for f in futures]
    all_answers = [r[0] for r in results]
    all_solutions = [r[1] for r in results]

    non_null_answers = [a for a in all_answers if not _is_null_answer(a) and a]
    n_non_null = len(non_null_answers)
    n_null = len(all_answers) - n_non_null

    if n_non_null < MIN_NON_NULL_ANSWERS:
        return -1.0, "", all_answers, all_solutions, n_non_null, n_null

    counter = Counter(non_null_answers)
    majority_answer, majority_count = counter.most_common(1)[0]
    agreement_rate = majority_count / n_non_null

    return agreement_rate, majority_answer, all_answers, all_solutions, n_non_null, n_null


# ---------------------------------------------------------------------------
# Verification helpers (re-used from verified_subproblem_gen)
# ---------------------------------------------------------------------------
VERIFY_TRIES = vsg.VERIFY_TRIES
VERIFY_ACCEPT_THRESHOLD = vsg.VERIFY_ACCEPT_THRESHOLD
verify_reasoning = vsg.verify_reasoning
pick_majority_trace = vsg.pick_majority_trace


# ---------------------------------------------------------------------------
# Core: build_null_verified_dataset
# ---------------------------------------------------------------------------
def build_null_verified_dataset(
    client,
    model: str,
    hard_problem: str,
    n_target: int = 100,
    n_samples_per_problem: int = 10,
    target_agreement_low: float = 0.60,
    target_agreement_high: float = 0.80,
    output_path: str | None = None,
    max_workers: int = 16,
    gen_workers: int = 8,
    failed_solutions: list[str] | None = None,
    solve_client=None,
    solve_model: str | None = None,
    quality_threshold: int | None = None,
    use_dedupe: bool = True,
    verify_model: str | None = None,
    min_non_null: int = MIN_NON_NULL_ANSWERS,
) -> Dataset:
    """Generate subproblems with NULL-aware self-consistency + verification.

    Differences from build_verified_dataset:
    - Solve prompt allows the model to return NULL when unsure.
    - Agreement is computed over non-NULL answers only.
    - Candidates with fewer than `min_non_null` non-NULL answers are skipped.
    - Reasoning verification still runs on passed candidates (same as parent).
    """
    s_client = solve_client or client
    s_model = solve_model or model
    v_model = verify_model or model

    dataset = Dataset(
        source_problem=hard_problem,
        target_agreement_low=target_agreement_low,
        target_agreement_high=target_agreement_high,
    )

    skipped_problems: list[GeneratedProblem] = []
    verify_log: list[dict] = []
    dedupe = DedupeIndex() if use_dedupe else None
    seen_problems: set[str] = set()

    if output_path and os.path.exists(output_path):
        try:
            with open(output_path) as f:
                existing = json.load(f)
            for p in existing.get("problems", []):
                entry = GeneratedProblem(**p)
                dataset.problems.append(entry)
                if use_dedupe:
                    dedupe.add(entry.problem)
                else:
                    seen_problems.add(entry.problem)
            if dataset.problems:
                print(f"  Resumed {len(dataset.problems)} existing problems from {output_path}")
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    if dedupe is not None:
        dedupe_baseline_kept = dedupe.n_kept
        dedupe_baseline_exact = dedupe.n_exact_dropped
        dedupe_baseline_fuzzy = dedupe.n_fuzzy_dropped
    else:
        dedupe_baseline_kept = 0
        dedupe_baseline_exact = 0
        dedupe_baseline_fuzzy = 0

    if output_path:
        skips_path = os.path.join(os.path.dirname(output_path), "skips.json")
        verify_log_path = os.path.join(os.path.dirname(output_path), "verify_log.json")
    else:
        skips_path = None
        verify_log_path = None

    def _flush() -> None:
        if not output_path:
            return
        s1._save_atomic(
            output_path,
            {
                "source_problem": dataset.source_problem,
                "target_agreement_low": dataset.target_agreement_low,
                "target_agreement_high": dataset.target_agreement_high,
                "n_problems": len(dataset.problems),
                "problems": [asdict(p) for p in dataset.problems],
            },
        )
        s1._save_atomic(
            skips_path,
            {
                "source_problem": dataset.source_problem,
                "target_agreement_low": dataset.target_agreement_low,
                "target_agreement_high": dataset.target_agreement_high,
                "n_problems": len(skipped_problems),
                "problems": [asdict(p) for p in skipped_problems],
            },
        )
        s1._save_atomic(
            verify_log_path,
            {
                "verify_model": v_model,
                "tries_per_problem": VERIFY_TRIES,
                "accept_threshold": VERIFY_ACCEPT_THRESHOLD,
                "records": list(verify_log),
            },
        )

    gen_label = model
    solve_label = s_model if solve_client else "(same)"

    print(f"\n{'=' * 70}")
    print(f"  TTT-Discover+Verify (NULL-aware): Building dataset from hard problem")
    print(
        f"  Target: {n_target} problems with "
        f"{target_agreement_low:.0%}-{target_agreement_high:.0%} agreement"
    )
    print(f"  Samples per problem:  {n_samples_per_problem}")
    print(f"  Min non-NULL answers: {min_non_null}  (skip if fewer)")
    print(f"  Failed solution attempts for context: {len(failed_solutions or [])}")
    print(f"  Generate model: {gen_label}")
    print(f"  Solve model:    {solve_label}")
    print(f"  Verify model:   {v_model}  ({VERIFY_TRIES} votes, threshold {VERIFY_ACCEPT_THRESHOLD})")
    print(f"  Max parallel solve workers: {max_workers}")
    print(f"  Gen pipeline workers:       {gen_workers}")
    if quality_threshold is not None:
        print(f"  Quality threshold:          >= {quality_threshold}/10 (inline judge)")
    if output_path:
        print(f"  Keeps file:  {output_path}")
        print(f"  Skips file:  {skips_path}")
        print(f"  Verify log:  {verify_log_path}")
    print(f"{'=' * 70}\n")

    seen_lock = threading.Lock()
    state_lock = threading.Lock()
    candidate_counter = {"n": 0}

    def _gen_eval_verify(solve_pool: concurrent.futures.ThreadPoolExecutor):
        with state_lock:
            candidate_counter["n"] += 1
            cn = candidate_counter["n"]

        # ── Step 1: Generate ─────────────────────────────────────────────
        t0 = time.time()
        candidates = s1.generate_similar_problems(
            client, model, hard_problem, failed_solutions=failed_solutions,
        )
        gen_time = time.time() - t0

        if not candidates:
            return {"kind": "gen_failed", "candidate_num": cn, "gen_time": gen_time}

        problem_text = candidates[0]["problem"]
        with seen_lock:
            if use_dedupe:
                if not dedupe.add(problem_text):
                    return {"kind": "duplicate", "candidate_num": cn, "gen_time": gen_time}
            else:
                if problem_text in seen_problems:
                    return {"kind": "duplicate", "candidate_num": cn, "gen_time": gen_time}
                seen_problems.add(problem_text)

        # ── Step 2: Solve N times (NULL-aware) + majority vote ───────────
        t1 = time.time()
        agreement, majority_ans, all_answers, all_solutions, n_non_null, n_null = (
            solve_and_check_agreement_null(
                s_client, s_model, problem_text,
                n_samples=n_samples_per_problem, pool=solve_pool,
            )
        )
        eval_time = time.time() - t1

        # Skip if too many NULLs
        if n_non_null < min_non_null:
            entry = GeneratedProblem(
                problem=problem_text,
                ground_truth_answer="",
                agreement_rate=0.0,
                all_answers=all_answers,
                all_solutions=all_solutions,
                n_samples=n_samples_per_problem,
            )
            return {
                "kind": "evaluated",
                "candidate_num": cn,
                "gen_time": gen_time,
                "eval_time": eval_time,
                "verify_time": 0.0,
                "kept": False,
                "status": f"skip (only {n_non_null}/{n_samples_per_problem} non-NULL, need {min_non_null})",
                "agreement": 0.0,
                "majority_ans": "",
                "n_non_null": n_non_null,
                "n_null": n_null,
                "quality_score": None,
                "verify_result": None,
                "entry": entry,
            }

        in_range = target_agreement_low <= agreement <= target_agreement_high
        numeric = s1._is_numeric_answer(s1.normalize_answer(majority_ans))
        kept = in_range and bool(majority_ans) and numeric
        if not bool(majority_ans):
            status = "skip (empty answer)"
        elif not numeric:
            status = f"skip (non-numeric: {majority_ans[:40]})"
        elif in_range:
            status = "KEEP"
        else:
            status = "skip"

        # ── Step 3: Verify reasoning trace ───────────────────────────────
        verify_result = None
        verify_time = 0.0
        if kept:
            trace = pick_majority_trace(majority_ans, all_answers, all_solutions)
            if trace:
                t2 = time.time()
                accepted, verify_result = verify_reasoning(
                    client, v_model, problem_text, majority_ans, trace,
                )
                verify_time = time.time() - t2
                if not accepted:
                    kept = False
                    n_inc = verify_result["n_incorrect"]
                    status = f"skip (verify rejected: {n_inc}/{VERIFY_TRIES} incorrect)"
                else:
                    status += " [verified]"
            else:
                kept = False
                status = "skip (no matching trace for majority answer)"

        # ── Step 4: Optional quality score ───────────────────────────────
        quality_score = None
        if kept and quality_threshold is not None:
            prompt = s1.QUALITY_SCORE_PROMPT.format(
                target=hard_problem, candidate=problem_text,
            )
            raw = s1.call_llm(client, model, prompt, temperature=0.3)
            quality_score = s1._parse_quality_score(raw)
            if quality_score < quality_threshold:
                kept = False
                status = f"skip (quality {quality_score}/10 < {quality_threshold})"
            else:
                status = f"KEEP (quality {quality_score}/10) [verified]"

        entry = GeneratedProblem(
            problem=problem_text,
            ground_truth_answer=majority_ans,
            agreement_rate=agreement,
            all_answers=all_answers,
            all_solutions=all_solutions,
            n_samples=n_samples_per_problem,
        )
        return {
            "kind": "evaluated",
            "candidate_num": cn,
            "gen_time": gen_time,
            "eval_time": eval_time,
            "verify_time": verify_time,
            "kept": kept,
            "status": status,
            "agreement": agreement,
            "majority_ans": majority_ans,
            "n_non_null": n_non_null,
            "n_null": n_null,
            "quality_score": quality_score,
            "verify_result": verify_result,
            "entry": entry,
        }

    # ── Worker pools ─────────────────────────────────────────────────────
    solve_pool_size = max(max_workers, gen_workers * n_samples_per_problem)

    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=gen_workers) as gen_pool,
        concurrent.futures.ThreadPoolExecutor(max_workers=solve_pool_size) as solve_pool,
    ):
        in_flight: set[Future] = set()

        def _submit_more():
            with state_lock:
                target_inflight = max(0, n_target - len(dataset.problems))
            target_inflight = min(target_inflight, gen_workers)
            while len(in_flight) < target_inflight:
                in_flight.add(gen_pool.submit(_gen_eval_verify, solve_pool))

        _submit_more()

        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in done:
                in_flight.discard(fut)
                try:
                    result = fut.result()
                except Exception as e:
                    print(f"  [error] worker raised: {e}")
                    continue

                cn = result["candidate_num"]
                kind = result["kind"]

                if kind == "gen_failed":
                    print(
                        f"--- Candidate {cn} ---  generation failed "
                        f"({result['gen_time']:.1f}s), continuing"
                    )
                elif kind == "duplicate":
                    print(f"--- Candidate {cn} ---  duplicate, continuing")
                elif kind == "evaluated":
                    entry = result["entry"]
                    with state_lock:
                        if result["kept"]:
                            dataset.problems.append(entry)
                        else:
                            skipped_problems.append(entry)
                        if result.get("verify_result"):
                            verify_log.append({
                                "candidate_num": cn,
                                "problem_snippet": entry.problem[:160],
                                "answer": entry.ground_truth_answer[:40],
                                **result["verify_result"],
                            })
                        kept_count = len(dataset.problems)
                        skipped_count = len(skipped_problems)
                        _flush()
                    vt = result.get("verify_time", 0)
                    nn = result.get("n_non_null", "?")
                    nz = result.get("n_null", "?")
                    timing = (
                        f"gen {result['gen_time']:.1f}s + eval {result['eval_time']:.1f}s"
                        + (f" + verify {vt:.1f}s" if vt > 0 else "")
                    )
                    null_info = f"  [non-null={nn}, null={nz}]"
                    print(
                        f"--- Candidate {cn} ---  {timing}  "
                        f"{result['agreement']:.0%} -> {result['status']}"
                        + null_info
                    )
                    if result["kept"]:
                        print(f"    majority answer: {result['majority_ans'][:80]}")
                    print(f"    totals: {kept_count} kept, {skipped_count} skipped")

            with state_lock:
                done_yet = len(dataset.problems) >= n_target
            if not done_yet:
                _submit_more()

    # ── Summary ──────────────────────────────────────────────────────────
    n_verified = sum(1 for r in verify_log if r.get("accepted"))
    n_rejected = sum(1 for r in verify_log if not r.get("accepted"))
    print(f"{'=' * 70}")
    print(f"  Dataset complete: {len(dataset.problems)} kept, {len(skipped_problems)} skipped")
    avg_agreement = (
        sum(p.agreement_rate for p in dataset.problems) / len(dataset.problems)
        if dataset.problems else 0
    )
    print(f"  Average agreement rate (kept, non-NULL): {avg_agreement:.1%}")
    print(f"  Verify: {n_verified} confirmed, {n_rejected} rejected")
    if dedupe is not None:
        kept_this_run = dedupe.n_kept - dedupe_baseline_kept
        exact_this_run = dedupe.n_exact_dropped - dedupe_baseline_exact
        fuzzy_this_run = dedupe.n_fuzzy_dropped - dedupe_baseline_fuzzy
        print(
            f"  Dedupe: kept {kept_this_run}, "
            f"dropped {exact_this_run + fuzzy_this_run} "
            f"(exact={exact_this_run}, fuzzy={fuzzy_this_run})"
        )
    print(f"{'=' * 70}\n")

    return dataset


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
    problem: str,
    *,
    n_problems: int = 100,
    n_samples: int = 10,
    agree_low: float = 0.60,
    agree_high: float = 0.80,
    output: str | None = None,
    model: str | None = None,
    max_workers: int = 16,
    gen_workers: int = 8,
    failed_solutions: list[str] | None = None,
    use_tinker: bool = False,
    tinker_checkpoint: str | None = None,
    tinker_checkpoint_step: int = 50,
    quality_threshold: int | None = None,
    use_dedupe: bool = True,
    verify_model: str | None = None,
    min_non_null: int = MIN_NON_NULL_ANSWERS,
) -> Dataset:
    if use_tinker:
        tinker_client, tinker_model = s1.get_tinker_client(
            tinker_checkpoint, checkpoint_step=tinker_checkpoint_step,
        )
        gen_client, gen_model = tinker_client, model or tinker_model
        solve_client, solve_model = tinker_client, tinker_model
    else:
        gen_client, gen_default_model = s1.get_client()
        gen_model = model or gen_default_model
        solve_client, solve_model = None, None

    from datetime import datetime

    run_dir = output or os.path.join(
        os.path.dirname(__file__),
        "runs",
        datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}",
    )
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "keeps.json")

    dataset = build_null_verified_dataset(
        client=gen_client,
        model=gen_model,
        hard_problem=problem,
        n_target=n_problems,
        n_samples_per_problem=n_samples,
        target_agreement_low=agree_low,
        target_agreement_high=agree_high,
        output_path=out_path,
        max_workers=max_workers,
        gen_workers=gen_workers,
        failed_solutions=failed_solutions,
        solve_client=solve_client,
        solve_model=solve_model,
        quality_threshold=quality_threshold,
        use_dedupe=use_dedupe,
        verify_model=verify_model or gen_model,
        min_non_null=min_non_null,
    )

    s1.save_dataset(dataset, out_path)
    return dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "TTT-Discover+Verify (NULL-aware): generate subproblems where the "
            "solver can abstain with NULL; agreement is computed over non-NULL "
            "responses only."
        )
    )
    parser.add_argument(
        "--problem-path", type=str, required=True,
        help="Path to a .txt file containing the problem statement",
    )
    parser.add_argument(
        "--runs-subdir", type=str, default=None,
        help="Subdirectory name under runs/ for outputs (default: .txt filename stem)",
    )
    parser.add_argument(
        "--failed-solutions", type=str, default=None,
        help="Path to failed-attempts JSON (default: data/reasoning-traces/<runs-subdir>.json)",
    )
    parser.add_argument("--tinker", action="store_true",
                        help="Use a tinker checkpoint instead of Vertex AI")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Tinker checkpoint tinker://... path")
    parser.add_argument("--tinker-step", type=int, default=50,
                        help="Tinker checkpoint step (default: 50)")
    parser.add_argument("--n-problems", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--gen-workers", type=int, default=8,
                        help="Concurrent gen+eval+verify pipeline workers (default 8)")
    parser.add_argument("--max-workers", type=int, default=16,
                        help="Solve-pool worker hint (default 16)")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--verify-model", type=str, default=None,
                        help=(
                            "Model for reasoning verification (default: same as --model). "
                            "Uses gpt-oss-120b by default."
                        ))
    parser.add_argument("--quality-threshold", type=int, default=None,
                        help="Inline 0-10 quality judge; only keep scoring >= this (e.g. 9)")
    parser.add_argument("--output", type=str, default=None,
                        help="Override run dir (default: runs/<runs-subdir>/stage1/<timestamp>/)")
    parser.add_argument("--no-dedupe", action="store_true",
                        help="Disable fuzzy dedup (for ablation)")
    parser.add_argument(
        "--min-non-null", type=int, default=MIN_NON_NULL_ANSWERS,
        help=(
            f"Minimum number of non-NULL answers required to keep a candidate "
            f"(default: {MIN_NON_NULL_ANSWERS}). Candidates where the model "
            f"abstains on more than (n-samples - min-non-null) calls are skipped."
        ),
    )
    args = parser.parse_args()

    problem_text = s1.load_problem_from_txt(args.problem_path)
    problem_stem = os.path.splitext(
        os.path.basename(os.path.abspath(args.problem_path))
    )[0]
    runs_subdir = (args.runs_subdir or "").strip() or problem_stem
    print(
        f"\nLoaded problem from {args.problem_path} "
        f"(runs/{runs_subdir}/, {len(problem_text)} chars)\n"
    )

    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    runs_root = os.path.join(repo_root, "runs", runs_subdir)

    failed_path = args.failed_solutions or os.path.join(
        repo_root, "data", "reasoning-traces", f"{runs_subdir}.json",
    )
    failed_solutions = s1._load_failed_solutions(failed_path)
    if not failed_solutions:
        print(
            f"No failed solutions at {failed_path}. "
            f"Add traces to data/reasoning-traces/{runs_subdir}.json "
            f"or pass --failed-solutions explicitly."
        )

    if args.output:
        run_dir = args.output
    else:
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
        run_dir = os.path.join(runs_root, "stage1", ts)

    run(
        problem=problem_text,
        n_problems=args.n_problems,
        n_samples=args.n_samples,
        model=args.model or "openai/gpt-oss-120b-maas",
        failed_solutions=failed_solutions,
        use_tinker=args.tinker,
        tinker_checkpoint=args.checkpoint,
        tinker_checkpoint_step=args.tinker_step,
        output=run_dir,
        gen_workers=args.gen_workers,
        quality_threshold=args.quality_threshold,
        max_workers=args.max_workers,
        use_dedupe=not args.no_dedupe,
        verify_model=args.verify_model,
        min_non_null=args.min_non_null,
    )


if __name__ == "__main__":
    main()
