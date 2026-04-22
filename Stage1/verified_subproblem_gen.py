"""
TTT-Discover + Verify: subproblem generation with reasoning-trace verification.

Drop-in replacement for distinct_llm_prompting.py that adds a verification
gate after majority-vote labelling.  For each candidate that passes the
agreement window, a reasoning trace from the majority answer is audited by
gpt-oss-120b.  Only candidates whose reasoning is confirmed correct are kept,
reducing bad ground-truth labels from correlated model errors.

Pipeline per candidate:
  1. Generate candidate subproblem  (same as Stage 1)
  2. Sample N solutions, compute majority vote  (same as Stage 1)
  3. Filter by agreement window + numeric answer  (same as Stage 1)
  4. **NEW** — Pick one trace that produced the majority answer, ask
     gpt-oss-120b to audit the reasoning step-by-step.  3-vote majority;
     need >= 2/3 "correct" to keep.
  5. (Optional) Quality-score filter  (same as Stage 1)

All heavy lifting (client setup, LLM calls, answer extraction, dedupe) is
imported from distinct_llm_prompting — no duplication.
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

# ---------------------------------------------------------------------------
# Imports from the existing Stage 1 module (no modifications needed there)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import Stage1.distinct_llm_prompting as s1  # noqa: E402
from pipeline_stages.dedupe import DedupeIndex  # noqa: E402

# Re-export for convenience
GeneratedProblem = s1.GeneratedProblem
Dataset = s1.Dataset

# ---------------------------------------------------------------------------
# Verification constants & prompt
# ---------------------------------------------------------------------------
VERIFY_TRIES = 3
VERIFY_ACCEPT_THRESHOLD = 2  # need >= 2/3 "correct" votes to keep

VERIFY_PROMPT = """\
You are a rigorous math reviewer. You will be given:
  1. A math problem.
  2. A claimed final answer.
  3. The full step-by-step solution that produced that answer.

Your job is to carefully check whether the reasoning is mathematically correct
and whether the final answer logically follows from the steps.

Do NOT solve the problem yourself from scratch. Instead, audit the given
solution line by line:
- Check each algebraic / arithmetic step.
- Check that any applied theorems or formulas are used correctly.
- Check that no steps are skipped in a way that hides an error.
- Check that the final numerical answer matches what the derivation produces.

## Problem

{problem}

## Claimed Answer

{answer}

## Solution Trace

{trace}

## Your Verdict

Return JSON only, no other text:
{{"verdict": "correct", "reason": "<1-2 sentence justification>"}}
or
{{"verdict": "incorrect", "reason": "<1-2 sentence explaining the error>"}}
"""

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_verify_verdict(raw: str) -> tuple[str, str]:
    """Return (verdict, reason). verdict is 'correct' | 'incorrect' | 'parse_error'."""
    m = _JSON_RE.search(raw or "")
    if not m:
        return "parse_error", (raw or "")[:200]
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return "parse_error", (raw or "")[:200]
    v = str(obj.get("verdict", "")).strip().lower()
    if v not in ("correct", "incorrect"):
        return "parse_error", (raw or "")[:200]
    return v, str(obj.get("reason", ""))[:300]


def pick_majority_trace(
    majority_answer: str,
    all_answers: list[str],
    all_solutions: list[str],
) -> str | None:
    """Select one solution trace whose extracted answer matches the majority."""
    for sol, ans in zip(all_solutions, all_answers):
        if ans == majority_answer and sol.strip():
            return sol
    norm = s1.normalize_answer(majority_answer)
    for sol, ans in zip(all_solutions, all_answers):
        if s1.normalize_answer(ans) == norm and sol.strip():
            return sol
    return None


def verify_reasoning(client, model: str, problem_text: str, answer: str, trace: str) -> tuple[bool, dict]:
    """Ask the model to audit a reasoning trace. Returns (accepted, details)."""
    prompt = VERIFY_PROMPT.format(problem=problem_text, answer=answer, trace=trace)
    votes: list[tuple[str, str]] = []
    for _ in range(VERIFY_TRIES):
        raw = s1.call_llm(client, model, prompt, temperature=0.3)
        votes.append(_parse_verify_verdict(raw))

    n_correct = sum(1 for v, _ in votes if v == "correct")
    accepted = n_correct >= VERIFY_ACCEPT_THRESHOLD
    details = {
        "accepted": accepted,
        "n_correct": n_correct,
        "n_incorrect": sum(1 for v, _ in votes if v == "incorrect"),
        "n_parse_errors": sum(1 for v, _ in votes if v == "parse_error"),
        "votes": [{"verdict": v, "reason": r} for v, r in votes],
    }
    return accepted, details


# ---------------------------------------------------------------------------
# Core: build_verified_dataset
# ---------------------------------------------------------------------------
def build_verified_dataset(
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
) -> Dataset:
    """Generate subproblems with integrated reasoning-trace verification.

    Identical to s1.build_dataset() except that every candidate passing the
    agreement + numeric filters is also verified: a majority-answer trace is
    audited by the verifier model (default: same as gen model).  Only
    candidates confirmed correct are kept.
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
    print(f"  TTT-Discover+Verify: Building verified dataset from hard problem")
    print(
        f"  Target: {n_target} problems with "
        f"{target_agreement_low:.0%}-{target_agreement_high:.0%} agreement"
    )
    print(f"  Samples per problem: {n_samples_per_problem}")
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

        # ── Step 2: Solve N times + majority vote ────────────────────────
        t1 = time.time()
        agreement, majority_ans, all_answers, all_solutions = s1.solve_and_check_agreement(
            s_client, s_model, problem_text,
            n_samples=n_samples_per_problem, pool=solve_pool,
        )
        eval_time = time.time() - t1

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
                    timing = (
                        f"gen {result['gen_time']:.1f}s + eval {result['eval_time']:.1f}s"
                        + (f" + verify {vt:.1f}s" if vt > 0 else "")
                    )
                    print(
                        f"--- Candidate {cn} ---  {timing}  "
                        f"{result['agreement']:.0%} -> {result['status']}"
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
    print(f"  Average agreement rate (kept): {avg_agreement:.1%}")
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

    dataset = build_verified_dataset(
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
    )

    s1.save_dataset(dataset, out_path)
    return dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="TTT-Discover+Verify: generate subproblems with reasoning verification"
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
                        help="Tinker checkpoint tinker://… path")
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
    )


if __name__ == "__main__":
    main()
