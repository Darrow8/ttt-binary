"""
TTT-Discover + Verify + Judge: subproblem generation with LLM judge pre-filter
and reasoning-trace verification.

Extends verified_subproblem_gen.py with an early LLM judge gate that screens
each candidate *before* the expensive solve step. The judge rejects candidates
that are:
  - Multi-part (asks the student to find A *then* B *then* C)
  - Built on a false premise (contradictory or impossible setup)
  - Intentionally confusing / trick questions

Pipeline per candidate:
  1. Generate candidate subproblem  (same as Stage 1)
  1b. **NEW** — LLM judge screens the problem statement for multi-part,
      false-premise, or intentionally-confusing qualities.  Rejected candidates
      are logged and skipped before any solve work.
  2. Sample N solutions, compute majority vote  (same as Stage 1)
  3. Filter by agreement window + numeric answer  (same as Stage 1)
  4. Verify reasoning traces  (same as verified_subproblem_gen)
  5. (Optional) Quality-score filter  (same as Stage 1)

All heavy lifting (client setup, LLM calls, answer extraction, dedupe) is
imported from distinct_llm_prompting — no duplication.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
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
# Verification constants & prompts
# ---------------------------------------------------------------------------

# Number of approve-pass votes per trace (at low temperature for consistency).
VERIFY_TRIES = 5
# Minimum approve votes across ALL trace audits combined to accept.
VERIFY_ACCEPT_THRESHOLD = 4  # require 4/5 unanimous-ish agreement

# How many distinct majority-answer traces to sample and audit.
VERIFY_N_TRACES = 3

# ---------------------------------------------------------------------------
# LLM Judge: pre-filter for multi-part / false-premise / confusing problems
# ---------------------------------------------------------------------------

JUDGE_TRIES = 3  # majority vote across this many judge calls
JUDGE_TEMPERATURE = 0.3

JUDGE_PROMPT = """\
You are a math-problem quality judge. You will be given a math problem \
statement and must decide whether it should be REJECTED.

Reject the problem if ANY of the following are true:
1. **Multi-part**: The problem asks for more than one distinct quantity (e.g. \
"find A, then find B, then find C" or "determine X and Y"). A single problem \
that naturally requires intermediate steps is fine — reject only when the \
problem explicitly requests multiple separate final answers.
2. **False premise**: The problem contains contradictory, impossible, or \
physically/mathematically nonsensical assumptions that make it unsolvable as \
stated (e.g. a triangle with sides 1, 2, 10).
3. **Intentionally confusing**: The problem uses deliberately misleading \
wording, trick phrasing, or ambiguous notation designed to confuse rather \
than test genuine mathematical skill.

A well-posed, single-answer problem — even if hard — should be ACCEPTED.

## Problem

{problem}

## Your Response

Return **only** a JSON object — no other text, no markdown fences:
{{
  "verdict": "accept" | "reject",
  "reason_code": "ok" | "multi_part" | "false_premise" | "confusing",
  "explanation": "<1-2 sentence justification>"
}}
"""


def _parse_judge_verdict(raw: str) -> tuple[str, str, str]:
    """Parse a judge response.

    Returns (verdict, reason_code, explanation).
    verdict is 'accept' | 'reject' | 'parse_error'.
    """
    blob = _extract_outermost_json(raw or "")
    if not blob:
        return "parse_error", "", (raw or "")[:200]
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return "parse_error", "", (raw or "")[:200]
    v = str(obj.get("verdict", "")).strip().lower()
    if v not in ("accept", "reject"):
        return "parse_error", "", (raw or "")[:200]
    reason_code = str(obj.get("reason_code", "")).strip().lower()
    explanation = str(obj.get("explanation") or "")[:300]
    return v, reason_code, explanation


def judge_problem(
    client,
    model: str,
    problem_text: str,
) -> tuple[bool, dict]:
    """Run the LLM judge on a problem statement.

    Returns (accepted, details) where accepted=True means the problem
    passed the judge (is NOT multi-part, false-premise, or confusing).
    """
    prompt = JUDGE_PROMPT.format(problem=problem_text)
    votes: list[dict] = []
    n_accept = 0
    n_reject = 0

    for attempt in range(JUDGE_TRIES):
        raw = None
        parsed = ("parse_error", "", "")
        for _try in range(2):
            raw = s1.call_llm(client, model, prompt, temperature=JUDGE_TEMPERATURE)
            parsed = _parse_judge_verdict(raw)
            if parsed[0] != "parse_error":
                break
        verdict, reason_code, explanation = parsed
        if verdict == "accept":
            n_accept += 1
        else:
            n_reject += 1
        votes.append({
            "attempt": attempt,
            "verdict": verdict,
            "reason_code": reason_code,
            "explanation": explanation,
        })

    accepted = n_accept > n_reject
    majority_reasons = [v["reason_code"] for v in votes if v["verdict"] == ("accept" if accepted else "reject")]
    details = {
        "accepted": accepted,
        "n_accept": n_accept,
        "n_reject": n_reject,
        "majority_reason": majority_reasons[0] if majority_reasons else "",
        "votes": votes,
    }
    return accepted, details


# ---------------------------------------------------------------------------
# Verification prompts & logic (same as verified_subproblem_gen)
# ---------------------------------------------------------------------------

# Approve-pass: model must cite concrete trace evidence, not re-solve.
VERIFY_PROMPT_APPROVE = """\
You are a rigorous math proof auditor. You will be given a math problem, a \
claimed answer, and a step-by-step solution trace.

Your ONLY job is to audit the *given trace* for correctness. You must NOT \
solve the problem yourself or rely on memorised results. Every verdict you \
issue must be grounded in a specific quoted or paraphrased step from the trace.

Audit checklist (work through in order):
1. Is every algebraic / arithmetic transformation in the trace valid?
2. Are all cited theorems or formulas applied correctly?
3. Are there any hidden jumps that could conceal an error?
4. Does the stated final answer follow directly from the last derivation step?

## Problem

{problem}

## Claimed Answer

{answer}

## Solution Trace

{trace}

## Your Response

Return **only** a JSON object — no other text, no markdown fences:
{{
  "verdict": "correct" | "incorrect",
  "critical_step": "<quote or close paraphrase of the key step you checked>",
  "trace_evidence": "<what the trace actually says that supports or refutes correctness>",
  "reason": "<1-2 sentence conclusion>"
}}

If you cannot point to a specific step in the trace that justifies your \
verdict, return verdict "incorrect" with reason "trace insufficient to audit".
"""

# Adversarial pass: specifically hunting for the first fatal flaw.
VERIFY_PROMPT_ADVERSARIAL = """\
You are an adversarial math checker. You will be given a math problem, a \
claimed answer, and a step-by-step solution trace.

Your job is to find the *first fatal flaw* in the trace — a step that is \
mathematically wrong, unjustified, or that silently changes the problem. \
Do NOT re-solve the problem; scrutinise the trace itself.

Look especially for:
- Off-by-one or sign errors,
- Incorrect application of a theorem (wrong hypotheses, wrong direction),
- A case that is silently dropped,
- An answer that does not match the last derivation step numerically.

## Problem

{problem}

## Claimed Answer

{answer}

## Solution Trace

{trace}

## Your Response

Return **only** a JSON object — no markdown fences:
{{
  "flaw_found": true | false,
  "flaw_step": "<quoted or paraphrased step where the flaw occurs, or null>",
  "flaw_description": "<what is wrong, or null if no flaw>",
  "reason": "<1-2 sentence summary>"
}}
"""

# Robust JSON extractor: finds the *last* top-level {...} block to avoid
# being fooled by prose braces earlier in the response, and handles nested
# braces inside string values.
def _extract_outermost_json(text: str) -> str | None:
    """Return the last outermost {...} block in text, or None."""
    best: str | None = None
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                best = text[start : i + 1]
    return best


def _parse_verify_verdict(raw: str) -> tuple[str, str, str, str]:
    """Parse an approve-pass response.

    Returns (verdict, critical_step, trace_evidence, reason).
    verdict is 'correct' | 'incorrect' | 'parse_error'.
    """
    blob = _extract_outermost_json(raw or "")
    if not blob:
        return "parse_error", "", "", (raw or "")[:200]
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return "parse_error", "", "", (raw or "")[:200]
    v = str(obj.get("verdict", "")).strip().lower()
    if v not in ("correct", "incorrect"):
        return "parse_error", "", "", (raw or "")[:200]
    critical_step = str(obj.get("critical_step") or "")[:300]
    trace_evidence = str(obj.get("trace_evidence") or "")[:300]
    reason = str(obj.get("reason") or "")[:300]
    # Reject votes that couldn't cite anything — model likely re-solved.
    if v == "correct" and not critical_step and not trace_evidence:
        return "incorrect", "", "", "no trace evidence cited — likely re-solved"
    return v, critical_step, trace_evidence, reason


def _parse_adversarial_verdict(raw: str) -> tuple[bool | None, str, str]:
    """Parse an adversarial-pass response.

    Returns (flaw_found, flaw_step, flaw_description).
    flaw_found is None on parse error.
    """
    blob = _extract_outermost_json(raw or "")
    if not blob:
        return None, "", (raw or "")[:200]
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return None, "", (raw or "")[:200]
    raw_flaw = obj.get("flaw_found")
    if raw_flaw is None:
        return None, "", (raw or "")[:200]
    flaw_found = bool(raw_flaw)
    flaw_step = str(obj.get("flaw_step") or "")[:300]
    flaw_desc = str(obj.get("flaw_description") or obj.get("reason") or "")[:300]
    return flaw_found, flaw_step, flaw_desc


def pick_majority_traces(
    majority_answer: str,
    all_answers: list[str],
    all_solutions: list[str],
    n: int = VERIFY_N_TRACES,
) -> list[str]:
    """Return up to *n* distinct non-empty traces whose answer matches the majority."""
    norm = s1.normalize_answer(majority_answer)
    seen: set[str] = set()
    traces: list[str] = []
    # Exact match first, then normalised match.
    for pass_exact in (True, False):
        for sol, ans in zip(all_solutions, all_answers):
            if not sol.strip():
                continue
            fingerprint = sol[:120]  # cheap dedup of near-identical solutions
            if fingerprint in seen:
                continue
            match = (ans == majority_answer) if pass_exact else (s1.normalize_answer(ans) == norm)
            if match:
                seen.add(fingerprint)
                traces.append(sol)
                if len(traces) >= n:
                    return traces
    return traces


# Keep old name as alias so any external callers don't break.
def pick_majority_trace(
    majority_answer: str,
    all_answers: list[str],
    all_solutions: list[str],
) -> str | None:
    traces = pick_majority_traces(majority_answer, all_answers, all_solutions, n=1)
    return traces[0] if traces else None


def _audit_single_trace(
    client,
    model: str,
    problem_text: str,
    answer: str,
    trace: str,
    trace_idx: int,
) -> dict:
    """Run VERIFY_TRIES approve votes + 1 adversarial vote on one trace.

    Returns a per-trace audit record.
    """
    approve_votes: list[dict] = []
    n_correct = 0
    n_incorrect = 0
    n_parse_errors = 0

    approve_prompt = VERIFY_PROMPT_APPROVE.format(
        problem=problem_text, answer=answer, trace=trace,
    )
    for attempt in range(VERIFY_TRIES):
        raw = None
        parsed = ("parse_error", "", "", "")
        # Retry once on parse error to avoid silently discarding a vote.
        for _try in range(2):
            raw = s1.call_llm(client, model, approve_prompt, temperature=0.3)
            parsed = _parse_verify_verdict(raw)
            if parsed[0] != "parse_error":
                break
        verdict, critical_step, trace_evidence, reason = parsed
        if verdict == "correct":
            n_correct += 1
        elif verdict == "incorrect":
            n_incorrect += 1
        else:
            n_parse_errors += 1
            # Parse errors count as negative evidence.
            n_incorrect += 1
        approve_votes.append({
            "attempt": attempt,
            "verdict": verdict,
            "critical_step": critical_step,
            "trace_evidence": trace_evidence,
            "reason": reason,
        })

    # Adversarial pass — single call, hunting for a fatal flaw.
    adv_prompt = VERIFY_PROMPT_ADVERSARIAL.format(
        problem=problem_text, answer=answer, trace=trace,
    )
    adv_raw = s1.call_llm(client, model, adv_prompt, temperature=0.6)
    flaw_found, flaw_step, flaw_desc = _parse_adversarial_verdict(adv_raw)

    return {
        "trace_idx": trace_idx,
        "trace_snippet": trace[:300],
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_parse_errors": n_parse_errors,
        "approve_votes": approve_votes,
        "adversarial": {
            "flaw_found": flaw_found,
            "flaw_step": flaw_step,
            "flaw_description": flaw_desc,
        },
    }


def verify_reasoning(
    client,
    model: str,
    problem_text: str,
    answer: str,
    traces: list[str],
) -> tuple[bool, dict]:
    """Audit multiple majority traces.  Returns (accepted, details).

    Acceptance requires:
    - Total approve votes across all traces >= VERIFY_ACCEPT_THRESHOLD
    - No trace had its adversarial pass unanimously flag a flaw (flaw_found=True
      with a specific flaw_step cited — pure "flaw_found: true, flaw_step: null"
      is treated as weak evidence and does not veto alone).
    """
    per_trace: list[dict] = []
    total_correct = 0
    total_votes = 0
    adv_vetoes = 0

    for idx, trace in enumerate(traces):
        record = _audit_single_trace(client, model, problem_text, answer, trace, idx)
        per_trace.append(record)
        total_correct += record["n_correct"]
        total_votes += VERIFY_TRIES
        adv = record["adversarial"]
        if adv["flaw_found"] is True and adv["flaw_step"]:
            adv_vetoes += 1

    approve_pass = total_correct >= VERIFY_ACCEPT_THRESHOLD
    # Veto only if ALL traces had a specific flaw identified (adversarial unanimity).
    adv_veto = len(traces) > 0 and adv_vetoes == len(traces)
    accepted = approve_pass and not adv_veto

    details = {
        "accepted": accepted,
        "total_correct": total_correct,
        "total_votes": total_votes,
        "n_traces_audited": len(traces),
        "adv_vetoes": adv_vetoes,
        "adv_veto_triggered": adv_veto,
        "per_trace": per_trace,
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
    judge_model: str | None = None,
    problems_per_call: int = 2,
) -> Dataset:
    """Generate subproblems with integrated reasoning-trace verification.

    Identical to s1.build_dataset() except that every candidate passing the
    agreement + numeric filters is also verified: a majority-answer trace is
    audited by the verifier model (default: same as gen model).  Only
    candidates confirmed correct are kept.

    Additionally, an LLM judge pre-screens each candidate before solving to
    reject multi-part, false-premise, or intentionally-confusing problems.
    """
    s_client = solve_client or client
    s_model = solve_model or model
    v_model = verify_model or model
    j_model = judge_model or model

    dataset = Dataset(
        source_problem=hard_problem,
        target_agreement_low=target_agreement_low,
        target_agreement_high=target_agreement_high,
    )

    skipped_problems: list[GeneratedProblem] = []
    verify_log: list[dict] = []
    judge_log: list[dict] = []
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
        judge_log_path = os.path.join(os.path.dirname(output_path), "judge_log.json")
    else:
        skips_path = None
        verify_log_path = None
        judge_log_path = None

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
                "tries_per_trace": VERIFY_TRIES,
                "n_traces_per_candidate": VERIFY_N_TRACES,
                "accept_threshold": VERIFY_ACCEPT_THRESHOLD,
                "records": list(verify_log),
            },
        )
        s1._save_atomic(
            judge_log_path,
            {
                "judge_model": j_model,
                "judge_tries": JUDGE_TRIES,
                "records": list(judge_log),
            },
        )

    gen_label = model
    solve_label = s_model if solve_client else "(same)"

    print(f"\n{'=' * 70}")
    print(f"  TTT-Discover+Verify+Judge: Building verified+judged dataset from hard problem")
    print(
        f"  Target: {n_target} problems with "
        f"{target_agreement_low:.0%}-{target_agreement_high:.0%} agreement"
    )
    print(f"  Samples per problem: {n_samples_per_problem}")
    print(f"  Problems per gen call:      {problems_per_call}")
    print(f"  Failed solution attempts for context: {len(failed_solutions or [])}")
    print(f"  Generate model: {gen_label}")
    print(f"  Solve model:    {solve_label}")
    print(f"  Verify model:   {v_model}  ({VERIFY_N_TRACES} traces × {VERIFY_TRIES} votes, threshold {VERIFY_ACCEPT_THRESHOLD})")
    print(f"  Judge model:    {j_model}  ({JUDGE_TRIES} votes, majority rule)")
    print(f"  Max parallel solve workers: {max_workers}")
    print(f"  Gen pipeline workers:       {gen_workers}")
    if quality_threshold is not None:
        print(f"  Quality threshold:          >= {quality_threshold}/10 (inline judge)")
    if output_path:
        print(f"  Keeps file:  {output_path}")
        print(f"  Skips file:  {skips_path}")
        print(f"  Verify log:  {verify_log_path}")
        print(f"  Judge log:   {judge_log_path}")
    print(f"{'=' * 70}\n")

    seen_lock = threading.Lock()
    state_lock = threading.Lock()
    candidate_counter = {"n": 0}

    def _gen_eval_verify(solve_pool: concurrent.futures.ThreadPoolExecutor):
        """Generate a batch of candidates, evaluate and verify each one.

        Returns a list of result dicts (one per candidate in the batch).
        """
        with state_lock:
            candidate_counter["n"] += 1
            cn = candidate_counter["n"]

        # ── Step 1: Generate ─────────────────────────────────────────────
        t0 = time.time()
        candidates = s1.generate_similar_problems(
            client, model, hard_problem,
            failed_solutions=failed_solutions,
            batch_size=problems_per_call,
        )
        gen_time = time.time() - t0

        if not candidates:
            return [{"kind": "gen_failed", "candidate_num": cn, "gen_time": gen_time}]

        results = []
        for ci, cand in enumerate(candidates):
            problem_text = cand["problem"]
            sub_label = f"{cn}.{ci + 1}"

            with seen_lock:
                if use_dedupe:
                    if not dedupe.add(problem_text):
                        results.append({"kind": "duplicate", "candidate_num": sub_label, "gen_time": gen_time})
                        continue
                else:
                    if problem_text in seen_problems:
                        results.append({"kind": "duplicate", "candidate_num": sub_label, "gen_time": gen_time})
                        continue
                    seen_problems.add(problem_text)

            # ── Step 1b: LLM Judge pre-filter ────────────────────────────
            t_judge = time.time()
            judge_accepted, judge_details = judge_problem(
                client, j_model, problem_text,
            )
            judge_time = time.time() - t_judge

            if not judge_accepted:
                reason_code = judge_details.get("majority_reason", "unknown")
                explanation = ""
                for v in judge_details.get("votes", []):
                    if v["verdict"] == "reject" and v.get("explanation"):
                        explanation = v["explanation"]
                        break
                entry = GeneratedProblem(
                    problem=problem_text,
                    ground_truth_answer="",
                    agreement_rate=0.0,
                    all_answers=[],
                    all_solutions=[],
                    n_samples=0,
                )
                results.append({
                    "kind": "judge_rejected",
                    "candidate_num": sub_label,
                    "gen_time": gen_time,
                    "judge_time": judge_time,
                    "reason_code": reason_code,
                    "explanation": explanation,
                    "judge_details": judge_details,
                    "entry": entry,
                })
                continue

            # ── Step 2: Solve N times + majority vote ────────────────────
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

            # ── Step 3: Verify reasoning traces ──────────────────────────
            verify_result = None
            verify_time = 0.0
            if kept:
                traces = pick_majority_traces(majority_ans, all_answers, all_solutions)
                if traces:
                    t2 = time.time()
                    accepted, verify_result = verify_reasoning(
                        client, v_model, problem_text, majority_ans, traces,
                    )
                    verify_time = time.time() - t2
                    if not accepted:
                        kept = False
                        n_tot = verify_result["total_correct"]
                        n_votes = verify_result["total_votes"]
                        adv = " [adv-veto]" if verify_result["adv_veto_triggered"] else ""
                        status = f"skip (verify rejected: {n_tot}/{n_votes} correct{adv})"
                    else:
                        status += " [verified]"
                else:
                    kept = False
                    status = "skip (no matching trace for majority answer)"

            # ── Step 4: Optional quality score ───────────────────────────
            quality_score = None
            if kept and quality_threshold is not None:
                qprompt = s1.QUALITY_SCORE_PROMPT.format(
                    target=hard_problem, candidate=problem_text,
                )
                raw = s1.call_llm(client, model, qprompt, temperature=0.3)
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
            results.append({
                "kind": "evaluated",
                "candidate_num": sub_label,
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
            })

        return results

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
                    batch_results = fut.result()
                except Exception as e:
                    print(f"  [error] worker raised: {e}")
                    continue

                for result in batch_results:
                    cn = result["candidate_num"]
                    kind = result["kind"]

                    if kind == "gen_failed":
                        print(
                            f"--- Candidate {cn} ---  generation failed "
                            f"({result['gen_time']:.1f}s), continuing"
                        )
                    elif kind == "duplicate":
                        print(f"--- Candidate {cn} ---  duplicate, continuing")
                    elif kind == "judge_rejected":
                        entry = result["entry"]
                        rc = result["reason_code"]
                        expl = result.get("explanation", "")[:80]
                        with state_lock:
                            skipped_problems.append(entry)
                            judge_log.append({
                                "candidate_num": cn,
                                "problem_snippet": entry.problem[:160],
                                "accepted": False,
                                "reason_code": rc,
                                "explanation": expl,
                                "details": result["judge_details"],
                            })
                            skipped_count = len(skipped_problems)
                            _flush()
                        print(
                            f"--- Candidate {cn} ---  gen {result['gen_time']:.1f}s + "
                            f"judge {result['judge_time']:.1f}s  "
                            f"skip (judge: {rc}) {expl}"
                        )
                        print(f"    totals: {len(dataset.problems)} kept, {skipped_count} skipped")
                    elif kind == "evaluated":
                        entry = result["entry"]
                        with state_lock:
                            if result["kept"]:
                                dataset.problems.append(entry)
                            else:
                                skipped_problems.append(entry)
                            if result.get("verify_result"):
                                vr = result["verify_result"]
                                verify_log.append({
                                    "candidate_num": cn,
                                    "problem_snippet": entry.problem[:160],
                                    "answer": entry.ground_truth_answer[:40],
                                    "accepted": vr["accepted"],
                                    "total_correct": vr["total_correct"],
                                    "total_votes": vr["total_votes"],
                                    "n_traces_audited": vr["n_traces_audited"],
                                    "adv_veto_triggered": vr["adv_veto_triggered"],
                                    "per_trace": vr["per_trace"],
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
    n_adv_vetoed = sum(1 for r in verify_log if r.get("adv_veto_triggered"))
    n_judge_rejected = sum(1 for r in judge_log if not r.get("accepted"))
    judge_reasons = Counter(r.get("reason_code", "unknown") for r in judge_log if not r.get("accepted"))
    print(f"{'=' * 70}")
    print(f"  Dataset complete: {len(dataset.problems)} kept, {len(skipped_problems)} skipped")
    avg_agreement = (
        sum(p.agreement_rate for p in dataset.problems) / len(dataset.problems)
        if dataset.problems else 0
    )
    print(f"  Average agreement rate (kept): {avg_agreement:.1%}")
    print(f"  Judge: {n_judge_rejected} rejected pre-solve", end="")
    if judge_reasons:
        breakdown = ", ".join(f"{k}={v}" for k, v in judge_reasons.most_common())
        print(f" ({breakdown})")
    else:
        print()
    print(f"  Verify: {n_verified} confirmed, {n_rejected} rejected ({n_adv_vetoed} adversarial-vetoed)")
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
    judge_model: str | None = None,
    problems_per_call: int = 2,
    backend: str = "azure",
) -> Dataset:
    if use_tinker:
        tinker_client, tinker_model = s1.get_tinker_client(
            tinker_checkpoint, checkpoint_step=tinker_checkpoint_step,
        )
        gen_client, gen_model = tinker_client, model or tinker_model
        solve_client, solve_model = tinker_client, tinker_model
    else:
        gen_client, gen_default_model = s1.get_client(backend)
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
        judge_model=judge_model or gen_model,
        problems_per_call=problems_per_call,
    )

    s1.save_dataset(dataset, out_path)
    return dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="TTT-Discover+Verify+Judge: generate subproblems with quality judge and reasoning verification"
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
                        help="Use a tinker checkpoint instead of the default API")
    parser.add_argument("--vertex", action="store_true",
                        help="Use Vertex AI backend instead of the default Azure AI Foundry")
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
    parser.add_argument("--judge-model", type=str, default=None,
                        help=(
                            "Model for the problem-quality judge (default: same as --model). "
                            "Screens out multi-part, false-premise, and confusing problems."
                        ))
    parser.add_argument("--quality-threshold", type=int, default=None,
                        help="Inline 0-10 quality judge; only keep scoring >= this (e.g. 9)")
    parser.add_argument("--output", type=str, default=None,
                        help="Override run dir (default: runs/<runs-subdir>/stage1/<timestamp>/)")
    parser.add_argument("--no-dedupe", action="store_true",
                        help="Disable fuzzy dedup (for ablation)")
    parser.add_argument(
        "--max-traces", type=int, default=None,
        help=(
            "Max number of failed-solution traces to include in each generation "
            "prompt (default: 5). Lower values reduce prompt length."
        ),
    )
    parser.add_argument(
        "--problems-per-call", type=int, default=2,
        help="Number of problems to request per generation LLM call (default 2)",
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
    k_top = args.max_traces if args.max_traces is not None else 5
    failed_solutions = s1._load_failed_solutions(failed_path, k_top=k_top)
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

    backend = "vertex" if args.vertex else "azure"

    run(
        problem=problem_text,
        n_problems=args.n_problems,
        n_samples=args.n_samples,
        model=args.model,
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
        judge_model=args.judge_model,
        problems_per_call=args.problems_per_call,
        backend=backend,
    )


if __name__ == "__main__":
    main()
