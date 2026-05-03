"""Stage 4 — generate solution traces for accepted subproblems.

For each accepted subproblem, sample K_solve attempts from a strong solver,
keep the ones that match the consensus answer (REVISIONS.md), and write
JSONL training data.

Comparison uses the SAME canonicalization as Stage 3 clustering, so a
predicted answer counts as correct iff it lies in the same cluster as the
consensus.

Optimization: re-uses the calibration attempts from Stage 3 if available, so
we only top up to K_solve total attempts.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ttt_binary.answer_extract import extract_boxed
from ttt_binary.cluster import _canonicalize  # cluster-equivalence, not raw equality
from ttt_binary.llm import call_openai


def _matches_consensus(predicted: str | None, consensus_canon: str | None) -> bool:
    if predicted is None or consensus_canon is None:
        return False
    return _canonicalize(predicted) == consensus_canon


SOLVE_SYSTEM = """\
You are a careful and rigorous math student. Solve the problem step by step, \
showing all important intermediate work. Compute a numerical final answer \
and round it to exactly 4 decimal places. Place ONLY that rounded number \
inside \\boxed{} at the end (e.g. \\boxed{866.0000}, \\boxed{0.5000}, \
\\boxed{-2.3457}). Always include trailing zeros to fill all 4 decimals."""


def _solve_one(prompt: str, *, model: str, temperature: float) -> dict:
    try:
        text = call_openai(
            prompt
            + "\n\nRound your final answer to 4 decimal places and place it"
              " inside \\boxed{} (e.g. \\boxed{866.0000}).",
            model=model,
            system=SOLVE_SYSTEM,
            temperature=temperature,
        )
    except Exception as e:
        return {"ok": False, "error": str(e)[:200], "predicted": None, "text": ""}
    return {"ok": True, "text": text, "predicted": extract_boxed(text)}


def solve_subproblems(
    accepted: list[dict],
    *,
    k_solve: int,
    solver_model: str,
    temperature: float,
    workers: int,
    out_path: Path,
) -> dict:
    """For each accepted subproblem (REVISIONS.md schema), harvest correct
    traces from Stage 3 calibration and top up with new solver calls until
    K_solve total attempts. Verification target is consensus_answer."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_traces = 0
    n_problems_with_any = 0
    per_problem_yield: list[dict] = []
    with out_path.open("w") as f:
        for sp in accepted:
            consensus = sp.get("consensus_answer")
            if consensus is None:
                # Backwards compatibility for old records that still carry
                # expected_answer instead of consensus_answer.
                consensus = sp.get("expected_answer")
            consensus_canon = _canonicalize(str(consensus)) if consensus else None
            problem_text = sp["problem_text"]

            # ---- harvest correct calibration traces (carry full text) ----
            cal_attempts = sp.get("calibration_attempts", [])
            cal_correct = [
                a for a in cal_attempts
                if a.get("ok")
                and a.get("text")
                and _matches_consensus(a.get("predicted"), consensus_canon)
            ]
            n_cal_ok = sum(1 for a in cal_attempts if a.get("ok"))
            attempts_needed = max(0, k_solve - n_cal_ok)

            # ---- top up with new solver calls if needed ----
            new_attempts: list[dict] = []
            if attempts_needed > 0:
                with cf.ThreadPoolExecutor(max_workers=min(workers, attempts_needed)) as pool:
                    new_attempts = list(pool.map(
                        lambda _i: _solve_one(
                            problem_text, model=solver_model, temperature=temperature,
                        ),
                        range(attempts_needed),
                    ))
            new_correct = [
                a for a in new_attempts
                if a.get("ok") and _matches_consensus(a.get("predicted"), consensus_canon)
            ]

            kept_local = 0
            for src, a in (
                [("calibration", c) for c in cal_correct]
                + [("new", c) for c in new_correct]
            ):
                f.write(json.dumps({
                    "combo_idx": sp.get("combo_idx"),
                    "skills_used": sp.get("skills_used"),
                    "problem_text": problem_text,
                    "consensus_answer": consensus,
                    "generator_proposed_answer": sp.get("generator_proposed_answer"),
                    "predicted_answer": a.get("predicted"),
                    "trace": a.get("text"),
                    "source": src,
                }) + "\n")
                kept_local += 1
                n_traces += 1

            if kept_local > 0:
                n_problems_with_any += 1
            per_problem_yield.append({
                "combo_idx": sp.get("combo_idx"),
                "k_calibration_correct": len(cal_correct),
                "k_new_attempted": len(new_attempts),
                "k_new_correct": len(new_correct),
                "k_total_correct": kept_local,
            })
            print(
                f"  [combo {sp.get('combo_idx')}] kept {kept_local} traces "
                f"(cal={len(cal_correct)}, new={len(new_correct)}/"
                f"{len(new_attempts)}, consensus={consensus!r})",
                flush=True,
            )

    summary = {
        "out_path": str(out_path),
        "k_solve": k_solve,
        "solver_model": solver_model,
        "n_subproblems": len(accepted),
        "n_problems_with_any_correct": n_problems_with_any,
        "n_correct_traces": n_traces,
        "per_problem_yield": per_problem_yield,
    }
    print(f"wrote {out_path} ({n_traces} correct traces across "
          f"{n_problems_with_any}/{len(accepted)} subproblems)")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem-id", required=True)
    ap.add_argument("--subproblems-file", required=True)
    ap.add_argument("--k-solve", type=int, default=16)
    ap.add_argument("--solver-model", default="openai/gpt-oss-120b-maas")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default="ttt_binary/data/solutions")
    args = ap.parse_args()

    obj = json.loads(Path(args.subproblems_file).read_text())
    accepted = obj["subproblems"]
    out_path = Path(args.out_dir) / f"{args.problem_id}.jsonl"
    summary = solve_subproblems(
        accepted,
        k_solve=args.k_solve,
        solver_model=args.solver_model,
        temperature=args.temperature,
        workers=args.workers,
        out_path=out_path,
    )
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"summary → {summary_path}")


if __name__ == "__main__":
    main()
