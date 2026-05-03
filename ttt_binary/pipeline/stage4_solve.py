"""Stage 4 — generate multi-part solution traces and continuous-reward labels.

For each accepted multi-part subproblem, sample K_solve attempts on the
full problem, score each attempt's per-part answers against the per-part
consensus, and write training records with reward = k/m (number of parts
correct / total parts) — the continuous reward signal that lets RL learn
on otherwise-binary verification tasks.

Comparison uses the SAME canonicalization as Stage 3 clustering. Calibration
attempts from Stage 3 are reused, topped up to K_solve total.
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

from ttt_binary.answer_extract import extract_answers_multipart
from ttt_binary.cluster import _canonicalize  # cluster-equivalence, not raw equality
from ttt_binary.llm import call_openai
from ttt_binary.pipeline.stage3_generate_subproblems import (
    SOLVE_SYSTEM as MULTIPART_SOLVE_SYSTEM,
    _render_full_problem,
)


def _matches_consensus(predicted: str | None, consensus_canon: str | None) -> bool:
    if predicted is None or consensus_canon is None:
        return False
    return _canonicalize(predicted) == consensus_canon


def _score_attempt(predicted_parts: dict, parts: list[dict]) -> tuple[int, list[bool]]:
    """Return (k, per_part_correct) where k = # parts correct, per_part_correct
    is a parallel boolean list aligned with *parts*."""
    flags = []
    k = 0
    for part in parts:
        consensus_canon = _canonicalize(str(part.get("consensus_answer"))) \
            if part.get("consensus_answer") else None
        pred = predicted_parts.get(part["label"]) if predicted_parts else None
        ok = _matches_consensus(pred, consensus_canon)
        flags.append(ok)
        if ok:
            k += 1
    return k, flags


def _solve_one(parts: list[dict], *, model: str, temperature: float) -> dict:
    """Solve a multi-part problem once. Returns {ok, predicted_parts, text}."""
    labels = [p["label"] for p in parts]
    rendered = _render_full_problem(parts)
    try:
        text = call_openai(
            rendered,
            model=model,
            system=MULTIPART_SOLVE_SYSTEM,
            temperature=temperature,
        )
    except Exception as e:
        return {
            "ok": False,
            "error": str(e)[:200],
            "predicted_parts": {label: None for label in labels},
            "text": "",
        }
    return {
        "ok": True,
        "text": text,
        "predicted_parts": extract_answers_multipart(text, labels),
    }


def solve_subproblems(
    accepted: list[dict],
    *,
    k_solve: int,
    solver_model: str,
    temperature: float,
    workers: int,
    out_path: Path,
) -> dict:
    """For each accepted multi-part subproblem, harvest Stage-3 calibration
    attempts and top up to K_solve total. Each kept record carries
    `reward = k/m` (number of correctly-answered parts) as the continuous
    training signal, plus per-part match flags for downstream analysis.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_records = 0
    per_problem_yield: list[dict] = []
    with out_path.open("w") as f:
        for sp in accepted:
            parts = sp.get("parts") or []
            m = len(parts)
            if m == 0:
                continue

            # ---- harvest calibration attempts (carry full text) ----
            cal_attempts = sp.get("calibration_attempts", [])
            cal_ok = [a for a in cal_attempts if a.get("ok") and a.get("text")]
            n_cal_ok = len(cal_ok)
            attempts_needed = max(0, k_solve - n_cal_ok)

            # ---- top up with new solver calls if needed ----
            new_attempts: list[dict] = []
            if attempts_needed > 0:
                with cf.ThreadPoolExecutor(max_workers=min(workers, attempts_needed)) as pool:
                    new_attempts = list(pool.map(
                        lambda _i: _solve_one(
                            parts, model=solver_model, temperature=temperature,
                        ),
                        range(attempts_needed),
                    ))

            kept_local = 0
            sum_reward = 0.0
            for src, attempt in (
                [("calibration", c) for c in cal_ok]
                + [("new", c) for c in new_attempts if c.get("ok") and c.get("text")]
            ):
                k, flags = _score_attempt(attempt.get("predicted_parts") or {}, parts)
                reward = k / m
                f.write(json.dumps({
                    "combo_idx": sp.get("combo_idx"),
                    "skills_used": sp.get("skills_used"),
                    "parts": [
                        {
                            "label": p["label"],
                            "skill": p.get("skill"),
                            "text": p["text"],
                            "consensus_answer": p.get("consensus_answer"),
                        }
                        for p in parts
                    ],
                    "predicted_parts": attempt.get("predicted_parts"),
                    "per_part_correct": flags,
                    "k_correct": k,
                    "m": m,
                    "reward": reward,
                    "trace": attempt.get("text"),
                    "source": src,
                }) + "\n")
                kept_local += 1
                sum_reward += reward
                n_records += 1

            mean_reward = sum_reward / kept_local if kept_local else 0.0
            per_problem_yield.append({
                "combo_idx": sp.get("combo_idx"),
                "m": m,
                "k_calibration": len(cal_ok),
                "k_new_attempted": len(new_attempts),
                "k_records_kept": kept_local,
                "mean_reward": mean_reward,
            })
            print(
                f"  [combo {sp.get('combo_idx')}] kept {kept_local} records, "
                f"mean reward = {mean_reward:.2f} (m={m}, "
                f"cal={len(cal_ok)}, new={len(new_attempts)})",
                flush=True,
            )

    n_problems_with_records = sum(
        1 for r in per_problem_yield if r.get("k_records_kept", 0) > 0
    )
    overall_mean_reward = (
        sum(r.get("mean_reward", 0.0) * r.get("k_records_kept", 0) for r in per_problem_yield)
        / n_records
    ) if n_records else 0.0
    summary = {
        "out_path": str(out_path),
        "k_solve": k_solve,
        "solver_model": solver_model,
        "n_subproblems": len(accepted),
        "n_problems_with_records": n_problems_with_records,
        "n_records_total": n_records,
        "overall_mean_reward": overall_mean_reward,
        "per_problem_yield": per_problem_yield,
    }
    print(
        f"wrote {out_path} ({n_records} records across "
        f"{n_problems_with_records}/{len(accepted)} subproblems, "
        f"overall mean reward = {overall_mean_reward:.2f})"
    )
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
