"""
Stage 2 — aggregate all Stage 1 timestamped runs for one hard problem.

Walks runs/<id>/stage1/*/keeps.json, unions the kept candidates, dedupes by
problem text (Stage 1 dedupes within a single run via `seen_problems`, but
across runs duplicates can re-appear), writes runs/<id>/aggregated_keeps.json
atomically.

Skipped problems (skips.json) are left in place per-run for debugging — only
the kept side is unioned here. If you want a unioned skips file, pass
--include-skips and a runs/<id>/aggregated_skips.json will be written too.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pipeline_stages.dedupe import DedupeIndex

REPO_ROOT = Path(__file__).resolve().parent.parent


def _save_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def aggregate_one(problem_id: str, *, include_skips: bool = False, no_dedupe: bool = False) -> dict:
    runs_root = REPO_ROOT / "runs" / problem_id
    stage1_root = runs_root / "stage1"

    if not stage1_root.is_dir():
        raise FileNotFoundError(
            f"No Stage 1 runs at {stage1_root}. Run Stage 1 first for id={problem_id!r}."
        )

    keeps_files = sorted(stage1_root.glob("*/keeps.json"))
    if not keeps_files:
        raise FileNotFoundError(
            f"No keeps.json files under {stage1_root}/*/. "
            f"Stage 1 may not have produced any output for id={problem_id!r}."
        )

    dedupe_enabled = not no_dedupe
    dedupe = DedupeIndex() if dedupe_enabled else None
    seen_fallback: set[str] = set()
    aggregated: list[dict] = []
    per_run_counts: list[dict] = []
    source_problem: str | None = None

    for kp in keeps_files:
        try:
            with open(kp) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  [warn] skipping unreadable {kp}: {e}")
            continue

        if source_problem is None:
            source_problem = data.get("source_problem")

        run_kept = data.get("problems", [])
        added = 0
        dup = 0
        for p in run_kept:
            text = p.get("problem", "")
            if not text:
                continue
            if dedupe_enabled:
                if not dedupe.add(text):
                    dup += 1
                    continue
            else:
                if text in seen_fallback:
                    dup += 1
                    continue
                seen_fallback.add(text)
            aggregated.append(p)
            added += 1

        per_run_counts.append({
            "run": kp.parent.name,
            "kept_in_run": len(run_kept),
            "added_after_dedup": added,
            "dropped_as_duplicate": dup,
        })
        print(f"  {kp.parent.name}: {len(run_kept)} kept, +{added} new, {dup} dup")

    out_path = runs_root / "aggregated_keeps.json"
    summary = {
        "id": problem_id,
        "source_problem": source_problem,
        "n_runs": len(per_run_counts),
        "n_problems": len(aggregated),
        "per_run": per_run_counts,
        "problems": aggregated,
    }
    if dedupe_enabled:
        summary["dedupe"] = {
            "n_kept": dedupe.n_kept,
            "n_dropped_exact": dedupe.n_exact_dropped,
            "n_dropped_fuzzy": dedupe.n_fuzzy_dropped,
        }
        print(
            f"  dedupe: kept {dedupe.n_kept}, "
            f"dropped {dedupe.n_exact_dropped + dedupe.n_fuzzy_dropped} "
            f"(exact={dedupe.n_exact_dropped}, fuzzy={dedupe.n_fuzzy_dropped})"
        )
    _save_atomic(out_path, summary)
    print(f"\nWrote {out_path}  ({len(aggregated)} unique problems from {len(keeps_files)} runs)")

    if include_skips:
        skips_files = sorted(stage1_root.glob("*/skips.json"))
        seen_skip: set[str] = set()
        agg_skips: list[dict] = []
        for sp in skips_files:
            try:
                with open(sp) as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"  [warn] skipping unreadable {sp}: {e}")
                continue
            for p in data.get("problems", []):
                text = p.get("problem", "")
                if not text or text in seen_skip:
                    continue
                seen_skip.add(text)
                agg_skips.append(p)
        skips_out = runs_root / "aggregated_skips.json"
        _save_atomic(skips_out, {
            "id": problem_id,
            "n_problems": len(agg_skips),
            "problems": agg_skips,
        })
        print(f"Wrote {skips_out}  ({len(agg_skips)} unique skipped)")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage 2: aggregate Stage 1 keeps")
    parser.add_argument("--id", type=str, required=True)
    parser.add_argument("--include-skips", action="store_true",
                        help="Also union skips.json across runs into aggregated_skips.json")
    parser.add_argument("--no-dedupe", action="store_true",
                        help="Disable near-duplicate dedup (use exact-string only, for ablation)")
    args = parser.parse_args()
    aggregate_one(args.id, include_skips=args.include_skips, no_dedupe=args.no_dedupe)


if __name__ == "__main__":
    main()
