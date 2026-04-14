"""
Stage 5 — export selected subproblems to GRPO-ready JSONL.

Reads:  runs/<id>/selected.json           (Stage 4 output)
        runs/<id>/filtered_keeps.json     (for ground_truth_answer fallback)
Writes: runs/<id>/subproblems.jsonl       per-problem GRPO input

For shared-adapter mode: --shared --ids id1,id2,... unions the per-id sets and
writes runs/shared/subproblems.jsonl with each row tagged by source_id.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _save_atomic_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp, path)


def _load_selected_problems(problem_id: str) -> list[dict]:
    runs_root = REPO_ROOT / "runs" / problem_id
    selected_path = runs_root / "selected.json"
    filtered_path = runs_root / "filtered_keeps.json"

    if not selected_path.exists():
        raise FileNotFoundError(
            f"Missing {selected_path}. Run Stage 4 (select) first for id={problem_id!r}."
        )
    with open(selected_path) as f:
        sel = json.load(f)

    # Build id -> ground_truth lookup from filtered_keeps in case selected.json
    # entries don't all carry the answer.
    gt_by_id: dict[int, str] = {}
    problem_text_by_id: dict[int, str] = {}
    if filtered_path.exists():
        with open(filtered_path) as f:
            filt = json.load(f)
        for i, p in enumerate(filt.get("problems", []), start=1):
            gt_by_id[i] = p.get("ground_truth_answer", "")
            problem_text_by_id[i] = p.get("problem", "")

    rows: list[dict] = []
    for entry in sel.get("selected", []):
        pid = entry.get("id")
        prompt = entry.get("problem") or problem_text_by_id.get(pid, "")
        reference = entry.get("ground_truth_answer") or gt_by_id.get(pid, "")
        if not prompt or not reference:
            print(f"  [warn] {problem_id}: skipping selected id={pid} "
                  f"(prompt_len={len(prompt)}, ref={reference!r})")
            continue
        rows.append({
            "prompt": prompt,
            "reference": str(reference),
            "source_id": problem_id,
            "selected_id": pid,
        })
    return rows


def export_one(problem_id: str) -> Path:
    rows = _load_selected_problems(problem_id)
    out_path = REPO_ROOT / "runs" / problem_id / "subproblems.jsonl"
    _save_atomic_jsonl(out_path, rows)
    print(f"Wrote {out_path}  ({len(rows)} rows)")
    return out_path


def export_shared(ids: list[str]) -> Path:
    all_rows: list[dict] = []
    per_id_counts: dict[str, int] = {}
    for pid in ids:
        rows = _load_selected_problems(pid)
        per_id_counts[pid] = len(rows)
        all_rows.extend(rows)
    out_path = REPO_ROOT / "runs" / "shared" / "subproblems.jsonl"
    _save_atomic_jsonl(out_path, all_rows)
    print(f"Wrote {out_path}  ({len(all_rows)} rows from {len(ids)} ids)")
    for pid, n in per_id_counts.items():
        print(f"  {pid}: {n}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Stage 5: export to GRPO JSONL")
    parser.add_argument("--id", type=str, default=None,
                        help="Single hard-problem id to export")
    parser.add_argument("--shared", action="store_true",
                        help="Export shared-adapter union to runs/shared/subproblems.jsonl")
    parser.add_argument("--ids", type=str, default=None,
                        help="Comma-separated ids for --shared (e.g. id1,id2,id3)")
    args = parser.parse_args()

    if args.shared:
        if not args.ids:
            raise SystemExit("--shared requires --ids id1,id2,...")
        ids = [s.strip() for s in args.ids.split(",") if s.strip()]
        export_shared(ids)
    elif args.id:
        export_one(args.id)
    else:
        raise SystemExit("Pass --id <id> for per-problem export or --shared --ids ...")


if __name__ == "__main__":
    main()
