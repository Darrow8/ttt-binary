"""Stage-1 stability diagnostic.

Run the current skill-generation prompt N times on the same problem and
report how stable the output is:

  - Skill-name overlap across runs (Jaccard, name multiplicity counts).
  - Token-level Jaccard on descriptions (loose semantic proxy).
  - Side-by-side skill-name table.

Usage:
    python scripts/stage1_stability.py \\
        --problem-file data/target-problems/conics.txt \\
        --n-runs 5 \\
        --workers 5 \\
        --out-dir ttt_binary/data/stability/conics
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv()

from ttt_binary.pipeline.stage1_generate_skills import generate_skills


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "is", "are",
    "be", "by", "with", "from", "this", "that", "these", "those", "it", "its",
    "as", "at", "if", "then", "else", "when", "while", "such", "each", "any",
    "into", "out", "we", "you", "your", "our", "their",
}


def _tokenize(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def run_one(problem: str, *, n_skills: int, model: str, run_idx: int) -> list[dict]:
    print(f"  [run {run_idx}] starting", flush=True)
    skills = generate_skills(problem, n_skills=n_skills, model=model, temperature=0.7)
    print(f"  [run {run_idx}] done ({len(skills)} skills)", flush=True)
    return skills


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem-file", required=True)
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--n-skills", type=int, default=10)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--model", default="openai/gpt-oss-120b-maas")
    ap.add_argument("--out-dir", default="ttt_binary/data/stability")
    args = ap.parse_args()

    problem = Path(args.problem_file).read_text().strip()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Run N independent skill generations in parallel ----
    print(f"running stage 1 {args.n_runs} times in parallel "
          f"(workers={args.workers})", flush=True)
    runs: list[list[dict]] = [None] * args.n_runs
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(run_one, problem, n_skills=args.n_skills,
                        model=args.model, run_idx=i): i
            for i in range(args.n_runs)
        }
        for fut in cf.as_completed(futs):
            i = futs[fut]
            try:
                runs[i] = fut.result()
            except Exception as e:
                print(f"  [run {i}] FAILED: {type(e).__name__}: {e}", flush=True)
                runs[i] = []

    # ---- Persist raw skill lists ----
    for i, skills in enumerate(runs):
        (out_dir / f"run_{i}.json").write_text(json.dumps({
            "run_idx": i, "n_skills": len(skills), "skills": skills,
        }, indent=2))

    # ---- Side-by-side skill-name table ----
    print("\n=== skill names per run ===")
    for i, skills in enumerate(runs):
        names = [s.get("name", "?") for s in skills]
        print(f"\nrun {i} ({len(names)} skills):")
        for n in names:
            print(f"  - {n}")

    # ---- Name multiplicity across runs ----
    from collections import Counter
    name_counts: Counter[str] = Counter()
    for skills in runs:
        for s in skills:
            name_counts[s["name"]] += 1
    print("\n=== name appearance counts (over all runs) ===")
    repeated = sorted(
        ((n, c) for n, c in name_counts.items() if c >= 2),
        key=lambda x: (-x[1], x[0]),
    )
    if repeated:
        for n, c in repeated:
            print(f"  {c}x  {n}")
    else:
        print("  (no skill name appeared in ≥2 runs)")
    n_unique = sum(1 for c in name_counts.values() if c == 1)
    n_total = sum(name_counts.values())
    print(f"\n  unique names (1x):     {n_unique} / {n_total}")
    print(f"  repeated names (≥2x):  {len(repeated)}")
    print(f"  most-shared count:     {max(name_counts.values()) if name_counts else 0}"
          f" / {args.n_runs} possible")

    # ---- Pairwise Jaccard on names + descriptions ----
    print("\n=== pairwise Jaccard ===")
    print(f"{'pair':<10} {'names':>8} {'desc-tokens':>14}")
    name_jaccs, desc_jaccs = [], []
    for i, j in combinations(range(args.n_runs), 2):
        names_i = {s["name"] for s in runs[i]}
        names_j = {s["name"] for s in runs[j]}
        # Token sets over all descriptions concatenated.
        desc_i = set().union(*(_tokenize(s.get("description", "")) for s in runs[i])) \
            if runs[i] else set()
        desc_j = set().union(*(_tokenize(s.get("description", "")) for s in runs[j])) \
            if runs[j] else set()
        nj = jaccard(names_i, names_j)
        dj = jaccard(desc_i, desc_j)
        name_jaccs.append(nj)
        desc_jaccs.append(dj)
        print(f"  {i}<->{j}     {nj:>8.2f} {dj:>14.2f}")
    if name_jaccs:
        print(f"\n  mean name Jaccard:        {sum(name_jaccs)/len(name_jaccs):.2f}")
        print(f"  mean desc-token Jaccard:  {sum(desc_jaccs)/len(desc_jaccs):.2f}")

    # ---- Verdict heuristic ----
    print("\n=== heuristic verdict ===")
    if name_jaccs:
        m = sum(name_jaccs) / len(name_jaccs)
        if m >= 0.6:
            print(f"  Stable name set (mean Jaccard {m:.2f} ≥ 0.6) — prompt is "
                  "already producing reproducible structure; tweaks will mostly "
                  "shape phrasing, not coverage.")
        elif m >= 0.3:
            print(f"  Moderately stable (mean name Jaccard {m:.2f}). Some core "
                  "concepts recur but several skills drift. The prompt may need "
                  "more concrete coverage requirements.")
        else:
            print(f"  Highly unstable (mean name Jaccard {m:.2f} < 0.3). The "
                  "skill set you happened to get for any single run is mostly "
                  "luck. Consider tightening the prompt with explicit required "
                  "categories or running multiple seeds and merging.")

    summary = {
        "n_runs": args.n_runs,
        "n_skills_per_run": args.n_skills,
        "name_jaccard_mean": (sum(name_jaccs)/len(name_jaccs)) if name_jaccs else None,
        "desc_token_jaccard_mean": (sum(desc_jaccs)/len(desc_jaccs)) if desc_jaccs else None,
        "name_counts": dict(name_counts),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nartifacts -> {out_dir}/run_*.json + summary.json")


if __name__ == "__main__":
    main()
