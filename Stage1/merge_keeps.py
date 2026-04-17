#!/usr/bin/env python3
import json
import sys
from pathlib import Path

runs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "runs"
out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else runs_dir / "merged_keeps.json"

all_problems = []
source_problem = None

for run_dir in sorted(runs_dir.iterdir()):
    keeps_path = run_dir / "keeps.json"
    if not keeps_path.is_file():
        continue
    with open(keeps_path) as f:
        data = json.load(f)
    problems = data.get("problems", [])
    all_problems.extend(problems)
    if source_problem is None:
        source_problem = data.get("source_problem")
    print(f"{run_dir.name}: {len(problems)} problems")

seen = set()
deduped = []
for p in all_problems:
    key = p["problem"]
    if key not in seen:
        seen.add(key)
        deduped.append(p)

merged = {
    "source_problem": source_problem,
    "n_problems": len(deduped),
    "problems": deduped,
}

with open(out_path, "w") as f:
    json.dump(merged, f, indent=2)

print(f"\nTotal: {len(deduped)} unique problems -> {out_path}")
