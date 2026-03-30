#!/usr/bin/env python3
import json
from pathlib import Path

runs_dir = Path(__file__).parent / "runs-1"

total = 0
for run_dir in sorted(runs_dir.iterdir()):
    skips_path = run_dir / "skips.json"
    if not skips_path.is_file():
        continue
    with open(skips_path) as f:
        data = json.load(f)
    n = data.get("n_problems", 0)
    print(f"{run_dir.name}: {n}")
    total += n

print(f"\nTotal n_problems: {total}")
