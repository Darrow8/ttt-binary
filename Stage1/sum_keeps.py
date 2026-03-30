#!/usr/bin/env python3
import json
import os
from pathlib import Path

runs_dir = Path(__file__).parent / "runs"

total = 0
for run_dir in sorted(runs_dir.iterdir()):
    keeps_path = run_dir / "keeps.json"
    if not keeps_path.is_file():
        continue
    with open(keeps_path) as f:
        data = json.load(f)
    n = data.get("n_problems", 0)
    print(f"{run_dir.name}: {n}")
    total += n

print(f"\nTotal n_problems: {total}")
