#!/usr/bin/env python3
"""Count how many problems have each score 1..10 in problems_ranked.json."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default=Path(__file__).resolve().parent / "problems_2_filters.json",
        type=Path,
        help="Path to problems_2_filters.json (default: alongside this script)",
    )
    args = parser.parse_args()

    data = json.loads(args.path.read_text(encoding="utf-8"))
    problems = data["problems"]
    counts = Counter(p["score"] for p in problems)

    print(f"File: {args.path}")
    print(f"Total problems: {len(problems)}")
    print()
    for s in range(1, 11):
        print(f"  score {s:2d}: {counts.get(s, 0)}")

    other = sum(c for k, c in counts.items() if k not in range(1, 11))
    if other:
        print()
        print(f"  (scores outside 1–10: {other} total)")
        other_keys = [k for k in counts if k not in range(1, 11)]
        for k in sorted(other_keys, key=lambda x: (str(type(x).__name__), str(x))):
            print(f"    score {k!r}: {counts[k]}")


if __name__ == "__main__":
    main()
