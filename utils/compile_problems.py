"""
Compile problems from every Stage1/runs/**/keeps.json into one JSON file.
Each record: prompt, reference, id (1..N). Default: unique by prompt text;
use --no-dedupe to keep every row from every run.
"""

import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def compile_problems(
    input_dir: Path,
    output_path: Path,
    *,
    dedupe: bool = True,
):
    input_dir = input_dir.resolve()
    keeps_files = sorted(input_dir.rglob("keeps.json"))
    if not keeps_files:
        print(f"No keeps.json files found under {input_dir}")
        return

    seen: set[str] = set()
    problems = []

    for keeps_file in keeps_files:
        with open(keeps_file) as f:
            data = json.load(f)

        for entry in data.get("problems", []):
            problem_text = entry.get("problem", "").strip()
            if not problem_text:
                continue
            if dedupe and problem_text in seen:
                continue
            if dedupe:
                seen.add(problem_text)
            problems.append({
                "prompt": problem_text,
                "reference": entry.get("ground_truth_answer", ""),
                "id": len(problems) + 1,
            })

    with open(output_path, "w") as f:
        json.dump(problems, f, indent=2)

    label = "unique" if dedupe else "total"
    print(
        f"Compiled {len(problems)} {label} problems from {len(keeps_files)} keeps files -> {output_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile problems from Stage1 keeps.json files")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "Stage1" / "runs",
        help="Root directory to search recursively for keeps.json (default: <repo>/Stage1/runs)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "Stage1" / "problems.json",
        help="Output path for compiled problems",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Include every problem from every keeps.json (same prompt may appear multiple times)",
    )
    args = parser.parse_args()
    compile_problems(args.input_dir, args.output, dedupe=not args.no_dedupe)
