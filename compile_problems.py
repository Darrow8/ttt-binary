"""
Compile unique problems from Stage1/runs/*/keeps.json
into a single problems.json file.
"""

import json
import argparse
from pathlib import Path


def compile_problems(input_dir: Path, output_path: Path):
    keeps_files = sorted(input_dir.glob("*/keeps.json"))
    if not keeps_files:
        print(f"No keeps.json files found in {input_dir}")
        return

    seen = set()
    problems = []

    for keeps_file in keeps_files:
        with open(keeps_file) as f:
            data = json.load(f)

        for entry in data.get("problems", []):
            problem_text = entry.get("problem", "").strip()
            if not problem_text or problem_text in seen:
                continue

            seen.add(problem_text)
            problems.append({
                "prompt": problem_text,
                "reference": entry.get("ground_truth_answer", ""),
                "agreement_rate": entry.get("agreement_rate", None),
                "source_run": keeps_file.parent.name,
            })

    with open(output_path, "w") as f:
        json.dump(problems, f, indent=2)

    print(f"Compiled {len(problems)} unique problems from {len(keeps_files)} keeps files -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile problems from Stage1 keeps.json files")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("Stage1/runs"),
        help="Directory containing timestamped run folders with keeps.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Stage1/problems-2.json"),
        help="Output path for compiled problems",
    )
    args = parser.parse_args()
    compile_problems(args.input_dir, args.output)
