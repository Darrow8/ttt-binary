"""End-to-end orchestrator for the skill-chained pipeline (stages 1, 3, 4).

Reads a YAML config (or accepts overrides on the CLI), runs each stage in
sequence, and emits one artifact per stage under ttt_binary/data/.

Usage:
    python -m ttt_binary.pipeline.run_pipeline \\
        --problem-id conics-tangent-5 \\
        --problem-file data/target-problems/conics.txt \\
        --config ttt_binary/configs/v1.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
load_dotenv()

from ttt_binary.pipeline.stage1_generate_skills import generate_skills
from ttt_binary.pipeline.stage3_generate_subproblems import generate_subproblems
from ttt_binary.pipeline.stage4_solve import solve_subproblems


DEFAULT_CONFIG = {
    "n_skills": 10,
    "m": 3,
    "band_lo": 0.4,
    "band_hi": 0.6,
    "ambiguity_threshold": 0.2,   # per-part max allowed second-cluster fraction
    "max_unparseable": 5,         # per-part cap; 50% allows for multi-part cascade failures
    "k_calibrate": 20,
    "k_solve": 16,
    "max_regen": 10,
    "generator_model": "openai/gpt-oss-120b-maas",
    "critic_model": "openai/gpt-oss-120b-maas",
    "judge_model": "openai/gpt-oss-120b-maas",   # step 2 shortcut-detection judge
    "solver_model": "openai/gpt-oss-120b-maas",
    "temperature": 0.7,
    "workers": 8,
    "max_combos": 0,             # 0 = process all C(X,M); >0 = smoke-test cap
}


def _load_yaml(path: str | None) -> dict:
    if not path:
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        # Tiny YAML subset parser: only flat key: value lines and comments.
        cfg: dict = {}
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            v = v.strip()
            if v.startswith("[") or v.startswith("{"):
                try:
                    cfg[k.strip()] = json.loads(v)
                except Exception:
                    cfg[k.strip()] = v
                continue
            try:
                cfg[k.strip()] = json.loads(v)  # int/float/bool/null/string
            except Exception:
                cfg[k.strip()] = v.strip("'\"")
        return cfg
    return yaml.safe_load(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem-id", required=True)
    ap.add_argument("--problem-file", required=True)
    ap.add_argument("--config", default="ttt_binary/configs/v1.yaml")
    ap.add_argument("--data-root", default="ttt_binary/data")
    ap.add_argument("--skip-stage1", action="store_true")
    ap.add_argument("--skip-stage3", action="store_true")
    ap.add_argument("--skip-stage4", action="store_true")
    # Knob overrides — any unset value falls back to the config file then defaults.
    for k in DEFAULT_CONFIG:
        ap.add_argument(f"--{k.replace('_', '-')}", default=None)
    args = ap.parse_args()

    cfg = {**DEFAULT_CONFIG, **_load_yaml(args.config)}
    for k in DEFAULT_CONFIG:
        v = getattr(args, k)
        if v is not None:
            # Cast to original type if possible.
            orig = DEFAULT_CONFIG[k]
            if isinstance(orig, bool):
                cfg[k] = str(v).lower() in {"1", "true", "yes"}
            elif isinstance(orig, int):
                cfg[k] = int(v)
            elif isinstance(orig, float):
                cfg[k] = float(v)
            else:
                cfg[k] = v
    print("config:", json.dumps(cfg, indent=2))

    data_root = Path(args.data_root)
    skills_path = data_root / "skills" / f"{args.problem_id}.json"
    sub_path = data_root / "subproblems" / f"{args.problem_id}.json"
    sol_path = data_root / "solutions" / f"{args.problem_id}.jsonl"
    keeps_path = data_root / "subproblems" / f"{args.problem_id}.keeps.jsonl"
    skips_path = data_root / "subproblems" / f"{args.problem_id}.skips.jsonl"

    problem = Path(args.problem_file).read_text().strip()

    # ---- Stage 1 ---------------------------------------------------------
    if args.skip_stage1 and skills_path.exists():
        print(f"\n[stage1] skipped, reusing {skills_path}")
    else:
        print("\n[stage1] generating skills")
        skills = generate_skills(
            problem,
            n_skills=cfg["n_skills"],
            model=cfg["generator_model"],
            temperature=cfg["temperature"],
        )
        skills_path.parent.mkdir(parents=True, exist_ok=True)
        skills_path.write_text(json.dumps({
            "problem_id": args.problem_id,
            "n_skills": len(skills),
            "model": cfg["generator_model"],
            "skills": skills,
        }, indent=2))
        print(f"  wrote {skills_path}")

    skills_obj = json.loads(skills_path.read_text())
    skills = skills_obj["skills"]

    # ---- Stage 3 ---------------------------------------------------------
    if args.skip_stage3 and sub_path.exists():
        print(f"\n[stage3] skipped, reusing {sub_path}")
    else:
        print("\n[stage3] generating subproblems with consensus difficulty filter")
        generate_subproblems(
            problem_id=args.problem_id,
            skills=skills,
            m=cfg["m"],
            band=(cfg["band_lo"], cfg["band_hi"]),
            ambiguity_threshold=cfg["ambiguity_threshold"],
            k_calibrate=cfg["k_calibrate"],
            max_regen=cfg["max_regen"],
            generator_model=cfg["generator_model"],
            critic_model=cfg["critic_model"],
            judge_model=cfg.get("judge_model") or cfg["critic_model"],
            temperature=cfg["temperature"],
            workers=cfg["workers"],
            max_unparseable=cfg["max_unparseable"],
            max_combos=(cfg["max_combos"] or None),
            out_path=sub_path,
            keeps_path=keeps_path,
            skips_path=skips_path,
        )

    sub_obj = json.loads(sub_path.read_text())
    accepted = sub_obj["subproblems"]
    print(f"\n[stage3] {len(accepted)} accepted subproblems")

    # ---- Stage 4 ---------------------------------------------------------
    if args.skip_stage4 and sol_path.exists():
        print(f"\n[stage4] skipped, reusing {sol_path}")
    else:
        print("\n[stage4] solving subproblems for training data")
        summary = solve_subproblems(
            accepted,
            k_solve=cfg["k_solve"],
            solver_model=cfg["solver_model"],
            temperature=cfg["temperature"],
            workers=cfg["workers"],
            out_path=sol_path,
        )
        sol_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))

    print("\nPipeline done.")
    print(f"  skills:     {skills_path}")
    print(f"  subproblems:{sub_path}")
    print(f"  solutions:  {sol_path}")


if __name__ == "__main__":
    main()
