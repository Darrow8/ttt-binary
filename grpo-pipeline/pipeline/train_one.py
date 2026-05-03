"""
Stage 6 — GRPO train one adapter (per-problem or shared).

Refactor of grpo-pipeline/grpo_subproblems.py and grpo_subproblems_resume.py.
All hardcoded id/path/wandb-name values become CLI arguments.

Usage::

    # Per-problem adapter
    python -m pipeline.train_one --id conics-tangent-5 --epochs 50

    # Shared adapter (reads runs/shared/subproblems.jsonl from Stage 5)
    python -m pipeline.train_one --id shared --epochs 50

    # Resume from a checkpoint
    python -m pipeline.train_one --id conics-tangent-5 --epochs 5 \\
        --resume-from "tinker://...weights/conics-tangent-5.ckpt-000040"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
load_dotenv()

from pipeline import (
    GRPOConfig,
    GRPOTrainer,
    boxed_format_bonus,
    boxed_match,
    combined,
    load_problems,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)


SYSTEM_PROMPT = """
You are a careful and rigorous math student working through an advanced mathematics problem. Your goal is to solve the problem step by step.

Show all important intermediate reasoning, derivations, and calculations. Explain why each step is valid and reference any relevant theorems or identities when appropriate. Avoid skipping logical steps or making large jumps in reasoning.

If the problem involves multiple cases or approaches, consider them systematically. Use clear mathematical notation and keep the solution organized.

After completing the reasoning, clearly state the final answer.
"""


def build_config(
    problem_id: str,
    *,
    log_dir: Path,
    resume_from: str | None,
    batch_size: int,
    group_size: int,
    learning_rate: float,
    lora_rank: int,
    max_tokens: int,
    save_every: int,
    wandb_run_name: str | None,
) -> GRPOConfig:
    return GRPOConfig(
        model_name="openai/gpt-oss-120b",
        log_dir=str(log_dir),
        resume_from=resume_from,
        batch_size=batch_size,
        group_size=group_size,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        max_tokens=max_tokens,
        save_every=save_every,
        wandb_project=f"grpo-{problem_id}",
        wandb_run_name=wandb_run_name,
        temperature=0.7,
        system_prompt=SYSTEM_PROMPT,
        prompt_suffix=" Put your final answer inside \\boxed{}.",
        few_shot=[],
    )


def train_problem(
    problem_id: str,
    *,
    epochs: int,
    resume_from: str | None = None,
    batch_size: int = 25,
    group_size: int = 16,
    learning_rate: float = 1e-4,
    lora_rank: int = 32,
    max_tokens: int = 100000,
    save_every: int = 5,
    wandb_run_name: str | None = None,
    subproblems_path: Path | None = None,
    log_dir: Path | None = None,
) -> None:
    """Train one GRPO adapter for one hard-problem id (or 'shared')."""
    runs_root = _REPO_ROOT / "runs" / problem_id
    if subproblems_path is None:
        subproblems_path = runs_root / "subproblems.jsonl"
    if log_dir is None:
        log_dir = runs_root / "grpo"

    if not subproblems_path.exists():
        raise FileNotFoundError(
            f"Missing {subproblems_path}. Run Stage 5 (export) first for id={problem_id!r}."
        )

    problems = load_problems(str(subproblems_path))
    print(f"\n{'='*70}")
    print(f"  Stage 6: GRPO train id={problem_id}")
    print(f"  Subproblems: {subproblems_path}  ({len(problems)} problems)")
    print(f"  Log dir:     {log_dir}")
    print(f"  Epochs:      {epochs}")
    if resume_from:
        print(f"  Resume from: {resume_from}")
    print(f"{'='*70}\n")

    cfg = build_config(
        problem_id,
        log_dir=log_dir,
        resume_from=resume_from,
        batch_size=batch_size,
        group_size=group_size,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        max_tokens=max_tokens,
        save_every=save_every,
        wandb_run_name=wandb_run_name,
    )

    reward_fn = combined(
        (1.0, boxed_match),
        (0.1, boxed_format_bonus),
    )

    trainer = GRPOTrainer(cfg, reward_fn)
    trainer.train(problems, epochs=epochs)


def main():
    parser = argparse.ArgumentParser(description="Stage 6: GRPO train one adapter")
    parser.add_argument("--id", type=str, required=True,
                        help="Hard-problem id, or 'shared' for the shared adapter")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Tinker checkpoint path to resume training from")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=30000,
                        help="Max generation tokens per rollout. Must satisfy "
                             "prompt_tokens + max_tokens <= model context. For "
                             "gpt-oss-120b (32768 ctx), 30000 leaves ~2.7k for "
                             "prompts and is safe for our problem set.")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--subproblems", type=Path, default=None,
                        help="Override path to subproblems.jsonl")
    parser.add_argument("--log-dir", type=Path, default=None,
                        help="Override grpo log dir (defaults to runs/<id>/grpo)")
    args = parser.parse_args()

    train_problem(
        problem_id=args.id,
        epochs=args.epochs,
        resume_from=args.resume_from,
        batch_size=args.batch_size,
        group_size=args.group_size,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        max_tokens=args.max_tokens,
        save_every=args.save_every,
        wandb_run_name=args.wandb_run_name,
        subproblems_path=args.subproblems,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
