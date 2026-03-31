"""
Resume GRPO training on subproblems from checkpoint at epoch 20.

The original run (pipeline.grpo_subproblems) crashed at epoch 20/25.
This script resumes from the saved checkpoint to complete the
remaining 5 epochs (global_step offset = 40).

Usage::

    python -m pipeline.grpo_subproblems_resume
"""

import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
load_dotenv()

from pipeline import (
    GRPOConfig,
    GRPOTrainer,
    load_problems,
    boxed_match,
    boxed_format_bonus,
    combined,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)


# ── Configuration ──────────────────────────────────────────────────────────

config = GRPOConfig(
    model_name="openai/gpt-oss-120b",
    log_dir=str(_REPO_ROOT / "subproblems-run-resumed"),

    resume_from="tinker://d7bcdbad-55ee-5d6d-bd99-a92db456ff1b:train:0/weights/subproblems-run.ckpt-000040",

    batch_size=25,
    group_size=16,
    learning_rate=1e-4,
    lora_rank=32,
    max_tokens=16384,

    save_every=5,

    wandb_project="grpo-conics-subproblems3",
    wandb_run_name="subproblems-resumed-ep20",

    temperature=0.7,
    system_prompt="""
You are a careful and rigorous math student working through an advanced mathematics problem. Your goal is to solve the problem step by step.

Show all important intermediate reasoning, derivations, and calculations. Explain why each step is valid and reference any relevant theorems or identities when appropriate. Avoid skipping logical steps or making large jumps in reasoning.

If the problem involves multiple cases or approaches, consider them systematically. Use clear mathematical notation and keep the solution organized.

After completing the reasoning, clearly state the final answer.
    """,

    prompt_suffix=" Put your final answer inside \\boxed{}.",

    few_shot=[],
)

EPOCHS = 5


# ── Problems ───────────────────────────────────────────────────────────────

problems = load_problems(str(_REPO_ROOT / "problems" / "conics-50.jsonl"))


# ── Reward function ────────────────────────────────────────────────────────

reward_fn = combined(
    (1.0, boxed_match),
    (0.1, boxed_format_bonus),
)


# ── Train ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trainer = GRPOTrainer(config, reward_fn)
    trainer.train(problems, epochs=EPOCHS)
