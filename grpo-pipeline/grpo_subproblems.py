"""
GRPO training on subproblems derived from a hard source problem.

Workflow:
    1. keeps.json contains a hard problem the model can't solve (0/100)
       and 21 easier subproblems with majority-vote ground truth.
    2. This script runs GRPO on those 21 subproblems for several epochs
       so the model learns the component reasoning skills.
    3. After training, evaluate the hard problem again to see if the
       model can now produce correct answers (target: >0/100).

Usage::

    python -m pipeline.grpo_subproblems
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
# 21 subproblems → batch_size=21 means 1 step per epoch.
# Run enough epochs for the model to learn all subproblems.

config = GRPOConfig(
    model_name="openai/gpt-oss-120b",
    log_dir=str(_REPO_ROOT / "subproblems-run"),

    batch_size=25,
    group_size=16,
    learning_rate=1e-4,
    lora_rank=32,
    max_tokens=100000,

    save_every=5,

    wandb_project="grpo-conics-subproblems",

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

EPOCHS = 50


# ── Problems ───────────────────────────────────────────────────────────────

problems = load_problems(str(_REPO_ROOT / "problems" / "subproblems.jsonl"))


# ── Reward function ────────────────────────────────────────────────────────

reward_fn = combined(
    (1.0, boxed_match),
    (0.1, boxed_format_bonus),
)


# ── Train ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trainer = GRPOTrainer(config, reward_fn)
    trainer.train(problems, epochs=EPOCHS)
