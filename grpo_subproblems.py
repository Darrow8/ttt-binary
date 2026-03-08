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

    python grpo_subproblems.py
"""

import logging

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
    log_dir="/tmp/tinker-grpo/subproblems-run",

    batch_size=21,
    group_size=16,
    learning_rate=4e-5,
    lora_rank=32,
    max_tokens=16384,

    save_every=5,

    wandb_project="grpo-tinker-subproblems",

    temperature=0.7,
    system_prompt="You are a helpful assistant. Show your reasoning step by step.",

    prompt_suffix=" Put your final answer inside \\boxed{}.",

    few_shot=[],
)

EPOCHS = 10


# ── Problems ───────────────────────────────────────────────────────────────

problems = load_problems("problems/subproblems.jsonl")


# ── Reward function ────────────────────────────────────────────────────────

reward_fn = combined(
    (1.0, boxed_match),
    (0.1, boxed_format_bonus),
)


# ── Train ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trainer = GRPOTrainer(config, reward_fn)
    trainer.train(problems, epochs=EPOCHS)
