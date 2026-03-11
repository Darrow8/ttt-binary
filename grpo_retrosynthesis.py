"""
GRPO training on retrosynthesis subproblems derived from a hard source problem.

Workflow:
    1. retro_suzuki_subproblems.jsonl contains 20 easier retrosynthesis
       problems with verified ground-truth reactant SMILES.
    2. This script runs GRPO on those 20 subproblems for several epochs
       so the model learns the component retrosynthetic skills (Suzuki
       coupling disconnection, boronic acid/halide assignment, etc.).
    3. After training, evaluate the hard problem again to see if the
       model can now produce correct answers (target: >0/10).

Usage::

    python grpo_retrosynthesis.py
"""

import logging

from dotenv import load_dotenv
load_dotenv()

from pipeline import (
    GRPOConfig,
    GRPOTrainer,
    load_problems,
    smiles_match,
    boxed_format_bonus,
    combined,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)


# ── Configuration ──────────────────────────────────────────────────────────
# 20 subproblems → batch_size=20 means 1 step per epoch.
# Run enough epochs for the model to learn all subproblems.

config = GRPOConfig(
    model_name="openai/gpt-oss-120b",
    log_dir="/tmp/tinker-grpo/retrosynthesis-run",

    batch_size=20,
    group_size=16,
    learning_rate=4e-5,
    lora_rank=32,
    max_tokens=16384,

    save_every=5,

    wandb_project="grpo-tinker-retrosynthesis",

    temperature=0.7,
    system_prompt="You are a helpful assistant. Show your reasoning step by step.",

    prompt_suffix=" Put your final answer inside \\boxed{}.",

    few_shot=[],
)

EPOCHS = 10


# ── Problems ───────────────────────────────────────────────────────────────

problems = load_problems("problems/retro_suzuki_subproblems.jsonl")


# ── Reward function ────────────────────────────────────────────────────────

reward_fn = combined(
    (1.0, smiles_match),
    (0.1, boxed_format_bonus),
)


# ── Train ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trainer = GRPOTrainer(config, reward_fn)
    trainer.train(problems, epochs=EPOCHS)
