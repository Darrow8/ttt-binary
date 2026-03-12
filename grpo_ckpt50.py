"""
GRPO training from checkpoint-50 on curated hard problems.

Continues GRPO from the subproblems-run checkpoint-50 model on a fresh
set of 50 problems derived from frontier-model agreement filtering.

Usage::

    python grpo_ckpt50.py
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

CHECKPOINT = (
    "tinker://e6b448b4-7e70-5e39-b0f7-06e0ef5b8e0d:train:0"
    "/weights/subproblems-run.ckpt-000050"
)

config = GRPOConfig(
    model_name="openai/gpt-oss-120b",
    log_dir="./ckpt50-run",
    resume_from=CHECKPOINT,

    batch_size=25,
    group_size=16,
    learning_rate=1e-4,
    lora_rank=32,
    max_tokens=16384,

    save_every=5,

    wandb_project="grpo-ckpt50-2",

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

EPOCHS = 25


# ── Problems ───────────────────────────────────────────────────────────────

def _normalize_fields(rows: list[dict]) -> list[dict]:
    """Map alternate field names (problem/ground_truth_answer) to prompt/reference."""
    out = []
    for row in rows:
        r = dict(row)
        if "prompt" not in r and "problem" in r:
            r["prompt"] = r.pop("problem")
        if "reference" not in r and "ground_truth_answer" in r:
            r["reference"] = r.pop("ground_truth_answer")
        out.append(r)
    return out


import json
from pathlib import Path

_raw = json.loads(Path("problems/ckpt-50-problems.json").read_text())
problems = load_problems(_normalize_fields(_raw))


# ── Reward function ────────────────────────────────────────────────────────

reward_fn = combined(
    (1.0, boxed_match),
    (0.1, boxed_format_bonus),
)


# ── Train ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trainer = GRPOTrainer(config, reward_fn)
    trainer.train(problems, epochs=EPOCHS)
