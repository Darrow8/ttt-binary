"""
GRPO training on subproblems derived from a hard source problem.

Uses the cookbook_grpo package with original DeepSeek GRPO settings:
  - Std-normalized advantages
  - Sequence-level loss normalization
  - KL penalty (β=0.04) against reference model

Workflow:
    1. conics-50.jsonl contains easier subproblems with majority-vote ground truth.
    2. This script runs GRPO on those subproblems for several epochs
       so the model learns the component reasoning skills.
    3. After training, evaluate the hard problem again to see if the
       model can now produce correct answers (target: >0/100).

Usage::

    python grpo_subproblems.py
"""

import asyncio
import logging

from dotenv import load_dotenv
load_dotenv()

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl import train

from cookbook_grpo.dataset import SubproblemDatasetBuilder
from cookbook_grpo.grpo_overrides import patch_grpo_advantages

patch_grpo_advantages()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)


# ── Configuration ──────────────────────────────────────────────────────────

MODEL_NAME = "openai/gpt-oss-120b"
RENDERER_NAME = model_info.get_recommended_renderer_name(MODEL_NAME)

SYSTEM_PROMPT = """\
You are a careful and rigorous math student working through an advanced mathematics problem. Your goal is to solve the problem step by step.

Show all important intermediate reasoning, derivations, and calculations. Explain why each step is valid and reference any relevant theorems or identities when appropriate. Avoid skipping logical steps or making large jumps in reasoning.

If the problem involves multiple cases or approaches, consider them systematically. Use clear mathematical notation and keep the solution organized.

After completing the reasoning, clearly state the final answer."""

dataset_builder = SubproblemDatasetBuilder(
    data_path="./conics-50.jsonl",
    batch_size=25,
    group_size=16,
    num_epochs=50,
    model_name=MODEL_NAME,
    renderer_name=RENDERER_NAME,
    prompt_suffix=" Put your final answer inside \\boxed{}.",
    system_prompt=SYSTEM_PROMPT,
    reward_correct=1.0,
    reward_wrong=0.01,
    reward_none=0.0,
    shuffle=True,
    seed=42,
)

config = train.Config(
    model_name=MODEL_NAME,
    renderer_name=RENDERER_NAME,
    log_path="./subproblems-run",
    dataset_builder=dataset_builder,
    learning_rate=1e-4,
    max_tokens=16384,
    lora_rank=32,
    save_every=5,
    eval_every=10,
    temperature=1.0,
    wandb_project="conics-50-16k-tokens",
    loss_fn="importance_sampling",
    remove_constant_reward_groups=True,
    kl_penalty_coef=0.04,
    kl_discount_factor=0.0,
    kl_reference_config=train.KLReferenceConfig(base_model=MODEL_NAME),
)


# ── Train ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))
