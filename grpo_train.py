"""
GRPO training entry point.

Configure the problem source, reward function, and model below, then run::

    python grpo_train.py

All keys are loaded from ``.env`` automatically.
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
    exact_match,
    combined,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)


# ── Configuration ──────────────────────────────────────────────────────────

config = GRPOConfig(
    model_name="openai/gpt-oss-120b",
    log_dir="/tmp/tinker-grpo/run",

    batch_size=16,
    group_size=16,
    learning_rate=4e-5,
    lora_rank=32,
    max_tokens=256,

    save_every=20,

    wandb_project="grpo-tinker",

    # Optional system prompt injected before every problem
    # system_prompt="You are a helpful assistant. Show your reasoning step by step.",

    # # Suffix appended to every problem prompt
    # prompt_suffix=" Put your final answer inside \\boxed{}.",

    # # Optional few-shot examples
    # few_shot=[
    #     {
    #         "role": "user",
    #         "content": "How many r's are in strawberry? Put your final answer inside \\boxed{}.",
    #     },
    #     {
    #         "role": "assistant",
    #         "content": (
    #             "Let me spell it out: s-t-r-a-w-b-e-r-r-y. "
    #             "Counting the r's: positions 3, 8, 9. That's 3. \\boxed{3}"
    #         ),
    #     },
    # ],
)


# ── Problems ───────────────────────────────────────────────────────────────
# Load from a JSONL file, CSV, HuggingFace dataset, or inline list.
# Examples:
#   problems = load_problems("problems/math.jsonl")
#   problems = load_problems("problems/reasoning.jsonl")
#   problems = load_problems("openai/gsm8k", prompt_field="question", reference_field="answer")
#   problems = load_problems([{"prompt": "What is 2+2?", "reference": "4"}, ...])

problems = load_problems("problems/math.jsonl")


# ── Reward function ────────────────────────────────────────────────────────
# Mix-and-match built-in rewards, or write your own:
#   from pipeline import exact_match, boxed_match, boxed_format_bonus, regex_match, contains_reference, combined
#
# Custom example:
#   def my_reward(response: str, problem) -> float:
#       return 1.0 if problem.reference.lower() in response.lower() else 0.0

reward_fn = combined(
    (1.0, boxed_match),          # full credit for correct \boxed{} answer
    (0.1, boxed_format_bonus),   # small bonus/penalty for format compliance
)


# ── Train ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trainer = GRPOTrainer(config, reward_fn)
    trainer.train(problems)
