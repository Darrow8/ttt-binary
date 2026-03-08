# GRPO Pipeline with the Tinker API

A modular pipeline for **Group Relative Policy Optimization (GRPO)** training via the [Thinking Machines Tinker API](https://thinkingmachines.ai/tinker/).

Bring your own problems, plug in a reward function, and train.

## Quick start

```bash
# 1. Virtual environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Set your keys in .env
#    TINKER_API_KEY, HF_TOKEN, WANDB_API_KEY, WANDB_ENTITY

# 3. Run
python grpo_train.py
```

## Project structure

```
grpo-project/
├── grpo_train.py              # Entry point — configure & run
├── pipeline/
│   ├── __init__.py
│   ├── problems.py            # Load problems from JSONL, CSV, HF datasets, or lists
│   ├── rewards.py             # Pluggable reward functions
│   ├── trainer.py             # GRPOTrainer class (the training loop)
│   └── logging.py             # Metrics logger (JSONL + W&B)
├── problems/
│   ├── math.jsonl             # Example: arithmetic problems
│   └── reasoning.jsonl        # Example: reasoning problems
├── requirements.txt
└── .env                       # API keys (not committed)
```

## How GRPO works

For each problem in a batch:

1. **Sample** a group of completions from the current policy
2. **Score** each completion with your reward function
3. **Center** rewards within the group (advantage = reward − group mean)
4. **Update** the policy via importance-sampled policy gradient

Problems where every completion scores the same are skipped (no learning signal).

## Loading problems

Problems can come from several sources:

```python
from pipeline import load_problems

# From a JSONL file (each line: {"prompt": "...", "reference": "..."})
problems = load_problems("problems/math.jsonl")

# From a CSV
problems = load_problems("data/questions.csv")

# From a HuggingFace dataset
problems = load_problems("openai/gsm8k", prompt_field="question", reference_field="answer")

# From a Python list
problems = load_problems([
    {"prompt": "What is 2+2?", "reference": "4"},
    {"prompt": "Capital of France?", "reference": "Paris"},
])
```

## Reward functions

Built-in rewards in `pipeline.rewards`:

| Function | Description |
|---|---|
| `exact_match` | 1.0 if reference appears in response (normalized) |
| `boxed_match` | 1.0 if `\boxed{...}` answer matches reference |
| `boxed_format_bonus` | +0.1 if `\boxed{}` present, −0.1 otherwise |
| `contains_reference` | 1.0 if reference appears in response (case-insensitive) |
| `regex_match(pattern)` | 1.0 if regex pattern matches response |
| `length_penalty(max)` | Small negative reward for overly long responses |

Combine them with weights:

```python
from pipeline import combined, boxed_match, boxed_format_bonus

reward_fn = combined(
    (1.0, boxed_match),
    (0.1, boxed_format_bonus),
)
```

Or write your own:

```python
def my_reward(response: str, problem) -> float:
    return 1.0 if problem.reference.lower() in response.lower() else 0.0
```

## Configuration

All settings live in `GRPOConfig`:

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `openai/gpt-oss-120b` | Base model to fine-tune |
| `batch_size` | `128` | Unique problems per training step |
| `group_size` | `16` | Rollouts per problem |
| `learning_rate` | `4e-5` | Adam learning rate |
| `lora_rank` | `32` | LoRA adapter rank |
| `max_tokens` | `256` | Max tokens per completion |
| `save_every` | `20` | Checkpoint interval (0 = disabled) |
| `system_prompt` | `None` | System message prepended to every prompt |
| `few_shot` | `[]` | Few-shot examples as chat messages |
| `prompt_suffix` | `""` | Text appended to every problem prompt |
| `wandb_project` | `"grpo-tinker"` | W&B project name (`None` to disable) |

## W&B dashboard

Every training step logs to Weights & Biases:

- **Scalars**: `reward/mean`, `reward/min`, `reward/max`, `advantage/std`, `tokens/mean_per_completion`, `time/total`
- **Histograms**: reward distribution, advantage distribution
- **Tables**: sample completions with prompt, response, expected/predicted answer, correctness

## Plotting locally

```python
import pandas as pd, matplotlib.pyplot as plt

df = pd.read_json("/tmp/tinker-grpo/run/metrics.jsonl", lines=True)
plt.plot(df["step"], df["reward/mean"])
plt.xlabel("Step"); plt.ylabel("Mean Reward")
plt.title("GRPO Training"); plt.grid(True); plt.show()
```
