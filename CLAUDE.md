# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **GRPO (Group Relative Policy Optimization)** training pipeline built on the [Thinking Machines Tinker API](https://thinkingmachines.ai/tinker/). The project implements TTT-Discover methodology: given a hard problem, generate similar problems at calibrated difficulty where LLM answers agree ~50-80% of the time, using majority-vote as pseudo ground truth for training.

The codebase supports:
- GRPO training with LoRA adapters on custom problem sets
- Multi-stage problem generation and subproblem decomposition
- Evaluation against frontier models (Claude, GPT, Gemini)
- Both remote (Vertex AI) and local checkpoint inference

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Set API keys in .env (see .env for required keys)
#    TINKER_API_KEY, HF_TOKEN, WANDB_API_KEY, WANDB_ENTITY
#    GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION
#    ANTHROPIC_API_KEY, OPENAI_API_KEY (for eval scripts)

# 3. For Google Cloud / Vertex AI access
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

## Common Commands

### Training

```bash
# Basic GRPO training (configure in the script first)
python grpo_train.py

# Train on subproblems dataset
python grpo_subproblems.py

# Resume from checkpoint
# Set `resume_from="path/to/checkpoint"` in GRPOConfig

# Train on specific checkpoint (checkpoint 50)
python grpo_ckpt50.py
```

Training checkpoints are saved to Tinker's cloud storage with labels like `{log_dir_name}.ckpt-{step:06d}`. Access them via `tinker.ServiceClient().create_training_client_from_state(path=checkpoint_path)`.

### Evaluation

```bash
# Evaluate frontier models (Claude Opus 4.6, GPT-5.4, Gemini 3.1)
python eval_frontier_models.py --problems problems/conics-100.jsonl --samples 5

# Batch evaluation on retrosynthesis problems
python batch_retro_eval.py --n-samples 3 --workers 10

# Evaluate with repeated sampling
python eval_gpt54_repeat.py
```

### Inference

```bash
# Using remote Vertex AI API
python inference/infer.py "Your problem statement here"

# Using local GRPO checkpoint
python inference/infer.py --local "Your problem here"

# Override sample count
python inference/infer.py --n-samples 50
```

### Problem Generation (Stage1)

```bash
# Generate calibrated difficulty problems via LLM self-consistency
python Stage1/llm_prompting.py --problem_statement "PROBLEM" --n_problems 100

# Generate problems with distinct domains
python Stage1/distinct_llm_prompting.py
```

## Architecture

### Core Pipeline (`pipeline/`)

- **`trainer.py`**: `GRPOTrainer` class implementing the complete GRPO loop
  - Samples completions from current policy (group_size rollouts per problem)
  - Computes rewards via pluggable reward functions
  - Calculates group-relative advantages: `(reward - mean) / std`
  - Updates policy via importance-sampled policy gradient
  - Supports multiple loss normalization strategies (token, sequence, dr_grpo)
  - Implements PPO-style clipping and optional KL penalty

- **`problems.py`**: Problem loading from JSONL, CSV, HuggingFace datasets, or lists
  - Each `Problem` has `prompt`, `reference`, and optional `meta` dict

- **`rewards.py`**: Pluggable reward functions with signature `(response: str, problem: Problem) -> float`
  - Built-in: `boxed_match`, `answer_tag_match`, `exact_match`, `smiles_match`
  - Format bonuses: `boxed_format_bonus`, `answer_tag_format_bonus`
  - Utilities: `extract_boxed()`, `extract_answer_tag()`, `_normalize()`
  - Combiner: `combined()` for weighted mixtures

- **`logging.py`**: `MetricsLogger` for JSONL + W&B integration

### Configuration (`GRPOConfig`)

Key hyperparameters in `pipeline/trainer.py`:
- `model_name`: Base model (default: `"openai/gpt-oss-120b"`)
- `resume_from`: Checkpoint path to resume training (default: `None`)
- `batch_size`: Unique problems per step (default: 128)
- `group_size`: Rollouts per problem (default: 16)
- `learning_rate`: Adam LR (default: 4e-5)
- `lora_rank`: LoRA adapter rank (default: 32)
- `max_tokens`: Max completion length (default: 256)
- `save_every`: Checkpoint interval (0 = disabled)
- `system_prompt`, `few_shot`, `prompt_suffix`: Prompt engineering
- `scale_rewards`: Enable advantage std normalization (default: True)
- `clip_epsilon`: PPO clipping (default: 0.2, None = no clipping)
- `loss_type`: `"token"` (DAPO), `"sequence"` (original GRPO), or `"dr_grpo"`

### Answer Extraction

The codebase uses two answer formats:
1. **`\boxed{answer}`** — Math problems (LaTeX style)
2. **`**ANSWER: value**`** — General format for frontier model evals

Extraction utilities:
- `extract_boxed(text)` in `pipeline/rewards.py` — handles nested braces
- `extract_answer_tag(text)` in `pipeline/rewards.py` — regex for `**ANSWER: ...**`
- Normalization removes commas, spaces, converts decimals (e.g., `"3264.0000"` → `"3264"`)

### Key Training Scripts

- `grpo_train.py`: Entry point template for standard GRPO training
- `grpo_subproblems.py`: Train on subproblem datasets (e.g., `conics-50.jsonl`)
- `grpo_subproblems_resume.py`: Resume subproblem training from checkpoint
- `grpo_ckpt50.py`: Train on checkpoint 50 specifically
- `grpo_retrosynthesis.py`: GRPO for chemistry/SMILES problems

### Evaluation and Inference

- `eval_frontier_models.py`: Compare Claude Opus 4.6, GPT-5.4, Gemini 3.1 Pro with majority vote
  - Uses `**ANSWER: ...**` format extraction
  - Supports async concurrent API calls (configurable `MAX_CONCURRENT`)

- `batch_retro_eval.py`: Batch evaluation for retrosynthesis (SMILES canonicalization via RDKit)

- `inference/infer.py`: Interactive problem solving
  - Remote: calls Vertex AI Maas endpoint
  - Local: loads Tinker checkpoint via `tinker.ServiceClient()`
  - Majority vote aggregation across n_samples

- `inference/infer_steps.py`: Step-by-step inference helper

### Stage1: Problem Generation

- `Stage1/llm_prompting.py`: TTT-Discover implementation
  - Generates N similar problems from a seed problem
  - Samples multiple solutions per problem
  - Filters by majority-vote agreement rate (50-80%)
  - Produces JSONL dataset with pseudo ground truth

- `Stage1/domains.py`: Domain-specific configurations (MATH, RETROSYNTHESIS)

## Important Details

### Tinker API Integration

The project uses Thinking Machines Tinker for distributed training:
- `tinker.ServiceClient()` manages training and sampling clients
- `create_lora_training_client()` for fresh training
- `create_training_client_from_state(path=...)` to resume from checkpoint
- Checkpoints saved with `training_client.save_state(label, ttl_seconds=...)`
- Returns checkpoint paths like `"tinker://..."` for later loading

### GRPO Algorithm Details

Standard GRPO from DeepSeekMath with corrections from TRL/DAPO/Dr.GRPO:
1. For each problem, sample G=group_size completions from current policy
2. Score each completion with reward function
3. Compute advantages: `A_i = (r_i - mean(r)) / std(r)` (if `scale_rewards=True`)
4. Skip groups where all rewards are identical (zero variance = no learning signal)
5. Build training datums with per-token loss weights (depends on `loss_type`)
6. Run forward-backward pass with `loss_fn="importance_sampling"`
7. Apply Adam optimizer step

### Loss Normalization

- `"token"` (DAPO): divide by total tokens across batch — avoids length bias
- `"sequence"`: divide by completion length (original GRPO)
- `"dr_grpo"`: divide by `group_size * max_tokens` — removes length bias entirely

### Vertex AI / Google Cloud

Scripts like `eval_frontier_models.py` and `batch_retro_eval.py` use Vertex AI Model-as-a-Service:
- Requires `gcloud auth application-default login`
- Set `GOOGLE_CLOUD_PROJECT` and optionally `GOOGLE_CLOUD_LOCATION`
- Uses OpenAI SDK with custom base URL pointing to Vertex endpoint

### Weights & Biases

All training runs log to W&B by default:
- Configure via `wandb_project`, `wandb_entity`, `wandb_run_name` in `GRPOConfig`
- Set `wandb_project=None` to disable
- Logs: reward stats, advantage stats, token counts, sample completions table

### Problem File Formats

JSONL format (one JSON object per line):
```json
{"prompt": "What is 2+2?", "reference": "4"}
```

The pipeline also supports CSV and HuggingFace datasets via `load_problems()`.

## Testing and Validation

When making changes to the training pipeline:
1. Test with a small `batch_size` (e.g., 4-16) and few epochs first
2. Check W&B dashboard for reward trends and sample completions
3. For reward functions, verify extraction with manual tests before training
4. Use `save_every` to checkpoint frequently during experiments

## Modified Files Status

Current uncommitted changes:
- `grpo_subproblems.py`: Modified subproblem sampling/config
- `pipeline/trainer.py`: Updated training loop robustness
- `inference/infer_steps.py`: New file (untracked)

See git status for details.
