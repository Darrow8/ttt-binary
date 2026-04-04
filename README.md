# TTT-Binary

**Test-Time Training with Binary Reward Signals for Mathematical Reasoning**

A pipeline for improving LLM mathematical reasoning through self-generated training data and Group Relative Policy Optimization (GRPO). Given a hard target problem the base model cannot solve, TTT-Binary discovers calibrated subproblems via LLM self-consistency, trains a LoRA adapter on those subproblems with GRPO, and evaluates whether the fine-tuned model can now solve the original problem.

This project was developed as part of Stanford CS224N (Spring 2026).

## Overview

TTT-Binary operates in three stages:

1. **TTT-Discover** — Generate a curriculum of subproblems calibrated by self-consistency. An LLM proposes related problems; parallel sampling estimates agreement rate; problems in a target difficulty band (e.g., 60–80% agreement) are kept as training data.
2. **GRPO Fine-Tuning** — Train a LoRA adapter on the discovered subproblems using Group Relative Policy Optimization with binary (correct/incorrect) reward signals via the Tinker API.
3. **Evaluation** — Compare the fine-tuned checkpoint against the base model and frontier models (Claude, GPT-5.4, Gemini) on the target problem and related tasks.

## Repository Structure

```
├── Stage1/                        # TTT-Discover: subproblem generation
│   ├── distinct_llm_prompting.py  # Main generation script
│   ├── attempted_answers.json     # Failed attempts for prompt conditioning
│   ├── sum_keeps.py               # Aggregate kept problems across runs
│   └── runs/                      # Timestamped output (keeps.json, skips.json)
│
├── grpo-pipeline/                 # GRPO training framework
│   ├── pipeline/
│   │   ├── __init__.py            # Public API re-exports
│   │   ├── trainer.py             # GRPOTrainer + GRPOConfig
│   │   ├── rewards.py             # Pluggable reward functions
│   │   ├── problems.py            # Problem loader (JSONL, JSON, CSV, HF)
│   │   └── logging.py             # MetricsLogger (JSONL + W&B)
│   ├── grpo_subproblems.py        # Driver: train on discovered subproblems
│   └── grpo_subproblems_resume.py # Driver: resume from a checkpoint
│
├── inference/
│   └── infer.py                   # Evaluate via Vertex AI or local Tinker checkpoint
│
├── problems/                      # Problem banks (JSONL)
│   ├── conics-50.jsonl
│   ├── conics-100.jsonl
│   └── subproblems.jsonl
│
├── eval-results/                  # Frontier model evaluation
│   ├── eval_frontier_models.py    # Compare Claude / GPT-5.4 / Gemini
│   └── eval_gpt54_repeat.py       # Pass@k-style repeated evaluation
│
├── utils/
│   ├── compile_problems.py        # Merge Stage1 keeps into one dataset
│   └── plot_comparison.py         # Base vs GRPO answer distribution plots
│
├── retrosynthesis/                # Chemistry domain: SMILES retrosynthesis eval
│   └── batch_retro_eval.py        # RDKit-based batch evaluation
│
├── simple-query/                  # Standalone query scripts
│   ├── query_gpt_oss.py           # One-shot Vertex query
│   └── query_tinker_ckpt50.py     # Tinker checkpoint sampling
│
├── runs/                          # Inference & eval outputs (gitignored)
├── requirements.txt
└── .env                           # API keys (not committed)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with the required credentials:

```
TINKER_API_KEY=<your-tinker-key>
HF_TOKEN=<your-huggingface-token>
```

Vertex AI access requires Google Cloud credentials configured via `gcloud auth application-default login`.

## Usage

### Stage 1: Generate Subproblems (TTT-Discover)

Generate calibrated subproblems by sampling and filtering on self-consistency:

```bash
python Stage1/distinct_llm_prompting.py --n-problems 50 --n-samples 32
```

Key flags:
- `--n-problems` — Number of subproblems to collect
- `--n-samples` — Parallel samples per candidate for agreement estimation
- `--tinker` — Use a Tinker checkpoint instead of Vertex AI
- `--checkpoint` — Explicit `tinker://…` checkpoint path

Output lands in `Stage1/runs/<timestamp>/keeps.json` and `skips.json`.

### Compile Training Data

Merge all kept subproblems into a single dataset:

```bash
python utils/compile_problems.py --input-dir Stage1/runs --output problems/subproblems.jsonl
```

### Stage 2: GRPO Training

Train a LoRA adapter on the discovered subproblems:

```bash
cd grpo-pipeline
python -m pipeline.grpo_subproblems
```

Configuration is set in `grpo_subproblems.py` — key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `openai/gpt-oss-120b` | Base model |
| `batch_size` | 25 | Problems per training step |
| `group_size` | 16 | Samples per problem for advantage estimation |
| `learning_rate` | 1e-4 | LoRA learning rate |
| `lora_rank` | 32 | LoRA rank |
| `max_tokens` | 100000 | Max generation length |
| `save_every` | 5 | Checkpoint frequency (steps) |

To resume from an existing checkpoint:

```bash
python -m pipeline.grpo_subproblems_resume
```

### Stage 3: Evaluation

**Inference with the fine-tuned model:**

```bash
python inference/infer.py --local --n-samples 100
```

**Inference with the base model (Vertex AI):**

```bash
python inference/infer.py --n-samples 100
```

**Frontier model comparison:**

```bash
python eval-results/eval_frontier_models.py
```

## Reward Functions

The GRPO pipeline supports pluggable reward functions defined in `grpo-pipeline/pipeline/rewards.py`:

| Function | Description |
|----------|-------------|
| `boxed_match` | 1.0 if `\boxed{...}` answer matches reference |
| `boxed_format_bonus` | ±0.1 for presence/absence of `\boxed{}` |
| `exact_match` | 1.0 if response contains the reference answer |
| `answer_tag_match` | 1.0 if `**ANSWER: ...**` matches reference |
| `smiles_match` | 1.0 if `\boxed{...}` SMILES matches reference (RDKit canonicalization) |
| `combined()` | Weighted combination of any reward functions |

## Problem Format

Problems are stored as JSONL with one object per line:

```json
{"prompt": "Find the value of ...", "reference": "42", "id": "prob_001"}
```

The loader also accepts JSON arrays, CSV files, and HuggingFace dataset identifiers.

## Dependencies

Core dependencies (see `requirements.txt`):

- `tinker` / `tinker-cookbook` — Tinker API for training and sampling
- `torch` — Tensor operations for GRPO advantage computation
- `transformers` — Tokenizer for chat template rendering
- `openai` — Vertex AI API client
- `google-auth` — Google Cloud authentication
- `wandb` — Experiment tracking
- `datasets` — HuggingFace dataset loading
- `rdkit` — SMILES canonicalization (retrosynthesis tasks only)
