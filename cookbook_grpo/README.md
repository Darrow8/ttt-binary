# Cookbook-based GRPO for TTT-Discover

Replaces the custom GRPO training loop with the official `tinker-cookbook` RL stack.
Same experiment semantics (50 subproblems, 3-tier reward, 500-sample target eval),
but advantage computation, batching, rollout scheduling, and optimization are handled
by the framework.

## Installation

```bash
# Activate the project venv
source .venv/bin/activate

# Required packages (already in requirements.txt)
pip install tinker tinker-cookbook chz pyyaml python-dotenv

# Set environment variables in .env:
#   TINKER_API_KEY=...
#   WANDB_API_KEY=...  (optional)
#   HF_TOKEN=...       (for tokenizer download)
```

## Quick Start

### Smoke test (verify parser + reward + dataset loading)

```bash
python3 -m pytest cookbook_grpo/tests/ -v
```

### Debug mode (inspect a few rollouts without training)

```bash
python3 debug_rollouts.py --data ./conics-50.jsonl --n-problems 2 --n-samples 4
```

### Run the reproduction training

```bash
python3 train_tinker_grpo.py
```

Or with a YAML config:

```bash
python3 train_tinker_grpo.py --config=cookbook_grpo/configs/conics50.yaml
```

### Evaluate checkpoints on the target problem

```bash
# Base model
python3 eval_target_problem.py --n-samples 500

# Specific checkpoint
python3 eval_target_problem.py --checkpoint "tinker://..." --n-samples 500 --output results/ckpt50.json
```

### Evaluate on subproblems

```bash
python3 eval_subproblems.py --data ./conics-50.jsonl --checkpoint "tinker://..." --n-samples 16
```

## Dataset Format

JSONL file, one record per line. Required fields:

```json
{"prompt": "Problem text...", "reference": "866"}
```

Alternative field names also accepted:

```json
{"problem": "Problem text...", "answer": "866", "id": "sub_01", "agreement_rate": 0.7}
```

## Architecture

```
cookbook_grpo/
├── __init__.py
├── parser.py          # extract_boxed_answer, normalize_answer
├── rewards.py         # compute_reward (1.0 / 0.01 / 0.0)
├── env.py             # SubproblemEnv (extends ProblemEnv)
├── dataset.py         # SubproblemDataset, SubproblemDatasetBuilder (extends RLDatasetBuilder)
├── configs/
│   ├── conics50.yaml
│   └── conics50_resume.yaml
└── tests/
    ├── test_parser.py
    ├── test_rewards.py
    ├── test_dataset.py
    └── test_eval.py

train_tinker_grpo.py    # Training entry point
eval_target_problem.py  # Target problem (866) evaluation
eval_subproblems.py     # Subproblem set evaluation
debug_rollouts.py       # Debug/sanity-check utility
```

### What the cookbook handles (you don't implement)

- GRPO advantage normalization and centering
- Importance-sampled policy gradient loss
- PPO-style clipping (if configured)
- Adam optimizer step pipelining
- Checkpoint save/resume with optimizer state
- Rollout parallelism and async scheduling
- KL penalty against reference policy (if configured)
- W&B metric logging integration
- Constant-reward group filtering

### What's custom (minimal surface area)

- `parser.py`: boxed answer extraction (40 lines)
- `rewards.py`: 3-tier reward function (20 lines)
- `env.py`: SubproblemEnv extending ProblemEnv (80 lines)
- `dataset.py`: JSONL loading + epoch cycling (100 lines)

## Configuration

Key hyperparameters matching the paper:

| Parameter | Value | Notes |
|-----------|-------|-------|
| model_name | openai/gpt-oss-120b | Base model |
| batch_size | 25 | Problems per step |
| group_size | 16 | Rollouts per problem |
| learning_rate | 1e-4 | Adam LR |
| temperature | 0.7 | Sampling temperature |
| num_epochs | 50 | Passes over 50 problems |
| max_tokens | 16384 | Max completion length |
| save_every | 5 | Checkpoint interval |
| loss_fn | importance_sampling | GRPO loss |
| loss_fn_config.clip_epsilon | 0.2 | PPO-style ratio clipping |
| reward: correct | 1.0 | Exact match |
| reward: wrong | 0.0 | Boxed but incorrect |
| reward: none | 0.0 | No boxed answer |

For the second-stage experiment (continued training from step 50):

| Parameter | Value |
|-----------|-------|
| load_checkpoint_path | (step-50 checkpoint) |
| learning_rate | 3e-5 |
| max_steps | 50 |

## Comparing with the Custom Implementation

The custom implementation is in `pipeline/trainer.py` + `grpo_subproblems.py`.
Key differences that this rewrite addresses:

1. **Advantage computation**: cookbook uses `compute_advantages()` from `rl.data_processing`
   which handles the centering + std normalization in a tested codepath.

2. **Loss normalization**: cookbook's `importance_sampling` loss handles token-level
   normalization server-side, eliminating the manual `_compute_loss_weights` logic.

3. **Constant-reward filtering**: `remove_constant_reward_groups=True` in config
   replaces the manual `all(a == 0.0 for a in advantages_g)` check.

4. **Datum assembly**: `assemble_training_data()` handles padding, alignment, and
   target token construction — the most error-prone part of the custom impl.

5. **Optimizer pipelining**: `train_step()` pipelines forward_backward + optim_step
   across substeps for GPU utilization, whereas the custom impl issues them sequentially.

To compare runs: both log to W&B. Compare reward curves, solve rates, and
checkpoint-level target evaluations side by side.

## Known Limitations & Design Notes

### `std(correction=0)` in advantage computation

The `grpo_compute_advantages` function uses population std (Bessel's correction=0)
to match the original DeepSeekMath paper. PyTorch's default is `correction=1` (sample std),
which would inflate advantages for small group sizes. If changing `group_size`, be aware
that this choice has a larger effect for small groups.

### No curriculum / difficulty adaptation

The `SubproblemDataset` cycles through the same problems for `num_epochs` without
mechanism to drop saturated problems (where all rollouts get the same reward) or
increase difficulty over time. As training progresses and the model improves,
an increasing fraction of problems produce zero-variance groups (all correct),
yielding zero gradient.

Mitigation approaches:
- Use `conics50_resume.yaml` pattern: manually resample harder problems at stage boundaries
- Implement a custom `RLDataset` that removes problems once solve rate exceeds a threshold
- Use the `eval_every` checkpoints to monitor saturation and trigger resampling

### Monkey-patching fragility

The `grpo_overrides.py` module patches `tinker_cookbook` internals. The patch includes
runtime assertions that verify the replacement took effect, so it will fail fast if
the cookbook changes its import structure. If upgrading `tinker_cookbook`, re-verify
that `patch_grpo_advantages()` still works by running the test suite.
