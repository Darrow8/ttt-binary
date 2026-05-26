"""Monkey-patches for tinker_cookbook to match the original DeepSeek GRPO algorithm.

The original GRPO (DeepSeekMath, arXiv:2402.03300) differs from the cookbook defaults:
  1. Advantages are std-normalized: A_i = (r_i - mean) / std  (cookbook only mean-centers)
  2. Loss uses sequence-level normalization: divide by |o_i| per trajectory
     (cookbook applies a flat scalar to all action tokens without length division)
  3. KL penalty must also be sequence-normalized to avoid asymmetric length bias

This module provides:
  - `grpo_compute_advantages`: drop-in replacement for cookbook's `compute_advantages`
    that adds std normalization
  - `apply_sequence_normalization`: post-processes datums to divide each datum's
    advantages by its action token count (mask sum)
  - `strip_env_all_prefix`: rewrites compute_trajectory_metrics output to drop the
    `env/all/` prefix, add reward/min/max/mean, and extract per-rollout sample rows
  - `_samples_table_log_metrics`: MultiplexLogger.log_metrics replacement that
    converts the extracted sample rows into a `samples_{step}` wandb.Table
  - `_terse_pretty_print_log_metrics`: PrettyPrintLogger.log_metrics replacement
    that emits a single `step=N reward=X.XXX time=X.Xs` INFO line per step
    instead of the cookbook's full Rich metrics table
  - `patch_grpo_advantages`: applies the monkey-patches at import time
"""

from __future__ import annotations

import torch

from tinker_cookbook.rl.data_processing import (
    assemble_training_data as _orig_assemble,
)
from tinker_cookbook.rl.metric_util import (
    compute_trajectory_metrics as _orig_compute_trajectory_metrics,
)
from tinker_cookbook.rl.metrics import (
    incorporate_kl_penalty as _orig_incorporate_kl_penalty,
)
from tinker_cookbook.rl.types import TrajectoryGroup

import tinker


def grpo_compute_advantages(trajectory_groups_P: list[TrajectoryGroup]) -> list[torch.Tensor]:
    """Compute advantages with full GRPO normalization: (r - mean) / std.

    Groups where all rewards are identical get zero advantages (the caller
    should filter these out via remove_constant_reward_groups=True).
    """
    advantages_P: list[torch.Tensor] = []

    for traj_group in trajectory_groups_P:
        rewards_G = torch.tensor(traj_group.get_total_rewards(), dtype=torch.float64)
        mean = rewards_G.mean()
        std = rewards_G.std(correction=0)
        if std < 1e-8:
            advantages_P.append(torch.zeros(len(rewards_G)))
        else:
            advantages_P.append(((rewards_G - mean) / std).float())

    return advantages_P


def apply_sequence_normalization(data_D: list[tinker.Datum]) -> list[tinker.Datum]:
    """Divide each datum's advantages by its action token count (mask sum).

    This implements the original GRPO's 1/|o_i| per-sequence normalization.
    The importance_sampling loss sums (ratio * advantage) across tokens, so
    dividing advantage by token count makes the effective loss per trajectory
    equal to: (1/|o_i|) * sum_t(ratio_t * A_i), matching the original paper.
    """
    for datum in data_D:
        mask = datum.loss_fn_inputs["mask"].to_torch()
        action_token_count = mask.sum()
        if action_token_count > 0:
            advantages = datum.loss_fn_inputs["advantages"].to_torch()
            normalized = advantages / action_token_count
            datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(normalized)
    return data_D


def grpo_assemble_training_data(
    trajectory_groups_P: list[TrajectoryGroup],
    advantages_P: list[torch.Tensor],
) -> tuple[list[tinker.Datum], list[dict[str, int]]]:
    """Assemble training data with sequence-level normalization applied."""
    data_D, metadata_D = _orig_assemble(trajectory_groups_P, advantages_P)
    apply_sequence_normalization(data_D)
    return data_D, metadata_D


async def grpo_incorporate_kl_penalty(
    data_D: list[tinker.Datum],
    base_sampling_client: tinker.SamplingClient,
    kl_penalty_coef: float,
    kl_discount_factor: float,
) -> dict[str, float]:
    """Wrapper around cookbook's incorporate_kl_penalty that applies 1/|o_i| normalization.

    The cookbook adds KL penalty terms directly to advantages. When sequence
    normalization is active, the policy gradient advantages are already divided
    by |o_i|, but the KL terms are not. This wrapper applies the same 1/|o_i|
    normalization to the KL contribution to keep the loss balanced regardless
    of sequence length.
    """
    # Snapshot pre-KL advantages
    pre_kl_advantages = [
        datum.loss_fn_inputs["advantages"].to_torch().clone() for datum in data_D
    ]

    # Run the original KL penalty incorporation (captured at import time, before the patch)
    metrics = await _orig_incorporate_kl_penalty(
        data_D, base_sampling_client, kl_penalty_coef, kl_discount_factor
    )

    # The KL delta is (post_advantages - pre_advantages). Normalize it by |o_i|.
    for i, datum in enumerate(data_D):
        post_advantages = datum.loss_fn_inputs["advantages"].to_torch()
        kl_delta = post_advantages - pre_kl_advantages[i]
        mask = datum.loss_fn_inputs["mask"].to_torch()
        action_token_count = mask.sum()
        if action_token_count > 0:
            normalized_kl_delta = kl_delta / action_token_count
        else:
            normalized_kl_delta = kl_delta
        final_advantages = pre_kl_advantages[i] + normalized_kl_delta
        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(final_advantages)

    return metrics


# Sentinel key used to pass per-rollout rows from compute_trajectory_metrics
# through the metrics dict to the W&B logger, where they're converted into a
# `wandb.Table` keyed `samples_{step}`. The key starts with `_` so callers
# treating the metrics dict as scalars won't accidentally try to log it.
_SAMPLE_ROWS_KEY = "_samples_rows"


def strip_env_all_prefix(
    trajectory_groups_P: list[TrajectoryGroup],
    taglist_P: list[list[str]],
) -> dict:
    """Match the old pipeline/logging.py format: reward/* at top level, plus samples table.

    Cookbook's compute_trajectory_metrics nests every aggregate metric under
    `env/all/...` and only emits `reward/total` (the mean). This wrapper:
      1. Strips the `env/all/` prefix so reward/by_group/etc. land in their
         own W&B sections (per-tag `env/<tag>/...` keys are left alone).
      2. Adds `reward/min`, `reward/max`, `reward/mean` to mirror the old
         pipeline/logging.py output.
      3. Extracts per-rollout (prompt, response, expected, predicted, reward,
         correct) rows from each transition's `logs` dict and stashes them
         under a sentinel key for the W&B logger to convert into a Table.
    """
    raw = _orig_compute_trajectory_metrics(trajectory_groups_P, taglist_P)
    out: dict = {
        (k[len("env/all/"):] if k.startswith("env/all/") else k): v
        for k, v in raw.items()
    }

    # reward/min, reward/max, reward/mean
    all_rewards: list[float] = [
        r for tg in trajectory_groups_P for r in tg.get_total_rewards()
    ]
    if all_rewards:
        out["reward/min"] = min(all_rewards)
        out["reward/max"] = max(all_rewards)
        out["reward/mean"] = sum(all_rewards) / len(all_rewards)

    # Per-rollout rows for the samples_{step} W&B Table. Pull the fields that
    # SubproblemEnv.step stashes in StepResult.logs.
    sample_rows: list[dict] = []
    for tg in trajectory_groups_P:
        for traj in tg.trajectories_G:
            for transition in traj.transitions:
                logs = transition.logs or {}
                if "sample_prompt" not in logs:
                    continue
                sample_rows.append(
                    {
                        "prompt": logs.get("sample_prompt", ""),
                        "response": logs.get("sample_response", ""),
                        "expected": logs.get("sample_expected", ""),
                        "predicted": logs.get("sample_predicted", ""),
                        "reward": logs.get("sample_reward", 0.0),
                        "correct": bool(logs.get("sample_correct", False)),
                    }
                )
    if sample_rows:
        out[_SAMPLE_ROWS_KEY] = sample_rows

    return out


def _terse_pretty_print_log_metrics(self, metrics: dict, step: int | None = None) -> None:
    """Replacement for PrettyPrintLogger.log_metrics: emit a single INFO line.

    The cookbook's default prints a Rich `Metric | Value` table covering every
    key in the metrics dict (one row each for by_group/*, total_episodes,
    env_metrics/*, etc.), which floods stdout/the W&B "Logs" tab. The old
    pipeline/logging.py logged only `step=N reward=X.XXX time=X.Xs` per step,
    which is what we want here. All metrics still flow through the WandbLogger
    / JsonLogger children unchanged.
    """
    import logging as _logging
    import time as _time

    now = _time.monotonic()
    last = getattr(self, "_last_t", None)
    elapsed = (now - last) if last is not None else 0.0
    self._last_t = now

    reward = metrics.get("reward/mean", metrics.get("reward/total", 0.0))
    try:
        reward_val = float(reward)
    except (TypeError, ValueError):
        reward_val = 0.0
    _logging.getLogger("tinker_cookbook.utils.ml_log").info(
        "step=%s  reward=%.3f  time=%.1fs", step, reward_val, elapsed
    )


def _samples_table_log_metrics(self, metrics: dict, step: int | None = None) -> None:
    """Replacement for MultiplexLogger.log_metrics that handles the samples table.

    Pops the _SAMPLE_ROWS_KEY entry (if present), builds a `wandb.Table`, and
    only attaches it when forwarding to a WandbLogger child. Other child
    loggers (JSON, pretty-print) see a clean dict with the sentinel removed.
    """
    from tinker_cookbook.utils.ml_log import WandbLogger

    sample_rows = metrics.pop(_SAMPLE_ROWS_KEY, None)

    for child in self.loggers:
        if sample_rows and isinstance(child, WandbLogger):
            try:
                import wandb
                table = wandb.Table(
                    columns=["prompt", "response", "expected", "predicted", "reward", "correct"],
                    data=[
                        [r["prompt"], r["response"], r["expected"], r["predicted"], r["reward"], r["correct"]]
                        for r in sample_rows
                    ],
                )
                key = f"samples_{step}" if step is not None else "samples"
                child.log_metrics({**metrics, key: table}, step)
                continue
            except Exception:
                # If wandb.Table construction fails for any reason, fall through
                # and log the scalars only so we don't crash training.
                pass
        child.log_metrics(metrics, step)


def patch_grpo_advantages():
    """Monkey-patch the cookbook to use original GRPO advantage computation and normalization.

    Call this before invoking `train.main()`.

    Patches both the source module (data_processing) and any module that imports
    the functions by name (train). Includes a runtime verification that the patch
    took effect to guard against future refactors adding new import sites.
    """
    import tinker_cookbook.rl.data_processing as dp
    import tinker_cookbook.rl.train as train_module

    dp.compute_advantages = grpo_compute_advantages
    train_module.compute_advantages = grpo_compute_advantages

    dp.assemble_training_data = grpo_assemble_training_data
    train_module.assemble_training_data = grpo_assemble_training_data

    # Patch KL penalty to apply sequence normalization to the KL terms
    import tinker_cookbook.rl.metrics as metrics_module
    train_module.incorporate_kl_penalty = grpo_incorporate_kl_penalty
    metrics_module.incorporate_kl_penalty = grpo_incorporate_kl_penalty

    # Strip env/all/ prefix so reward metrics get their own W&B section, and
    # emit reward/min, reward/max, reward/mean + per-rollout sample rows.
    import tinker_cookbook.rl.metric_util as metric_util
    metric_util.compute_trajectory_metrics = strip_env_all_prefix
    train_module.compute_trajectory_metrics = strip_env_all_prefix

    # Patch MultiplexLogger so the sample rows attached by strip_env_all_prefix
    # are turned into a `samples_{step}` wandb.Table only on the WandbLogger
    # child (other loggers see a cleaned dict without the sentinel).
    import tinker_cookbook.utils.ml_log as ml_log
    ml_log.MultiplexLogger.log_metrics = _samples_table_log_metrics

    # Replace PrettyPrintLogger's rich-table console output with a terse
    # `step=N reward=X.XXX time=X.Xs` one-liner, matching old pipeline/logging.py.
    # Wandb/JsonLogger children still get the full metrics dict.
    ml_log.PrettyPrintLogger.log_metrics = _terse_pretty_print_log_metrics

    # Verify the patch took effect at all known call sites
    assert train_module.compute_advantages is grpo_compute_advantages, (
        "Patch failed: train_module.compute_advantages was not replaced. "
        "tinker_cookbook may have changed its import structure."
    )
    assert train_module.assemble_training_data is grpo_assemble_training_data, (
        "Patch failed: train_module.assemble_training_data was not replaced. "
        "tinker_cookbook may have changed its import structure."
    )
    assert train_module.incorporate_kl_penalty is grpo_incorporate_kl_penalty, (
        "Patch failed: train_module.incorporate_kl_penalty was not replaced. "
        "tinker_cookbook may have changed its import structure."
    )
    assert train_module.compute_trajectory_metrics is strip_env_all_prefix, (
        "Patch failed: train_module.compute_trajectory_metrics was not replaced. "
        "tinker_cookbook may have changed its import structure."
    )
    assert ml_log.MultiplexLogger.log_metrics is _samples_table_log_metrics, (
        "Patch failed: MultiplexLogger.log_metrics was not replaced. "
        "tinker_cookbook may have changed its import structure."
    )
    assert ml_log.PrettyPrintLogger.log_metrics is _terse_pretty_print_log_metrics, (
        "Patch failed: PrettyPrintLogger.log_metrics was not replaced. "
        "tinker_cookbook may have changed its import structure."
    )
