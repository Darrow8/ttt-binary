"""Metrics logging to JSONL and Weights & Biases."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Dual-sink logger: local JSONL file + optional W&B run."""

    def __init__(
        self,
        log_dir: Path,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        wandb_run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        log_dir.mkdir(parents=True, exist_ok=True)
        self._file = open(log_dir / "metrics.jsonl", "a")
        self._wandb_run = None

        if wandb_project:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=wandb_run_name,
                    config=config or {},
                    reinit=True,
                )
                logger.info("W&B logging → %s/%s", wandb_entity or "(default)", wandb_project)
            except Exception as exc:
                logger.warning("W&B init failed (%s); continuing without it", exc)

    def log(
        self,
        metrics: dict[str, float],
        step: int,
        reward_list: list[float] | None = None,
        advantage_list: list[float] | None = None,
        samples: list[dict] | None = None,
    ):
        record = {"step": step, **metrics}
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

        logger.info(
            "step=%d  reward=%.3f  time=%.1fs",
            step,
            metrics.get("reward/mean", 0),
            metrics.get("time/total", 0),
        )

        if self._wandb_run is not None:
            import wandb

            log_dict: dict[str, Any] = {**metrics}

            if reward_list:
                log_dict["reward/histogram"] = wandb.Histogram(reward_list)
            if advantage_list:
                log_dict["advantage/histogram"] = wandb.Histogram(advantage_list)
            if samples:
                columns = ["prompt", "response", "expected", "predicted", "reward", "correct"]
                rows = [
                    [s["prompt"], s["response"], s["expected"], s["predicted"], s.get("reward", 0), s["correct"]]
                    for s in samples
                ]
                log_dict[f"samples_{step}"] = wandb.Table(columns=columns, data=rows)

            wandb.log(log_dict, step=step)

    def close(self):
        self._file.close()
        if self._wandb_run is not None:
            import wandb
            wandb.finish()
