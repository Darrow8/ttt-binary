"""
GRPOTrainer — reusable GRPO training loop on the Tinker API.

Accepts any list of :class:`Problem` instances and any :class:`RewardFunc`.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from tqdm import tqdm

from pipeline.logging import MetricsLogger
from pipeline.problems import Problem
from pipeline.rewards import RewardFunc, extract_boxed, extract_answer_tag

logger = logging.getLogger(__name__)


def _extract_predicted(response: str) -> str:
    """Best-effort extraction of the predicted answer for display."""
    try:
        return extract_answer_tag(response)
    except ValueError:
        pass
    try:
        return extract_boxed(response)
    except ValueError:
        pass
    return "(no answer extracted)"


class _Renderer:
    """Thin wrapper: chat-template → token ids."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def build_prompt(self, messages: list[dict]) -> types.ModelInput:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return types.ModelInput.from_ints(ids)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


@dataclass
class GRPOConfig:
    model_name: str = "openai/gpt-oss-120b"
    log_dir: str = "/tmp/tinker-grpo/run"

    batch_size: int = 128
    group_size: int = 16
    learning_rate: float = 4e-5
    lora_rank: int = 32
    max_tokens: int = 256

    save_every: int = 20
    ttl_seconds: int = 604_800

    wandb_project: str | None = "grpo-tinker"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None

    temperature: float = 1.0
    system_prompt: str | None = None
    few_shot: list[dict] = field(default_factory=list)
    prompt_suffix: str = ""
    override_chat_template: bool = False

    num_sample_rows: int = 10

    def __post_init__(self):
        if self.wandb_entity is None:
            self.wandb_entity = os.environ.get("WANDB_ENTITY")


class GRPOTrainer:
    """End-to-end GRPO trainer: problems in, fine-tuned model out."""

    def __init__(self, config: GRPOConfig, reward_fn: RewardFunc):
        self.cfg = config
        self.reward_fn = reward_fn
        self.log_path = Path(config.log_dir)

        self._init_logger()
        self._init_tokenizer()
        self._init_tinker()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_logger(self):
        self.ml_logger = MetricsLogger(
            log_dir=self.log_path,
            wandb_project=self.cfg.wandb_project,
            wandb_entity=self.cfg.wandb_entity,
            wandb_run_name=self.cfg.wandb_run_name,
            config={
                "model_name": self.cfg.model_name,
                "batch_size": self.cfg.batch_size,
                "group_size": self.cfg.group_size,
                "learning_rate": self.cfg.learning_rate,
                "lora_rank": self.cfg.lora_rank,
                "max_tokens": self.cfg.max_tokens,
                "reward_fn": getattr(self.reward_fn, "__qualname__", str(self.reward_fn)),
            },
        )

    _SIMPLE_CHAT_TEMPLATE = (
        "{%- for message in messages -%}"
        "<|start|>{{ message['role'] }}<|message|>{{ message['content'] }}<|end|>"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}<|start|>assistant{%- endif -%}"
    )

    def _init_tokenizer(self):
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login as hf_login
            hf_login(token=hf_token)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        if self.cfg.override_chat_template:
            self.tokenizer.chat_template = self._SIMPLE_CHAT_TEMPLATE
            logger.info("Overriding chat template (no injected default system msg)")
        self.renderer = _Renderer(self.tokenizer)

    def _init_tinker(self):
        if not os.environ.get("TINKER_API_KEY"):
            raise RuntimeError(
                "Set TINKER_API_KEY in your .env or environment. "
                "Get one at https://tinker-console.thinkingmachines.ai/"
            )
        self.service = tinker.ServiceClient()
        self.training_client = self.service.create_lora_training_client(
            base_model=self.cfg.model_name,
            rank=self.cfg.lora_rank,
        )
        self.sampling_params = tinker.types.SamplingParams(
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
        )
        self.adam_params = types.AdamParams(
            learning_rate=self.cfg.learning_rate,
            beta1=0.9, beta2=0.95, eps=1e-8,
        )

    # ------------------------------------------------------------------
    # Build the chat prompt for a single problem
    # ------------------------------------------------------------------

    def _build_messages(self, problem: Problem) -> list[dict]:
        msgs: list[dict] = []
        if self.cfg.system_prompt:
            msgs.append({"role": "system", "content": self.cfg.system_prompt})
        msgs.extend(self.cfg.few_shot)
        msgs.append({"role": "user", "content": problem.prompt + self.cfg.prompt_suffix})
        return msgs

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def train(self, problems: list[Problem], *, epochs: int = 1):
        """Run GRPO training over *problems* for *epochs* passes."""
        cfg = self.cfg
        n_batches = len(problems) // cfg.batch_size
        if n_batches == 0:
            raise ValueError(
                f"Not enough problems ({len(problems)}) for batch_size={cfg.batch_size}. "
                "Lower batch_size or add more problems."
            )

        logger.info(
            "GRPO training  model=%s  problems=%d  batches=%d  epochs=%d",
            cfg.model_name, len(problems), n_batches, epochs,
        )

        global_step = 0
        for epoch in range(epochs):
            for batch_idx in range(n_batches):
                start = batch_idx * cfg.batch_size
                batch = problems[start : start + cfg.batch_size]
                self._train_step(batch, global_step, n_batches * epochs)
                global_step += 1

                if cfg.save_every and global_step > 0 and global_step % cfg.save_every == 0:
                    self._save_checkpoint(f"ckpt-{global_step:06d}")

        self._save_checkpoint("ckpt-final")
        self.ml_logger.close()
        logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def _train_step(self, batch: list[Problem], step: int, total_steps: int):
        t0 = time.time()
        cfg = self.cfg
        metrics: dict[str, float] = {
            "progress/step": step,
            "progress/done_frac": (step + 1) / total_steps,
            "optim/lr": cfg.learning_rate,
        }

        sampling_client = self.training_client.save_weights_and_get_sampling_client()

        # ---- Sampling ------------------------------------------------
        futures, prompts = [], []
        for problem in batch:
            msgs = self._build_messages(problem)
            prompt = self.renderer.build_prompt(msgs)
            future = sampling_client.sample(
                prompt=prompt,
                num_samples=cfg.group_size,
                sampling_params=self.sampling_params,
            )
            futures.append(future)
            prompts.append(prompt)

        # ---- Rewards & advantages ------------------------------------
        datums: list[types.Datum] = []
        all_rewards: list[float] = []
        all_advantages: list[float] = []
        all_token_counts: list[int] = []
        skipped = 0
        sample_rows: list[dict] = []

        for future, prompt, problem in tqdm(
            zip(futures, prompts, batch),
            total=len(futures),
            desc=f"Step {step}",
        ):
            result = future.result()

            rewards_g: list[float] = []
            tokens_g: list[list[int]] = []
            logprobs_g: list[list[float]] = []
            decoded_g: list[str] = []

            for seq in result.sequences:
                tokens_g.append(seq.tokens)
                logprobs_g.append(seq.logprobs)
                all_token_counts.append(len(seq.tokens))
                decoded = self.renderer.decode(seq.tokens)
                decoded_g.append(decoded)
                rewards_g.append(self.reward_fn(decoded, problem))

            mean_r = sum(rewards_g) / len(rewards_g)
            advantages_g = [r - mean_r for r in rewards_g]
            all_rewards.extend(rewards_g)
            all_advantages.extend(advantages_g)

            for i, (resp, rew) in enumerate(zip(decoded_g, rewards_g)):
                predicted = _extract_predicted(resp)
                sample_rows.append({
                    "prompt": problem.prompt,
                    "response": resp,
                    "expected": problem.reference,
                    "predicted": predicted,
                    "reward": rew,
                    "correct": rew >= 1.0,
                })

            if all(a == 0.0 for a in advantages_g):
                skipped += 1
                continue

            ob_len = prompt.length - 1
            for toks, lps, adv in zip(tokens_g, logprobs_g, advantages_g):
                model_input = prompt.append(types.EncodedTextChunk(tokens=toks[:-1]))
                target_tokens = [0] * ob_len + toks
                padded_logprobs = [0.0] * ob_len + lps
                padded_advantages = [0.0] * ob_len + [adv] * (model_input.length - ob_len)

                datums.append(types.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                    },
                ))

        # ---- Policy update -------------------------------------------
        fwd_bwd = self.training_client.forward_backward(datums, loss_fn="importance_sampling")
        optim = self.training_client.optim_step(self.adam_params)
        fwd_bwd.result()
        optim_result = optim.result()

        if optim_result.metrics:
            metrics.update(optim_result.metrics)

        n = max(len(all_rewards), 1)
        metrics["time/total"] = time.time() - t0
        metrics["reward/mean"] = sum(all_rewards) / n
        metrics["reward/max"] = max(all_rewards) if all_rewards else 0.0
        metrics["reward/min"] = min(all_rewards) if all_rewards else 0.0
        metrics["advantage/std"] = (sum(a ** 2 for a in all_advantages) / max(len(all_advantages), 1)) ** 0.5
        metrics["tokens/mean_per_completion"] = sum(all_token_counts) / max(len(all_token_counts), 1)
        metrics["problems/skipped_frac"] = skipped / max(len(futures), 1)
        metrics["datums/count"] = len(datums)

        self.ml_logger.log(
            metrics, step=step,
            reward_list=all_rewards,
            advantage_list=all_advantages,
            samples=sample_rows,
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, name: str):
        path = str(self.log_path / name)
        response = self.training_client.save_state(path, ttl_seconds=self.cfg.ttl_seconds).result()
        logger.info("Checkpoint → %s  (tinker_path: %s)", path, response.path)
