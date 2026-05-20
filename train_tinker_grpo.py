"""Train GRPO on TTT-Discover subproblems using the tinker-cookbook RL stack.

Applies monkey-patches to match the original DeepSeek GRPO algorithm:
  - Std-normalized advantages (not just mean-centered)
  - Sequence-level loss normalization (1/|o_i|)
  - KL penalty against reference model (β=0.04)

Usage:
    python train_tinker_grpo.py --config configs/conics50.yaml

    # Or directly with CLI overrides:
    python train_tinker_grpo.py \
        --dataset_builder.data_path=./conics-50.jsonl \
        --learning_rate=1e-4 \
        --max_tokens=16384
"""

import asyncio
import sys

import chz
from dotenv import load_dotenv

load_dotenv()

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl import train

from cookbook_grpo.dataset import SubproblemDatasetBuilder
from cookbook_grpo.grpo_overrides import patch_grpo_advantages

patch_grpo_advantages()


SYSTEM_PROMPT = """You are a careful and rigorous math student working through an advanced mathematics problem. Your goal is to solve the problem step by step.

Show all important intermediate reasoning, derivations, and calculations. Explain why each step is valid and reference any relevant theorems or identities when appropriate. Avoid skipping logical steps or making large jumps in reasoning.

If the problem involves multiple cases or approaches, consider them systematically. Use clear mathematical notation and keep the solution organized.

After completing the reasoning, clearly state the final answer."""


def build_config_blueprint() -> chz.Blueprint[train.Config]:
    model_name = "openai/gpt-oss-120b"
    renderer_name = model_info.get_recommended_renderer_name(model_name)

    builder = SubproblemDatasetBuilder(
        data_path="./conics-50.jsonl",
        batch_size=25,
        group_size=16,
        num_epochs=50,
        model_name=model_name,
        renderer_name=renderer_name,
        prompt_suffix=" Put your final answer inside \\boxed{}.",
        system_prompt=SYSTEM_PROMPT,
        reward_correct=1.0,
        reward_wrong=0.01,
        reward_none=0.0,
        shuffle=True,
        seed=42,
    )

    return chz.Blueprint(train.Config).apply(
        {
            "model_name": model_name,
            "renderer_name": renderer_name,
            "log_path": "./cookbook-grpo-run",
            "dataset_builder": builder,
            "learning_rate": 1e-4,
            "max_tokens": 16384,
            "lora_rank": 32,
            "save_every": 5,
            "eval_every": 10,
            "temperature": 1.0,
            "wandb_project": "ttt-cookbook-conics50",
            "loss_fn": "importance_sampling",
            "remove_constant_reward_groups": True,
            "kl_penalty_coef": 0.04,
            "kl_discount_factor": 0.0,
            "kl_reference_config": train.KLReferenceConfig(base_model=model_name),
        }
    )


def main(config: train.Config):
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


def config_from_yaml(yaml_path: str) -> train.Config:
    """Build a Config from a YAML file, handling nested dataset_builder."""
    import yaml

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    # PyYAML safe_load doesn't parse "1e-4" as float; coerce known float fields
    for key in ("learning_rate", "temperature", "kl_penalty_coef", "kl_discount_factor"):
        if key in cfg and isinstance(cfg[key], str):
            cfg[key] = float(cfg[key])

    # Extract dataset_builder dict and build the object
    ds_cfg = cfg.pop("dataset_builder", {})
    model_name = cfg.get("model_name", "openai/gpt-oss-120b")
    renderer_name = cfg.get("renderer_name") or model_info.get_recommended_renderer_name(model_name)

    builder = SubproblemDatasetBuilder(
        data_path=ds_cfg.get("data_path", "./conics-50.jsonl"),
        batch_size=ds_cfg.get("batch_size", 25),
        group_size=ds_cfg.get("group_size", 16),
        num_epochs=ds_cfg.get("num_epochs", 50),
        model_name=model_name,
        renderer_name=renderer_name,
        prompt_suffix=ds_cfg.get("prompt_suffix", " Put your final answer inside \\boxed{}."),
        system_prompt=ds_cfg.get("system_prompt"),
        reward_correct=ds_cfg.get("reward_correct", 1.0),
        reward_wrong=ds_cfg.get("reward_wrong", 0.01),
        reward_none=ds_cfg.get("reward_none", 0.0),
        shuffle=ds_cfg.get("shuffle", True),
        seed=ds_cfg.get("seed", 42),
        eval_data_path=ds_cfg.get("eval_data_path"),
    )

    cfg["dataset_builder"] = builder
    if "renderer_name" not in cfg:
        cfg["renderer_name"] = renderer_name

    # Build KLReferenceConfig from nested dict if present
    kl_ref_dict = cfg.pop("kl_reference_config", None)
    if kl_ref_dict is not None:
        cfg["kl_reference_config"] = train.KLReferenceConfig(
            base_model=kl_ref_dict.get("base_model", model_name),
            load_checkpoint_path=kl_ref_dict.get("load_checkpoint_path"),
        )

    blueprint = chz.Blueprint(train.Config).apply(cfg)
    return blueprint.make()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("--config"):
        config_path = sys.argv[1].split("=", 1)[1] if "=" in sys.argv[1] else sys.argv[2]
        config = config_from_yaml(config_path)
    else:
        blueprint = build_config_blueprint()
        blueprint.make_from_argv(sys.argv[1:])
        config = blueprint.make()
    main(config)
