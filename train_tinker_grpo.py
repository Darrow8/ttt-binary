"""Train GRPO on TTT-Discover subproblems using the tinker-cookbook RL stack.

Applies monkey-patches to match the original DeepSeek GRPO algorithm:
  - Std-normalized advantages (not just mean-centered)
  - Sequence-level loss normalization (1/|o_i|)
  - KL penalty against reference model (β=0.04)

Usage:
    python train_tinker_grpo.py --config=cookbook_grpo/configs/conics50.yaml
"""

import asyncio
import sys

import chz
import yaml
from dotenv import load_dotenv

load_dotenv()

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl import train

from cookbook_grpo.dataset import SubproblemDatasetBuilder
from cookbook_grpo.grpo_overrides import patch_grpo_advantages, patch_sparse_logging

patch_grpo_advantages()

DEFAULT_CONFIG = "cookbook_grpo/configs/conics50.yaml"


def config_from_yaml(yaml_path: str) -> train.Config:
    """Build a Config from a YAML file, handling nested dataset_builder."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    for key in ("learning_rate", "temperature", "kl_penalty_coef", "kl_discount_factor"):
        if key in cfg and isinstance(cfg[key], str):
            cfg[key] = float(cfg[key])

    # Extract our custom logging keys before passing to Config
    log_samples_every = cfg.pop("log_samples_every", 0)

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
        reward_wrong=ds_cfg.get("reward_wrong", 0.0),
        reward_none=ds_cfg.get("reward_none", 0.0),
        shuffle=ds_cfg.get("shuffle", True),
        seed=ds_cfg.get("seed", 42),
        eval_data_path=ds_cfg.get("eval_data_path"),
        weight_by_inverse_frequency=ds_cfg.get("weight_by_inverse_frequency", False),
    )

    cfg["dataset_builder"] = builder
    if "renderer_name" not in cfg:
        cfg["renderer_name"] = renderer_name

    kl_ref_dict = cfg.pop("kl_reference_config", None)
    if kl_ref_dict is not None:
        cfg["kl_reference_config"] = train.KLReferenceConfig(
            base_model=kl_ref_dict.get("base_model", model_name),
            load_checkpoint_path=kl_ref_dict.get("load_checkpoint_path"),
        )

    # Apply sparse logging patch based on config
    patch_sparse_logging(log_samples_every=log_samples_every)

    blueprint = chz.Blueprint(train.Config).apply(cfg)
    return blueprint.make()


def main(config: train.Config):
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("--config"):
        config_path = sys.argv[1].split("=", 1)[1] if "=" in sys.argv[1] else sys.argv[2]
    else:
        config_path = DEFAULT_CONFIG
        print(f"No --config specified, using default: {DEFAULT_CONFIG}")

    config = config_from_yaml(config_path)
    main(config)
