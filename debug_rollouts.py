"""Debug utility: sample a small number of rollouts and inspect the reward path.

Shows the full reward computation for each response, useful for verifying
parser behavior and reward assignment before running expensive training.

Usage:
    python debug_rollouts.py --data ./conics-50.jsonl --n-problems 3 --n-samples 4
    python debug_rollouts.py --data ./conics-50.jsonl --checkpoint "tinker://..."
"""

import argparse
import logging
from collections import Counter

import tinker
from dotenv import load_dotenv

load_dotenv()

from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from cookbook_grpo.parser import extract_boxed_answer, normalize_answer, answers_match
from cookbook_grpo.rewards import compute_reward
from cookbook_grpo.dataset import load_subproblems

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a careful and rigorous math student working through an advanced mathematics problem. Your goal is to solve the problem step by step.

Show all important intermediate reasoning, derivations, and calculations. Explain why each step is valid and reference any relevant theorems or identities when appropriate. Avoid skipping logical steps or making large jumps in reasoning.

If the problem involves multiple cases or approaches, consider them systematically. Use clear mathematical notation and keep the solution organized.

After completing the reasoning, clearly state the final answer."""


def debug_rollouts(
    data_path: str,
    checkpoint_path: str | None,
    model_name: str,
    n_problems: int,
    n_samples: int,
    temperature: float,
    max_tokens: int,
):
    records = load_subproblems(data_path)
    logger.info(f"Loaded {len(records)} subproblems, will debug first {n_problems}")

    renderer_name = model_info.get_recommended_renderer_name(model_name)
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    service_client = tinker.ServiceClient()

    if checkpoint_path:
        sampling_client = service_client.create_sampling_client(model_path=checkpoint_path)
    else:
        sampling_client = service_client.create_sampling_client(base_model=model_name)

    stop_sequences = renderer.get_stop_sequences()
    sampling_params = tinker.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop_sequences,
    )

    for i, rec in enumerate(records[:n_problems]):
        print("\n" + "=" * 80)
        print(f"PROBLEM {i+1}/{n_problems}")
        print("=" * 80)
        print(f"Problem:   {rec.problem[:200]}...")
        print(f"Reference: {rec.answer}")
        print("-" * 80)

        convo = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": rec.problem + " Put your final answer inside \\boxed{}."},
        ]
        prompt = renderer.build_generation_prompt(convo)

        future = sampling_client.sample(
            prompt=prompt,
            num_samples=n_samples,
            sampling_params=sampling_params,
        )
        result = future.result()

        rewards = []
        for j, seq in enumerate(result.sequences):
            message, _ = renderer.parse_response(seq.tokens)
            content = renderers.get_text_content(message)
            predicted = extract_boxed_answer(content)
            reward = compute_reward(content, rec.answer)
            rewards.append(reward)

            print(f"\n  Sample {j+1}/{n_samples}:")
            print(f"    Token count:    {len(seq.tokens)}")
            print(f"    Extracted:      {predicted or '(none)'}")
            print(f"    Reference:      {rec.answer}")
            if predicted:
                print(f"    Normalized:     '{normalize_answer(predicted)}' vs '{normalize_answer(rec.answer)}'")
                print(f"    Match:          {answers_match(predicted, rec.answer)}")
            print(f"    Reward:         {reward}")
            # Show last 300 chars of response for context
            tail = content[-300:] if len(content) > 300 else content
            print(f"    Response tail:  ...{tail}")

        print(f"\n  Group summary:")
        print(f"    Rewards: {rewards}")
        print(f"    Mean:    {sum(rewards)/len(rewards):.3f}")
        print(f"    Correct: {sum(1 for r in rewards if r >= 1.0)}/{n_samples}")
        mean = sum(rewards) / len(rewards)
        advantages = [r - mean for r in rewards]
        print(f"    Advantages (raw): {[f'{a:.3f}' for a in advantages]}")
        all_same = all(a == 0.0 for a in advantages)
        print(f"    Would skip (zero variance): {all_same}")

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Debug rollouts for subproblems")
    parser.add_argument("--data", type=str, default="./conics-50.jsonl")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--n-problems", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=16384)
    args = parser.parse_args()

    debug_rollouts(
        data_path=args.data,
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        n_problems=args.n_problems,
        n_samples=args.n_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
