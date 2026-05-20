"""Evaluate a checkpoint on the subproblem training/validation set.

Reports per-problem accuracy, mean reward, boxed-answer rate, and
answer frequency distributions.

Usage:
    python eval_subproblems.py --data ./conics-50.jsonl
    python eval_subproblems.py --data ./conics-50.jsonl --checkpoint "tinker://..."
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

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


def evaluate_subproblems(
    data_path: str,
    checkpoint_path: str | None,
    model_name: str,
    n_samples: int,
    temperature: float,
    max_tokens: int,
    output_path: str | None,
) -> dict:
    """Evaluate checkpoint on all subproblems."""
    records = load_subproblems(data_path)
    logger.info(f"Loaded {len(records)} subproblems from {data_path}")

    renderer_name = model_info.get_recommended_renderer_name(model_name)
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    service_client = tinker.ServiceClient()

    if checkpoint_path:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        sampling_client = service_client.create_sampling_client(model_path=checkpoint_path)
    else:
        logger.info(f"Evaluating base model: {model_name}")
        sampling_client = service_client.create_sampling_client(base_model=model_name)

    stop_sequences = renderer.get_stop_sequences()
    sampling_params = tinker.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop_sequences,
    )

    per_problem_results = []
    all_rewards = []

    for i, rec in enumerate(records):
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

        problem_rewards = []
        problem_answers = []
        problem_correct = 0
        problem_has_boxed = 0

        for seq in result.sequences:
            message, _ = renderer.parse_response(seq.tokens)
            content = renderers.get_text_content(message)
            reward = compute_reward(content, rec.answer)
            predicted = extract_boxed_answer(content)

            problem_rewards.append(reward)
            all_rewards.append(reward)
            if predicted is not None:
                problem_has_boxed += 1
                problem_answers.append(normalize_answer(predicted))
            if reward >= 1.0:
                problem_correct += 1

        answer_freq = Counter(problem_answers).most_common(5)

        per_problem_results.append({
            "id": rec.id or f"problem_{i}",
            "reference": rec.answer,
            "accuracy": problem_correct / n_samples,
            "mean_reward": sum(problem_rewards) / len(problem_rewards),
            "boxed_rate": problem_has_boxed / n_samples,
            "top_answers": {str(k): v for k, v in answer_freq},
        })

        if (i + 1) % 10 == 0 or i == len(records) - 1:
            logger.info(
                f"  [{i+1}/{len(records)}] acc={problem_correct/n_samples:.2f} "
                f"reward={sum(problem_rewards)/len(problem_rewards):.3f}"
            )

    # Aggregate metrics
    n_total = len(all_rewards)
    mean_reward = sum(all_rewards) / n_total if n_total > 0 else 0
    exact_match_rate = sum(1 for r in all_rewards if r >= 1.0) / n_total if n_total > 0 else 0
    boxed_rate = sum(1 for r in all_rewards if r > 0.0) / n_total if n_total > 0 else 0
    mean_per_problem_accuracy = (
        sum(p["accuracy"] for p in per_problem_results) / len(per_problem_results)
        if per_problem_results else 0
    )

    results = {
        "checkpoint": checkpoint_path or f"base:{model_name}",
        "data_path": data_path,
        "n_problems": len(records),
        "n_samples_per_problem": n_samples,
        "mean_reward": mean_reward,
        "exact_match_rate": exact_match_rate,
        "boxed_rate": boxed_rate,
        "mean_per_problem_accuracy": mean_per_problem_accuracy,
        "reward_1.0_frac": sum(1 for r in all_rewards if r >= 1.0) / n_total,
        "reward_0.01_frac": sum(1 for r in all_rewards if 0 < r < 1.0) / n_total,
        "reward_0.0_frac": sum(1 for r in all_rewards if r == 0.0) / n_total,
        "per_problem": per_problem_results,
    }

    # Print summary
    print("\n" + "=" * 60)
    print("SUBPROBLEM EVALUATION")
    print("=" * 60)
    print(f"Checkpoint:          {results['checkpoint']}")
    print(f"Problems:            {len(records)}")
    print(f"Samples/problem:     {n_samples}")
    print(f"Mean reward:         {mean_reward:.4f}")
    print(f"Exact-match rate:    {exact_match_rate:.4f}")
    print(f"Boxed-answer rate:   {boxed_rate:.4f}")
    print(f"Per-problem acc:     {mean_per_problem_accuracy:.4f}")
    print(f"Reward breakdown:    1.0={results['reward_1.0_frac']:.3f}  "
          f"0.01={results['reward_0.01_frac']:.3f}  "
          f"0.0={results['reward_0.0_frac']:.3f}")
    print("=" * 60 + "\n")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on subproblems")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to subproblem JSONL file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Tinker checkpoint path (omit for base model)")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    evaluate_subproblems(
        data_path=args.data,
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        n_samples=args.n_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
