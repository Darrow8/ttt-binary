"""Evaluate a checkpoint on the FrontierMath target problem.

Samples N completions and measures the solve rate (fraction matching answer 866).

Usage:
    # Evaluate base model
    python eval_target_problem.py

    # Evaluate a specific checkpoint
    python eval_target_problem.py --checkpoint "tinker://path/to/checkpoint"

    # Override sample count
    python eval_target_problem.py --n-samples 100 --checkpoint "tinker://..."
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_ANSWER = "866"

TARGET_PROBLEM = r"""A classical result of Chasles, completed by the work of Bézout, Schubert, and others using the method of characteristics (later made rigorous by intersection theory), determines the number of smooth conics tangent to five given general smooth conics in the complex projective plane $\mathbb{P}^2_\mathbb{C}$. Determine this number. Give just the integer as your answer."""

SYSTEM_PROMPT = """You are a careful and rigorous math student working through an advanced mathematics problem. Your goal is to solve the problem step by step.

Show all important intermediate reasoning, derivations, and calculations. Explain why each step is valid and reference any relevant theorems or identities when appropriate. Avoid skipping logical steps or making large jumps in reasoning.

If the problem involves multiple cases or approaches, consider them systematically. Use clear mathematical notation and keep the solution organized.

After completing the reasoning, clearly state the final answer."""


def evaluate_checkpoint(
    checkpoint_path: str | None,
    model_name: str,
    n_samples: int,
    temperature: float,
    max_tokens: int,
    output_path: str | None,
) -> dict:
    """Sample completions and measure solve rate."""
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

    convo = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TARGET_PROBLEM + " Put your final answer inside \\boxed{}."},
    ]
    prompt = renderer.build_generation_prompt(convo)
    stop_sequences = renderer.get_stop_sequences()

    sampling_params = tinker.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop_sequences,
    )

    # Sample in batches to avoid overwhelming the API
    batch_size = min(50, n_samples)
    all_answers: list[str | None] = []
    all_correct: list[bool] = []

    logger.info(f"Sampling {n_samples} completions (batch_size={batch_size})...")

    remaining = n_samples
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        future = sampling_client.sample(
            prompt=prompt,
            num_samples=current_batch,
            sampling_params=sampling_params,
        )
        result = future.result()

        for seq in result.sequences:
            message, _ = renderer.parse_response(seq.tokens)
            content = renderers.get_text_content(message)
            predicted = extract_boxed_answer(content)
            all_answers.append(normalize_answer(predicted) if predicted else None)
            all_correct.append(
                predicted is not None and answers_match(predicted, TARGET_ANSWER)
            )

        remaining -= current_batch
        logger.info(
            f"  Progress: {n_samples - remaining}/{n_samples} sampled, "
            f"{sum(all_correct)} correct so far"
        )

    # Compute metrics
    correct_count = sum(all_correct)
    solve_rate = correct_count / n_samples
    parseable = sum(1 for a in all_answers if a is not None)
    parseable_rate = parseable / n_samples

    answer_counts = Counter(a for a in all_answers if a is not None)
    top_answers = answer_counts.most_common(20)

    results = {
        "checkpoint": checkpoint_path or f"base:{model_name}",
        "target_answer": TARGET_ANSWER,
        "n_samples": n_samples,
        "correct_count": correct_count,
        "solve_rate": solve_rate,
        "parseable_count": parseable,
        "parseable_rate": parseable_rate,
        "no_answer_count": n_samples - parseable,
        "answer_frequency": {str(k): v for k, v in top_answers},
    }

    # Print results
    print("\n" + "=" * 60)
    print("TARGET PROBLEM EVALUATION")
    print("=" * 60)
    print(f"Checkpoint:     {results['checkpoint']}")
    print(f"Target answer:  {TARGET_ANSWER}")
    print(f"Samples:        {n_samples}")
    print(f"Correct:        {correct_count}/{n_samples} = {solve_rate:.4f}")
    print(f"Parseable:      {parseable}/{n_samples} = {parseable_rate:.4f}")
    print(f"No answer:      {n_samples - parseable}")
    print("\nTop answers:")
    for ans, count in top_answers[:10]:
        marker = " <-- CORRECT" if normalize_answer(ans) == normalize_answer(TARGET_ANSWER) else ""
        print(f"  {ans:>12s}: {count:4d} ({count/n_samples:.3f}){marker}")
    print("=" * 60 + "\n")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on target problem")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Tinker checkpoint path (omit for base model)")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save JSON results")
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        n_samples=args.n_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
