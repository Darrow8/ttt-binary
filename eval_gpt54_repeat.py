#!/usr/bin/env python3
"""
Run GPT-5.4 (thinking/reasoning mode) N times on a single problem
to measure pass@k accuracy.

Usage:
    python eval_gpt54_repeat.py --problem-id 18 --n 100
    python eval_gpt54_repeat.py --problem-file problems/conics-100.jsonl --problem-id 18 --n 100
    python eval_gpt54_repeat.py --problem-text "Find ..." --reference "42" --n 50
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

import openai

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = "gpt-5.4"
REASONING_EFFORT = None  # None = no reasoning, or "low"/"medium"/"high"/"xhigh"
MAX_CONCURRENT = 20
DEFAULT_REFERENCE = "326400"

DEFAULT_PROBLEM = r"""
Let \(U \subset PH^0_{\mathbb{Z}}(\mathbb{P}^2,\mathcal{O}(2))\) be the space of smooth conics in \(\mathbb{P}^2\), and let \(Z \subset U^6\) be the closed subscheme parametrizing \(6\)-tuples \((C_1,\dots,C_6)\) with \(C_1\) tangent to \(C_2,\dots,C_6\). Let
\[
\pi : Z \to U^5
\]
be the map induced by the projection onto the last \(5\) coordinates, and let \(V \subset U^5\) be the dense open subscheme over which \(\pi\) is finite étale. Let
\[
L=\lim_{p\to\infty}\frac{1}{\#V(\mathbb{F}_p)}\sum_{x\in V(\mathbb{F}_p)} \#\pi^{-1}(x),
\]
that is, the limit of the average number of components of the space of conics tangent to \(5\) smooth conics over \(\mathbb{F}_p\), as \(p\) tends to infinity. Find \(\lfloor 100L \rfloor\).
"""

SOLVE_PROMPT = """\
## Problem

{problem}

## Instructions

Solve this problem step by step. You MUST show all of your reasoning, \
calculations, and intermediate steps IN YOUR RESPONSE — do not skip ahead \
to the answer. Think carefully and work through the math explicitly. \
Write out every key derivation.

Round your answer to 4 decimal places if necessary. \

After you have fully worked through the solution, write your final \
numerical answer on the very last line in exactly this format \
(including the double asterisks):

**ANSWER: <your numerical answer here>**

Your answer must be a number, not an expression. Begin your solution now.

"""


# ── Answer extraction ─────────────────────────────────────────────────────────
def extract_answer(solution: str) -> str:
    if not solution:
        return ""
    matches = re.findall(r"\*\*ANSWER:\s*(.+?)\*\*", solution, re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    boxed = re.findall(r"\\boxed\{([^}]+)\}", solution)
    if boxed:
        return boxed[-1].strip()
    m = re.search(
        r"(?:final answer|the answer)(?:\s+is)?[:\s]+([^\n.]+)",
        solution, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return ""


# ── Query GPT-5.4 ────────────────────────────────────────────────────────────
async def query_once(
    client: openai.AsyncOpenAI,
    prompt: str,
    sem: asyncio.Semaphore,
    idx: int,
    total: int,
    reasoning_effort: str | None = None,
    temperature: float = 0.7,
) -> dict:
    async with sem:
        t0 = time.time()
        try:
            kwargs = dict(
                model=MODEL,
                input=[{"role": "user", "content": prompt}],
            )
            if reasoning_effort:
                kwargs["reasoning"] = {"effort": reasoning_effort}
            if temperature is not None:
                kwargs["temperature"] = temperature
            resp = await client.responses.create(**kwargs)
            elapsed = time.time() - t0

            parts = []
            for item in resp.output:
                if item.type == "message":
                    for content in item.content:
                        if content.type == "output_text":
                            parts.append(content.text)
            text = "\n".join(parts)
            answer = extract_answer(text)

            print(f"  [{idx+1}/{total}] answer={answer!r}  ({elapsed:.1f}s)", flush=True)
            return {
                "sample_idx": idx,
                "answer": answer,
                "raw_output": text,
                "elapsed_s": round(elapsed, 1),
            }
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{total}] ERROR: {e}  ({elapsed:.1f}s)", flush=True)
            return {
                "sample_idx": idx,
                "answer": "",
                "raw_output": "",
                "elapsed_s": round(elapsed, 1),
                "error": str(e),
            }


# ── Save helper ───────────────────────────────────────────────────────────────
def save_atomic(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(description="Run GPT-5.4 N times on a single problem")
    parser.add_argument("--problem-file", default="problems/conics-100.jsonl",
                        help="JSONL file containing problems")
    parser.add_argument("--problem-id", type=int, default=None,
                        help="Problem ID from the JSONL file")
    parser.add_argument("--problem-text", type=str, default=None,
                        help="Raw problem text (overrides --problem-file/--problem-id)")
    parser.add_argument("--reference", type=str, default=None,
                        help="Reference answer (required if using --problem-text)")
    parser.add_argument("--n", type=int, default=100, help="Number of samples")
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENT,
                        help="Max concurrent API calls")
    parser.add_argument("--reasoning-effort", type=str, default=REASONING_EFFORT,
                        choices=["low", "medium", "high", "xhigh", "none"],
                        help="Reasoning effort level (default: none)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (default: not set)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    # Parse reasoning effort ("none" -> None)
    reasoning_effort = args.reasoning_effort
    if reasoning_effort == "none" or reasoning_effort is None:
        reasoning_effort = None

    # GPT-5.4: temperature only works with reasoning=none
    if reasoning_effort is not None and args.temperature is not None:
        print("Warning: GPT-5.4 does not support temperature with reasoning enabled. Ignoring temperature.")
        args.temperature = None

    # Load problem
    if args.problem_text:
        problem_text = args.problem_text
        reference = args.reference or ""
        problem_id = "custom"
    elif args.problem_id is not None:
        problems = []
        with open(args.problem_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    problems.append(json.loads(line))
        match = [p for p in problems if p["id"] == args.problem_id]
        if not match:
            print(f"Problem ID {args.problem_id} not found in {args.problem_file}")
            sys.exit(1)
        problem_text = match[0]["prompt"]
        reference = str(match[0]["reference"]).strip()
        problem_id = args.problem_id
    else:
        # Use default Steiner conics problem
        problem_text = DEFAULT_PROBLEM.strip()
        reference = args.reference or DEFAULT_REFERENCE
        problem_id = "default"

    # Output path
    if args.output:
        out_path = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"runs/gpt54_repeat/pid{problem_id}_{ts}.json"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Init
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set")
        sys.exit(1)

    client = openai.AsyncOpenAI(api_key=api_key)
    prompt = SOLVE_PROMPT.format(problem=problem_text)
    sem = asyncio.Semaphore(args.concurrency)

    print(f"Model:       {MODEL} (reasoning={REASONING_EFFORT})")
    print(f"Problem ID:  {problem_id}")
    print(f"Reference:   {reference}")
    print(f"Samples:     {args.n}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output:      {out_path}")
    print(f"Problem:     {problem_text[:120]}...")
    print()

    # Check for existing partial results to resume
    completed = []
    done_indices = set()
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                existing = json.load(f)
            completed = existing.get("results", [])
            done_indices = {r["sample_idx"] for r in completed}
            print(f"Resuming: {len(done_indices)} samples already done")
        except (json.JSONDecodeError, KeyError):
            pass

    remaining = [i for i in range(args.n) if i not in done_indices]
    if not remaining:
        print("All samples already completed!")
        return

    print(f"Running {len(remaining)} remaining samples...\n")

    # Run queries in batches of concurrency size, saving after each batch
    t0 = time.time()
    output_data = {
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "problem_id": problem_id,
        "problem_text": problem_text,
        "reference": reference,
        "n_samples": args.n,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed": False,
        "results": completed,
    }

    batch_size = args.concurrency
    for batch_start in range(0, len(remaining), batch_size):
        batch_indices = remaining[batch_start : batch_start + batch_size]
        tasks = [
            query_once(client, prompt, sem, idx, args.n,
                       reasoning_effort=reasoning_effort, temperature=args.temperature)
            for idx in batch_indices
        ]
        batch_results = await asyncio.gather(*tasks)
        completed.extend(batch_results)

        # Update and save after each batch
        output_data["results"] = sorted(completed, key=lambda r: r["sample_idx"])
        answers = [r["answer"] for r in completed if r.get("answer")]
        n_correct = sum(1 for a in answers if a == reference) if reference else 0
        output_data["progress"] = {
            "done": len(completed),
            "total": args.n,
            "n_correct": n_correct,
            "n_answered": len(answers),
            "accuracy": round(100 * n_correct / len(answers), 2) if answers else 0,
        }
        save_atomic(out_path, output_data)

        # Progress line
        elapsed = time.time() - t0
        print(f"  --- batch done: {len(completed)}/{args.n} samples, "
              f"{n_correct}/{len(answers)} correct ({100*n_correct/len(answers) if answers else 0:.1f}%), "
              f"{elapsed:.0f}s elapsed ---\n", flush=True)

    total_time = time.time() - t0

    # Final summary
    answers = [r["answer"] for r in completed if r.get("answer")]
    counter = Counter(answers)
    n_correct = sum(1 for a in answers if a == reference) if reference else 0
    errors = [r for r in completed if r.get("error")]

    output_data["completed"] = True
    output_data["finished_at"] = datetime.now(timezone.utc).isoformat()
    output_data["total_time_s"] = round(total_time, 1)
    output_data["summary"] = {
        "reference": reference,
        "n_samples": args.n,
        "n_answered": len(answers),
        "n_errors": len(errors),
        "n_correct": n_correct,
        "accuracy": round(100 * n_correct / len(answers), 2) if answers else 0,
        "answer_distribution": dict(counter.most_common()),
        "majority_answer": counter.most_common(1)[0][0] if counter else None,
    }
    save_atomic(out_path, output_data)

    print(f"\n{'='*60}")
    print(f"  Reference:    {reference}")
    print(f"  Samples:      {args.n} ({len(answers)} answered, {len(errors)} errors)")
    print(f"  Correct:      {n_correct}/{len(answers)} ({100*n_correct/len(answers):.1f}%)" if answers else "  No answers")
    print(f"  Majority:     {counter.most_common(1)[0] if counter else 'N/A'}")
    print(f"  Distribution: {dict(counter.most_common())}")
    print(f"  Total time:   {total_time:.1f}s")
    print(f"{'='*60}")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
