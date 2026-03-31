#!/usr/bin/env python3
"""
Evaluate three frontier models on math problems and compare via majority vote.

Usage:
    python3 eval_frontier_models.py [--problems problems/conics-100.jsonl] [--output eval_results.json] [--samples 1]

Keys are loaded from .env:
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GCP_PROJECT, GCP_REGION
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

import anthropic
import openai
from google import genai
from google.genai.types import GenerateContentConfig

# ── Model config ──────────────────────────────────────────────────────────────
CLAUDE_MODEL = "claude-opus-4-6"
OPENAI_MODEL = "gpt-5.4"
OPENAI_REASONING_EFFORT = "xhigh"
GEMINI_MODEL = "gemini-3.1-pro-preview"

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

MAX_CONCURRENT = 10  # per-provider concurrency cap


# ── Answer extraction (from inference/infer.py) ──────────────────────────────
def extract_answer(solution: str) -> str:
    """Extract the answer via regex. Returns empty string if nothing found."""
    if not solution:
        return ""
    # **ANSWER: ...** pattern
    matches = re.findall(r"\*\*ANSWER:\s*(.+?)\*\*", solution, re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    # \boxed{...} fallback
    boxed = re.findall(r"\\boxed\{([^}]+)\}", solution)
    if boxed:
        return boxed[-1].strip()
    # "final answer is ..." fallback
    m = re.search(
        r"(?:final answer|the answer)(?:\s+is)?[:\s]+([^\n.]+)",
        solution, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return ""


# ── Provider wrappers ─────────────────────────────────────────────────────────
async def query_claude(client: anthropic.AsyncAnthropic, prompt: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        t0 = time.time()
        # Must use streaming for extended thinking (long requests)
        async with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=65536,
            thinking={
                "type": "adaptive",
            },
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            resp = await stream.get_final_message()
        print(f"    [claude] {time.time()-t0:.1f}s", flush=True)
        # Extract the text block (skip thinking blocks)
        for block in resp.content:
            if block.type == "text":
                return block.text
        return ""


async def query_openai(client: openai.AsyncOpenAI, prompt: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        t0 = time.time()
        resp = await client.responses.create(
            model=OPENAI_MODEL,
            reasoning={"effort": OPENAI_REASONING_EFFORT},
            input=[{"role": "user", "content": prompt}],
        )
        print(f"    [openai] {time.time()-t0:.1f}s", flush=True)
        # Extract text from response output items
        parts = []
        for item in resp.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        parts.append(content.text)
        return "\n".join(parts)


async def query_gemini(gemini_client: genai.Client, prompt: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        t0 = time.time()
        resp = await gemini_client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        print(f"    [gemini] {time.time()-t0:.1f}s", flush=True)
        return resp.text


# ── Main eval logic ───────────────────────────────────────────────────────────
async def evaluate_problem(
    problem: dict,
    claude_client: anthropic.AsyncAnthropic | None,
    openai_client: openai.AsyncOpenAI | None,
    gemini_client: genai.Client | None,
    sems: dict[str, asyncio.Semaphore],
    n_samples: int,
) -> dict:
    """Query all three models on a single problem, return results."""
    raw_problem = problem["prompt"]
    pid = problem["id"]
    reference = str(problem["reference"]).strip()

    # Format the prompt
    prompt = SOLVE_PROMPT.format(problem=raw_problem)

    tasks = {}
    for sample_idx in range(n_samples):
        if claude_client:
            tasks[f"claude_{sample_idx}"] = query_claude(claude_client, prompt, sems["claude"])
        if openai_client:
            tasks[f"openai_{sample_idx}"] = query_openai(openai_client, prompt, sems["openai"])
        if gemini_client:
            tasks[f"gemini_{sample_idx}"] = query_gemini(gemini_client, prompt, sems["gemini"])

    # Run all queries for this problem concurrently
    keys = list(tasks.keys())
    raw_results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    model_answers = {"claude": [], "openai": [], "gemini": []}
    raw_outputs = {"claude": [], "openai": [], "gemini": []}

    for key, result in zip(keys, raw_results):
        provider = key.rsplit("_", 1)[0]
        if isinstance(result, Exception):
            print(f"  [!] {provider} error on problem {pid}: {result}", file=sys.stderr)
            raw_outputs[provider].append(f"ERROR: {result}")
            continue
        raw_outputs[provider].append(result)
        ans = extract_answer(result)
        if ans:
            model_answers[provider].append(ans)

    # Per-model best answer (most common among its samples)
    all_answers = []
    per_model_best = {}
    for provider in ["claude", "openai", "gemini"]:
        if model_answers[provider]:
            best = Counter(model_answers[provider]).most_common(1)[0][0]
            per_model_best[provider] = best
            all_answers.append(best)
        else:
            per_model_best[provider] = None

    # Cross-model majority vote
    if all_answers:
        majority_answer = Counter(all_answers).most_common(1)[0][0]
    else:
        majority_answer = None

    correct = majority_answer == reference if majority_answer else False

    return {
        "id": pid,
        "prompt": raw_problem,
        "reference": reference,
        "model_answers": per_model_best,
        "majority_vote": majority_answer,
        "majority_agrees_with_reference": correct,
        "raw_outputs": raw_outputs,
    }


async def main():
    parser = argparse.ArgumentParser(description="Evaluate frontier models on math problems")
    parser.add_argument("--problems", default="problems/conics-50.jsonl", help="Path to JSONL problems file")
    parser.add_argument("--output", default="eval_results.json", help="Output JSON file")
    parser.add_argument("--samples", type=int, default=1, help="Number of samples per model per problem")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of problems to process concurrently")
    args = parser.parse_args()

    # Load problems
    all_problems = []
    with open(args.problems) as f:
        for line in f:
            line = line.strip()
            if line:
                all_problems.append(json.loads(line))

    # Resume: load existing results and skip already-completed problem IDs
    results = []
    full_output = args.output.replace(".json", "_full.json")
    done_ids = set()
    if os.path.exists(full_output):
        try:
            with open(full_output) as f:
                results = json.load(f)
            done_ids = {r["id"] for r in results}
            print(f"Resuming: found {len(done_ids)} completed problems in {full_output}")
        except (json.JSONDecodeError, KeyError):
            print("Warning: could not parse existing results, starting fresh")
            results = []

    problems = [p for p in all_problems if p["id"] not in done_ids]
    print(f"Loaded {len(all_problems)} problems from {args.problems}, {len(problems)} remaining")

    # Init clients
    claude_client = None
    openai_client = None
    gemini_client = None

    gcp_project = os.environ.get("GCP_PROJECT")
    gcp_region = os.environ.get("GCP_REGION", "us-central1")

    if False and os.environ.get("ANTHROPIC_API_KEY"):  # Claude disabled — unreliable responses
        claude_client = anthropic.AsyncAnthropic()
        print(f"  Claude: {CLAUDE_MODEL} (direct API)")
    else:
        print("  [SKIP] Claude -- set ANTHROPIC_API_KEY to enable")

    if os.environ.get("OPENAI_API_KEY"):
        openai_client = openai.AsyncOpenAI()
        print(f"  OpenAI: {OPENAI_MODEL}")
    else:
        print("  [SKIP] OpenAI -- set OPENAI_API_KEY to enable")

    if gcp_project:
        gemini_client = genai.Client(
            vertexai=True,
            project=gcp_project,
            location="global",
        )
        print(f"  Gemini: {GEMINI_MODEL} (Vertex AI, project={gcp_project}, region=global)")
    else:
        print("  [SKIP] Gemini -- set GCP_PROJECT to enable (uses Vertex AI + cloud credits)")

    if not claude_client and not openai_client and not gemini_client:
        print("\nNo API keys set. Export at least one of:")
        print("  OPENAI_API_KEY, GCP_PROJECT")
        sys.exit(1)

    sems = {
        "claude": asyncio.Semaphore(MAX_CONCURRENT),
        "openai": asyncio.Semaphore(MAX_CONCURRENT),
        "gemini": asyncio.Semaphore(MAX_CONCURRENT),
    }

    # Process in batches, saving incrementally
    t0 = time.time()

    def save_progress():
        """Save current results to disk after each batch."""
        n_correct = sum(1 for r in results if r["majority_agrees_with_reference"])
        summary = [{k: v for k, v in r.items() if k != "raw_outputs"} for r in results]
        output_data = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "models": {"claude": CLAUDE_MODEL, "openai": OPENAI_MODEL, "gemini": GEMINI_MODEL},
                "n_problems": len(results),
                "n_total": len(all_problems),
                "n_correct_majority": n_correct,
                "accuracy_majority": round(100 * n_correct / len(results), 2) if results else 0,
                "samples_per_model": args.samples,
                "completed": len(results) == len(all_problems),
                "elapsed_s": round(time.time() - t0, 1),
            },
            "results": summary,
        }
        tmp = args.output + ".tmp"
        with open(tmp, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, args.output)
        # Also save full results with raw outputs
        tmp_full = full_output + ".tmp"
        with open(tmp_full, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        os.replace(tmp_full, full_output)

    for i in range(0, len(problems), args.batch_size):
        batch = problems[i : i + args.batch_size]
        batch_results = await asyncio.gather(
            *[evaluate_problem(p, claude_client, openai_client, gemini_client, sems, args.samples) for p in batch]
        )
        results.extend(batch_results)
        save_progress()

        n_done = len(results)
        n_correct = sum(1 for r in results if r["majority_agrees_with_reference"])
        elapsed = time.time() - t0
        print(f"  [{n_done}/{len(all_problems)}] correct so far: {n_correct}/{n_done} "
              f"({100*n_correct/n_done:.1f}%) -- {elapsed:.0f}s elapsed")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_correct = sum(1 for r in results if r["majority_agrees_with_reference"])
    print(f"\n{'='*60}")
    print(f"RESULTS: {n_correct}/{len(results)} correct by majority vote ({100*n_correct/len(results):.1f}%)")

    # Per-model accuracy
    for provider in ["claude", "openai", "gemini"]:
        answered = [(r["model_answers"].get(provider), r["reference"]) for r in results if r["model_answers"].get(provider)]
        if answered:
            acc = sum(1 for a, ref in answered if a == ref)
            print(f"  {provider:>8}: {acc}/{len(answered)} ({100*acc/len(answered):.1f}%)")
        else:
            print(f"  {provider:>8}: skipped")

    # Agreement stats
    agree_count = 0
    disagree_examples = []
    for r in results:
        answers = [v for v in r["model_answers"].values() if v is not None]
        if len(set(answers)) <= 1 and len(answers) > 1:
            agree_count += 1
        elif len(set(answers)) > 1:
            disagree_examples.append(r)

    print(f"\nModel agreement: {agree_count}/{len(results)} problems")
    if disagree_examples:
        print(f"Disagreements on {len(disagree_examples)} problems:")
        for r in disagree_examples[:10]:
            print(f"  id={r['id']}: {r['model_answers']} (ref={r['reference']})")

    print(f"\nResults saved to {args.output} and {full_output}")


if __name__ == "__main__":
    asyncio.run(main())
