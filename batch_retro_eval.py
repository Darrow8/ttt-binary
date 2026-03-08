"""Batch evaluate gpt-oss-120b on retrosynthesis problems.

Usage:
    python batch_retro_eval.py [--n-samples 3] [--workers 10]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import threading
import time
from collections import Counter
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from rdkit import Chem


# ---------------------------------------------------------------------------
# Vertex AI client
# ---------------------------------------------------------------------------
REMOTE_MODEL = "openai/gpt-oss-120b-maas"


def _get_remote_client():
    from google.auth import default
    from google.auth.transport.requests import Request
    from openai import OpenAI

    credentials, _ = default()
    credentials.refresh(Request())
    token = credentials.token

    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    if location == "global":
        location = "us-central1"
    base_url = (
        f"https://{location}-aiplatform.googleapis.com/v1/"
        f"projects/{project}/locations/{location}/endpoints/openapi"
    )
    return OpenAI(api_key=token, base_url=base_url)


# ---------------------------------------------------------------------------
# SMILES helpers
# ---------------------------------------------------------------------------
def canonicalize(smiles: str) -> str | None:
    parts = smiles.strip().split(".")
    canonical = []
    for part in parts:
        mol = Chem.MolFromSmiles(part.strip())
        if mol is None:
            return None
        canonical.append(Chem.MolToSmiles(mol))
    return ".".join(sorted(canonical))


def extract_boxed(text: str) -> str:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    depth, start = 0, idx + len("\\boxed{")
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return ""


def extract_answer(text: str) -> str:
    """Try \\boxed{} first, then **ANSWER: ...**, then 'final answer' pattern."""
    ans = extract_boxed(text)
    if ans:
        return ans
    m = re.findall(r"\*\*ANSWER:\s*(.+?)\*\*", text, re.IGNORECASE)
    if m:
        return m[-1].strip()
    m = re.search(r"(?:final answer|the answer)(?:\s+is)?[:\s]+([^\n.]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def check_correct(predicted: str, reference: str) -> bool:
    # Clean up \text{} wrappers
    predicted = re.sub(r"\\text\{([^}]*)\}", r"\1", predicted)
    pred_canon = canonicalize(predicted)
    ref_canon = canonicalize(reference)
    if pred_canon is not None and ref_canon is not None:
        return pred_canon == ref_canon
    return predicted.strip() == reference.strip()


# ---------------------------------------------------------------------------
# Solve one problem once
# ---------------------------------------------------------------------------
def solve_once(client, problem_text: str, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=REMOTE_MODEL,
                messages=[{"role": "user", "content": problem_text}],
                temperature=0.7,
            )
            elapsed = time.time() - t0
            solution = resp.choices[0].message.content or ""
            answer = extract_answer(solution)
            return {"answer": answer, "reasoning": solution, "elapsed_s": round(elapsed, 2)}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return {"answer": "", "reasoning": "", "error": "all retries exhausted"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=3, help="Samples per problem")
    parser.add_argument("--workers", type=int, default=10, help="Parallel workers")
    parser.add_argument("--problems", default="problems/retrosynthesis.jsonl")
    args = parser.parse_args()

    # Load problems
    problems = []
    with open(args.problems) as f:
        for line in f:
            problems.append(json.loads(line))

    print(f"Model:    {REMOTE_MODEL}")
    print(f"Problems: {len(problems)}")
    print(f"Samples:  {args.n_samples} per problem")
    print(f"Workers:  {args.workers}")
    print()

    client = _get_remote_client()

    out_dir = os.path.join("runs", f"retro_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.json")

    results = []
    lock = threading.Lock()
    completed = [0]

    def eval_problem(idx, prob):
        answers = []
        correct_count = 0
        sample_results = []
        for s in range(args.n_samples):
            r = solve_once(client, prob["prompt"])
            ans = r["answer"]
            is_correct = check_correct(ans, prob["reference"]) if ans else False
            if is_correct:
                correct_count += 1
            answers.append(ans)
            sample_results.append({
                "answer": ans,
                "correct": is_correct,
                "elapsed_s": r.get("elapsed_s", 0),
            })

        result = {
            "problem_idx": idx,
            "product_smiles": prob.get("product_smiles", ""),
            "reference": prob["reference"],
            "reaction_type": prob.get("reaction_type", ""),
            "reaction_type_name": prob.get("reaction_type_name", ""),
            "difficulty_score": prob.get("difficulty_score", 0),
            "n_correct": correct_count,
            "n_samples": args.n_samples,
            "accuracy": correct_count / args.n_samples,
            "samples": sample_results,
        }

        with lock:
            results.append(result)
            completed[0] += 1
            status = "PASS" if correct_count > 0 else "FAIL"
            print(f"  [{completed[0]:3d}/{len(problems)}] {status} "
                  f"({correct_count}/{args.n_samples}) "
                  f"type={prob.get('reaction_type', '?')} "
                  f"diff={prob.get('difficulty_score', 0):.0f} "
                  f"product={prob.get('product_smiles', '')[:40]}...")
            # Save incrementally
            _save(out_path, results, problems, args)

        return result

    def _save(path, res, probs, a):
        sorted_res = sorted(res, key=lambda x: x["problem_idx"])
        n_problems_done = len(sorted_res)
        n_any_correct = sum(1 for r in sorted_res if r["n_correct"] > 0)
        data = {
            "model": REMOTE_MODEL,
            "n_problems": len(probs),
            "n_samples_per_problem": a.n_samples,
            "n_evaluated": n_problems_done,
            "n_any_correct": n_any_correct,
            "overall_accuracy": n_any_correct / n_problems_done if n_problems_done else 0,
            "results": sorted_res,
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

    t_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(eval_problem, i, p) for i, p in enumerate(problems)]
        concurrent.futures.wait(futures)
    total_time = time.time() - t_start

    # Final save & summary
    _save(out_path, results, problems, args)

    sorted_results = sorted(results, key=lambda x: x["problem_idx"])
    n_any_correct = sum(1 for r in sorted_results if r["n_correct"] > 0)
    n_all_correct = sum(1 for r in sorted_results if r["n_correct"] == args.n_samples)
    total_correct = sum(r["n_correct"] for r in sorted_results)
    total_samples = len(problems) * args.n_samples

    print(f"\n{'='*60}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Problems with >= 1 correct: {n_any_correct}/{len(problems)} ({n_any_correct/len(problems):.1%})")
    print(f"  Problems with all correct:  {n_all_correct}/{len(problems)} ({n_all_correct/len(problems):.1%})")
    print(f"  Sample-level accuracy:      {total_correct}/{total_samples} ({total_correct/total_samples:.1%})")
    print(f"\n  By reaction type:")
    by_type = {}
    for r in sorted_results:
        rt = r["reaction_type_name"]
        by_type.setdefault(rt, []).append(r)
    for rt, rs in sorted(by_type.items()):
        n_pass = sum(1 for r in rs if r["n_correct"] > 0)
        print(f"    {rt}: {n_pass}/{len(rs)} pass")
    print(f"\n  Results saved to {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
