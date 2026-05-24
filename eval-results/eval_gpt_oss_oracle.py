#!/usr/bin/env python3
"""
Run each problem in a dataset N times with GPT-OSS-120B (Vertex AI) and
record the majority-vote answer as the pseudo-label reference.

Usage:
    python3 eval-results/eval_gpt_oss_oracle.py problems/conics-50.jsonl
    python3 eval-results/eval_gpt_oss_oracle.py problems/conics-50.jsonl \
        --n-samples 20 --workers 10 --out-dir eval-results/oracle_results

Reads GCP credentials from Application Default Credentials (gcloud auth).
Reads GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION from .env or shell.
"""

import sys
print("[debug] python started", flush=True)

import argparse
import concurrent.futures
import json
import os
import re
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

print("[debug] stdlib imports done", flush=True)

from dotenv import load_dotenv
print("[debug] dotenv imported", flush=True)
load_dotenv()
print("[debug] dotenv loaded", flush=True)

from tenacity import retry, stop_after_attempt, wait_random_exponential
print("[debug] tenacity imported", flush=True)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = "openai/gpt-oss-120b-maas"
TEMPERATURE = 0.7
DEFAULT_N_SAMPLES = 20
DEFAULT_WORKERS = 10

SOLVE_PROMPT = """\
## Problem

{problem}

## Instructions

Solve this problem step by step. You MUST show all of your reasoning, \
calculations, and intermediate steps IN YOUR RESPONSE — do not skip ahead \
to the answer. Think carefully and work through the math explicitly. \
Write out every key derivation.

After you have fully worked through the solution, write your final answer \
on the very last line in exactly this format:

\\boxed{{<answer>}}

For example: \\boxed{{0.6079}} or \\boxed{{42}}

If the answer is a finite number, write it as a decimal rounded to 4 decimal \
places (e.g. 0.6079, 42.0000). Do NOT write symbolic expressions like 1/π² or √2. \
If the answer is infinite, write \\boxed{{\\infty}} or \\boxed{{-\\infty}}. \
If the problem is ill-posed or the answer does not exist, write \\boxed{{\\text{{undefined}}}}.

"""


# ── Answer normalization ──────────────────────────────────────────────────────
def normalize_answer(ans: str) -> str:
    """Normalize numeric answers so '32.0000' and '32' count as the same."""
    try:
        f = float(ans.replace(",", ""))
        return str(int(f)) if f == int(f) else str(round(f, 4))
    except (ValueError, AttributeError):
        return ans  # keep non-numeric as-is (\\infty, \\text{undefined}, etc.)


# ── Answer extraction ─────────────────────────────────────────────────────────
def _extract_boxed_raw(text: str) -> str:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    depth, start = 0, idx + len("\\boxed{")
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth == 0:
                return text[start:i]
            depth -= 1
    return ""


def extract_answer(solution: str) -> str:
    if not solution:
        return ""
    raw = _extract_boxed_raw(solution)
    if raw:
        return raw.strip()
    m = re.search(
        r"(?:final answer|the answer)(?:\s+is)?[:\s]+([^\n.]+)",
        solution, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return ""


# ── Dataset loading ───────────────────────────────────────────────────────────
def load_dataset(path: str) -> list[dict]:
    p = Path(path)
    text = p.read_text()
    raw_items: list[dict] = []
    if path.endswith(".jsonl"):
        for line in text.splitlines():
            line = line.strip()
            if line:
                raw_items.append(json.loads(line))
    else:
        data = json.loads(text)
        raw_items = data if isinstance(data, list) else data.get("problems", [])

    out = []
    for i, item in enumerate(raw_items):
        prompt = item.get("prompt") or item.get("problem")
        reference = item.get("reference") or item.get("ground_truth_answer")
        ident = item.get("id", i)
        out.append({
            "id": ident,
            "prompt": prompt,
            "reference": str(reference).strip() if reference is not None else None,
        })
    return out


# ── Vertex AI client ──────────────────────────────────────────────────────────
def _get_remote_client():
    from google.auth import default
    from google.auth.transport.requests import Request
    from openai import OpenAI

    print("Authenticating with GCP...", flush=True)
    credentials, _ = default()
    credentials.refresh(Request())
    print("Authenticated.", flush=True)

    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    host = (
        "aiplatform.googleapis.com"
        if location == "global"
        else f"{location}-aiplatform.googleapis.com"
    )
    base_url = f"https://{host}/v1/projects/{project}/locations/{location}/endpoints/openapi"
    return OpenAI(api_key=credentials.token, base_url=base_url)


# ── Single sample ─────────────────────────────────────────────────────────────
def _solve_once(client, problem_prompt: str, sample_idx: int) -> dict:
    prompt = SOLVE_PROMPT.format(problem=problem_prompt)

    @retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(8))
    def _call():
        return client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
        )

    try:
        t0 = time.time()
        resp = _call()
        elapsed = time.time() - t0
        msg = resp.choices[0].message if resp.choices else None
        content = (msg.content or "") if msg else ""
        reasoning = getattr(msg, "reasoning_content", None) or ""
        solution = (reasoning + "\n\n" + content) if reasoning else content
        answer = extract_answer(solution)
        print(f"    sample {sample_idx+1}: {answer!r}  ({elapsed:.1f}s)", flush=True)
        return {"sample_idx": sample_idx, "answer": answer, "elapsed_s": round(elapsed, 2)}
    except Exception as e:
        print(f"    sample {sample_idx+1}: ERROR {e}", flush=True)
        return {"sample_idx": sample_idx, "answer": "", "elapsed_s": 0, "error": str(e)}


def _build_problem_result(problem: dict, sample_results: list[dict], n_samples: int) -> dict:
    answers = [r["answer"] for r in sample_results if r.get("answer")]
    norm_answers = [normalize_answer(a) for a in answers]
    counter = Counter(norm_answers)
    best = counter.most_common(1)[0][0] if counter else None
    best_count = counter[best] if best else 0
    print(
        f"  id={problem['id']}: {best!r} ({best_count}/{n_samples})  "
        f"dist={dict(counter.most_common(5))}", flush=True,
    )
    return {
        "id": problem["id"],
        "prompt": problem["prompt"],
        "reference": problem["reference"],
        "gpt_oss_answers": answers,
        "gpt_oss_best_answer": best,
        "gpt_oss_best_count": best_count,
        "gpt_oss_n_samples": n_samples,
        "gpt_oss_answer_dist": dict(counter),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--out-dir", default="eval-results/oracle_results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = _get_remote_client()
    print(f"Model   : {MODEL}", flush=True)
    print(f"Samples : {args.n_samples}", flush=True)
    print(f"Workers : {args.workers}", flush=True)

    for path in args.datasets:
        if not Path(path).exists():
            print(f"[SKIP] {path} not found", flush=True)
            continue

        stem = Path(path).stem
        out_path = out_dir / f"{stem}_gpt_oss_oracle.json"

        problems = load_dataset(path)
        print(f"\n=== {stem} ({len(problems)} problems) ===", flush=True)

        # Resume
        done: dict = {}
        if out_path.exists():
            with open(out_path) as f:
                existing = json.load(f)
            done = {r["id"]: r for r in existing.get("results", [])}
            print(f"  Resuming: {len(done)} already done", flush=True)

        results = list(done.values())
        results_lock = threading.Lock()
        pending = [p for p in problems if p["id"] not in done]
        print(f"  {len(pending)} remaining", flush=True)

        # Track per-problem sample accumulation
        sample_buckets: dict = {p["id"]: [] for p in pending}
        sample_buckets_lock = threading.Lock()
        n_samples = args.n_samples

        def save():
            output = {
                "metadata": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": MODEL,
                    "dataset": path,
                    "n_samples": n_samples,
                    "n_problems": len(results),
                    "n_total": len(problems),
                    "completed": len(results) == len(problems),
                },
                "results": results,
            }
            tmp = str(out_path) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            os.replace(tmp, out_path)

        def _submit_sample(problem, sample_idx):
            r = _solve_once(client, problem["prompt"], sample_idx)
            pid = problem["id"]
            with sample_buckets_lock:
                sample_buckets[pid].append(r)
                n_done = len(sample_buckets[pid])
            if n_done == n_samples:
                problem_result = _build_problem_result(problem, sample_buckets[pid], n_samples)
                with results_lock:
                    results.append(problem_result)
                    save()

        # Submit all samples for all problems into one shared pool
        print(f"  Submitting {len(pending) * n_samples} total samples...", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for problem in pending:
                print(f"  queuing id={problem['id']}", flush=True)
                for i in range(n_samples):
                    pool.submit(_submit_sample, problem, i)

        print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
