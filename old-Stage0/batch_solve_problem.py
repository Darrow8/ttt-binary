"""Solve a math problem by querying gpt-oss-120b 20 times in parallel.

Results (reasoning + extracted answer) are streamed to a JSON file as each
query completes, so partial progress is never lost.

Usage:
    python solve_problem.py                       # uses the default problem
    python solve_problem.py "Find the value of …" # custom problem
"""

import concurrent.futures
import json
import os
import re
import sys
import threading
import time
from datetime import datetime

from openai import OpenAI

# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                val = val.strip().strip('"').strip("'")
                os.environ.setdefault(key.strip(), val)

# ---------------------------------------------------------------------------
# Vertex AI client
# ---------------------------------------------------------------------------
MODEL = "openai/gpt-oss-120b-maas"
N_SAMPLES = 20
MAX_WORKERS = 10


def _get_vertex_access_token() -> str:
    from google.auth import default
    from google.auth.transport.requests import Request

    credentials, _ = default()
    credentials.refresh(Request())
    return credentials.token


def _build_vertex_base_url() -> str:
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    if location == "global":
        location = "us-central1"
    return (
        f"https://{location}-aiplatform.googleapis.com/v1/"
        f"projects/{project}/locations/{location}/endpoints/openapi"
    )


def get_client() -> OpenAI:
    token = _get_vertex_access_token()
    base_url = _build_vertex_base_url()
    return OpenAI(api_key=token, base_url=base_url)


# ---------------------------------------------------------------------------
# Answer extraction (regex first, LLM fallback)
# ---------------------------------------------------------------------------
EXTRACT_ANSWER_PROMPT = """\
Below is a student's solution to a math problem. What is the student's \
final answer? Reply with ONLY the answer value — nothing else.

Rules:
- If the answer is a number, return just the number (e.g. 42, 3/2, 0.75).
- If the answer is "infinity" or "does not exist" or similar, return \
  that phrase in lowercase (e.g. "infinity", "does not exist").
- No LaTeX, no units, no extra words.
- If there is no clear final answer, reply with exactly: NONE

## Solution

{solution}"""


def _regex_extract(solution: str) -> str:
    """Try to extract the answer from structured patterns."""
    if not solution:
        return ""

    matches = re.findall(
        r"\*\*ANSWER:\s*(.+?)\*\*",
        solution,
        re.IGNORECASE,
    )
    if matches:
        return matches[-1].strip()

    boxed = re.findall(r"\\boxed\{([^}]+)\}", solution)
    if boxed:
        return boxed[-1].strip()

    m = re.search(
        r"(?:final answer|the answer)(?:\s+is)?[:\s]+([^\n.]+)",
        solution,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    return ""


def extract_answer(client: OpenAI, solution: str) -> str:
    """Extract the answer: regex first, LLM fallback."""
    answer = _regex_extract(solution)
    if answer:
        return answer

    if not solution:
        return ""

    prompt = EXTRACT_ANSWER_PROMPT.format(solution=solution)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        answer = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"  [warn] answer-extraction LLM call failed: {e}")
        return ""

    if answer.upper() == "NONE":
        return ""
    return answer


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------
DEFAULT_PROBLEM = r"""
Consider the projective space \(\mathbb{P}^5\) over an algebraically closed field of characteristic 0, with homogeneous coordinates \([z_0:\dots:z_5]\). Let \(Q_1,Q_2,Q_3\) be three generic homogeneous quadratic polynomials and \(C_1,C_2\) be two generic homogeneous cubic polynomials in these variables. Define the zero‑dimensional scheme
\[ X = \{[z]\in \mathbb{P}^5 \mid Q_1([z]) = Q_2([z]) = Q_3([z]) = C_1([z]) = C_2([z]) = 0\}.\]
Let \(\mathcal{Q}\) be the parameter space of ordered 5‑tuples \((Q_1,Q_2,Q_3,C_1,C_2)\) of such polynomials (each up to scaling), which is a dense open subset of a product of projective spaces. Define the incidence variety
\[ Z = \{(x,\mathbf{Q}) \in \mathbb{P}^5 \times \mathcal{Q} \mid x \text{ satisfies all five equations of } \mathbf{Q}\} \]
and let \(\pi : Z \to \mathcal{Q}\) be the projection onto the second factor. Over a dense open subset \(V\subset \mathcal{Q}\) the map \(\pi\) is finite étale.
Define
\[ L = \lim_{p\to\infty} \frac{1}{\#V(\mathbb{F}_p)} \sum_{\mathbf{Q}\in V(\mathbb{F}_p)} \#\pi^{-1}(\mathbf{Q}).\]
Compute \(\lfloor 100L \rfloor\)."""

SOLVE_PROMPT = """\
## Problem

{problem}

## Instructions

Solve this problem step by step. You MUST show all of your reasoning, \
calculations, and intermediate steps IN YOUR RESPONSE — do not skip ahead \
to the answer. Think carefully and work through the math explicitly. \
Write out every key derivation.

After you have fully worked through the solution, write your final \
numerical answer on the very last line in exactly this format \
(including the double asterisks):

**ANSWER: <your numerical answer here>**

Your answer must be a number, not an expression. Begin your solution now.

"""

# ---------------------------------------------------------------------------
# Atomic JSON save
# ---------------------------------------------------------------------------
def _save_atomic(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Single solve attempt (with retries)
# ---------------------------------------------------------------------------
def _solve_once(
    client: OpenAI,
    problem: str,
    sample_idx: int,
    max_retries: int = 3,
) -> dict:
    """Run one solve attempt and return a result dict."""
    prompt = SOLVE_PROMPT.format(problem=problem)

    for attempt in range(max_retries):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            elapsed = time.time() - t0
            solution = resp.choices[0].message.content or ""

            if not solution:
                print(f"  [sample {sample_idx+1}] empty response "
                      f"(attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                continue

            answer = extract_answer(client, solution)

            return {
                "sample_idx": sample_idx,
                "answer": answer,
                "reasoning": solution,
                "elapsed_s": round(elapsed, 2),
            }

        except Exception as e:
            print(f"  [sample {sample_idx+1}] error "
                  f"(attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return {
        "sample_idx": sample_idx,
        "answer": "",
        "reasoning": "",
        "elapsed_s": 0,
        "error": "all retries exhausted",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) > 1:
        problem = " ".join(sys.argv[1:])
    else:
        problem = DEFAULT_PROBLEM

    out_dir = os.path.join(
        os.path.dirname(__file__), "runs",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.json")

    print(f"Model:   {MODEL}")
    print(f"Samples: {N_SAMPLES}")
    print(f"Workers: {MAX_WORKERS}")
    print(f"Output:  {out_path}")
    print()

    client = get_client()

    results: list[dict] = []
    lock = threading.Lock()

    output_data = {
        "model": MODEL,
        "n_samples": N_SAMPLES,
        "problem": problem.strip(),
        "started_at": datetime.now().isoformat(),
        "completed": False,
        "results": results,
    }
    _save_atomic(out_path, output_data)

    def _on_complete(future: concurrent.futures.Future) -> None:
        try:
            result = future.result()
        except Exception as e:
            result = {"error": str(e)}

        with lock:
            results.append(result)
            output_data["results"] = sorted(results, key=lambda r: r.get("sample_idx", 0))
            _save_atomic(out_path, output_data)

        idx = result.get("sample_idx", "?")
        ans = result.get("answer", "")
        t = result.get("elapsed_s", 0)
        print(f"  [sample {idx+1}/{N_SAMPLES}] answer={ans!r}  ({t:.1f}s)  "
              f"[{len(results)}/{N_SAMPLES} done]")

    t_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for i in range(N_SAMPLES):
            fut = pool.submit(_solve_once, client, problem, i)
            fut.add_done_callback(_on_complete)

    total_time = time.time() - t_start

    output_data["completed"] = True
    output_data["finished_at"] = datetime.now().isoformat()
    output_data["total_time_s"] = round(total_time, 2)

    answers = [r.get("answer", "") for r in results if r.get("answer")]
    if answers:
        from collections import Counter
        counter = Counter(answers)
        majority_answer, majority_count = counter.most_common(1)[0]
        output_data["summary"] = {
            "majority_answer": majority_answer,
            "agreement_rate": round(majority_count / len(answers), 3),
            "answer_distribution": dict(counter.most_common()),
            "n_valid_answers": len(answers),
            "n_empty": N_SAMPLES - len(answers),
        }
        print(f"\n{'='*60}")
        print(f"  Majority answer: {majority_answer}")
        print(f"  Agreement: {majority_count}/{len(answers)} "
              f"({majority_count/len(answers):.0%})")
        print(f"  Distribution: {dict(counter.most_common())}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"{'='*60}")

    _save_atomic(out_path, output_data)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
