"""Solve a math problem using the remote Vertex AI API or a local GRPO checkpoint.

Usage:
    python Inference/infer.py                          # remote API, default problem
    python Inference/infer.py --local                  # local checkpoint, default problem
    python Inference/infer.py "Find the value of …"    # custom problem (remote)
    python Inference/infer.py --local "Find …"         # custom problem (local)
    python Inference/infer.py --n-samples 50           # override sample count
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

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
Your answer must be a number, not an expression. \
Put your final answer inside \\boxed{{}}.

"""

TEMPERATURE = 0.7
BASE_MODEL = "openai/gpt-oss-120b"

# ---------------------------------------------------------------------------
# Answer extraction (regex first, optional LLM fallback)
# ---------------------------------------------------------------------------

EXTRACT_ANSWER_PROMPT = """\
Below is a student's solution to a math problem. The student was asked to \
put their final answer inside \\boxed{{}}. What is the value inside the \
last \\boxed{{}}? Reply with ONLY the answer value — nothing else.

Rules:
- If the answer is a number, return just the number (e.g. 42, 3/2, 0.75).
- If the answer is "infinity" or "does not exist" or similar, return \
  that phrase in lowercase (e.g. "infinity", "does not exist").
- No LaTeX, no units, no extra words.
- If there is no clear final answer, reply with exactly: NONE

## Solution

{solution}"""


def _extract_boxed_raw(text: str) -> str:
    """Return the raw contents of the last ``\\boxed{...}``, handling nested braces."""
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


_LATEX_NOISE = re.compile(
    r"\\(?:displaystyle|bigl?|Bigl?|bigr?|Bigr?|left|right|text\{[^}]*\}|lfloor|rfloor|,|;| )"
)


def _clean_boxed(raw: str) -> str:
    """Extract a plain numeric answer from the raw boxed content."""
    if "=" in raw:
        raw = raw.rsplit("=", 1)[1]
    cleaned = _LATEX_NOISE.sub("", raw)
    cleaned = cleaned.replace(",", "").replace(" ", "").strip().rstrip(".")
    if not cleaned:
        return ""
    try:
        num = float(cleaned)
        if num == int(num):
            return str(int(num))
        return str(num)
    except (ValueError, OverflowError):
        pass
    return cleaned


def _regex_extract(solution: str) -> str:
    if not solution:
        return ""
    raw = _extract_boxed_raw(solution)
    if raw:
        return _clean_boxed(raw) or raw
    m = re.search(
        r"(?:final answer|the answer)(?:\s+is)?[:\s]+([^\n.]+)",
        solution, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return ""


def extract_answer(solution: str, *, client=None, model: str = "") -> str:
    """Extract the answer: regex first, LLM fallback if a client is provided."""
    answer = _regex_extract(solution)
    if answer:
        return answer
    if not solution or client is None:
        return ""
    prompt = EXTRACT_ANSWER_PROMPT.format(solution=solution)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
        )
        answer = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"  [warn] answer-extraction LLM call failed: {e}")
        return ""
    if answer.upper() == "NONE":
        return ""
    return answer


def _save_atomic(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _print_summary(answers: list[str], n_samples: int, total_time: float | None = None) -> dict:
    counter = Counter(answers)
    majority_answer, majority_count = counter.most_common(1)[0]
    summary = {
        "majority_answer": majority_answer,
        "agreement_rate": round(majority_count / len(answers), 3),
        "answer_distribution": dict(counter.most_common()),
        "n_valid_answers": len(answers),
        "n_empty": n_samples - len(answers),
    }
    print(f"\n{'='*60}")
    print(f"  Majority answer: {majority_answer}")
    print(f"  Agreement: {majority_count}/{len(answers)} "
          f"({majority_count/len(answers):.0%})")
    print(f"  Distribution: {dict(counter.most_common())}")
    if total_time is not None:
        print(f"  Total time: {total_time:.1f}s")
    print(f"{'='*60}")
    return summary


# ---------------------------------------------------------------------------
# Remote (Vertex AI) backend
# ---------------------------------------------------------------------------

REMOTE_MODEL = "openai/gpt-oss-120b-maas"
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


def _get_remote_client():
    from openai import OpenAI
    token = _get_vertex_access_token()
    base_url = _build_vertex_base_url()
    return OpenAI(api_key=token, base_url=base_url)


def _solve_once_remote(client, problem: str, sample_idx: int, max_retries: int = 3) -> dict:
    prompt = SOLVE_PROMPT.format(problem=problem)
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=REMOTE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
            )
            elapsed = time.time() - t0
            msg = resp.choices[0].message
            content = msg.content or ""
            reasoning_content = getattr(msg, "reasoning_content", None) or ""
            if reasoning_content:
                solution = reasoning_content + "\n\n" + content
            else:
                solution = content
            if not solution:
                print(f"  [sample {sample_idx+1}] empty response "
                      f"(attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                continue
            answer = extract_answer(solution, client=client, model=REMOTE_MODEL)
            result = {
                "sample_idx": sample_idx,
                "answer": answer,
                "reasoning": solution,
                "elapsed_s": round(elapsed, 2),
            }
            if reasoning_content:
                result["has_hidden_reasoning"] = True
                result["reasoning_content_length"] = len(reasoning_content)
            return result
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


def run_remote(problem: str, n_samples: int) -> None:
    out_dir = os.path.join("runs", "base_model_inference", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.json")

    print(f"Mode:    remote")
    print(f"Model:   {REMOTE_MODEL}")
    print(f"Samples: {n_samples}")
    print(f"Workers: {MAX_WORKERS}")
    print(f"Output:  {out_path}\n")

    client = _get_remote_client()

    results: list[dict] = []
    lock = threading.Lock()

    output_data = {
        "mode": "remote",
        "model": REMOTE_MODEL,
        "n_samples": n_samples,
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
            output_data["results"] = sorted(
                results, key=lambda r: r.get("sample_idx", 0),
            )
            _save_atomic(out_path, output_data)
        idx = result.get("sample_idx", "?")
        ans = result.get("answer", "")
        t = result.get("elapsed_s", 0)
        print(f"  [sample {idx+1}/{n_samples}] answer={ans!r}  ({t:.1f}s)  "
              f"[{len(results)}/{n_samples} done]")

    t_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for i in range(n_samples):
            fut = pool.submit(_solve_once_remote, client, problem, i)
            fut.add_done_callback(_on_complete)
    total_time = time.time() - t_start

    output_data["completed"] = True
    output_data["finished_at"] = datetime.now().isoformat()
    output_data["total_time_s"] = round(total_time, 2)

    answers = [r.get("answer", "") for r in results if r.get("answer")]
    if answers:
        output_data["summary"] = _print_summary(answers, n_samples, total_time)

    _save_atomic(out_path, output_data)
    print(f"\nResults saved to {out_path}")


# ---------------------------------------------------------------------------
# Local (tinker checkpoint) backend
# ---------------------------------------------------------------------------

def _get_tinker_service():
    import tinker
    if not os.environ.get("TINKER_API_KEY"):
        sys.exit(
            "TINKER_API_KEY not set. "
            "Add it to your .env or export it in your shell."
        )
    return tinker.ServiceClient()


def _find_latest_checkpoint(service):
    rest = service.create_rest_client()
    response = rest.list_user_checkpoints(limit=100).result()
    training_ckpts = [
        c for c in response.checkpoints
        if c.checkpoint_type == "training"
    ]
    if not training_ckpts:
        sys.exit("No training checkpoints found. Run training first.")
    latest = training_ckpts[0]
    print(f"Using checkpoint: {latest.tinker_path}  (created {latest.time})")
    return latest.tinker_path


def _build_local_clients(service, tinker_path: str):
    from transformers import AutoTokenizer
    print(f"Loading checkpoint: {tinker_path}")
    training_client = service.create_training_client_from_state(tinker_path)
    sampling_client = training_client.save_weights_and_get_sampling_client()
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login as hf_login
        hf_login(token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    print("Ready.\n")
    return sampling_client, tokenizer


def run_local(problem: str, n_samples: int, checkpoint: str | None = None) -> None:
    from tinker import types

    service = _get_tinker_service()
    tinker_path = checkpoint or _find_latest_checkpoint(service)
    sampling_client, tokenizer = _build_local_clients(service, tinker_path)

    print(f"Mode:       local")
    print(f"Model:      {BASE_MODEL}")
    print(f"Checkpoint: {tinker_path}")
    print(f"Samples:    {n_samples}\n")

    solve_prompt = SOLVE_PROMPT.format(problem=problem)
    messages = [{"role": "user", "content": solve_prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    prompt = types.ModelInput.from_ints(ids)

    params = types.SamplingParams(temperature=TEMPERATURE)
    result = sampling_client.sample(
        prompt=prompt,
        num_samples=n_samples,
        sampling_params=params,
    ).result()

    responses = [
        tokenizer.decode(seq.tokens, skip_special_tokens=True)
        for seq in result.sequences
    ]

    extract_client = _get_remote_client()
    results = []
    for i, resp in enumerate(responses):
        answer = extract_answer(resp, client=extract_client, model=REMOTE_MODEL)
        results.append({
            "sample_idx": i,
            "answer": answer,
            "reasoning": resp,
        })

    output = {
        "mode": "local",
        "model": BASE_MODEL,
        "checkpoint": tinker_path,
        "n_samples": n_samples,
        "problem": problem.strip(),
        "temperature": TEMPERATURE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "completed": True,
        "results": results,
    }

    answers = [r["answer"] for r in results if r["answer"]]
    if answers:
        output["summary"] = _print_summary(answers, n_samples)

    out_dir = os.path.join("runs", "local_inference", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(responses)} response(s) to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Solve a math problem with the remote API or a local GRPO checkpoint.",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Use local GRPO checkpoint instead of remote Vertex AI API",
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="Number of samples (default: 50 for local, 100 for remote)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Tinker checkpoint path (default: auto-detect latest)",
    )
    args = parser.parse_args()

    # problem = " ".join(args.problem) if args.problem else DEFAULT_PROBLEM

    if args.local:
        run_local(DEFAULT_PROBLEM, args.n_samples or 100, checkpoint=args.checkpoint)
    else:
        run_remote(DEFAULT_PROBLEM, args.n_samples or 100)


if __name__ == "__main__":
    main()
