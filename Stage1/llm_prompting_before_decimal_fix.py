"""
TTT-Discover: Dataset generation via LLM self-consistency.

Given a hard problem, generates similar problems at a calibrated difficulty
where the LLM's answers agree ~50-80% of the time across repeated samples.
The majority-vote answer is treated as pseudo ground truth, producing a
training dataset of ~100 (problem, answer) pairs with binary reward.

Usage:
    gcloud auth application-default login
    export GOOGLE_CLOUD_PROJECT="your-project-id"

    python llm_prompting.py --problem_statement PROBLEM_STATEMENT --n_problems 100
"""

import argparse
import concurrent.futures
import json
import os
import threading
import time
import re
from collections import Counter
from dataclasses import dataclass, field, asdict

from openai import OpenAI

# ---------------------------------------------------------------------------
# .env loader (same as eval.py)
# ---------------------------------------------------------------------------
_env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                val = val.strip().strip('"').strip("'")
                os.environ.setdefault(key.strip(), val)

# ---------------------------------------------------------------------------
# Vertex AI client (mirrors eval.py)
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "openai/gpt-oss-120b-maas"

PROBLEM_STATEMENT = r"""There are unique constants \(C, \alpha, \beta\) such that the number of solutions to the equation
\[
ab + 1 = cde
\]
with \(a, b, c, d, e \in \mathbb{N}\) and \(ab \le x\) is asymptotic to
\[
C x^{\alpha} \log^{\beta} x
\]
as \(x \to \infty\). Compute
\[
\left\lfloor 1000C \right\rfloor .
\]
"""


print(PROBLEM_STATEMENT)

# Let $\mathbb{N}$ denote the set of positive integers. A function
# $f : \mathbb{N} \to \mathbb{N}$ is said to be \emph{bonza} if
# \[
# f(a) \mid b^a - f(b)^{f(a)}
# \]
# for all positive integers $a$ and $b$.

# Determine the smallest real constant $c$ such that
# \[
# f(n) \le cn
# \]
# for all bonza functions $f$ and all positive integers $n$.
# """

def _get_vertex_access_token() -> str:
    from google.auth import default
    from google.auth.transport.requests import Request

    credentials, _ = default()
    credentials.refresh(Request())
    return credentials.token


def _build_vertex_base_url() -> str:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError(
            "Set GOOGLE_CLOUD_PROJECT environment variable.\n"
            "  export GOOGLE_CLOUD_PROJECT='your-project-id'"
        )
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    if location == "global":
        location = "us-central1"
    return (
        f"https://{location}-aiplatform.googleapis.com/v1/"
        f"projects/{project}/locations/{location}/endpoints/openapi"
    )


def get_client() -> tuple[OpenAI, str]:
    token = _get_vertex_access_token()
    base_url = _build_vertex_base_url()
    client = OpenAI(api_key=token, base_url=base_url)
    return client, DEFAULT_MODEL


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------
def call_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            content = response.choices[0].message.content or ""
            if content:
                return content
            print(f"  [warn] empty response (attempt {attempt+1}/{max_retries})")
        except Exception as e:
            print(f"  [warn] LLM call failed (attempt {attempt+1}/{max_retries}): {e}")
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
    return ""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class GeneratedProblem:
    problem: str
    ground_truth_answer: str
    agreement_rate: float
    all_answers: list[str] = field(default_factory=list)
    all_solutions: list[str] = field(default_factory=list)
    n_samples: int = 0


@dataclass
class Dataset:
    source_problem: str
    problems: list[GeneratedProblem] = field(default_factory=list)
    target_agreement_low: float = 0.60
    target_agreement_high: float = 0.80


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
GENERATE_PROBLEMS_PROMPT = """\
You are designing training problems for test-time training on ONE hard source problem.
Your job is to create problems that preserve the same mathematical bottleneck as the
source problem, while being smaller, cleaner, and self-contained.

Your goal is NOT to create random "similar-looking" problems or parametric variants
(e.g. changing the number of variables). Your goal is to create problems whose solution
would train the model on a reusable subskill needed for the original problem.

## Source Problem

{problem}

## Attempted Solutions (may be incorrect)

Below are solution attempts on the source problem. They may contain errors — use them
ONLY to identify what mathematical techniques and subskills are being attempted, NOT
to determine the correct answer.

{failed_solutions}

## Instructions

1. Read the attempted solutions above and identify the DISTINCT mathematical techniques
   they use (e.g. Dirichlet series, CRT, Euler products, divisor sums, asymptotic
   estimates, Tauberian theorems, etc.)
2. For each technique, consider: what is a simpler, self-contained problem that would
   drill EXACTLY that technique?
3. Generate {batch_size} problems, each targeting a DIFFERENT technique or subskill.
   DO NOT generate multiple problems that only differ in a parameter (like the number
   of factors in a product). Each problem must be structurally distinct.

Each generated problem MUST satisfy all of the following:
- It has a SINGLE numerical final answer.
- It is self-contained.
- It isolates one real bottleneck or subskill from the source problem.
- It is NOT a parametric variant of the source problem (e.g. changing ab+1=cde to
  ab+1=cdef is NOT acceptable — that tests the same skill at the same difficulty).
- It avoids fake complexity and decorative algebraic clutter.
- CRITICAL: The problem must be HARD — comparable to a research-level or competition
  math problem. A strong LLM should get it WRONG 20-40% of the time.
  DO NOT generate textbook exercises, definitions, or routine calculations like
  "compute ζ(2)", "count divisors of 60", "evaluate a standard limit", or
  "state a well-known asymptotic formula". These are TOO EASY and useless for training.
  The problem should require COMBINING techniques or applying them in a non-obvious way.

## Output Format

Return ONLY valid JSON, with no markdown fences and no extra text.
Return a JSON array of objects in exactly this shape:
[
  {{"problem": "..."}}
]
"""

SOLVE_PROMPT = """\
## Problem

{problem}

When you are done, write your final numerical answer on the very last line \
in exactly this format (including the double asterisks):

**ANSWER: <your numerical answer here>**

Solve the problem above step by step.

"""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def generate_similar_problems(
    client: OpenAI,
    model: str,
    hard_problem: str,
    batch_size: int = 10,
    failed_solutions: list[str] | None = None,
) -> list[dict]:
    """Prompt the LLM to generate a batch of similar problems.

    Returns a list of dicts with a `problem` key.
    """
    if failed_solutions:
        solutions_block = "\n\n---\n\n".join(
            f"### Attempt {i+1}\n\n{sol}"
            for i, sol in enumerate(failed_solutions)
        )
    else:
        solutions_block = "(No attempted solutions available.)"

    prompt = GENERATE_PROBLEMS_PROMPT.format(
        problem=hard_problem,
        batch_size=batch_size,
        failed_solutions=solutions_block,
    )
    raw = call_llm(client, model, prompt, temperature=0.8)

    # Extract JSON from response (handle markdown fences)
    json_str = raw
    if "```" in json_str:
        parts = json_str.split("```")
        for part in parts[1:]:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            if cleaned.startswith("["):
                json_str = cleaned
                break

    # Fix invalid JSON escapes from LaTeX (e.g. \alpha, \infty, \le).
    # The alternation first matches \\\\ (an already-escaped backslash pair)
    # and leaves it alone, then matches a lone \\ not followed by a valid
    # JSON escape char and doubles it.
    json_str = re.sub(
        r'\\\\|\\(?!["\\/bfnrtu])',
        lambda m: m.group() if m.group() == '\\\\' else '\\\\',
        json_str,
    )

    try:
        problems = json.loads(json_str)
        if not isinstance(problems, list):
            raise ValueError("Expected a JSON array")
        cleaned_problems = []
        for item in problems:
            if not isinstance(item, dict):
                continue
            problem_text = item.get("problem", "")
            if not isinstance(problem_text, str):
                continue
            problem_text = problem_text.strip()
            if not problem_text:
                continue
            cleaned_problems.append({"problem": problem_text})
        return cleaned_problems
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  [warn] Failed to parse generated problems: {e}")
        print(f"  [warn] Raw response (first 500 chars): {raw[:500]}")
        return []



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


_NUMERIC_ANSWER_RE = re.compile(
    r"^[+-]?\d+([.,]\d+)?(/\d+)?$"
)


def _is_numeric_answer(answer: str) -> bool:
    """Return True if the normalized answer is a number."""
    return bool(_NUMERIC_ANSWER_RE.match(answer))


def _regex_extract(solution: str) -> str:
    """Try to extract the answer from structured patterns. Returns "" on failure."""
    if not solution:
        return ""

    # 1. **ANSWER: <value>** — take the last occurrence
    matches = re.findall(
        r"\*\*ANSWER:\s*(.+?)\*\*",
        solution,
        re.IGNORECASE,
    )
    if matches:
        return matches[-1].strip()

    # 2. \boxed{...} — take the last occurrence
    boxed = re.findall(r"\\boxed\{([^}]+)\}", solution)
    if boxed:
        return boxed[-1].strip()

    # 3. "the answer is …" / "final answer is …"
    m = re.search(
        r"(?:final answer|the answer)(?:\s+is)?[:\s]+([^\n.]+)",
        solution, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    return ""


def extract_answer(
    client: OpenAI,
    model: str,
    solution: str,
) -> str:
    """Extract the answer: regex first, LLM fallback if regex finds nothing."""
    answer = _regex_extract(solution)
    if answer:
        return answer

    if not solution:
        return ""

    prompt = EXTRACT_ANSWER_PROMPT.format(solution=solution)
    raw = call_llm(client, model, prompt, temperature=0.0)
    answer = raw.strip()
    if answer.upper() == "NONE":
        return ""
    return answer


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison.

    Strips LaTeX markup, whitespace, and converts fractions / decimals to a
    canonical decimal form so that e.g. "1/2" and "0.5" compare equal.
    """
    a = answer.strip().lower()
    a = re.sub(r"[\\${}]", "", a)
    a = re.sub(r"\s+", "", a)
    a = a.replace(",", ".")

    # Try to evaluate as a number (handles fractions like "1/2", "6/pi^2", etc.)
    try:
        from fractions import Fraction
        if "/" in a and a.replace("/", "").replace("-", "").replace(".", "").isdigit():
            val = float(Fraction(a))
            a = f"{val:.10g}"
        elif a.replace(".", "").replace("-", "").isdigit():
            val = float(a)
            a = f"{val:.10g}"
    except (ValueError, ZeroDivisionError):
        pass

    return a


_SOLVE_MAX_RETRIES = 3


def _solve_one(
    client: OpenAI,
    model: str,
    problem: str,
    _use_llm_extract: bool,
) -> tuple[str, str]:
    """Solve a problem once and return (normalized_answer, full_solution).

    Retries if the solution is empty or truncated (no answer pattern found).
    """
    prompt = SOLVE_PROMPT.format(problem=problem)
    for attempt in range(_SOLVE_MAX_RETRIES):
        solution = call_llm(client, model, prompt, temperature=0.7)
        if not solution:
            continue
        answer = extract_answer(client, model, solution)
        if normalize_answer(answer):
            return normalize_answer(answer), solution
        # Solution exists but no answer extracted — likely truncated, retry
        if attempt < _SOLVE_MAX_RETRIES - 1:
            print("r", end="", flush=True)
    # Return whatever we got on the last attempt
    answer = extract_answer(client, model, solution) if solution else ""
    return normalize_answer(answer), solution


def solve_and_check_agreement(
    client: OpenAI,
    model: str,
    problem: str,
    n_samples: int = 10,
    use_llm_extract: bool = False,
    pool: concurrent.futures.ThreadPoolExecutor | None = None,
) -> tuple[float, str, list[str], list[str]]:
    """Solve a problem n_samples times in parallel and compute agreement rate.

    Returns:
        (agreement_rate, majority_answer, all_answers, all_solutions)
    """
    futures = [
        pool.submit(_solve_one, client, model, problem, use_llm_extract)
        for _ in range(n_samples)
    ]
    results = [f.result() for f in futures]
    answers = [r[0] for r in results]
    solutions = [r[1] for r in results]

    if not answers:
        return 0.0, "", answers, solutions

    counter = Counter(answers)
    majority_answer, majority_count = counter.most_common(1)[0]
    agreement_rate = majority_count / len(answers)

    return agreement_rate, majority_answer, answers, solutions


def _save_atomic(path: str, data: dict | list) -> None:
    """Atomically write JSON data to disk via a temp file."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)



def build_dataset(
    client: OpenAI,
    model: str,
    hard_problem: str,
    n_target: int = 100,
    n_samples_per_problem: int = 10,
    target_agreement_low: float = 0.60,
    target_agreement_high: float = 0.80,
    generation_batch_size: int = 15,
    max_generation_rounds: int = 20,
    use_llm_extract: bool = False,
    output_path: str | None = None,
    max_workers: int = 16,
    failed_solutions: list[str] | None = None,
) -> Dataset:
    """Build a dataset of problems with calibrated difficulty.

    Generates similar problems in batches, filters for those where the LLM's
    self-consistency falls within [target_agreement_low, target_agreement_high],
    and uses the majority-vote answer as ground truth.

    API calls are parallelised with a shared thread pool (max_workers controls
    the total number of concurrent LLM requests across all candidates).

    When output_path is provided, two JSON files are maintained incrementally:
      - <output_path>          — kept problems (in-band agreement)
      - <stem>_skips<ext>      — skipped problems (out-of-band agreement)
    Both files are updated atomically after every candidate is evaluated.
    """
    dataset = Dataset(
        source_problem=hard_problem,
        target_agreement_low=target_agreement_low,
        target_agreement_high=target_agreement_high,
    )

    skipped_problems: list[GeneratedProblem] = []
    seen_problems: set[str] = set()
    round_num = 0
    save_lock = threading.Lock()

    if output_path:
        skips_path = os.path.join(os.path.dirname(output_path), "skips.json")
    else:
        skips_path = None

    def _flush() -> None:
        if not output_path:
            return
        _save_atomic(output_path, {
            "source_problem": dataset.source_problem,
            "target_agreement_low": dataset.target_agreement_low,
            "target_agreement_high": dataset.target_agreement_high,
            "n_problems": len(dataset.problems),
            "problems": [asdict(p) for p in dataset.problems],
        })
        _save_atomic(skips_path, {
            "source_problem": dataset.source_problem,
            "target_agreement_low": dataset.target_agreement_low,
            "target_agreement_high": dataset.target_agreement_high,
            "n_problems": len(skipped_problems),
            "problems": [asdict(p) for p in skipped_problems],
        })

    def _evaluate_candidate(
        problem_text: str,
        pool: concurrent.futures.ThreadPoolExecutor,
    ) -> tuple[str, float, str, list[str], list[str], float]:
        """Evaluate one candidate (runs in a worker thread)."""
        t1 = time.time()
        agreement, majority_ans, all_answers, all_solutions = solve_and_check_agreement(
            client, model, problem_text,
            n_samples=n_samples_per_problem,
            use_llm_extract=use_llm_extract,
            pool=pool,
        )
        elapsed = time.time() - t1
        return problem_text, agreement, majority_ans, all_answers, all_solutions, elapsed

    print(f"\n{'='*70}")
    print(f"  TTT-Discover: Building dataset from hard problem")
    print(f"  Target: {n_target} problems with {target_agreement_low:.0%}-{target_agreement_high:.0%} agreement")
    print(f"  Samples per problem: {n_samples_per_problem}")
    print(f"  Failed solution attempts for context: {len(failed_solutions)}")
    print(f"  Model: {model}")
    print(f"  Max parallel API calls: {max_workers}")
    if output_path:
        print(f"  Keeps file:  {output_path}")
        print(f"  Skips file:  {skips_path}")
    print(f"{'='*70}\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        while len(dataset.problems) < n_target and round_num < max_generation_rounds:
            round_num += 1
            batch_size = generation_batch_size

            print(f"--- Round {round_num}: generating {batch_size} candidate problems "
                  f"({len(dataset.problems)}/{n_target} collected) ---")

            t0 = time.time()
            candidates = generate_similar_problems(
                client, model, hard_problem,
                batch_size=batch_size,
                failed_solutions=failed_solutions,
            )
            gen_time = time.time() - t0
            print(f"  Generated {len(candidates)} candidates ({gen_time:.1f}s)")

            # Deduplicate and collect valid candidates
            to_evaluate: list[tuple[int, str]] = []
            for j, cand in enumerate(candidates):
                problem_text = cand.get("problem", "")
                if not problem_text or problem_text in seen_problems:
                    continue
                seen_problems.add(problem_text)
                to_evaluate.append((j, problem_text))

            if not to_evaluate:
                print("  No new unique candidates, retrying...")
                continue

            # Submit all candidates for parallel evaluation
            future_to_idx: dict[concurrent.futures.Future, tuple[int, str]] = {}
            for j, problem_text in to_evaluate:
                fut = pool.submit(_evaluate_candidate, problem_text, pool)
                future_to_idx[fut] = (j, problem_text)

            print(f"  Evaluating {len(to_evaluate)} candidates in parallel...\n")

            for fut in concurrent.futures.as_completed(future_to_idx):
                j, problem_text = future_to_idx[fut]
                try:
                    _, agreement, majority_ans, all_answers, all_solutions, elapsed = fut.result()
                except Exception as e:
                    print(f"  [{j+1}] ERROR: {e}")
                    continue

                in_range = target_agreement_low <= agreement <= target_agreement_high
                numeric = _is_numeric_answer(normalize_answer(majority_ans))
                kept = in_range and bool(majority_ans) and numeric
                if not bool(majority_ans):
                    status = "skip (empty answer)"
                elif not numeric:
                    status = f"skip (non-numeric: {majority_ans[:40]})"
                elif in_range:
                    status = "KEEP"
                else:
                    status = "skip"

                entry = GeneratedProblem(
                    problem=problem_text,
                    ground_truth_answer=majority_ans,
                    agreement_rate=agreement,
                    all_answers=all_answers,
                    all_solutions=all_solutions,
                    n_samples=n_samples_per_problem,
                )

                with save_lock:
                    if kept:
                        dataset.problems.append(entry)
                    else:
                        skipped_problems.append(entry)
                    _flush()
                    n_keeps = len(dataset.problems)
                    n_skips = len(skipped_problems)

                print(f"  [{j+1}] {agreement:.0%} agreement ({elapsed:.1f}s) -> {status}")
                if kept:
                    print(f"    majority answer: {majority_ans[:80]}")
                print(f"    -> saved (keeps: {n_keeps}, skips: {n_skips})")

            print()

    print(f"{'='*70}")
    print(f"  Dataset complete: {len(dataset.problems)} kept, {len(skipped_problems)} skipped")
    avg_agreement = (
        sum(p.agreement_rate for p in dataset.problems) / len(dataset.problems)
        if dataset.problems else 0
    )
    print(f"  Average agreement rate (kept): {avg_agreement:.1%}")
    print(f"{'='*70}\n")

    return dataset


def save_dataset(dataset: Dataset, path: str) -> None:
    """Save the dataset to a JSON file."""
    data = {
        "source_problem": dataset.source_problem,
        "target_agreement_low": dataset.target_agreement_low,
        "target_agreement_high": dataset.target_agreement_high,
        "n_problems": len(dataset.problems),
        "problems": [asdict(p) for p in dataset.problems],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Dataset saved to {path}")


def load_dataset(path: str) -> Dataset:
    """Load a dataset from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    ds = Dataset(
        source_problem=data["source_problem"],
        target_agreement_low=data.get("target_agreement_low", 0.60),
        target_agreement_high=data.get("target_agreement_high", 0.80),
    )
    for p in data["problems"]:
        ds.problems.append(GeneratedProblem(**p))
    return ds


def run(
    problem: str,
    *,
    n_problems: int = 100,
    n_samples: int = 10,
    agree_low: float = 0.60,
    agree_high: float = 0.80,
    batch_size: int = 10,
    output: str | None = None,
    model: str | None = None,
    llm_extract: bool = False,
    max_workers: int = 16,
    failed_solutions: list[str] | None = None,
) -> Dataset:
    """Run the full pipeline programmatically.

    Returns the generated Dataset.
    """
    client, default_model = get_client()
    model = model or default_model

    from datetime import datetime
    run_dir = output or os.path.join(
        os.path.dirname(__file__),
        "runs",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "keeps.json")

    dataset = build_dataset(
        client=client,
        model=model,
        hard_problem=problem,
        n_target=n_problems,
        n_samples_per_problem=n_samples,
        target_agreement_low=agree_low,
        target_agreement_high=agree_high,
        generation_batch_size=batch_size,
        use_llm_extract=llm_extract,
        output_path=out_path,
        max_workers=max_workers,
        failed_solutions=failed_solutions,
    )

    save_dataset(dataset, out_path)
    return dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_failed_solutions() -> list[str]:
    path = os.path.join(os.path.dirname(__file__), "failed_solutions.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return [s for s in data if isinstance(s, str)]
    return []


def main():
    run(
        problem=PROBLEM_STATEMENT,
        n_problems=20,
        n_samples=10,
        model="openai/gpt-oss-120b-maas",
        llm_extract=True,
        failed_solutions=_load_failed_solutions(),
    )


if __name__ == "__main__":
    main()
