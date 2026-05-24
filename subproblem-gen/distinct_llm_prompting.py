"""
TTT-Discover (distinct variant): Dataset generation via LLM self-consistency.

Like llm_prompting.py, but prioritises quality over quantity in generation:
each round launches N_GENERATORS independent LLM calls that each produce only
PROBLEMS_PER_GENERATOR problems (default 5 × 2 = 10 per round). Because each
call only asks for 2 problems, the LLM can spend more effort on each one, and
because the calls are independent (high temperature), we get natural diversity.
"""

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
DEFAULT_MODEL = "openai/gpt-oss-120b-maas"

# ---------------------------------------------------------------------------
# Tinker checkpoint client (adapter over tinker sampling API)
# ---------------------------------------------------------------------------
TINKER_BASE_MODEL = "openai/gpt-oss-120b"
TINKER_CKPT_50 = "tinker://e6b448b4-7e70-5e39-b0f7-06e0ef5b8e0d:train:0/weights/subproblems-run.ckpt-000050"


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _TinkerCompletions:
    """Mimics client.chat.completions so call_llm() works unchanged."""

    def __init__(self, sampling_client, tokenizer):
        self._sampling_client = sampling_client
        self._tokenizer = tokenizer

    def create(self, *, model: str = "", messages: list | None = None,
               temperature: float = 0.7, **_kwargs) -> _FakeResponse:
        from tinker import types

        text = self._tokenizer.apply_chat_template(
            messages or [], tokenize=False, add_generation_prompt=True,
        )
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        prompt = types.ModelInput.from_ints(ids)
        params = types.SamplingParams(temperature=temperature)
        result = self._sampling_client.sample(
            prompt=prompt, num_samples=1, sampling_params=params,
        ).result()
        content = self._tokenizer.decode(
            result.sequences[0].tokens, skip_special_tokens=True,
        )
        return _FakeResponse(content)


class _TinkerChat:
    def __init__(self, sampling_client, tokenizer):
        self.completions = _TinkerCompletions(sampling_client, tokenizer)


class TinkerClient:
    """Drop-in replacement for OpenAI() backed by a tinker checkpoint."""

    def __init__(self, tinker_path: str):
        import tinker
        if not os.environ.get("TINKER_API_KEY"):
            raise RuntimeError(
                "TINKER_API_KEY not set. Add it to .env or export it."
            )
        service = tinker.ServiceClient()
        print(f"Loading tinker checkpoint: {tinker_path}")
        sampling_client = service.create_sampling_client(model_path=tinker_path)

        from transformers import AutoTokenizer
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login as hf_login
            hf_login(token=hf_token)
        tokenizer = AutoTokenizer.from_pretrained(TINKER_BASE_MODEL)

        self.chat = _TinkerChat(sampling_client, tokenizer)
        self._tinker_path = tinker_path
        print(f"Tinker checkpoint ready: {tinker_path}\n")


def get_tinker_client(checkpoint: str | None = None) -> tuple["TinkerClient", str]:
    path = checkpoint or TINKER_CKPT_50
    client = TinkerClient(path)
    return client, TINKER_BASE_MODEL

PROBLEM_STATEMENT = r"""
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

print(PROBLEM_STATEMENT)


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
   teach EXACTLY that technique? Write out a set of techniques and concepts that are needed to solve the problem.
3. Generate {batch_size} problems, each targeting a DIFFERENT technique or subskill.
   DO NOT generate multiple problems that only differ in a parameter (like the number
   of factors in a product). Each problem must be structurally distinct.

Each generated problem MUST satisfy all of the following:
- It has a SINGLE numerical final answer that is a DECIMAL NUMBER (not a symbolic
  expression like 1/π², 6/π², √2, ln 2, etc.). If the exact answer is irrational
  or a fraction, the problem must ask the solver to ROUND to 4 decimal places.
  For example, instead of "Find the value of C" where C = 6/π², write
  "Find the value of C, rounded to 4 decimal places" (answer: 0.6079).
- It is self-contained.
- It isolates one real bottleneck or subskill from the source problem.
- It is NOT a parametric variant of the source problem (e.g. changing ab+1=cde to
  ab+1=cdef is NOT acceptable — that tests the same skill at the same difficulty).
- It avoids fake complexity and decorative algebraic clutter.
- CRITICAL: The problem must be HARD — comparable to a research-level or competition
  math problem. A strong LLM should get it WRONG 20-40% of the time.
  DO NOT generate textbook exercises, definitions, or routine calculations like
  "compute ζ(2)", "count divisors of 60", "evaluate a standard limit", or
  "state a well-known asymptotic formula". These are TOO EASY.
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

Your answer MUST be a single decimal number (e.g. 0.6079, 42, 3.1416). \
Do NOT write symbolic expressions like 1/π², 6/π², √2, or ln 2. \
If the answer is irrational or a fraction, round to 4 decimal places.

Solve the problem above step by step.

"""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def generate_similar_problems(
    client: OpenAI,
    model: str,
    hard_problem: str,
    batch_size: int = 2,
    failed_solutions: list[str] | None = None,
) -> list[dict]:
    """Prompt the LLM to generate a small batch of similar problems."""
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


def generate_problems_parallel(
    client: OpenAI,
    model: str,
    hard_problem: str,
    n_generators: int = 5,
    problems_per_generator: int = 2,
    failed_solutions: list[str] | None = None,
    pool: concurrent.futures.ThreadPoolExecutor | None = None,
) -> list[dict]:
    """Launch n_generators independent LLM calls, each producing problems_per_generator problems.

    Returns the merged, deduplicated list of candidate dicts.
    """
    futures = [
        pool.submit(
            generate_similar_problems,
            client, model, hard_problem,
            batch_size=problems_per_generator,
            failed_solutions=failed_solutions,
        )
        for _ in range(n_generators)
    ]

    all_candidates: list[dict] = []
    seen: set[str] = set()
    for i, fut in enumerate(concurrent.futures.as_completed(futures)):
        try:
            batch = fut.result()
            for cand in batch:
                text = cand.get("problem", "")
                if text and text not in seen:
                    seen.add(text)
                    all_candidates.append(cand)
            print(f"    generator {i+1}/{n_generators}: {len(batch)} problems")
        except Exception as e:
            print(f"    generator {i+1}/{n_generators}: ERROR {e}")

    return all_candidates


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
    return bool(_NUMERIC_ANSWER_RE.match(answer))


def _regex_extract(solution: str) -> str:
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
    a = answer.strip().lower()
    a = re.sub(r"[\\${}]", "", a)
    a = re.sub(r"\s+", "", a)
    a = a.replace(",", ".")

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
    prompt = SOLVE_PROMPT.format(problem=problem)
    for attempt in range(_SOLVE_MAX_RETRIES):
        solution = call_llm(client, model, prompt, temperature=0.7)
        if not solution:
            continue
        answer = extract_answer(client, model, solution)
        if normalize_answer(answer):
            return normalize_answer(answer), solution
        if attempt < _SOLVE_MAX_RETRIES - 1:
            print("r", end="", flush=True)
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
    n_generators: int = 5,
    problems_per_generator: int = 2,
    max_generation_rounds: int = 20,
    use_llm_extract: bool = False,
    output_path: str | None = None,
    max_workers: int = 16,
    failed_solutions: list[str] | None = None,
    solve_client: OpenAI | None = None,
    solve_model: str | None = None,
) -> Dataset:
    """Build a dataset of problems with calibrated difficulty.

    Uses `client`/`model` for problem generation (needs instruction-following)
    and `solve_client`/`solve_model` for solving/evaluation (needs math
    reasoning). When solve_client is None, falls back to client/model for both.
    """
    s_client = solve_client or client
    s_model = solve_model or model

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
        t1 = time.time()
        agreement, majority_ans, all_answers, all_solutions = solve_and_check_agreement(
            s_client, s_model, problem_text,
            n_samples=n_samples_per_problem,
            use_llm_extract=use_llm_extract,
            pool=pool,
        )
        elapsed = time.time() - t1
        return problem_text, agreement, majority_ans, all_answers, all_solutions, elapsed

    total_per_round = n_generators * problems_per_generator

    gen_label = model
    solve_label = s_model if solve_client else "(same)"

    print(f"\n{'='*70}")
    print(f"  TTT-Discover (distinct): Building dataset from hard problem")
    print(f"  Target: {n_target} problems with {target_agreement_low:.0%}-{target_agreement_high:.0%} agreement")
    print(f"  Generation: {n_generators} generators × {problems_per_generator} problems = {total_per_round}/round")
    print(f"  Samples per problem: {n_samples_per_problem}")
    print(f"  Failed solution attempts for context: {len(failed_solutions or [])}")
    print(f"  Generate model: {gen_label}")
    print(f"  Solve model:    {solve_label}")
    print(f"  Max parallel API calls: {max_workers}")
    if output_path:
        print(f"  Keeps file:  {output_path}")
        print(f"  Skips file:  {skips_path}")
    print(f"{'='*70}\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        while len(dataset.problems) < n_target and round_num < max_generation_rounds:
            round_num += 1

            print(f"--- Round {round_num}: {n_generators} generators × {problems_per_generator} problems "
                  f"({len(dataset.problems)}/{n_target} collected) ---")

            t0 = time.time()
            candidates = generate_problems_parallel(
                client, model, hard_problem,
                n_generators=n_generators,
                problems_per_generator=problems_per_generator,
                failed_solutions=failed_solutions,
                pool=pool,
            )
            gen_time = time.time() - t0
            print(f"  Generated {len(candidates)} unique candidates across {n_generators} generators ({gen_time:.1f}s)")

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
    n_generators: int = 5,
    problems_per_generator: int = 2,
    output: str | None = None,
    model: str | None = None,
    llm_extract: bool = False,
    max_workers: int = 16,
    failed_solutions: list[str] | None = None,
    use_tinker: bool = False,
    tinker_checkpoint: str | None = None,
) -> Dataset:
    # Generation always uses Vertex AI (needs instruction-following for JSON).
    gen_client, gen_default_model = get_client()
    gen_model = model or gen_default_model

    # Solving uses tinker checkpoint when requested (calibrates difficulty
    # against the fine-tuned model), otherwise falls back to the same client.
    solve_client, solve_model = None, None
    if use_tinker:
        solve_client, solve_model = get_tinker_client(tinker_checkpoint)

    from datetime import datetime
    run_dir = output or os.path.join(
        os.path.dirname(__file__),
        "runs",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "keeps.json")

    dataset = build_dataset(
        client=gen_client,
        model=gen_model,
        hard_problem=problem,
        n_target=n_problems,
        n_samples_per_problem=n_samples,
        target_agreement_low=agree_low,
        target_agreement_high=agree_high,
        n_generators=n_generators,
        problems_per_generator=problems_per_generator,
        use_llm_extract=llm_extract,
        output_path=out_path,
        max_workers=max_workers,
        failed_solutions=failed_solutions,
        solve_client=solve_client,
        solve_model=solve_model,
    )

    save_dataset(dataset, out_path)
    return dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_FAILED_SOLUTIONS_FILES = [
    "hard_attempts_ckpt50.json",
]


def _load_failed_solutions() -> list[str]:
    base = os.path.dirname(__file__)
    for name in _FAILED_SOLUTIONS_FILES:
        path = os.path.join(base, name)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        solutions: list[str] = []
        for item in data:
            if isinstance(item, str):
                solutions.append(item)
            elif isinstance(item, dict) and "reasoning" in item:
                solutions.append(item["reasoning"])
        if solutions:
            print(f"Loaded {len(solutions)} failed solutions from {name}")
            return solutions
    return []


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="TTT-Discover (distinct): generate training subproblems"
    )
    parser.add_argument(
        "--tinker", action="store_true",
        help="Use a tinker checkpoint instead of the Vertex AI API",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help=f"Tinker checkpoint path (default: ckpt-50 = {TINKER_CKPT_50})",
    )
    parser.add_argument("--n-problems", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--n-generators", type=int, default=5)
    parser.add_argument("--problems-per-generator", type=int, default=2)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    failed_solutions = _load_failed_solutions()
    if not failed_solutions:
        print("No failed solutions found. Place hard_attempts_ckpt50.json or failed_solutions.json in Stage1/.")

    run(
        problem=PROBLEM_STATEMENT,
        n_problems=args.n_problems,
        n_samples=args.n_samples,
        n_generators=args.n_generators,
        problems_per_generator=args.problems_per_generator,
        model=args.model or "openai/gpt-oss-120b-maas",
        llm_extract=True,
        failed_solutions=failed_solutions,
        use_tinker=args.tinker,
        tinker_checkpoint=args.checkpoint,
        output=args.output,
    )


if __name__ == "__main__":
    main()
