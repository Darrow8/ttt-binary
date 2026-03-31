"""
TTT-Discover: Dataset generation via LLM self-consistency.

Generates subproblems one at a time, evaluates each by sampling N solutions
in parallel, and accumulates problems that hit a target self-consistency
(agreement rate) window.
"""

import concurrent.futures
import json
import os
import random
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
                val = val.strip()
                if val and val[0] in ('"', "'"):
                    quote = val[0]
                    end = val.find(quote, 1)
                    if end != -1:
                        val = val[1:end]
                    else:
                        val = val[1:]
                else:
                    if " #" in val:
                        val = val[:val.index(" #")]
                    val = val.strip()
                os.environ.setdefault(key.strip(), val)

# ---------------------------------------------------------------------------
# Vertex AI client
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "openai/gpt-oss-120b-maas"

# ---------------------------------------------------------------------------
# Tinker checkpoint client (adapter over tinker sampling API)
# ---------------------------------------------------------------------------
TINKER_BASE_MODEL = "openai/gpt-oss-120b"


def _resolve_tinker_checkpoint_path(
    explicit: str | None, *, step: int = 50
) -> str:
    """CLI --checkpoint / TINKER_CHECKPOINT, else list this account's training checkpoints."""
    if explicit and explicit.strip():
        return explicit.strip()
    env_path = (os.environ.get("TINKER_CHECKPOINT") or "").strip()
    if env_path:
        return env_path
    import tinker

    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError(
            "TINKER_API_KEY not set. Add it to .env or export it."
        )
    service = tinker.ServiceClient()
    rest = service.create_rest_client()
    response = rest.list_user_checkpoints(limit=200).result()
    training_ckpts = [
        c for c in response.checkpoints if c.checkpoint_type == "training"
    ]
    if not training_ckpts:
        raise RuntimeError(
            "No training checkpoints on this Tinker account. Train first, "
            "or set TINKER_CHECKPOINT / pass --checkpoint with a tinker://… path."
        )
    step_tag = f"ckpt-{step:06d}"
    for ckpt in training_ckpts:
        if step_tag in ckpt.tinker_path:
            print(
                f"Using Tinker checkpoint (step {step}): {ckpt.tinker_path}  "
                f"(created {ckpt.time})"
            )
            return ckpt.tinker_path
    sample = "\n".join(
        f"  {c.tinker_path}  ({c.time})" for c in training_ckpts[:25]
    )
    raise RuntimeError(
        f"No training checkpoint matching step {step} ({step_tag!r}). "
        "Pass --checkpoint, set TINKER_CHECKPOINT, or use --tinker-step.\n"
        f"Available on this account:\n{sample}"
    )


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
        params = types.SamplingParams(temperature=temperature, context_length=32768)
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
        training_client = service.create_training_client_from_state(tinker_path)
        sampling_client = training_client.save_weights_and_get_sampling_client()

        from transformers import AutoTokenizer
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login as hf_login
            hf_login(token=hf_token)
        tokenizer = AutoTokenizer.from_pretrained(TINKER_BASE_MODEL)

        self.chat = _TinkerChat(sampling_client, tokenizer)
        self._tinker_path = tinker_path
        print(f"Tinker checkpoint ready: {tinker_path}\n")


def get_tinker_client(
    checkpoint: str | None = None,
    *,
    checkpoint_step: int = 50,
) -> tuple["TinkerClient", str]:
    path = _resolve_tinker_checkpoint_path(checkpoint, step=checkpoint_step)
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
You are designing a training problem for training on ONE hard source problem.
Your job is to create a problem that preserves the same mathematical bottleneck as the
source problem, while being smaller, cleaner, and self-contained.

Your goal is NOT to create a random "similar-looking" problem or a parametric variant
(e.g. changing the number of variables). Your goal is to create a problem whose solution
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
2. Pick ONE technique or subskill and design a single problem that would teach
   EXACTLY that technique. The problem must be structurally distinct from the source.

The generated problem MUST satisfy all of the following:
- It has a SINGLE numerical final answer that is a DECIMAL NUMBER (not a symbolic
  expression like 1/π², 6/π², √2, ln 2, etc.). If the exact answer is irrational
  or a fraction, the problem must ask the solver to ROUND to 4 decimal places.
  For example, instead of "Find the value of C" where C = 6/π², write
  "Find the value of C, rounded to 4 decimal places" (answer: 0.6079).
- It is self-contained.
- It isolates one real bottleneck or subskill from the source problem.
- It is NOT a parametric variant of the source problem.
- It avoids fake complexity and decorative algebraic clutter.
- CRITICAL: The problem must be HARD — comparable to a research-level or competition
  math problem. A strong LLM should get it WRONG 20-40% of the time.
  DO NOT generate textbook exercises, definitions, or routine calculations.
  The problem should require COMBINING techniques or applying them in a non-obvious way.
- The problem cannot have subparts, it must be a single problem with a single answer.

## Output

First briefly state which technique/subskill you are targeting and why.
Then write the problem statement between these exact delimiters:

===PROBLEM START===
<the problem text>
===PROBLEM END===
"""

SOLVE_PROMPT = """\
## Problem

{problem}

When you are done, write your final numerical answer on the very last line \
in exactly this format:

\\boxed{{<answer>}}

For example: \\boxed{{0.6079}} or \\boxed{{42}}

Your answer MUST be a single decimal number (e.g. 0.6079, 42, 3.1416). \
Do NOT write symbolic expressions like 1/π², 6/π², √2, or ln 2. \
If the answer is irrational or a fraction, round to 4 decimal places.

Solve the problem above step by step.

"""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
_PROBLEM_DELIM_RE = re.compile(
    r"===\s*PROBLEM\s+START\s*===\s*\n(.*?)\n\s*===\s*PROBLEM\s+END\s*===",
    re.DOTALL,
)


def generate_similar_problems(
    client: OpenAI,
    model: str,
    hard_problem: str,
    failed_solutions: list[str] | None = None,
) -> list[dict]:
    """Prompt the LLM to generate one similar problem.

    To stay within context limits, only one randomly-chosen failed solution
    is included per call. Across many calls this still exposes the generator
    to all attempted techniques.
    """
    if failed_solutions:
        pick = random.choice(failed_solutions)
        solutions_block = f"### Attempt 1\n\n{pick}"
    else:
        solutions_block = "(No attempted solutions available.)"

    prompt = GENERATE_PROBLEMS_PROMPT.format(
        problem=hard_problem,
        failed_solutions=solutions_block,
    )
    raw = call_llm(client, model, prompt, temperature=0.8)

    problems = [m.group(1).strip() for m in _PROBLEM_DELIM_RE.finditer(raw)
                if m.group(1).strip()]

    if not problems:
        print(f"  [warn] No delimited problem found in response")
        print(f"  [warn] Raw response (first 500 chars): {raw[:500]}")
        return []

    return [{"problem": p} for p in problems]


_NUMERIC_ANSWER_RE = re.compile(
    r"^[+-]?\d+([.,]\d+)?(/\d+)?$"
)


def _is_numeric_answer(answer: str) -> bool:
    return bool(_NUMERIC_ANSWER_RE.match(answer))


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


def extract_answer(solution: str) -> str:
    """Extract the answer from a solution using regex only (no LLM)."""
    ans = _regex_extract(solution)
    if ans and "<" in ans and ">" in ans:
        return ""
    return ans


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
) -> tuple[str, str]:
    prompt = SOLVE_PROMPT.format(problem=problem)
    for attempt in range(_SOLVE_MAX_RETRIES):
        solution = call_llm(client, model, prompt, temperature=0.7)
        if not solution:
            continue
        answer = extract_answer(solution)
        if answer:
            return answer, solution
        if attempt < _SOLVE_MAX_RETRIES - 1:
            print("r", end="", flush=True)
    answer = extract_answer(solution) if solution else ""
    return answer, solution


def solve_and_check_agreement(
    client: OpenAI,
    model: str,
    problem: str,
    n_samples: int = 10,
    pool: concurrent.futures.ThreadPoolExecutor | None = None,
) -> tuple[float, str, list[str], list[str]]:
    futures = [
        pool.submit(_solve_one, client, model, problem)
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
    output_path: str | None = None,
    max_workers: int = 16,
    failed_solutions: list[str] | None = None,
    solve_client: OpenAI | None = None,
    solve_model: str | None = None,
) -> Dataset:
    """Build a dataset of problems with calibrated difficulty.

    Generates one problem at a time, evaluates it with parallel solve samples,
    and keeps it if it lands in the target agreement window.

    Uses `client`/`model` for generation and `solve_client`/`solve_model` for
    evaluation. When solve_client is None, falls back to client/model for both.
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

    if output_path and os.path.exists(output_path):
        try:
            with open(output_path) as f:
                existing = json.load(f)
            for p in existing.get("problems", []):
                entry = GeneratedProblem(**p)
                dataset.problems.append(entry)
                seen_problems.add(entry.problem)
            if dataset.problems:
                print(f"  Resumed {len(dataset.problems)} existing problems from {output_path}")
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

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

    gen_label = model
    solve_label = s_model if solve_client else "(same)"

    print(f"\n{'='*70}")
    print(f"  TTT-Discover: Building dataset from hard problem")
    print(f"  Target: {n_target} problems with {target_agreement_low:.0%}-{target_agreement_high:.0%} agreement")
    print(f"  Samples per problem: {n_samples_per_problem}")
    print(f"  Failed solution attempts for context: {len(failed_solutions or [])}")
    print(f"  Generate model: {gen_label}")
    print(f"  Solve model:    {solve_label}")
    print(f"  Max parallel solve workers: {max_workers}")
    if output_path:
        print(f"  Keeps file:  {output_path}")
        print(f"  Skips file:  {skips_path}")
    print(f"{'='*70}\n")

    candidate_num = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        while len(dataset.problems) < n_target:
            candidate_num += 1
            print(f"--- Candidate {candidate_num} ({len(dataset.problems)}/{n_target} kept) ---")

            t0 = time.time()
            candidates = generate_similar_problems(
                client, model, hard_problem,
                failed_solutions=failed_solutions,
            )
            gen_time = time.time() - t0

            if not candidates:
                print(f"  Generation failed ({gen_time:.1f}s), retrying...")
                continue

            problem_text = candidates[0]["problem"]
            if problem_text in seen_problems:
                print(f"  Duplicate problem, retrying...")
                continue
            seen_problems.add(problem_text)

            print(f"  Generated problem ({gen_time:.1f}s), evaluating with {n_samples_per_problem} samples...")

            t1 = time.time()
            agreement, majority_ans, all_answers, all_solutions = solve_and_check_agreement(
                s_client, s_model, problem_text,
                n_samples=n_samples_per_problem,
                pool=pool,
            )
            eval_time = time.time() - t1

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

            if kept:
                dataset.problems.append(entry)
            else:
                skipped_problems.append(entry)
            _flush()

            print(f"  {agreement:.0%} agreement ({eval_time:.1f}s) -> {status}")
            if kept:
                print(f"    majority answer: {majority_ans[:80]}")
            print(f"    totals: {len(dataset.problems)} kept, {len(skipped_problems)} skipped\n")

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
    output: str | None = None,
    model: str | None = None,
    max_workers: int = 16,
    failed_solutions: list[str] | None = None,
    use_tinker: bool = False,
    tinker_checkpoint: str | None = None,
    tinker_checkpoint_step: int = 50,
) -> Dataset:
    if use_tinker:
        tinker_client, tinker_model = get_tinker_client(
            tinker_checkpoint, checkpoint_step=tinker_checkpoint_step
        )
        gen_client, gen_model = tinker_client, model or tinker_model
        solve_client, solve_model = tinker_client, tinker_model
    else:
        gen_client, gen_default_model = get_client()
        gen_model = model or gen_default_model
        solve_client, solve_model = None, None

    from datetime import datetime
    run_dir = output or os.path.join(
        os.path.dirname(__file__),
        "runs",
        datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}",
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
    "attempted_answers.json",
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
        help=(
            "Tinker checkpoint tinker://… path. If omitted, uses TINKER_CHECKPOINT "
            "env or lists your account's checkpoints (--tinker-step)."
        ),
    )
    parser.add_argument(
        "--tinker-step",
        type=int,
        default=50,
        help=(
            "When --checkpoint is omitted, use the training checkpoint whose path "
            "contains ckpt-NNNNNN for this step (default: 50)."
        ),
    )
    parser.add_argument("--n-problems", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=10)
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
        model=args.model or "openai/gpt-oss-120b-maas",
        failed_solutions=failed_solutions,
        use_tinker=args.tinker,
        tinker_checkpoint=args.checkpoint,
        tinker_checkpoint_step=args.tinker_step,
        output=args.output,
    )


if __name__ == "__main__":
    main()
