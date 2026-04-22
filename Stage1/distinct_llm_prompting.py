"""
TTT-Discover: Dataset generation via LLM self-consistency.

Generates subproblems in small batches (default 2 per call), evaluates each by
sampling N solutions in parallel, and accumulates problems that hit a target
self-consistency (agreement rate) window. Each generation call sees all available
failed solution traces for richer context and asks the LLM to target a different
technique per problem.
"""

import concurrent.futures
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import asdict, dataclass, field

from openai import OpenAI
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Ensure the repo root is on sys.path so `from pipeline_stages.*` resolves
# regardless of how this script is invoked.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline_stages.dedupe import DedupeIndex  # noqa: E402

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
                        val = val[: val.index(" #")]
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


def _resolve_tinker_checkpoint_path(explicit: str | None, *, step: int = 50) -> str:
    """CLI --checkpoint / TINKER_CHECKPOINT, else list this account's training checkpoints."""
    if explicit and explicit.strip():
        return explicit.strip()
    env_path = (os.environ.get("TINKER_CHECKPOINT") or "").strip()
    if env_path:
        return env_path
    import tinker

    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY not set. Add it to .env or export it.")
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
    sample = "\n".join(f"  {c.tinker_path}  ({c.time})" for c in training_ckpts[:25])
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

    def create(
        self,
        *,
        model: str = "",
        messages: list | None = None,
        temperature: float = 1.2,
        **_kwargs,
    ) -> _FakeResponse:
        from tinker import types

        text = self._tokenizer.apply_chat_template(
            messages or [],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        prompt = types.ModelInput.from_ints(ids)
        params = types.SamplingParams(temperature=temperature, context_length=32768)
        result = self._sampling_client.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=params,
        ).result()
        content = self._tokenizer.decode(
            result.sequences[0].tokens,
            skip_special_tokens=True,
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
            raise RuntimeError("TINKER_API_KEY not set. Add it to .env or export it.")
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


def load_hard_problem(problem_set: str, problem_id: str) -> tuple[str, dict]:
    """Look up one hard problem in a JSONL set file. Returns (statement, full_row)."""
    with open(problem_set) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("id") == problem_id:
                return row["statement"], row
    raise KeyError(f"id {problem_id!r} not found in {problem_set}")


def load_problem_from_txt(path: str) -> str:
    """Load a single problem statement from a plain .txt file."""
    p = os.path.abspath(path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Problem file not found: {p}")
    if not p.lower().endswith(".txt"):
        raise ValueError(f"--problem-path must be a .txt file, got: {path}")
    with open(p, encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        raise ValueError(f"Problem file is empty: {p}")
    return text


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
    temperature: float = 1.2,
) -> str:
    @retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(8))
    def _call():
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        if isinstance(response, str):
            raise ValueError(f"vertex returned raw string: {response[:300]!r}")
        if not response.choices:
            raise ValueError("LLM response has no choices")
        content = response.choices[0].message.content or ""
        if not content:
            raise ValueError("empty response content")
        return content

    try:
        return _call()
    except Exception as e:
        print(f"  [warn] LLM call failed after all retries: {e}")
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
You should also not create a problem with a flawed or confusing premise.

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
   estimates, Tauberian theorems, algebraic geometry, intersection theory, etc.)
2. For each technique, consider: what is a simpler, self-contained problem that would
   teach EXACTLY that technique? Write out a set of techniques and concepts that are
   needed to solve the problem.
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
- The problem cannot have subparts, it must be a single problem with a single answer.
- The problem must not have a confusing or contradictory premise.
- CRITICAL: The problem must be HARD — comparable to a research-level or competition
  math problem. A strong LLM should get it WRONG 20-40% of the time.
  DO NOT generate textbook exercises, definitions, or routine calculations like
  "compute ζ(2)", "count divisors of 60", "evaluate a standard limit", or
  "state a well-known asymptotic formula". These are TOO EASY.
  The problem should require COMBINING techniques or applying them in a non-obvious way.

## Output

For EACH problem, first briefly state which technique/subskill you are targeting and why.
Then write the problem statement between these exact delimiters:

===PROBLEM START===
<the problem text>
===PROBLEM END===

You must produce exactly {batch_size} delimited problem blocks.
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
    batch_size: int = 2,
) -> list[dict]:
    """Prompt the LLM to generate a small batch of similar problems.

    All available failed solutions are included so the generator sees the
    full range of attempted techniques. Use ``_load_failed_solutions(…,
    k_top=N)`` upstream to cap how many traces are fed in.
    """
    if failed_solutions:
        solutions_block = "\n\n---\n\n".join(
            f"### Attempt {i + 1}\n\n{sol}"
            for i, sol in enumerate(failed_solutions)
        )
    else:
        solutions_block = "(No attempted solutions available.)"

    prompt = GENERATE_PROBLEMS_PROMPT.format(
        problem=hard_problem,
        failed_solutions=solutions_block,
        batch_size=batch_size,
    )
    raw = call_llm(client, model, prompt, temperature=1.2)

    problems = [
        m.group(1).strip()
        for m in _PROBLEM_DELIM_RE.finditer(raw)
        if m.group(1).strip()
    ]

    if not problems:
        print(f"  [warn] No delimited problem found in response")
        print(f"  [warn] Raw response (first 500 chars): {raw[:500]}")
        return []

    return [{"problem": p} for p in problems]


_NUMERIC_ANSWER_RE = re.compile(r"^[+-]?\d+([.,]\d+)?(/\d+)?$")


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
        solution,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return ""


def extract_answer(solution: str) -> str:
    """Extract the answer from a solution using regex."""
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
        solution = call_llm(client, model, prompt, temperature=1.0)
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


QUALITY_SCORE_PROMPT = """\
You are evaluating a candidate training problem for its usefulness in learning
to solve a specific hard target problem. Score the candidate on a 0 to 10 integer
scale, where:

  10 = excellent — directly teaches a reusable subskill needed for the target,
       well-posed with a single unambiguous numerical answer, appropriately
       difficult (neither trivial nor impossible), isolates one real bottleneck.
   9 = very good — clearly relevant, well-posed, only minor imperfections.
 7-8 = useful but not perfectly targeted.
 5-6 = weakly related; teaches tangential skills.
 3-4 = mostly off-topic or poorly constructed.
 0-2 = bad (ill-posed, multi-part, contradictory, or irrelevant).

## Target problem (the one we ultimately want the model to solve)

{target}

## Candidate training problem

{candidate}

## Your task

Rate the candidate as a single integer 0-10. Be strict — only give 9 or 10 to
problems you would actually include in a careful training curriculum for this
target.

Return JSON only, no other text:
{{"score": <integer 0-10>, "reason": "<1-2 sentence justification>"}}
"""


def _parse_quality_score(raw: str) -> int:
    m = re.search(r"\{.*?\}", raw or "", re.DOTALL)
    if not m:
        return -1
    try:
        return int(json.loads(m.group(0)).get("score", -1))
    except (json.JSONDecodeError, TypeError, ValueError):
        return -1


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
    gen_workers: int = 8,
    failed_solutions: list[str] | None = None,
    solve_client: OpenAI | None = None,
    solve_model: str | None = None,
    quality_threshold: int | None = None,
    use_dedupe: bool = True,
    problems_per_call: int = 2,
) -> Dataset:
    """Build a dataset of problems with calibrated difficulty.

    Pipelines `gen_workers` candidates concurrently through gen+eval. Each
    worker generates one candidate problem and evaluates it with up to
    `max_workers` parallel solve samples; results stream back to a shared
    keeps/skips list as soon as they're ready.

    When `quality_threshold` is set (e.g. 9), each candidate that passes the
    60-80% agreement window is ALSO scored 0-10 by the same model, inline.
    Only candidates scoring >= threshold count toward `n_target`. This lets
    you do ``--n-problems 100 --quality-threshold 9`` and get exactly 100
    quality-rated subproblems without needing a separate judging pass.

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
    dedupe = DedupeIndex() if use_dedupe else None
    seen_problems: set[str] = set()  # used only when use_dedupe=False

    if output_path and os.path.exists(output_path):
        try:
            with open(output_path) as f:
                existing = json.load(f)
            for p in existing.get("problems", []):
                entry = GeneratedProblem(**p)
                dataset.problems.append(entry)
                if use_dedupe:
                    dedupe.add(entry.problem)
                else:
                    seen_problems.add(entry.problem)
            if dataset.problems:
                print(
                    f"  Resumed {len(dataset.problems)} existing problems from {output_path}"
                )
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    # Snapshot counters after any resume pre-seed so the final summary reports
    # only this run's adds/drops, not the ones re-hydrated from disk.
    if dedupe is not None:
        dedupe_baseline_kept = dedupe.n_kept
        dedupe_baseline_exact = dedupe.n_exact_dropped
        dedupe_baseline_fuzzy = dedupe.n_fuzzy_dropped
    else:
        dedupe_baseline_kept = 0
        dedupe_baseline_exact = 0
        dedupe_baseline_fuzzy = 0

    if output_path:
        skips_path = os.path.join(os.path.dirname(output_path), "skips.json")
    else:
        skips_path = None

    def _flush() -> None:
        if not output_path:
            return
        _save_atomic(
            output_path,
            {
                "source_problem": dataset.source_problem,
                "target_agreement_low": dataset.target_agreement_low,
                "target_agreement_high": dataset.target_agreement_high,
                "n_problems": len(dataset.problems),
                "problems": [asdict(p) for p in dataset.problems],
            },
        )
        _save_atomic(
            skips_path,
            {
                "source_problem": dataset.source_problem,
                "target_agreement_low": dataset.target_agreement_low,
                "target_agreement_high": dataset.target_agreement_high,
                "n_problems": len(skipped_problems),
                "problems": [asdict(p) for p in skipped_problems],
            },
        )

    gen_label = model
    solve_label = s_model if solve_client else "(same)"

    print(f"\n{'=' * 70}")
    print(f"  TTT-Discover: Building dataset from hard problem")
    print(
        f"  Target: {n_target} problems with {target_agreement_low:.0%}-{target_agreement_high:.0%} agreement"
    )
    print(f"  Samples per problem: {n_samples_per_problem}")
    print(f"  Problems per gen call:      {problems_per_call}")
    print(f"  Failed solution attempts for context: {len(failed_solutions or [])}")
    print(f"  Generate model: {gen_label}")
    print(f"  Solve model:    {solve_label}")
    print(f"  Max parallel solve workers: {max_workers}")
    print(f"  Gen pipeline workers:       {gen_workers}")
    if quality_threshold is not None:
        print(f"  Quality threshold:          >= {quality_threshold}/10 (inline judge)")
    if output_path:
        print(f"  Keeps file:  {output_path}")
        print(f"  Skips file:  {skips_path}")
    print(f"{'=' * 70}\n")

    seen_lock = threading.Lock()
    state_lock = threading.Lock()
    candidate_counter = {"n": 0}

    def _gen_and_eval_one(solve_pool: concurrent.futures.ThreadPoolExecutor):
        """Generate a batch of candidates and evaluate each one.

        Returns a list of result dicts (one per candidate in the batch).
        """
        with state_lock:
            candidate_counter["n"] += 1
            cn = candidate_counter["n"]

        t0 = time.time()
        candidates = generate_similar_problems(
            client,
            model,
            hard_problem,
            failed_solutions=failed_solutions,
            batch_size=problems_per_call,
        )
        gen_time = time.time() - t0

        if not candidates:
            return [{"kind": "gen_failed", "candidate_num": cn, "gen_time": gen_time}]

        results = []
        for ci, cand in enumerate(candidates):
            problem_text = cand["problem"]
            sub_label = f"{cn}.{ci + 1}"

            with seen_lock:
                if use_dedupe:
                    if not dedupe.add(problem_text):
                        results.append({"kind": "duplicate", "candidate_num": sub_label, "gen_time": gen_time})
                        continue
                else:
                    if problem_text in seen_problems:
                        results.append({"kind": "duplicate", "candidate_num": sub_label, "gen_time": gen_time})
                        continue
                    seen_problems.add(problem_text)

            t1 = time.time()
            agreement, majority_ans, all_answers, all_solutions = solve_and_check_agreement(
                s_client,
                s_model,
                problem_text,
                n_samples=n_samples_per_problem,
                pool=solve_pool,
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

            quality_score = None
            if kept and quality_threshold is not None:
                qprompt = QUALITY_SCORE_PROMPT.format(
                    target=hard_problem, candidate=problem_text,
                )
                raw = call_llm(client, model, qprompt, temperature=0.4)
                quality_score = _parse_quality_score(raw)
                if quality_score < quality_threshold:
                    kept = False
                    status = f"skip (quality {quality_score}/10 < {quality_threshold})"
                else:
                    status = f"KEEP (quality {quality_score}/10)"

            entry = GeneratedProblem(
                problem=problem_text,
                ground_truth_answer=majority_ans,
                agreement_rate=agreement,
                all_answers=all_answers,
                all_solutions=all_solutions,
                n_samples=n_samples_per_problem,
            )
            results.append({
                "kind": "evaluated",
                "candidate_num": sub_label,
                "gen_time": gen_time,
                "eval_time": eval_time,
                "kept": kept,
                "status": status,
                "agreement": agreement,
                "majority_ans": majority_ans,
                "quality_score": quality_score,
                "entry": entry,
            })

        return results

    # Two pools: outer pool runs gen+eval workers, inner pool runs the
    # parallel solve samples for whichever candidate is currently evaluating.
    # Sizing: gen_workers concurrent candidates × n_samples_per_problem solve
    # calls each → up to gen_workers * n_samples concurrent solve calls.
    solve_pool_size = max(max_workers, gen_workers * n_samples_per_problem)

    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=gen_workers) as gen_pool,
        concurrent.futures.ThreadPoolExecutor(
            max_workers=solve_pool_size
        ) as solve_pool,
    ):
        in_flight: set[Future] = set()

        def _submit_more():
            with state_lock:
                target_inflight = max(0, n_target - len(dataset.problems))
            target_inflight = min(target_inflight, gen_workers)
            while len(in_flight) < target_inflight:
                in_flight.add(gen_pool.submit(_gen_and_eval_one, solve_pool))

        _submit_more()

        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in done:
                in_flight.discard(fut)
                try:
                    batch_results = fut.result()
                except Exception as e:
                    print(f"  [error] worker raised: {e}")
                    continue

                for result in batch_results:
                    cn = result["candidate_num"]
                    kind = result["kind"]

                    if kind == "gen_failed":
                        print(
                            f"--- Candidate {cn} ---  generation failed "
                            f"({result['gen_time']:.1f}s), continuing"
                        )
                    elif kind == "duplicate":
                        print(f"--- Candidate {cn} ---  duplicate, continuing")
                    elif kind == "evaluated":
                        entry = result["entry"]
                        with state_lock:
                            if result["kept"]:
                                dataset.problems.append(entry)
                            else:
                                skipped_problems.append(entry)
                            kept_count = len(dataset.problems)
                            skipped_count = len(skipped_problems)
                            _flush()
                        print(
                            f"--- Candidate {cn} ---  gen {result['gen_time']:.1f}s "
                            f"+ eval {result['eval_time']:.1f}s  "
                            f"{result['agreement']:.0%} -> {result['status']}"
                        )
                        if result["kept"]:
                            print(f"    majority answer: {result['majority_ans'][:80]}")
                        print(f"    totals: {kept_count} kept, {skipped_count} skipped")

            with state_lock:
                done_yet = len(dataset.problems) >= n_target
            if not done_yet:
                _submit_more()

    print(f"{'=' * 70}")
    print(
        f"  Dataset complete: {len(dataset.problems)} kept, {len(skipped_problems)} skipped"
    )
    avg_agreement = (
        sum(p.agreement_rate for p in dataset.problems) / len(dataset.problems)
        if dataset.problems
        else 0
    )
    print(f"  Average agreement rate (kept): {avg_agreement:.1%}")
    if dedupe is not None:
        kept_this_run = dedupe.n_kept - dedupe_baseline_kept
        exact_this_run = dedupe.n_exact_dropped - dedupe_baseline_exact
        fuzzy_this_run = dedupe.n_fuzzy_dropped - dedupe_baseline_fuzzy
        print(
            f"  dedupe: kept {kept_this_run}, "
            f"dropped {exact_this_run + fuzzy_this_run} "
            f"(exact={exact_this_run}, fuzzy={fuzzy_this_run})"
        )
    print(f"{'=' * 70}\n")

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
    gen_workers: int = 8,
    failed_solutions: list[str] | None = None,
    use_tinker: bool = False,
    tinker_checkpoint: str | None = None,
    tinker_checkpoint_step: int = 50,
    quality_threshold: int | None = None,
    use_dedupe: bool = True,
    problems_per_call: int = 2,
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
        gen_workers=gen_workers,
        failed_solutions=failed_solutions,
        solve_client=solve_client,
        solve_model=solve_model,
        quality_threshold=quality_threshold,
        use_dedupe=use_dedupe,
        problems_per_call=problems_per_call,
    )

    save_dataset(dataset, out_path)
    return dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_failed_solutions(path: str, k_top: int = 3, seed: int = 0) -> list[str]:
    """Load top-K most-common-answer reasoning traces from a Stage 0 file.

    Stage 0 saves all N (typically 500) per-sample results in `results`. Per
    the paper methodology, we group by extracted answer, take the K most
    common answer groups, and return one randomly-chosen reasoning trace per
    group. This gives Stage 1 K diverse failed traces that represent the
    most prevalent ways the base model gets the target wrong.
    """
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        results = data
    elif isinstance(data, dict):
        results = data.get("results") or []
    else:
        return []

    by_answer: dict[str, list[str]] = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        ans = r.get("answer", "")
        reasoning = r.get("reasoning", "")
        if not ans or not reasoning:
            continue
        by_answer.setdefault(ans, []).append(reasoning)

    if not by_answer:
        return []

    top_groups = sorted(by_answer.items(), key=lambda x: -len(x[1]))[:k_top]
    rng = random.Random(seed)
    solutions = [rng.choice(traces) for _, traces in top_groups]
    if solutions:
        print(
            f"Loaded {len(solutions)} failed solutions "
            f"(top-{k_top} answer groups out of {len(by_answer)} unique) "
            f"from {path}"
        )
    return solutions


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="TTT-Discover (distinct): generate training subproblems"
    )
    parser.add_argument(
        "--problem-path",
        type=str,
        required=True,
        help="Path to a .txt file containing the problem statement (e.g. data/target-problems/conics.txt)",
    )
    parser.add_argument(
        "--runs-subdir",
        type=str,
        default=None,
        help=(
            "Subdirectory name under runs/ for outputs (default: the .txt filename "
            "without extension). Set when the filename does not match how later stages "
            "name their run folders."
        ),
    )
    parser.add_argument(
        "--failed-solutions",
        type=str,
        default=None,
        help=(
            "Path to failed-attempts JSON. Defaults to runs/<runs-subdir>/base_attempts.json "
            "(--runs-subdir defaults to the .txt basename). Pass an explicit path to override."
        ),
    )
    parser.add_argument(
        "--tinker",
        action="store_true",
        help="Use a tinker checkpoint instead of the Vertex AI API",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
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
    parser.add_argument(
        "--gen-workers",
        type=int,
        default=8,
        help="Concurrent gen+eval pipeline workers (default 8)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Solve-pool worker hint (default 16)",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--quality-threshold", type=int, default=None,
                        help=(
                            "If set, each candidate that passes the 60-80%% agreement "
                            "window is also scored 0-10 inline by the same model. "
                            "Only candidates >= this threshold count toward --n-problems. "
                            "Example: --quality-threshold 9"
                        ))
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Override run dir. Defaults to runs/<runs-subdir>/stage1/<timestamp>/ "
            "(see --runs-subdir)."
        ),
    )
    parser.add_argument("--no-dedupe", action="store_true",
                        help="Disable fuzzy dedup; fall back to exact-string equality only (for ablation)")
    parser.add_argument(
        "--max-traces",
        type=int,
        default=None,
        help=(
            "Max number of failed-solution traces to include in each generation "
            "prompt (default: all). Lower values reduce prompt length at the cost "
            "of less context per call."
        ),
    )
    parser.add_argument(
        "--problems-per-call",
        type=int,
        default=2,
        help="Number of problems to request per generation LLM call (default 2)",
    )
    args = parser.parse_args()

    problem_text = load_problem_from_txt(args.problem_path)
    problem_stem = os.path.splitext(
        os.path.basename(os.path.abspath(args.problem_path))
    )[0]
    runs_subdir = (args.runs_subdir or "").strip() or problem_stem
    print(
        f"\nLoaded problem from {args.problem_path} "
        f"(runs/{runs_subdir}/, {len(problem_text)} chars)\n"
    )

    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    runs_root = os.path.join(repo_root, "runs", runs_subdir)

    failed_path = args.failed_solutions or os.path.join(repo_root, "data", "reasoning-traces", f"{runs_subdir}.json")
    k_top = args.max_traces if args.max_traces is not None else 5
    failed_solutions = _load_failed_solutions(failed_path, k_top=k_top)
    if not failed_solutions:
        print(
            f"No failed solutions at {failed_path}. "
            f"Add traces to data/reasoning-traces/{runs_subdir}.json or pass --failed-solutions explicitly."
        )

    if args.output:
        run_dir = args.output
    else:
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
        run_dir = os.path.join(runs_root, "stage1", ts)

    run(
        problem=problem_text,
        n_problems=args.n_problems,
        n_samples=args.n_samples,
        model=args.model or "openai/gpt-oss-120b-maas",
        failed_solutions=failed_solutions,
        use_tinker=args.tinker,
        tinker_checkpoint=args.checkpoint,
        tinker_checkpoint_step=args.tinker_step,
        output=run_dir,
        gen_workers=args.gen_workers,
        quality_threshold=args.quality_threshold,
        max_workers=args.max_workers,
        use_dedupe=not args.no_dedupe,
        problems_per_call=args.problems_per_call,
    )


if __name__ == "__main__":
    main()
