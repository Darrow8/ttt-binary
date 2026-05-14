#!/usr/bin/env python3
"""
Query Claude (Anthropic) and GPT-5.5 (Azure AI Foundry) on math-problem
subproblem datasets and record each model's answer per problem.

Companion to eval_frontier_models.py, but uses adaptive extended thinking on
Claude and the Azure Responses API on GPT-5.5, and is aware of the
heterogeneous schemas used across this project's datasets.

Supported input shapes (auto-detected by file contents):
  1. JSONL where each line is {"id", "prompt", "reference"}
       e.g. problems/conics-50.jsonl
  2. JSON list of {"id", "prompt", "reference"}
       e.g. conics-verifier-filter.json, general-concepts.json,
            conics-bigboy.json
  3. JSON dict with a "problems" list whose items use
     {"problem", "ground_truth_answer", ...} (plus optional skill/agreement_rate)
       e.g. keeps_deduped_100.json

Usage:
    python3 eval_claude_datasets.py <dataset> [<dataset> ...] \
        [--out-dir claude_eval_results] \
        [--samples 1] [--batch-size 5] \
        [--model claude-opus-4-6] [--azure-deployment gpt-5.5-1] \
        [--gpt55-reasoning-effort high] \
        [--skip-claude] [--skip-gpt55]

Required env vars (in .env or shell):
    ANTHROPIC_API_KEY                  – for Claude
    AZURE_AI_FOUNDRY_API_KEY_FRONTIER  – for Azure GPT-5.5

Optional Azure overrides (have sensible defaults):
    AZURE_OPENAI_ENDPOINT              – default: https://hdarrow3-8892-resource.openai.azure.com
    AZURE_OPENAI_API_VERSION           – default: 2025-04-01-preview
    AZURE_OPENAI_DEPLOYMENT            – default: gpt-5.5-1

Outputs (per dataset, named after the input stem):
    <out-dir>/<stem>_eval.json         summary (no raw outputs)
    <out-dir>/<stem>_eval_full.json    full results including raw responses

Both files are written incrementally after every batch and the script will
resume from <stem>_eval_full.json if it already exists.
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
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Boto3 (Bedrock) calls run in a thread pool via run_in_executor — make sure
# we have enough threads to actually run MAX_CONCURRENT_BEDROCK in parallel.
import concurrent.futures
asyncio_default_executor = concurrent.futures.ThreadPoolExecutor(max_workers=32)

import anthropic
import boto3
from openai import AsyncAzureOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

try:
    from google import genai
    from google.genai.types import GenerateContentConfig, ThinkingConfig
except ImportError:
    genai = None
    GenerateContentConfig = None
    ThinkingConfig = None

# ── Model config ──────────────────────────────────────────────────────────────
DEFAULT_CLAUDE_MODEL = "claude-opus-4-6"
CLAUDE_THINKING_BUDGET = 32000
MAX_TOKENS = 65536
MAX_CONCURRENT = 10        # concurrent requests in flight per provider
MAX_CONCURRENT_BEDROCK = 6 # Bedrock TPM is tight with 32k thinking
MAX_CONCURRENT_GEMINI = 3  # Gemini Vertex AI has TPM limits

DEFAULT_AZURE_ENDPOINT = "https://hdarrow3-8892-resource.openai.azure.com"
DEFAULT_AZURE_API_VERSION = "2025-04-01-preview"
DEFAULT_AZURE_DEPLOYMENT = "gpt-5.5-1"
DEFAULT_GPT55_REASONING_EFFORT = "high"

BEDROCK_CLAUDE_MODEL = "us.anthropic.claude-opus-4-6-v1"
BEDROCK_REGION = "us-east-2"  # primary region (for logging)
BEDROCK_REGIONS = ["us-east-2", "us-east-1", "us-west-2", "us-west-1"]  # rotate on throttle
BEDROCK_THINKING_BUDGET = 16000

GEMINI_MODEL = "gemini-3.1-pro-preview"
GEMINI_THINKING_BUDGET = 32768

# Same prompt as eval_frontier_models.py / eval_gpt54_repeat.py so results are
# directly comparable to the rest of the eval suite.
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


# ── Numeric-aware answer comparison ──────────────────────────────────────────
def answers_match(a: str, b: str) -> bool:
    """Return True if a and b represent the same number, or are equal strings."""
    if a == b:
        return True
    try:
        fa, fb = float(a.replace(",", "")), float(b.replace(",", ""))
        return abs(fa - fb) < 1e-6
    except (ValueError, AttributeError):
        return False


def _is_non_numeric_answer(ans: str) -> bool:
    """True if the answer is empty / infinity / undefined / not parseable as a number."""
    s = str(ans).lower().strip()
    if not s:
        return True
    if "undefined" in s or "infty" in s or "infinity" in s:
        return True
    try:
        float(s.replace(",", ""))
        return False
    except ValueError:
        return True


def _normalize_numeric_answer(ans: str) -> str:
    """Canonicalize numeric answers so '32' and '32.0000' bucket together."""
    try:
        f = float(str(ans).replace(",", ""))
        return str(int(f)) if f == int(f) else str(round(f, 4))
    except (ValueError, AttributeError):
        return str(ans)


# ── Answer extraction (matches inference/infer.py / eval_frontier_models.py) ──
def extract_answer(solution: str) -> str:
    """Extract the answer via regex. Returns empty string if nothing found."""
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


# ── Dataset loading: handle the three shapes used in this project ────────────
def load_dataset(path: str) -> list[dict]:
    """Return a list of {id, prompt, reference} dicts regardless of source."""
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
        if isinstance(data, list):
            raw_items = data
        elif isinstance(data, dict) and "problems" in data:
            raw_items = data["problems"]
        else:
            raise ValueError(
                f"Unrecognized JSON shape in {path}: top-level "
                f"{type(data).__name__} without a 'problems' key."
            )

    out: list[dict] = []
    for i, item in enumerate(raw_items):
        # Tolerate {prompt, reference, id} and {problem, ground_truth_answer, skill}
        prompt = item.get("prompt") or item.get("problem")
        if prompt is None:
            raise ValueError(
                f"{path} item {i}: missing 'prompt' (or 'problem') field"
            )
        reference = item.get("reference")
        if reference is None:
            reference = item.get("ground_truth_answer")
        ident = item.get("id")
        if ident is None:
            # Use skill+index for keeps_deduped_100.json which has no id field
            skill = item.get("skill")
            ident = f"{skill}_{i}" if skill else i
        out.append({
            "id": ident,
            "prompt": prompt,
            "reference": str(reference).strip() if reference is not None else None,
        })
    return out


# ── Claude query ──────────────────────────────────────────────────────────────
async def query_claude(
    client: anthropic.AsyncAnthropic,
    model: str,
    prompt: str,
    sem: asyncio.Semaphore,
) -> str:
    """One Claude completion. Uses streaming + adaptive extended thinking."""
    async with sem:
        t0 = time.time()
        async with client.messages.stream(
            model=model,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            resp = await stream.get_final_message()
        print(f"    [claude] {time.time()-t0:.1f}s", flush=True)
        for block in resp.content:
            if block.type == "text":
                return block.text
        return ""


# ── Azure GPT-5.5 query ───────────────────────────────────────────────────────
async def query_gpt55(
    client: AsyncAzureOpenAI,
    deployment: str,
    prompt: str,
    sem: asyncio.Semaphore,
    reasoning_effort: str = DEFAULT_GPT55_REASONING_EFFORT,
) -> str:
    """One GPT-5.5 completion via the Azure Responses API with reasoning.
    Retries on bare refusals from the content filter (no math content)."""
    async with sem:
        t0 = time.time()

        @retry(
            wait=wait_random_exponential(multiplier=1, max=30),
            stop=stop_after_attempt(4),
            reraise=True,
        )
        async def _call():
            resp = await client.responses.create(
                model=deployment,
                input=[{"role": "user", "content": prompt}],
                reasoning={"effort": reasoning_effort},
                max_output_tokens=MAX_TOKENS,
            )
            parts: list[str] = []
            for item in resp.output:
                if item.type == "message":
                    for content in item.content:
                        if content.type == "output_text":
                            parts.append(content.text)
            text = "\n".join(parts)
            # Detect Azure content-filter refusals so retry kicks in
            if text.strip().lower().startswith("i'm sorry") and len(text) < 200:
                raise RuntimeError(f"Azure content filter refusal: {text!r}")
            return text

        text = await _call()
        print(f"    [gpt55]  {time.time()-t0:.1f}s", flush=True)
        return text


# ── AWS Bedrock Claude query ──────────────────────────────────────────────────
async def query_claude_bedrock(api_key: str, prompt: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        t0 = time.time()
        loop = asyncio.get_event_loop()

        def _call():
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key
            from botocore.config import Config
            cfg = Config(retries={"max_attempts": 1}, read_timeout=600, connect_timeout=30)
            clients = {r: boto3.client("bedrock-runtime", region_name=r, config=cfg) for r in BEDROCK_REGIONS}
            attempt = {"n": 0}
            max_attempts = max(10, len(BEDROCK_REGIONS) * 3)

            @retry(wait=wait_random_exponential(multiplier=2, max=120), stop=stop_after_attempt(max_attempts))
            def _invoke():
                attempt["n"] += 1
                region = BEDROCK_REGIONS[(attempt["n"] - 1) % len(BEDROCK_REGIONS)]
                if attempt["n"] > 1:
                    print(f"    [bedrock] retry {attempt['n']}/{max_attempts} ({region})", flush=True)
                try:
                    return clients[region].converse(
                        modelId=BEDROCK_CLAUDE_MODEL,
                        messages=[{"role": "user", "content": [{"text": prompt}]}],
                        inferenceConfig={"maxTokens": 64000, "temperature": 1.0},
                        additionalModelRequestFields={
                            "thinking": {"type": "enabled", "budget_tokens": BEDROCK_THINKING_BUDGET},
                        },
                        performanceConfig={"latency": "standard"},
                    )
                except Exception as e:
                    print(f"    [bedrock] attempt {attempt['n']} ({region}) failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
                    raise

            resp = _invoke()
            # With thinking enabled, content has a reasoningContent block before text.
            for block in resp["output"]["message"]["content"]:
                if "text" in block:
                    return block["text"]
            return ""

        result = await loop.run_in_executor(asyncio_default_executor, _call)
        print(f"    [bedrock] {time.time()-t0:.1f}s", flush=True)
        return result


# ── Gemini query (Vertex AI) ──────────────────────────────────────────────────
async def query_gemini(client, prompt: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        t0 = time.time()

        @retry(wait=wait_random_exponential(multiplier=2, max=120), stop=stop_after_attempt(8))
        async def _call():
            return await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=GenerateContentConfig(
                    thinking_config=ThinkingConfig(thinking_budget=GEMINI_THINKING_BUDGET),
                ),
            )

        resp = await _call()
        print(f"    [gemini] {time.time()-t0:.1f}s", flush=True)
        return resp.text or ""


# ── Per-provider helper: gather n_samples in parallel and pick best answer ───
async def _gather_provider(
    name: str,
    coro_factory,
    n_samples: int,
    pid,
    on_sample=None,
) -> tuple[list[str], list[str], str | None]:
    """Run n_samples concurrent calls; return (answers, raw_outputs, best).
    If `on_sample` is provided, it is called after each sample lands with
    (provider_name, answer_str, raw_output_str)."""
    async def _one(idx):
        try:
            r = await coro_factory()
        except Exception as e:
            print(f"  [!] {name} error on problem {pid}: {e}", file=sys.stderr)
            return idx, None, f"ERROR: {e}"
        ans = extract_answer(r)
        print(f"    pid={pid} {name} sample {idx+1}: {ans!r}", flush=True)
        return idx, ans, r

    tasks = [asyncio.create_task(_one(i)) for i in range(n_samples)]
    answers: list[str] = []
    raw_outputs: list[str] = []
    for fut in asyncio.as_completed(tasks):
        idx, ans, raw = await fut
        raw_outputs.append(raw)
        if ans:
            answers.append(ans)
        if on_sample is not None:
            await on_sample(name, ans, raw)
    best = Counter(answers).most_common(1)[0][0] if answers else None
    return answers, raw_outputs, best


def _has_provider_result(record: dict, provider: str) -> bool:
    """True if `record` has a non-empty result for `provider`."""
    val = record.get(f"{provider}_answers")
    return bool(val)  # None or [] ⇒ not yet queried / no answer extracted


def _migrate_legacy_record(record: dict) -> dict:
    """Bring a record from the old single-provider schema up to the multi-provider one.

    Old format:  raw_outputs is a flat list of Claude responses.
    New format:  raw_outputs is a dict keyed by provider name.
    """
    raw = record.get("raw_outputs")
    if isinstance(raw, list):
        record = dict(record)  # don't mutate the caller's copy
        record["raw_outputs"] = {"claude": raw}
    elif raw is None:
        record = dict(record)
        record["raw_outputs"] = {}
    return record


# ── Per-problem evaluation ────────────────────────────────────────────────────
async def evaluate_problem(
    problem: dict,
    claude_client: anthropic.AsyncAnthropic | None,
    azure_client: AsyncAzureOpenAI | None,
    bedrock_api_key: str | None,
    gemini_client,
    claude_model: str,
    azure_deployment: str,
    gpt55_reasoning_effort: str,
    sems: dict[str, asyncio.Semaphore],
    n_samples: int,
    existing_record: dict | None = None,
    on_sample=None,
) -> dict:
    """Query whichever providers are enabled-and-not-yet-recorded, then merge
    their results into ``existing_record`` (if provided) so that prior work is
    preserved across runs."""
    pid = problem["id"]
    raw_problem = problem["prompt"]
    reference = problem["reference"]
    prompt = SOLVE_PROMPT.format(problem=raw_problem)

    existing = _migrate_legacy_record(existing_record) if existing_record else {
        "id": pid,
        "prompt": raw_problem,
        "reference": reference,
        "raw_outputs": {},
    }

    # Per-provider sample-landed callback (forwards to the run-level callback,
    # passing the problem id so run_dataset can patch the right record).
    async def _wrap_on_sample(provider, ans, raw):
        if on_sample is not None:
            await on_sample(pid, provider, ans, raw)

    # Decide which providers actually need to be (re-)queried for this problem.
    coros = {}
    if claude_client is not None and not _has_provider_result(existing, "claude"):
        coros["claude"] = _gather_provider(
            "claude",
            lambda: query_claude(claude_client, claude_model, prompt, sems["claude"]),
            n_samples, pid, on_sample=_wrap_on_sample,
        )
    if azure_client is not None and not _has_provider_result(existing, "gpt55"):
        coros["gpt55"] = _gather_provider(
            "gpt55",
            lambda: query_gpt55(
                azure_client, azure_deployment, prompt, sems["gpt55"],
                gpt55_reasoning_effort,
            ),
            n_samples, pid, on_sample=_wrap_on_sample,
        )
    if bedrock_api_key is not None and not _has_provider_result(existing, "bedrock"):
        coros["bedrock"] = _gather_provider(
            "bedrock",
            lambda: query_claude_bedrock(bedrock_api_key, prompt, sems["bedrock"]),
            n_samples, pid, on_sample=_wrap_on_sample,
        )
    if gemini_client is not None and not _has_provider_result(existing, "gemini"):
        coros["gemini"] = _gather_provider(
            "gemini",
            lambda: query_gemini(gemini_client, prompt, sems["gemini"]),
            n_samples, pid, on_sample=_wrap_on_sample,
        )

    if coros:
        keys = list(coros.keys())
        gathered = await asyncio.gather(*coros.values())
        by_provider = dict(zip(keys, gathered))
    else:
        by_provider = {}

    # Start from the existing record so untouched providers carry through.
    out: dict = {
        "id": pid,
        "prompt": raw_problem,
        "reference": reference,
        "raw_outputs": dict(existing.get("raw_outputs") or {}),
    }
    for provider in ("claude", "gpt55", "bedrock", "gemini"):
        out[f"{provider}_answers"] = existing.get(f"{provider}_answers")
        out[f"{provider}_best_answer"] = existing.get(f"{provider}_best_answer")
        out[f"{provider}_agrees_with_reference"] = existing.get(f"{provider}_agrees_with_reference")

    # Overlay anything we just queried.
    for provider, (answers, raw_outputs, best) in by_provider.items():
        correct = (
            best is not None
            and reference is not None
            and answers_match(best, reference)
        )
        out[f"{provider}_answers"] = answers
        out[f"{provider}_best_answer"] = best
        out[f"{provider}_agrees_with_reference"] = correct
        out["raw_outputs"][provider] = raw_outputs

    # Cross-model majority vote, filtering out infinity / undefined responses.
    all_samples = []
    for provider in ("claude", "gpt55", "bedrock", "gemini"):
        for ans in (out.get(f"{provider}_answers") or []):
            all_samples.append({"provider": provider, "answer": ans})

    n_infinity = sum(
        1 for s in all_samples
        if "infty" in str(s["answer"]).lower() or "infinity" in str(s["answer"]).lower()
    )
    n_undefined = sum(
        1 for s in all_samples
        if "undefined" in str(s["answer"]).lower()
    )
    numeric_norm = [
        _normalize_numeric_answer(s["answer"])
        for s in all_samples
        if not _is_non_numeric_answer(s["answer"])
    ]
    counter = Counter(numeric_norm)
    majority = counter.most_common(1)[0][0] if counter else None

    out["all_samples"] = all_samples
    out["n_infinity"] = n_infinity
    out["n_undefined"] = n_undefined
    out["pooled_distribution"] = dict(counter)
    out["majority_vote"] = majority
    out["majority_count"] = counter[majority] if majority else 0
    out["pooled_n"] = len(all_samples)
    out["pooled_n_numeric"] = len(numeric_norm)

    return out


# ── Per-dataset driver ────────────────────────────────────────────────────────
async def run_dataset(
    path: str,
    out_dir: Path,
    claude_client: anthropic.AsyncAnthropic | None,
    azure_client: AsyncAzureOpenAI | None,
    bedrock_api_key: str | None,
    gemini_client,
    claude_model: str,
    azure_deployment: str,
    gpt55_reasoning_effort: str,
    sems: dict[str, asyncio.Semaphore],
    n_samples: int,
    batch_size: int,
):
    dataset_name = Path(path).stem
    print(f"\n=== {dataset_name} ===")
    all_problems = load_dataset(path)
    print(f"  Loaded {len(all_problems)} problems from {path}")

    summary_path = out_dir / f"{dataset_name}_eval.json"
    full_path = out_dir / f"{dataset_name}_eval_full.json"

    # Resume from prior full-results file if present.
    # Backwards-compatible: also accept the old <stem>_claude_full.json.
    legacy_full_path = out_dir / f"{dataset_name}_claude_full.json"
    by_id: dict = {}  # id -> record (the source of truth; merged across runs)
    source_for_resume = None
    if full_path.exists():
        source_for_resume = full_path
    elif legacy_full_path.exists():
        source_for_resume = legacy_full_path
    if source_for_resume is not None:
        try:
            with open(source_for_resume) as f:
                prior = json.load(f)
            for rec in prior:
                by_id[rec["id"]] = _migrate_legacy_record(rec)
            print(f"  Resuming: {len(by_id)} prior records in {source_for_resume.name}")
        except (json.JSONDecodeError, KeyError):
            print(f"  Warning: could not parse {source_for_resume}, starting fresh")
            by_id = {}

    # A problem is "pending" if at least one *enabled* provider hasn't been
    # queried for it yet. Carry-over records keep their existing data;
    # evaluate_problem will only run the missing providers and merge.
    enabled_providers: list[str] = []
    if claude_client is not None:
        enabled_providers.append("claude")
    if azure_client is not None:
        enabled_providers.append("gpt55")
    if bedrock_api_key is not None:
        enabled_providers.append("bedrock")
    if gemini_client is not None:
        enabled_providers.append("gemini")

    def _record_needs_work(rec: dict) -> bool:
        return any(not _has_provider_result(rec, p) for p in enabled_providers)

    pending = [
        p for p in all_problems
        if _record_needs_work(by_id.get(p["id"], {}))
    ]
    n_fully_done = len(all_problems) - len(pending)
    if n_fully_done:
        print(f"  {n_fully_done} fully done for enabled providers, {len(pending)} pending")
    else:
        print(f"  {len(pending)} pending")

    t0 = time.time()

    def _accuracy(provider: str) -> tuple[int, int]:
        recs = list(by_id.values())
        answered = [r for r in recs if r.get(f"{provider}_best_answer") is not None]
        n_correct = sum(1 for r in answered if r.get(f"{provider}_agrees_with_reference"))
        return n_correct, len(answered)

    def _ordered_results() -> list[dict]:
        """Return records in dataset (problem) order, dropping ids not in by_id."""
        seen = set()
        ordered = []
        for p in all_problems:
            if p["id"] in by_id and p["id"] not in seen:
                ordered.append(by_id[p["id"]])
                seen.add(p["id"])
        # Append any orphan records (id not in current dataset) at the end so
        # we never silently drop data on save.
        for rid, rec in by_id.items():
            if rid not in seen:
                ordered.append(rec)
        return ordered

    def save_progress():
        n_claude_correct, n_claude_answered = _accuracy("claude")
        n_gpt55_correct, n_gpt55_answered = _accuracy("gpt55")
        n_bedrock_correct, n_bedrock_answered = _accuracy("bedrock")
        n_gemini_correct, n_gemini_answered = _accuracy("gemini")
        ordered = _ordered_results()
        summary = [{k: v for k, v in r.items() if k != "raw_outputs"} for r in ordered]
        output_data = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "models": {
                    "claude": claude_model if claude_client else None,
                    "gpt55": azure_deployment if azure_client else None,
                    "bedrock": BEDROCK_CLAUDE_MODEL if bedrock_api_key else None,
                    "gemini": GEMINI_MODEL if gemini_client else None,
                },
                "gpt55_reasoning_effort": gpt55_reasoning_effort if azure_client else None,
                "dataset": str(path),
                "n_problems": len(ordered),
                "n_total": len(all_problems),
                "claude": {
                    "n_answered": n_claude_answered,
                    "n_correct": n_claude_correct,
                    "accuracy": round(100 * n_claude_correct / n_claude_answered, 2) if n_claude_answered else 0,
                },
                "gpt55": {
                    "n_answered": n_gpt55_answered,
                    "n_correct": n_gpt55_correct,
                    "accuracy": round(100 * n_gpt55_correct / n_gpt55_answered, 2) if n_gpt55_answered else 0,
                },
                "bedrock": {
                    "n_answered": n_bedrock_answered,
                    "n_correct": n_bedrock_correct,
                    "accuracy": round(100 * n_bedrock_correct / n_bedrock_answered, 2) if n_bedrock_answered else 0,
                },
                "gemini": {
                    "n_answered": n_gemini_answered,
                    "n_correct": n_gemini_correct,
                    "accuracy": round(100 * n_gemini_correct / n_gemini_answered, 2) if n_gemini_answered else 0,
                },
                "infinity_total": sum(r.get("n_infinity", 0) for r in ordered),
                "undefined_total": sum(r.get("n_undefined", 0) for r in ordered),
                "problems_with_any_infinity": sum(1 for r in ordered if r.get("n_infinity", 0) > 0),
                "problems_with_any_undefined": sum(1 for r in ordered if r.get("n_undefined", 0) > 0),
                "samples_per_problem": n_samples,
                "completed": len(ordered) == len(all_problems),
                "elapsed_s": round(time.time() - t0, 1),
            },
            "results": summary,
        }
        tmp = str(summary_path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, summary_path)

        tmp_full = str(full_path) + ".tmp"
        with open(tmp_full, "w") as f:
            json.dump(ordered, f, indent=2, ensure_ascii=False)
        os.replace(tmp_full, full_path)

    sem_problems = asyncio.Semaphore(batch_size)
    save_lock = asyncio.Lock()

    async def _on_sample(pid, provider, answer, raw):
        """Patch the partial record for this problem and save."""
        async with save_lock:
            rec = by_id.get(pid)
            if rec is None:
                # find the problem prompt/reference for the partial record
                prob = next((p for p in all_problems if p["id"] == pid), None)
                rec = {
                    "id": pid,
                    "prompt": prob["prompt"] if prob else "",
                    "reference": prob["reference"] if prob else None,
                    "raw_outputs": {},
                }
                by_id[pid] = rec
            rec.setdefault("raw_outputs", {}).setdefault(provider, []).append(raw)
            answers_field = f"{provider}_answers"
            if rec.get(answers_field) is None:
                rec[answers_field] = []
            if answer:
                rec[answers_field].append(answer)
            save_progress()

    async def _run_one(p):
        async with sem_problems:
            return await evaluate_problem(
                p, claude_client, azure_client, bedrock_api_key, gemini_client,
                claude_model, azure_deployment, gpt55_reasoning_effort,
                sems, n_samples,
                existing_record=by_id.get(p["id"]),
                on_sample=_on_sample,
            )

    tasks = [asyncio.create_task(_run_one(p)) for p in pending]
    for coro in asyncio.as_completed(tasks):
        rec = await coro
        by_id[rec["id"]] = rec
        save_progress()
        n_done = len(by_id)
        nc, na = _accuracy("claude")
        ng, ga = _accuracy("gpt55")
        nb, ba = _accuracy("bedrock")
        ngm, gma = _accuracy("gemini")
        elapsed = time.time() - t0
        bits = []
        if claude_client:
            bits.append(f"claude {nc}/{na}" + (f" ({100*nc/na:.1f}%)" if na else ""))
        if azure_client:
            bits.append(f"gpt55 {ng}/{ga}" + (f" ({100*ng/ga:.1f}%)" if ga else ""))
        if bedrock_api_key:
            bits.append(f"bedrock {nb}/{ba}" + (f" ({100*nb/ba:.1f}%)" if ba else ""))
        if gemini_client:
            bits.append(f"gemini {ngm}/{gma}" + (f" ({100*ngm/gma:.1f}%)" if gma else ""))
        print(f"  [{n_done}/{len(all_problems)}] " + ", ".join(bits) + f" -- {elapsed:.0f}s")

    nc, na = _accuracy("claude")
    ng, ga = _accuracy("gpt55")
    nb, ba = _accuracy("bedrock")
    ngm, gma = _accuracy("gemini")
    print(f"  DONE  ({len(by_id)} problems)")
    if claude_client:
        print(f"    claude  : {nc}/{na} correct" + (f" ({100*nc/na:.1f}%)" if na else ""))
    if azure_client:
        print(f"    gpt55   : {ng}/{ga} correct" + (f" ({100*ng/ga:.1f}%)" if ga else ""))
    if bedrock_api_key:
        print(f"    bedrock : {nb}/{ba} correct" + (f" ({100*nb/ba:.1f}%)" if ba else ""))
    if gemini_client:
        print(f"    gemini  : {ngm}/{gma} correct" + (f" ({100*ngm/gma:.1f}%)" if gma else ""))
    print(f"  Summary: {summary_path}")
    print(f"  Full   : {full_path}")


# ── Fixed-dataset writer ──────────────────────────────────────────────────────
def _write_fixed_dataset(datasets: list[str], out_dir: Path, output_path: str) -> None:
    """After eval, write a JSONL with `reference` replaced by majority_vote per problem.
    Skip problems with no consensus (majority_vote=None)."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    dropped = 0
    changed = 0
    entries: list[dict] = []

    for ds_path in datasets:
        stem = Path(ds_path).stem
        full_path = out_dir / f"{stem}_eval_full.json"
        if not full_path.exists():
            print(f"[fixed-dataset] skip {ds_path}: no eval results at {full_path}", file=sys.stderr)
            continue
        with open(full_path) as f:
            records = json.load(f)
        for r in records:
            mv = r.get("majority_vote")
            if mv is None:
                dropped += 1
                continue
            ref = r.get("reference")
            if ref is not None and not answers_match(str(mv), str(ref)):
                changed += 1
            entries.append({"id": r["id"], "prompt": r["prompt"], "reference": str(mv)})
            written += 1

    is_jsonl = out_path.suffix.lower() == ".jsonl"
    with open(out_path, "w") as f:
        if is_jsonl:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        else:
            json.dump(entries, f, indent=2, ensure_ascii=False)

    print(f"\n[fixed-dataset] wrote {written} entries to {out_path}")
    print(f"[fixed-dataset]   dropped (no consensus): {dropped}")
    print(f"[fixed-dataset]   changed vs original ref: {changed}")


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(
        description="Query Claude and GPT-5.5 (Azure) on math problem datasets",
    )
    parser.add_argument(
        "datasets", nargs="+",
        help="One or more dataset paths (.json or .jsonl)",
    )
    parser.add_argument(
        "--out-dir", default="claude_eval_results",
        help="Output directory for per-dataset results (default: claude_eval_results)",
    )
    parser.add_argument(
        "--samples", type=int, default=1,
        help="Samples per problem per model (default: 1)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10,
        help="Number of problems to process concurrently (default: 10)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_CLAUDE_MODEL,
        help=f"Claude model id (default: {DEFAULT_CLAUDE_MODEL})",
    )
    parser.add_argument(
        "--azure-deployment", default=None,
        help=f"Azure deployment name (default: $AZURE_OPENAI_DEPLOYMENT or {DEFAULT_AZURE_DEPLOYMENT})",
    )
    parser.add_argument(
        "--azure-endpoint", default=None,
        help=f"Azure endpoint (default: $AZURE_OPENAI_ENDPOINT or {DEFAULT_AZURE_ENDPOINT})",
    )
    parser.add_argument(
        "--azure-api-version", default=None,
        help=f"Azure API version (default: $AZURE_OPENAI_API_VERSION or {DEFAULT_AZURE_API_VERSION})",
    )
    parser.add_argument(
        "--gpt55-reasoning-effort", default=DEFAULT_GPT55_REASONING_EFFORT,
        choices=["minimal", "low", "medium", "high"],
        help=f"GPT-5.5 reasoning effort (default: {DEFAULT_GPT55_REASONING_EFFORT})",
    )
    parser.add_argument(
        "--skip-claude", action="store_true",
        help="Disable the Claude provider for this run",
    )
    parser.add_argument(
        "--skip-gpt55", action="store_true",
        help="Disable the GPT-5.5 provider for this run",
    )
    parser.add_argument(
        "--skip-bedrock", action="store_true",
        help="Disable the Claude (Bedrock) provider for this run",
    )
    parser.add_argument(
        "--skip-gemini", action="store_true",
        help="Disable the Gemini provider for this run",
    )
    parser.add_argument(
        "--write-fixed-dataset", default=None,
        help="If set, after eval write a JSONL with `reference` replaced by the cross-model majority vote.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Provider setup ────────────────────────────────────────────────────────
    claude_client: anthropic.AsyncAnthropic | None = None
    azure_client: AsyncAzureOpenAI | None = None
    bedrock_api_key: str | None = None
    gemini_client = None

    # Claude
    if args.skip_claude:
        print("  [SKIP] Claude (--skip-claude)")
    elif os.environ.get("ANTHROPIC_API_KEY"):
        claude_client = anthropic.AsyncAnthropic()
        print(f"  Claude  : {args.model}")
    else:
        print("  [SKIP] Claude -- ANTHROPIC_API_KEY not set")

    # Azure GPT-5.5
    azure_endpoint = args.azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", DEFAULT_AZURE_ENDPOINT)
    azure_api_version = args.azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION", DEFAULT_AZURE_API_VERSION)
    azure_deployment = args.azure_deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", DEFAULT_AZURE_DEPLOYMENT)
    azure_api_key = os.environ.get("AZURE_AI_FOUNDRY_API_KEY_FRONTIER")

    if args.skip_gpt55:
        print("  [SKIP] GPT-5.5 (--skip-gpt55)")
    elif azure_api_key:
        azure_client = AsyncAzureOpenAI(
            api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            api_version=azure_api_version,
        )
        print(f"  GPT-5.5 : {azure_deployment} via {azure_endpoint} (api={azure_api_version}, effort={args.gpt55_reasoning_effort})")
    else:
        print("  [SKIP] GPT-5.5 -- AZURE_AI_FOUNDRY_API_KEY_FRONTIER not set")

    if args.skip_bedrock:
        print("  [SKIP] Claude Bedrock (--skip-bedrock)")
    elif os.environ.get("AWS_BEDROCK_API_KEY"):
        bedrock_api_key = os.environ["AWS_BEDROCK_API_KEY"]
        print(f"  Bedrock : {BEDROCK_CLAUDE_MODEL}")
    else:
        print("  [SKIP] Claude Bedrock -- AWS_BEDROCK_API_KEY not set")

    if args.skip_gemini:
        print("  [SKIP] Gemini (--skip-gemini)")
    elif genai is None:
        print("  [SKIP] Gemini -- google-genai not installed (pip install google-genai)")
    elif os.environ.get("GOOGLE_CLOUD_PROJECT"):
        gemini_client = genai.Client(
            vertexai=True,
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
        )
        print(f"  Gemini  : {GEMINI_MODEL} (Vertex AI, project={os.environ['GOOGLE_CLOUD_PROJECT']})")
    else:
        print("  [SKIP] Gemini -- GOOGLE_CLOUD_PROJECT not set")

    if (claude_client is None and azure_client is None
        and bedrock_api_key is None and gemini_client is None):
        print("\nERROR: no providers enabled. Set ANTHROPIC_API_KEY and/or "
              "AZURE_AI_FOUNDRY_API_KEY_FRONTIER and/or AWS_BEDROCK_API_KEY "
              "and/or GOOGLE_CLOUD_PROJECT in .env, "
              "or remove the corresponding --skip-* flag.", file=sys.stderr)
        sys.exit(1)

    sems = {
        "claude":  asyncio.Semaphore(MAX_CONCURRENT),
        "gpt55":   asyncio.Semaphore(MAX_CONCURRENT),
        "bedrock": asyncio.Semaphore(MAX_CONCURRENT_BEDROCK),
        "gemini":  asyncio.Semaphore(MAX_CONCURRENT_GEMINI),
    }

    print(f"  Datasets        : {args.datasets}")
    print(f"  Output dir      : {out_dir}/")
    print(f"  Samples / problem: {args.samples}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Concurrency cap : {MAX_CONCURRENT} per provider")

    for path in args.datasets:
        if not Path(path).exists():
            print(f"[SKIP] {path} does not exist", file=sys.stderr)
            continue
        await run_dataset(
            path, out_dir,
            claude_client, azure_client, bedrock_api_key, gemini_client,
            args.model, azure_deployment, args.gpt55_reasoning_effort,
            sems, args.samples, args.batch_size,
        )

    if args.write_fixed_dataset:
        _write_fixed_dataset(args.datasets, out_dir, args.write_fixed_dataset)

    print("\nAll datasets processed.")


if __name__ == "__main__":
    asyncio.run(main())
