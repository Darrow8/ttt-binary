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

import anthropic
import boto3
from openai import AsyncAzureOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

# ── Model config ──────────────────────────────────────────────────────────────
DEFAULT_CLAUDE_MODEL = "claude-opus-4-7"
MAX_TOKENS = 65536
MAX_CONCURRENT = 10       # concurrent requests in flight per provider
MAX_CONCURRENT_BEDROCK = 2  # Bedrock has strict TPM limits

DEFAULT_AZURE_ENDPOINT = "https://hdarrow3-8892-resource.openai.azure.com"
DEFAULT_AZURE_API_VERSION = "2025-04-01-preview"
DEFAULT_AZURE_DEPLOYMENT = "gpt-5.5-1"
DEFAULT_GPT55_REASONING_EFFORT = "high"

BEDROCK_CLAUDE_MODEL = "arn:aws:bedrock:us-east-1:322356185166:inference-profile/us.anthropic.claude-opus-4-7"
BEDROCK_REGION = "us-east-1"

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

After you have fully worked through the solution, write your final answer \
on the very last line in exactly this format:

\\boxed{{<answer>}}

For example: \\boxed{{0.6079}} or \\boxed{{42}}

If the answer is a finite number, write it as a decimal rounded to 4 decimal \
places (e.g. 0.6079, 42.0000). Do NOT write symbolic expressions like 1/π² or √2. \
If the answer is infinite, write \\boxed{{\\infty}} or \\boxed{{-\\infty}}. \
If the problem is ill-posed or the answer does not exist, write \\boxed{{\\text{{undefined}}}}.

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
    """One GPT-5.5 completion via the Azure Responses API with reasoning."""
    async with sem:
        t0 = time.time()
        resp = await client.responses.create(
            model=deployment,
            input=[
                {"role": "system", "content": "You are an expert mathematician. Answer all academic math problems fully and completely."},
                {"role": "user", "content": prompt},
            ],
            reasoning={"effort": reasoning_effort},
            max_output_tokens=MAX_TOKENS,
        )
        print(f"    [gpt55]  {time.time()-t0:.1f}s", flush=True)
        parts: list[str] = []
        for item in resp.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        parts.append(content.text)
        return "\n".join(parts)


# ── AWS Bedrock Claude query ──────────────────────────────────────────────────
async def query_claude_bedrock(api_key: str, prompt: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        t0 = time.time()
        loop = asyncio.get_event_loop()

        def _call():
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key
            client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

            @retry(wait=wait_random_exponential(multiplier=2, max=120), stop=stop_after_attempt(8))
            def _invoke():
                return client.converse(
                    modelId=BEDROCK_CLAUDE_MODEL,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    inferenceConfig={"maxTokens": 32000},
                    performanceConfig={"latency": "standard"},
                )

            resp = _invoke()
            return resp["output"]["message"]["content"][0]["text"]

        result = await loop.run_in_executor(None, _call)
        print(f"    [bedrock] {time.time()-t0:.1f}s", flush=True)
        return result


# ── Per-provider helper: gather n_samples in parallel and pick best answer ───
async def _gather_provider(
    name: str,
    coro_factory,
    n_samples: int,
    pid,
) -> tuple[list[str], list[str], str | None]:
    """Run n_samples concurrent calls; return (answers, raw_outputs, best)."""
    tasks = [coro_factory() for _ in range(n_samples)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    answers: list[str] = []
    raw_outputs: list[str] = []
    for r in results:
        if isinstance(r, Exception):
            print(f"  [!] {name} error on problem {pid}: {r}", file=sys.stderr)
            raw_outputs.append(f"ERROR: {r}")
            continue
        raw_outputs.append(r)
        ans = extract_answer(r)
        if ans:
            answers.append(ans)
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
    claude_model: str,
    azure_deployment: str,
    gpt55_reasoning_effort: str,
    sems: dict[str, asyncio.Semaphore],
    n_samples: int,
    existing_record: dict | None = None,
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

    # Decide which providers actually need to be (re-)queried for this problem.
    coros = {}
    if claude_client is not None and not _has_provider_result(existing, "claude"):
        coros["claude"] = _gather_provider(
            "claude",
            lambda: query_claude(claude_client, claude_model, prompt, sems["claude"]),
            n_samples, pid,
        )
    if azure_client is not None and not _has_provider_result(existing, "gpt55"):
        coros["gpt55"] = _gather_provider(
            "gpt55",
            lambda: query_gpt55(
                azure_client, azure_deployment, prompt, sems["gpt55"],
                gpt55_reasoning_effort,
            ),
            n_samples, pid,
        )
    if bedrock_api_key is not None and not _has_provider_result(existing, "bedrock"):
        coros["bedrock"] = _gather_provider(
            "bedrock",
            lambda: query_claude_bedrock(bedrock_api_key, prompt, sems["bedrock"]),
            n_samples, pid,
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
    for provider in ("claude", "gpt55", "bedrock"):
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

    return out


# ── Per-dataset driver ────────────────────────────────────────────────────────
async def run_dataset(
    path: str,
    out_dir: Path,
    claude_client: anthropic.AsyncAnthropic | None,
    azure_client: AsyncAzureOpenAI | None,
    bedrock_api_key: str | None,
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
        ordered = _ordered_results()
        summary = [{k: v for k, v in r.items() if k != "raw_outputs"} for r in ordered]
        output_data = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "models": {
                    "claude": claude_model if claude_client else None,
                    "gpt55": azure_deployment if azure_client else None,
                    "bedrock": BEDROCK_CLAUDE_MODEL if bedrock_api_key else None,
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

    async def _run_one(p):
        async with sem_problems:
            return await evaluate_problem(
                p, claude_client, azure_client, bedrock_api_key,
                claude_model, azure_deployment, gpt55_reasoning_effort,
                sems, n_samples,
                existing_record=by_id.get(p["id"]),
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
        elapsed = time.time() - t0
        bits = []
        if claude_client:
            bits.append(f"claude {nc}/{na}" + (f" ({100*nc/na:.1f}%)" if na else ""))
        if azure_client:
            bits.append(f"gpt55 {ng}/{ga}" + (f" ({100*ng/ga:.1f}%)" if ga else ""))
        if bedrock_api_key:
            bits.append(f"bedrock {nb}/{ba}" + (f" ({100*nb/ba:.1f}%)" if ba else ""))
        print(f"  [{n_done}/{len(all_problems)}] " + ", ".join(bits) + f" -- {elapsed:.0f}s")

    nc, na = _accuracy("claude")
    ng, ga = _accuracy("gpt55")
    nb, ba = _accuracy("bedrock")
    print(f"  DONE  ({len(by_id)} problems)")
    if claude_client:
        print(f"    claude  : {nc}/{na} correct" + (f" ({100*nc/na:.1f}%)" if na else ""))
    if azure_client:
        print(f"    gpt55   : {ng}/{ga} correct" + (f" ({100*ng/ga:.1f}%)" if ga else ""))
    if bedrock_api_key:
        print(f"    bedrock : {nb}/{ba} correct" + (f" ({100*nb/ba:.1f}%)" if ba else ""))
    print(f"  Summary: {summary_path}")
    print(f"  Full   : {full_path}")


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
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Provider setup ────────────────────────────────────────────────────────
    claude_client: anthropic.AsyncAnthropic | None = None
    azure_client: AsyncAzureOpenAI | None = None
    bedrock_api_key: str | None = None

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

    if claude_client is None and azure_client is None and bedrock_api_key is None:
        print("\nERROR: no providers enabled. Set ANTHROPIC_API_KEY and/or "
              "AZURE_AI_FOUNDRY_API_KEY_FRONTIER in .env, "
              "or remove the corresponding --skip-* flag.", file=sys.stderr)
        sys.exit(1)

    sems = {
        "claude":  asyncio.Semaphore(MAX_CONCURRENT),
        "gpt55":   asyncio.Semaphore(MAX_CONCURRENT),
        "bedrock": asyncio.Semaphore(MAX_CONCURRENT_BEDROCK),
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
            claude_client, azure_client, bedrock_api_key,
            args.model, azure_deployment, args.gpt55_reasoning_effort,
            sems, args.samples, args.batch_size,
        )

    print("\nAll datasets processed.")


if __name__ == "__main__":
    asyncio.run(main())
