"""
Stage 4 — LLM-judge selection of K subproblems from N filtered candidates.

Refactor of simple-query/query_tinker_ckpt50.py with everything parameterized
per hard-problem id. Two judge backends supported:

  --judge base       (default) Vertex base model, no tinker dependency.
                     Uniform across all ids, no circularity.
  --judge tinker     Load a tinker checkpoint as the judge (matches the
                     historical conics run; pass --judge-checkpoint).

Reads:  runs/<id>/filtered_keeps.json
Writes: runs/<id>/selected.json          (id-keyed list of selected problems)
        runs/<id>/selection_log.json     (per-batch responses for reproducibility)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "Stage1"))
import distinct_llm_prompting as s1  # noqa: E402

DEFAULT_BATCH_SIZE = 20
DEFAULT_TARGET = 50
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MODEL_CONTEXT = 32768
DEFAULT_MAX_OUTPUT_TOKENS = 6144


def _save_atomic(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ── Prompt (port of simple-query/query_tinker_ckpt50.py:72) ──────────────────

def build_prompt(
    *,
    target_problem: str,
    batch_index: int,
    batch_json: str,
    num_batches: int,
    batch_size: int,
    batch_len: int,
    select_count: int,
) -> str:
    b = batch_index + 1
    lo = batch_index * batch_size + 1
    hi = batch_index * batch_size + batch_len
    return f"""
You are given a difficult target problem in advanced mathematics. Your task is \\textbf{{not to solve it yet}}, but to determine which earlier problems would best prepare someone to solve it.

You are processing \\textbf{{batch {b} of {num_batches}}}. The JSON below lists exactly {batch_len} candidate problems (problem ids roughly {lo}--{hi} in the original list order).

From \\textbf{{only}} this list of {batch_len} problems, select \\textbf{{exactly {select_count} problems}} that a student should study first to build knowledge and intuition for the target problem.

When selecting, prioritize problems that help develop:

\\begin{{itemize}}
\\item Relevant \\textbf{{concepts and theory}}
\\item \\textbf{{Intermediate techniques}} used in the target problem
\\item \\textbf{{Simpler versions or special cases}} of the same ideas
\\item Problems that introduce the ideas appearing in the target problem
\\end{{itemize}}

Avoid problems that are:
\\begin{{itemize}}
\\item Unrelated to the key ideas
\\item Much harder than the target problem
\\item Purely computational with no conceptual overlap
\\end{{itemize}}

Return \\textbf{{exactly {select_count} problems}} from this batch.

\\textbf{{Output format:}} Return a JSON array of objects, each with an \\texttt{{"id"}} field (integer, matching the problem id) and a \\texttt{{"reason"}} field (1--2 sentence explanation).
Example: \\texttt{{[{{"id": 5, "reason": "..."}}, {{"id": 12, "reason": "..."}}]}}

Return ONLY the JSON array—no other text before or after it.

Focus on building a \\textbf{{progressive learning path}} toward the final problem.

\\bigskip

\\textbf{{Target problem}}

{target_problem}

\\bigskip

\\textbf{{Available problems (batch {b} of {num_batches})}}

{batch_json}
"""


def parse_ids(response: str) -> list[int]:
    """Extract problem ids from a JSON-array response."""
    match = re.search(r"\[.*\]", response, re.DOTALL)
    if match:
        try:
            arr = json.loads(match.group())
            return [int(item["id"]) for item in arr if isinstance(item, dict) and "id" in item]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    return [int(m) for m in re.findall(r'"id"\s*:\s*(\d+)', response)]


# ── Judge backends ───────────────────────────────────────────────────────────

class _BaseJudge:
    """Vertex base-model judge. No tinker dependency."""

    def __init__(self, model: str | None = None, temperature: float = DEFAULT_TEMPERATURE):
        client, default_model = s1.get_client()
        self.client = client
        self.model = model or default_model
        self.temperature = temperature

    def label(self) -> str:
        return f"vertex:{self.model}"

    def query(self, prompt: str, *, max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> str:
        # call_llm in s1 doesn't expose max_tokens, but Vertex chat/completions
        # honors it via extra kwargs. We don't strictly need a token cap for
        # the judge — the small judge prompts are well within limits.
        return s1.call_llm(self.client, self.model, prompt, temperature=self.temperature)


class _TinkerJudge:
    """Optional tinker-checkpoint judge (matches the historical conics run)."""

    def __init__(self, checkpoint: str | None = None, *, temperature: float = DEFAULT_TEMPERATURE):
        import tinker
        from tinker import types
        from transformers import AutoTokenizer

        if not os.environ.get("TINKER_API_KEY"):
            raise RuntimeError("TINKER_API_KEY not set; cannot use --judge tinker")

        ckpt = (checkpoint or os.environ.get("TINKER_CHECKPOINT") or "").strip()
        if not ckpt:
            raise RuntimeError(
                "Pass --judge-checkpoint or set TINKER_CHECKPOINT for --judge tinker."
            )

        service = tinker.ServiceClient()
        print(f"Loading tinker checkpoint: {ckpt}")
        training_client = service.create_training_client_from_state(ckpt)
        self.sampling_client = training_client.save_weights_and_get_sampling_client()

        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login as hf_login
            hf_login(token=hf_token)
        self.tokenizer = AutoTokenizer.from_pretrained(s1.TINKER_BASE_MODEL)

        self._types = types
        self._ckpt = ckpt
        self.temperature = temperature

    def label(self) -> str:
        return f"tinker:{self._ckpt}"

    def query(self, prompt: str, *, max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        budget = DEFAULT_MODEL_CONTEXT - max_tokens - 32
        if len(ids) > budget:
            print(f"  [warn] truncating prompt {len(ids)} -> {budget} tokens")
            ids = ids[:budget]
        model_input = self._types.ModelInput.from_ints(ids)
        params = self._types.SamplingParams(temperature=self.temperature, max_tokens=max_tokens)
        result = self.sampling_client.sample(
            prompt=model_input, num_samples=1, sampling_params=params,
        ).result()
        return self.tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)


# ── Selection driver ─────────────────────────────────────────────────────────

def select_one(
    problem_id: str,
    *,
    target: int = DEFAULT_TARGET,
    batch_size: int = DEFAULT_BATCH_SIZE,
    judge: str = "base",
    judge_model: str | None = None,
    judge_checkpoint: str | None = None,
) -> dict:
    runs_root = REPO_ROOT / "runs" / problem_id
    in_path = runs_root / "filtered_keeps.json"
    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing {in_path}. Run Stage 3 (filter) first for id={problem_id!r}."
        )

    with open(in_path) as f:
        filt = json.load(f)
    problems = filt.get("problems", [])
    n_problems = len(problems)
    if n_problems == 0:
        raise ValueError(f"{in_path} has no problems to select from.")
    if n_problems < target:
        print(
            f"  [warn] only {n_problems} problems available, target was {target}. "
            f"Will select all {n_problems}."
        )
        target = n_problems

    statement, _ = s1.load_hard_problem(
        str(REPO_ROOT / "problems" / "hard_problems.jsonl"), problem_id
    )

    num_batches = (n_problems + batch_size - 1) // batch_size

    # Proportional selection per batch (port of query_tinker_ckpt50.py:199-211)
    select_per_batch: list[int] = []
    for i in range(num_batches):
        start = i * batch_size
        batch_len = min(batch_size, n_problems - start)
        share = round(target * batch_len / n_problems)
        select_per_batch.append(share)
    diff = target - sum(select_per_batch)
    for i in range(abs(diff)):
        idx = i % num_batches
        select_per_batch[idx] += 1 if diff > 0 else -1
    print(f"Selection plan: {select_per_batch} (total {sum(select_per_batch)})")

    if judge == "base":
        judge_impl: _BaseJudge | _TinkerJudge = _BaseJudge(model=judge_model)
    elif judge == "tinker":
        judge_impl = _TinkerJudge(checkpoint=judge_checkpoint)
    else:
        raise ValueError(f"Unknown --judge {judge!r}; expected 'base' or 'tinker'")

    print(f"\n{'='*70}")
    print(f"  Stage 4: select id={problem_id}")
    print(f"  Input:    {in_path}  ({n_problems} problems)")
    print(f"  Target:   {target} selected ({num_batches} batches of <= {batch_size})")
    print(f"  Judge:    {judge_impl.label()}")
    print(f"{'='*70}\n")

    log: dict = {
        "meta": {
            "id": problem_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "judge": judge_impl.label(),
            "input_path": str(in_path),
            "n_problems": n_problems,
            "num_batches": num_batches,
            "batch_size": batch_size,
            "target_selected": target,
            "select_per_batch": select_per_batch,
        },
        "batches": [],
    }

    all_selected: list[dict] = []

    for b in range(num_batches):
        start = b * batch_size
        batch = problems[start : start + batch_size]
        slim = [
            {"id": start + j + 1, "problem": p["problem"]}
            for j, p in enumerate(batch)
        ]
        n_select = select_per_batch[b]
        prompt = build_prompt(
            target_problem=statement,
            batch_index=b,
            batch_json=json.dumps(slim, indent=2),
            num_batches=num_batches,
            batch_size=batch_size,
            batch_len=len(batch),
            select_count=n_select,
        )
        print(
            f"--- Batch {b + 1}/{num_batches}  problems {start + 1}–{start + len(batch)}  "
            f"select {n_select} ---"
        )

        t0 = time.time()
        response = judge_impl.query(prompt)
        elapsed = time.time() - t0

        ids = parse_ids(response)
        print(f"  → {len(ids)} ids in {elapsed:.1f}s: {ids[:10]}{'...' if len(ids) > 10 else ''}")

        batch_entry: dict = {
            "batch_index": b,
            "elapsed_seconds": round(elapsed, 1),
            "response_text": response,
            "parsed_ids": ids,
            "selected": [],
        }
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            try:
                arr = json.loads(match.group())
                for item in arr:
                    if isinstance(item, dict) and "id" in item:
                        all_selected.append(item)
                        batch_entry["selected"].append(item)
            except (json.JSONDecodeError, TypeError):
                for pid in ids:
                    all_selected.append({"id": pid})
                    batch_entry["selected"].append({"id": pid})
        else:
            for pid in ids:
                all_selected.append({"id": pid})
                batch_entry["selected"].append({"id": pid})
        log["batches"].append(batch_entry)

    selected_ids = [item["id"] for item in all_selected]
    log["meta"]["finished_at"] = datetime.now(timezone.utc).isoformat()
    log["final"] = {
        "selected_count": len(selected_ids),
        "selected_ids": selected_ids,
    }

    output = {
        "id": problem_id,
        "target_problem": statement,
        "judge": judge_impl.label(),
        "n_selected": len(selected_ids),
        "selected": [],
    }
    for item in all_selected:
        pid = item["id"]
        idx = pid - 1
        entry: dict = {"id": pid}
        if 0 <= idx < n_problems:
            entry["problem"] = problems[idx]["problem"]
            entry["ground_truth_answer"] = problems[idx].get("ground_truth_answer")
        if "reason" in item:
            entry["reason"] = item["reason"]
        output["selected"].append(entry)

    out_path = runs_root / "selected.json"
    log_path = runs_root / "selection_log.json"
    _save_atomic(out_path, output)
    _save_atomic(log_path, log)
    print(f"\nWrote {out_path}  ({len(selected_ids)} selected)")
    print(f"Wrote {log_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Stage 4: LLM-judge selection")
    parser.add_argument("--id", type=str, required=True)
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--judge", type=str, default="base", choices=["base", "tinker"])
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Override Vertex model for --judge base")
    parser.add_argument("--judge-checkpoint", type=str, default=None,
                        help="Tinker path for --judge tinker (or set TINKER_CHECKPOINT)")
    args = parser.parse_args()
    select_one(
        args.id,
        target=args.target,
        batch_size=args.batch_size,
        judge=args.judge,
        judge_model=args.judge_model,
        judge_checkpoint=args.judge_checkpoint,
    )


if __name__ == "__main__":
    main()
