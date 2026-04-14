"""Query a Tinker checkpoint: from 100 problems, five batches of 20, pick 10 each (50 total).

Outputs a final JSON list of the 50 selected problem IDs.
"""

import json
import os
import re
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

import tinker
from tinker import types
from transformers import AutoTokenizer

MODEL_NAME = "openai/gpt-oss-120b"
TEMPERATURE = 0.7
MODEL_CONTEXT = 32768
MAX_OUTPUT_TOKENS = 6144

PROBLEMS_PATH = "Stage1/filtered_keeps.json"
BATCH_SIZE = 20
TARGET_SELECTED = 50

# subproblems-run-resumed, step 10
TINKER_CHECKPOINT = (
    "tinker://858c48a0-b533-5d70-8c41-4d2d787508bb:train:0/"
    "weights/subproblems-run-resumed.ckpt-000010"
)

TARGET_PROBLEM = r"""
    \textbf{Target problem}

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


def get_service() -> tinker.ServiceClient:
    if not os.environ.get("TINKER_API_KEY"):
        sys.exit(
            "TINKER_API_KEY not set. "
            "Add it to your .env or export it in your shell."
        )
    return tinker.ServiceClient()


def build_clients(service: tinker.ServiceClient, tinker_path: str):
    print(f"Loading checkpoint: {tinker_path}")
    training_client = service.create_training_client_from_state(tinker_path)
    sampling_client = training_client.save_weights_and_get_sampling_client()

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login as hf_login
        hf_login(token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Ready.\n")
    return sampling_client, tokenizer


def build_prompt(batch_index: int, batch_json: str, num_batches: int, batch_len: int, select_count: int) -> str:
    """batch_index is 0-based."""
    b = batch_index + 1
    lo = batch_index * BATCH_SIZE + 1
    hi = batch_index * BATCH_SIZE + batch_len
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

    {TARGET_PROBLEM}

    \\bigskip

    \\textbf{{Available problems (batch {b} of {num_batches})}}

    {batch_json}
    """


def prompt_token_count(tokenizer, prompt: str) -> int:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    return len(tokenizer.encode(text, add_special_tokens=False))


def query(
    sampling_client,
    tokenizer,
    prompt: str,
    *,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    warnings_out: list[str] | None = None,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    budget = MODEL_CONTEXT - max_tokens - 32
    if len(ids) > budget:
        msg = (
            f"Warning: truncating prompt from {len(ids)} to {budget} tokens."
        )
        print(msg, file=sys.stderr)
        if warnings_out is not None:
            warnings_out.append(msg)
        ids = ids[:budget]
    model_input = types.ModelInput.from_ints(ids)

    params = types.SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=max_tokens,
    )
    result = sampling_client.sample(
        prompt=model_input,
        num_samples=1,
        sampling_params=params,
    ).result()

    return tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)


def parse_ids(response: str) -> list[int]:
    """Extract problem IDs from the model response (expects a JSON array)."""
    # Try to find a JSON array in the response
    match = re.search(r'\[.*\]', response, re.DOTALL)
    if match:
        try:
            arr = json.loads(match.group())
            return [item["id"] for item in arr if isinstance(item, dict) and "id" in item]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    # Fallback: grab all "id": <int> occurrences
    return [int(m) for m in re.findall(r'"id"\s*:\s*(\d+)', response)]


OUTPUT_PATH = "Stage1/selected_50_ids.json"
LOG_JSON_PATH = "Stage1/query_tinker_ckpt50_log.json"


if __name__ == "__main__":
    with open(PROBLEMS_PATH) as f:
        raw = json.load(f)

    # Support both a plain list and a dict with a "problems" key
    if isinstance(raw, dict):
        all_problems = raw["problems"]
    else:
        all_problems = raw

    NUM_PROBLEMS = len(all_problems)
    NUM_BATCHES = (NUM_PROBLEMS + BATCH_SIZE - 1) // BATCH_SIZE  # ceiling division
    problems_100 = all_problems

    # Distribute TARGET_SELECTED proportionally across batches
    select_per_batch: list[int] = []
    remaining = TARGET_SELECTED
    for i in range(NUM_BATCHES):
        start = i * BATCH_SIZE
        batch_len = min(BATCH_SIZE, NUM_PROBLEMS - start)
        # proportional share, rounded
        share = round(TARGET_SELECTED * batch_len / NUM_PROBLEMS)
        select_per_batch.append(share)
    # Adjust to hit exactly TARGET_SELECTED
    diff = TARGET_SELECTED - sum(select_per_batch)
    for i in range(abs(diff)):
        idx = i % NUM_BATCHES
        select_per_batch[idx] += 1 if diff > 0 else -1
    print(f"Selection plan: {select_per_batch} (total {sum(select_per_batch)})")

    service = get_service()
    print(f"Using checkpoint: {TINKER_CHECKPOINT}")
    sampling_client, tokenizer = build_clients(service, TINKER_CHECKPOINT)

    run_warnings: list[str] = []
    log: dict = {
        "meta": {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "tinker_checkpoint": TINKER_CHECKPOINT,
            "model_name": MODEL_NAME,
            "temperature": TEMPERATURE,
            "problems_path": PROBLEMS_PATH,
            "num_problems": NUM_PROBLEMS,
            "num_batches": NUM_BATCHES,
            "batch_size": BATCH_SIZE,
            "target_selected": TARGET_SELECTED,
            "select_per_batch": select_per_batch,
            "output_path": OUTPUT_PATH,
            "log_path": LOG_JSON_PATH,
        },
        "batches": [],
        "warnings": run_warnings,
    }

    all_selected: list[dict] = []

    for batch_index in range(NUM_BATCHES):
        start = batch_index * BATCH_SIZE
        batch = problems_100[start : start + BATCH_SIZE]
        # Only send id + problem text to stay within token limits
        slim_batch = [
            {"id": start + j + 1, "problem": p["problem"]}
            for j, p in enumerate(batch)
        ]
        batch_json = json.dumps(slim_batch, indent=2)
        n_select = select_per_batch[batch_index]
        prompt = build_prompt(batch_index, batch_json, NUM_BATCHES, len(batch), n_select)
        n_tok = prompt_token_count(tokenizer, prompt)
        print(
            f"--- Batch {batch_index + 1}/{NUM_BATCHES} "
            f"(problems {start + 1}–{start + len(batch)}, select {n_select}, ~{n_tok} prompt tokens) ---",
            flush=True,
        )

        batch_warnings: list[str] = []
        response = query(
            sampling_client,
            tokenizer,
            prompt,
            warnings_out=batch_warnings,
        )
        run_warnings.extend(batch_warnings)
        print(response, "\n")

        ids = parse_ids(response)
        print(f"  → Parsed {len(ids)} IDs: {ids}\n")

        batch_entry: dict = {
            "batch_index": batch_index,
            "batch_number": batch_index + 1,
            "problem_indices_1based": [start + 1, start + len(batch)],
            "prompt_token_count": n_tok,
            "response_text": response,
            "parsed_ids": ids,
            "warnings": list(batch_warnings),
        }

        # Try to preserve id+reason pairs
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            try:
                arr = json.loads(match.group())
                selected_from_batch: list[dict] = []
                for item in arr:
                    if isinstance(item, dict) and "id" in item:
                        all_selected.append(item)
                        selected_from_batch.append(item)
                batch_entry["selected"] = selected_from_batch
                log["batches"].append(batch_entry)
                continue
            except (json.JSONDecodeError, TypeError):
                pass
        for pid in ids:
            all_selected.append({"id": pid})
        batch_entry["selected"] = [{"id": pid} for pid in ids]
        log["batches"].append(batch_entry)

    selected_ids = [item["id"] for item in all_selected]

    log["meta"]["finished_at"] = datetime.now(timezone.utc).isoformat()
    log["final"] = {
        "selected_count": len(selected_ids),
        "selected_ids": selected_ids,
        "all_selected_objects": all_selected,
    }

    print("=" * 60)
    print(f"FINAL SELECTION: {len(selected_ids)} problem IDs")
    print(json.dumps(selected_ids))
    print("=" * 60)

    # Build output with full original problem data + the target problem
    output = {
        "target_problem": TARGET_PROBLEM.strip(),
        "selected": [],
    }
    for item in all_selected:
        pid = item["id"]
        idx = pid - 1  # 1-based id -> 0-based index
        entry = {"id": pid}
        if 0 <= idx < len(problems_100):
            entry["problem"] = problems_100[idx]["problem"]
            entry["ground_truth_answer"] = problems_100[idx].get("ground_truth_answer")
        if "reason" in item:
            entry["reason"] = item["reason"]
        output["selected"].append(entry)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {OUTPUT_PATH}")

    with open(LOG_JSON_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Log saved to {LOG_JSON_PATH}")
