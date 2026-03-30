"""
Run inference on the most recent GRPO checkpoint.

Usage::

    python infer.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

import tinker
from tinker import types
from transformers import AutoTokenizer

# ── Configuration ─────────────────────────────────────────────────────────

QUESTION = r"""
There are unique constants \(C, \alpha, \beta\) such that the number of solutions to the equation
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
\]"""



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
NUM_SAMPLES = 50
MODEL_NAME = "openai/gpt-oss-120b"
MAX_TOKENS = 16384
TEMPERATURE = 0.7

# ── Helpers ───────────────────────────────────────────────────────────────

def get_service() -> tinker.ServiceClient:
    if not os.environ.get("TINKER_API_KEY"):
        sys.exit(
            "TINKER_API_KEY not set. "
            "Add it to your .env or export it in your shell."
        )
    return tinker.ServiceClient()


def find_latest_checkpoint(service: tinker.ServiceClient) -> str:
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


def sample(sampling_client, tokenizer, question: str) -> list[str]:
    solve_prompt = SOLVE_PROMPT.format(problem=question)
    messages = [
        # {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": solve_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    prompt = types.ModelInput.from_ints(ids)

    params = types.SamplingParams(temperature=TEMPERATURE)
    result = sampling_client.sample(
        prompt=prompt,
        num_samples=NUM_SAMPLES,
        sampling_params=params,
    ).result()

    return [
        tokenizer.decode(seq.tokens, skip_special_tokens=True)
        for seq in result.sequences
    ]


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    service = get_service()
    tinker_path = find_latest_checkpoint(service)
    sampling_client, tokenizer = build_clients(service, tinker_path)

    responses = sample(sampling_client, tokenizer, QUESTION)

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": tinker_path,
        "question": QUESTION.strip(),
        "num_samples": NUM_SAMPLES,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "responses": responses,
    }

    out_path = f"infer_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(responses)} response(s) to {out_path}")
