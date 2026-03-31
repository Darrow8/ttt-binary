"""Query tinker checkpoint 50 to pick the 50 best problems from problems-2.json."""

import os
import sys

from dotenv import load_dotenv
load_dotenv()

import tinker
from tinker import types
from transformers import AutoTokenizer

MODEL_NAME = "openai/gpt-oss-120b"
TEMPERATURE = 0.7


def get_service() -> tinker.ServiceClient:
    if not os.environ.get("TINKER_API_KEY"):
        sys.exit(
            "TINKER_API_KEY not set. "
            "Add it to your .env or export it in your shell."
        )
    return tinker.ServiceClient()


def find_checkpoint(service: tinker.ServiceClient, step: int) -> str:
    rest = service.create_rest_client()
    response = rest.list_user_checkpoints(limit=200).result()
    training_ckpts = [
        c for c in response.checkpoints
        if c.checkpoint_type == "training"
    ]
    if not training_ckpts:
        sys.exit("No training checkpoints found. Run training first.")

    step_tag = f"ckpt-{step:06d}"
    for ckpt in training_ckpts:
        if step_tag in ckpt.tinker_path:
            print(f"Using checkpoint: {ckpt.tinker_path}  (created {ckpt.time})")
            return ckpt.tinker_path

    print(f"No checkpoint matching step {step}. Available checkpoints:")
    for c in training_ckpts[:20]:
        print(f"  {c.tinker_path}  ({c.time})")
    sys.exit(1)


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


def query(sampling_client, tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    model_input = types.ModelInput.from_ints(ids)

    params = types.SamplingParams(temperature=TEMPERATURE)
    result = sampling_client.sample(
        prompt=model_input,
        num_samples=1,
        sampling_params=params,
    ).result()

    return tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)


if __name__ == "__main__":
    with open("problems-2.json") as f:
        file_contents = f.read()

    instruction = """
    You are given a difficult target problem in advanced mathematics. Your task is \\textbf{not to solve it yet}, but to determine which earlier problems would best prepare someone to solve it.

    From the list of available problems, \\textbf{select 50 problems that a student should study first} in order to build the knowledge and intuition needed to solve the target problem.

    When selecting the problems, prioritize ones that help develop:

    \\begin{itemize}
    \\item Relevant \\textbf{concepts and theory}
    \\item \\textbf{Intermediate techniques} used in the target problem
    \\item \\textbf{Simpler versions or special cases} of the same ideas
    \\item Problems that introduce the ideas appearing in the target problem
    \\end{itemize}

    Avoid selecting problems that are:
    \\begin{itemize}
    \\item Unrelated to the key ideas
    \\item Much harder than the target problem
    \\item Purely computational with no conceptual overlap
    \\end{itemize}

    Return \\textbf{exactly 50 problems}.

    For each selected problem include the problem id given in the json below. Provide a brief explanation (1--2 sentences) of why it helps prepare for the target problem.

    Focus on building a \\textbf{progressive learning path} that gradually develops the tools needed for the final problem.

    \\bigskip

    \\textbf{Target problem}

    Let \\(U \\subset PH^0_{\\mathbb{Z}}(\\mathbb{P}^2,\\mathcal{O}(2))\\) be the space of smooth conics in \\(\\mathbb{P}^2\\), and let \\(Z \\subset U^6\\) be the closed subscheme parametrizing \\(6\\)-tuples \\((C_1,\\dots,C_6)\\) with \\(C_1\\) tangent to \\(C_2,\\dots,C_6\\). Let
    \\[
    \\pi : Z \\to U^5
    \\]
    be the map induced by the projection onto the last \\(5\\) coordinates, and let \\(V \\subset U^5\\) be the dense open subscheme over which \\(\\pi\\) is finite étale. Let
    \\[
    L=\\lim_{p\\to\\infty}\\frac{1}{\\#V(\\mathbb{F}_p)}\\sum_{x\\in V(\\mathbb{F}_p)} \\#\\pi^{-1}(x),
    \\]
    that is, the limit of the average number of components of the space of conics tangent to \\(5\\) smooth conics over \\(\\mathbb{F}_p\\), as \\(p\\) tends to infinity. Find \\(\\lfloor 100L \\rfloor\\).

    \\bigskip

    \\textbf{Available problems below}

    """
    prompt = f"{instruction} \n \n {file_contents}"

    service = get_service()
    tinker_path = find_checkpoint(service, 50)
    sampling_client, tokenizer = build_clients(service, tinker_path)

    print(query(sampling_client, tokenizer, prompt))
