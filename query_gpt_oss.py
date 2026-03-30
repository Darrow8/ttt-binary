"""Simple script to query gpt-oss-120b via Vertex AI."""

import os
from dotenv import load_dotenv
from google.auth import default
from google.auth.transport.requests import Request
from openai import OpenAI

load_dotenv()


def get_client() -> OpenAI:
    credentials, _ = default()
    credentials.refresh(Request())
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    if location == "global":
        location = "us-central1"
    base_url = (
        f"https://{location}-aiplatform.googleapis.com/v1/"
        f"projects/{project}/locations/{location}/endpoints/openapi"
    )
    return OpenAI(api_key=credentials.token, base_url=base_url)


def query(prompt: str, temperature: float = 0.7) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b-maas",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


if __name__ == "__main__":
    import sys

    with open("sp-problems.jsonl") as f:
        file_contents = f.read()

    instruction = """
    You are given a difficult target problem in advanced mathematics. Your task is \textbf{not to solve it yet}, but to determine which earlier problems would best prepare someone to solve it.

    From the list of available problems, \textbf{select 20 problems that a student should study first} in order to build the knowledge and intuition needed to solve the target problem.

    When selecting the problems, prioritize ones that help develop:

    \begin{itemize}
    \item Relevant \textbf{concepts and theory}
    \item \textbf{Intermediate techniques} used in the target problem
    \item \textbf{Simpler versions or special cases} of the same ideas
    \item Problems that introduce the ideas appearing in the target problem
    \end{itemize}

    Avoid selecting problems that are:
    \begin{itemize}
    \item Unrelated to the key ideas
    \item Much harder than the target problem
    \item Purely computational with no conceptual overlap
    \end{itemize}

    Return \textbf{exactly 20 problems}.

    For each selected problem include the problem id given in the json below. Provide a brief explanation (1--2 sentences) of why it helps prepare for the target problem.

    Focus on building a \textbf{progressive learning path} that gradually develops the tools needed for the final problem.

    \bigskip

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

    \bigskip

    \textbf{Available problems below}

    """
    prompt = f"{instruction} \n \n {file_contents}"
    print(query(prompt))
