#!/usr/bin/env python3
"""
Rank lemma relevance, redundancy, and complexity for training set evaluation.

Takes a JSON file of lemmas and answers along with a seed problem,
uses GPT-OSS to rank:
- Relevance (0-5): How useful is this lemma for proving the seed problem?
- Redundancy (0-1): Are there redundant premises in the lemma?
- Complexity (0-4): How complex is the conclusion of the lemma?
"""

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

TARGET_PROBLEM = """
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
# LLM backend client configuration
# ---------------------------------------------------------------------------
VERTEX_DEFAULT_MODEL = "openai/gpt-oss-120b-maas"
AZURE_DEFAULT_MODEL = os.environ.get("AZURE_AI_FOUNDRY_MODEL", "FW-GPT-OSS-120B")


def _get_vertex_access_token() -> str:
    """Get Vertex AI access token using Google Auth."""
    from google.auth import default
    from google.auth.transport.requests import Request

    credentials, _ = default()
    credentials.refresh(Request())
    return credentials.token


def _build_vertex_base_url() -> str:
    """Build Vertex AI base URL from environment variables."""
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


def _get_vertex_client() -> tuple[OpenAI, str]:
    """Create OpenAI client configured for Vertex AI."""
    token = _get_vertex_access_token()
    base_url = _build_vertex_base_url()
    client = OpenAI(api_key=token, base_url=base_url)
    return client, VERTEX_DEFAULT_MODEL


def _get_azure_client() -> tuple[OpenAI, str]:
    """Create OpenAI client configured for Azure AI Foundry."""
    endpoint = os.environ.get("AZURE_AI_FOUNDRY_ENDPOINT")
    api_key = os.environ.get("AZURE_AI_FOUNDRY_API_KEY")
    if not endpoint or not api_key:
        raise RuntimeError(
            "Set AZURE_AI_FOUNDRY_ENDPOINT and AZURE_AI_FOUNDRY_API_KEY "
            "environment variables for Azure AI Foundry."
        )
    client = OpenAI(api_key=api_key, base_url=endpoint)
    return client, AZURE_DEFAULT_MODEL


def get_client(backend: str = "azure") -> tuple[OpenAI, str]:
    """Create OpenAI client for the requested backend ('azure' or 'vertex')."""
    backend = backend.lower()
    if backend == "azure":
        return _get_azure_client()
    if backend == "vertex":
        return _get_vertex_client()
    raise ValueError(f"Unknown backend: {backend!r} (expected 'azure' or 'vertex')")


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------
def call_llm(
    client: OpenAI,
    model: str,
    prompt: str,
) -> str:
    """Call LLM with retry logic."""

    @retry(
        wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(8)
    )
    def _call() -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        if not response.choices:
            raise RuntimeError("LLM response has no choices")
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM response has empty content")
        return content

    try:
        return _call()
    except Exception as e:
        print(f"  [warn] LLM call failed after all retries: {e}")
        return ""


# ---------------------------------------------------------------------------
# Ranking functions
# ---------------------------------------------------------------------------
@dataclass
class ProblemRanking:
    """Container for problem rankings."""

    score: Optional[int] = None  # Single 0-10 score for quality and relevance
    raw_response: str = ""


def parse_ranking_response(response: str) -> ProblemRanking:
    """Parse LLM response to extract single 0-10 score from \boxed{} format."""
    ranking = ProblemRanking(raw_response=response)

    # Look for score in \boxed{} format
    boxed_pattern = r"\\boxed\{(\d+)\}"
    boxed_matches = re.findall(boxed_pattern, response)

    if len(boxed_matches) >= 1:
        try:
            score = int(boxed_matches[0])
            if 0 <= score <= 10:
                ranking.score = score
        except ValueError:
            pass

    return ranking


def build_ranking_prompt(
    target_problem: str, subproblem: str, subproblem_answer: str
) -> str:
    """Build prompt for ranking subproblem quality and relevance for curriculum learning."""
    return f"""You are evaluating a subproblem and its answer for their usefulness in a curriculum for learning to solve a target problem.

**TARGET PROBLEM** (the problem a student needs to learn to solve):
{target_problem}

**SUBPROBLEM** (a candidate subproblem in the learning curriculum):
{subproblem}

**SUBPROBLEM ANSWER**:
{subproblem_answer}

Rate this subproblem on a scale of 0 to 10 based on its QUALITY and RELEVANCE for learning how to solve the target problem.

DISQUALIFYING CONDITIONS (automatic 0):
- The subproblem has multiple parts (e.g., multiple questions or sub-problems to answer)
- The subproblem uses a false premise or a purposefully confusing premise
- The subproblem is identical to the target problem (no decomposition)
- The subproblem is trivial and teaches nothing useful for the target

Rubric:
- 0: Disqualified (see conditions above) or completely irrelevant/useless
- 1-2: Poor quality - unclear, confusing, poorly stated, or answer is incorrect/unhelpful; minimally related to target
- 3-4: Low quality - some ambiguity or minor issues; related to target domain but not directly useful
- 5-6: Moderate quality - adequately clear, somewhat useful - includes relevant concepts but only slightly helps
- 7-8: Good quality - clear, well-stated, directly useful - solving this would meaningfully help solve the target
- 9-10: Excellent quality - crystal clear, precise, pedagogically well-formed; essential stepping stone for learning to solve the target

Format your response with a single boxed score:
\\boxed{{score}}

Give an explanation for your score and think step by step.
"""


def rank_problem(
    client: OpenAI,
    model: str,
    target_problem: str,
    subproblem: str,
    subproblem_answer: str,
    verbose: bool = False,
) -> ProblemRanking:
    """Rank a subproblem for relevance, redundancy, and quality."""
    prompt = build_ranking_prompt(target_problem, subproblem, subproblem_answer)

    if verbose:
        print(f"\n[Prompt]\n{prompt[:300]}...")

    response = call_llm(client, model, prompt)

    if verbose:
        print(f"\n[Response]\n{response}")

    ranking = parse_ranking_response(response)

    if ranking.score is None:
        print("  [warn] Ill-formed response - could not parse all scores")
        print(f"  [debug] Raw response: {response[:500]}...")

    return ranking


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------
def process_json_file(
    input_path: str,
    target_problem: Optional[str] = None,
    target_file: Optional[str] = None,
    output_path: Optional[str] = None,
    verbose: bool = False,
    limit: Optional[int] = None,
    workers: int = 8,
    backend: str = "azure",
) -> list[dict]:
    """Process JSON file of lemmas and add relevance, redundancy, and complexity rankings."""
    # Get target problem from file if specified
    if target_file:
        with open(target_file, "r") as f:
            target_data = json.load(f)
            if isinstance(target_data, dict):
                target_problem = (
                    target_data.get("problem")
                    or target_data.get("question")
                    or target_data.get("target", "")
                )
            else:
                target_problem = str(target_data)

    if not target_problem:
        raise ValueError(
            "Target problem is required. Use --target to specify it directly, "
            "or --target-file to load from a JSON file."
        )

    # Load input data
    with open(input_path, "r") as f:
        data = json.load(f)

    # Handle both list and dict formats
    if isinstance(data, dict):
        if "problems" in data:
            problems = data["problems"]
        else:
            problems = [data]
    else:
        problems = data

    # Apply limit if specified
    if limit is not None:
        problems = problems[:limit]

    # Get client (shared across threads — OpenAI client is thread-safe)
    print(f"Initializing {backend} client...")
    client, model = get_client(backend=backend)
    print(f"Using model: {model}")
    print(f"Workers: {workers}")
    print(f"\nTarget problem: {target_problem[:100]}...")

    total = len(problems)
    print_lock = threading.Lock()

    def process_item(args: tuple[int, dict]) -> dict:
        i, item = args
        with print_lock:
            print(f"\nProcessing {i}/{total}...")

        problem = item.get("problem") or item.get("question") or item.get("prompt", "")
        answer = item.get("answer") or item.get("solution") or item.get("response") or item.get("reference", "")

        if not problem:
            with print_lock:
                print(f"  [{i}] [warn] No problem text found")
            return {
                "id": item.get("id"),
                "prompt": problem,
                "reference": answer,
                "score": None,
                "error": "No problem text",
            }

        ranking = rank_problem(
            client, model, target_problem, problem, answer, verbose=verbose
        )

        result = {
            "id": item.get("id"),
            "prompt": problem,
            "reference": answer,
            "score": ranking.score,
            "reasoning": ranking.raw_response,
        }

        with print_lock:
            if ranking.score is None:
                print(f"  [{i}] [ILL-FORMED] Could not parse score")
            else:
                print(f"  [{i}] Score: {ranking.score}/10")

        return result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # map preserves input order in the returned results
        results = list(executor.map(process_item, enumerate(problems, 1)))

    # Save results if output path specified
    if output_path:
        output_data = {"target_problem": target_problem, "problems": results}
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Rank subproblems by quality and relevance (0-10) for curriculum learning"
    )
    parser.add_argument("input", help="Input JSON file containing lemmas and answers")
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON file for results (default: input_ranked.json)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed prompts and responses",
    )
    parser.add_argument(
        "-l", "--limit", type=int, help="Limit number of problems to process"
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=8,
        help="Number of parallel worker threads (default: 8)",
    )
    parser.add_argument(
        "--backend",
        choices=["azure", "vertex"],
        default="azure",
        help="LLM backend to use (default: azure)",
    )

    args = parser.parse_args()

    # Set default output path
    output_path = args.output
    if not output_path:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_ranked{ext}"

    # Process file
    try:
        results = process_json_file(
            args.input,
            target_problem=TARGET_PROBLEM,
            target_file=None,
            output_path=output_path,
            verbose=args.verbose,
            limit=args.limit,
            workers=args.workers,
            backend=args.backend,
        )

        # Print summary statistics
        scores = [r["score"] for r in results if r.get("score") is not None]
        ill_formed_count = sum(1 for r in results if r.get("score") is None)

        print("\n=== Summary ===")
        print(f"Subproblems processed: {len(results)}")
        print(f"Ill-formed responses: {ill_formed_count}")

        if scores:
            print(f"\nScore Statistics:")
            print(f"  Average: {sum(scores) / len(scores):.2f}")
            print(f"  Range: {min(scores)} - {max(scores)}")
            print(f"  Successfully parsed: {len(scores)}/{len(results)}")

    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
