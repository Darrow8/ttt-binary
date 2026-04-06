"""
Baseline GRPO run with dynamic consensus (TTRL-style).

Instead of a fixed ground-truth reference, the reward signal comes from
majority vote over the model's own sampled completions each step.
This faithfully replicates the TTRL paper's approach.

Usage::

    python -m pipeline.baseline_run_consensus
"""

import logging
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
load_dotenv()

from pipeline import (
    GRPOConfig,
    GRPOTrainer,
    Problem,
    boxed_match,
    boxed_format_bonus,
    combined,
)
from pipeline.rewards import extract_boxed, _normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARN)

logger = logging.getLogger(__name__)


# ── Consensus trainer ─────────────────────────────────────────────────────

class ConsensusGRPOTrainer(GRPOTrainer):
    """GRPO trainer that derives the reference answer via majority vote
    over the current model's sampled completions (TTRL approach)."""

    def _get_reference(self, problem: Problem, decoded_responses: list[str]) -> str:
        """Extract answers from all completions and return the majority vote."""
        answers: list[str] = []
        for resp in decoded_responses:
            try:
                answers.append(_normalize(extract_boxed(resp)))
            except ValueError:
                continue  # skip responses without a \boxed{}

        if not answers:
            logger.warning("No extractable answers in group — falling back to empty reference")
            return ""

        counter = Counter(answers)
        consensus, count = counter.most_common(1)[0]
        total = len(decoded_responses)
        extractable = len(answers)

        logger.info(
            "Consensus: %s  (%d/%d extractable, %d/%d agree)",
            consensus, extractable, total, count, extractable,
        )
        return consensus


# ── Single training problem (no ground truth needed) ──────────────────────

DEFAULT_PROBLEM = Problem(
    prompt=r"""Let \(U \subset PH^0_{\mathbb{Z}}(\mathbb{P}^2,\mathcal{O}(2))\) be the space of smooth conics in \(\mathbb{P}^2\), and let \(Z \subset U^6\) be the closed subscheme parametrizing \(6\)-tuples \((C_1,\dots,C_6)\) with \(C_1\) tangent to \(C_2,\dots,C_6\). Let
\[
\pi : Z \to U^5
\]
be the map induced by the projection onto the last \(5\) coordinates, and let \(V \subset U^5\) be the dense open subscheme over which \(\pi\) is finite étale. Let
\[
L=\lim_{p\to\infty}\frac{1}{\#V(\mathbb{F}_p)}\sum_{x\in V(\mathbb{F}_p)} \#\pi^{-1}(x),
\]
that is, the limit of the average number of components of the space of conics tangent to \(5\) smooth conics over \(\mathbb{F}_p\), as \(p\) tends to infinity. Find \(\lfloor 100L \rfloor\).""",
    reference="",  # intentionally blank — consensus provides the reference
)

problems = [DEFAULT_PROBLEM]


# ── Configuration ─────────────────────────────────────────────────────────

config = GRPOConfig(
    model_name="openai/gpt-oss-120b",
    log_dir=str(_REPO_ROOT / "baseline-run-consensus"),

    batch_size=1,
    group_size=16,
    learning_rate=1e-4,
    lora_rank=32,
    max_tokens=16384,

    save_every=5,

    wandb_project="grpo-baseline-consensus",

    temperature=0.7,
    system_prompt="""
You are a careful and rigorous math student working through an advanced mathematics problem. Your goal is to solve the problem step by step.

Show all important intermediate reasoning, derivations, and calculations. Explain why each step is valid and reference any relevant theorems or identities when appropriate. Avoid skipping logical steps or making large jumps in reasoning.

If the problem involves multiple cases or approaches, consider them systematically. Use clear mathematical notation and keep the solution organized.

After completing the reasoning, clearly state the final answer.
    """,

    prompt_suffix=" Put your final answer inside \\boxed{}.",

    few_shot=[],
)

EPOCHS = 100


# ── Reward function ───────────────────────────────────────────────────────

reward_fn = combined(
    (1.0, boxed_match),
    (0.1, boxed_format_bonus),
)


# ── Train ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trainer = ConsensusGRPOTrainer(config, reward_fn)
    trainer.train(problems, epochs=EPOCHS)
