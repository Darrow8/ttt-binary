"""
Pluggable reward functions for GRPO.

A reward function has the signature::

    (response: str, problem: Problem) -> float

Use the built-in helpers or write your own.  ``combined()`` lets you
mix multiple reward signals with weights.
"""

from __future__ import annotations

import re
import logging
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.problems import Problem

logger = logging.getLogger(__name__)

RewardFunc = Callable[[str, "Problem"], float]


# ---------------------------------------------------------------------------
# Text extraction utilities
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str:
    """Return the contents of the last ``\\boxed{...}`` in *text*."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        raise ValueError("no \\boxed{} found")
    depth, start = 0, idx + len("\\boxed{")
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    raise ValueError("unmatched braces in \\boxed{}")


def _normalize(s: str) -> str:
    s = s.replace(",", "").replace(" ", "").strip().rstrip(".")
    # Normalize numeric strings so "3264.0000" == "3264"
    try:
        num = float(s)
        if num == int(num):
            return str(int(num))
        return str(num)
    except (ValueError, OverflowError):
        pass
    return s


# ---------------------------------------------------------------------------
# Built-in reward functions
# ---------------------------------------------------------------------------

def exact_match(response: str, problem: "Problem") -> float:
    """1.0 if ``response`` contains the reference answer (normalized), else 0.0."""
    ref = _normalize(problem.reference)
    if not ref:
        return 0.0
    return 1.0 if ref in _normalize(response) else 0.0


def boxed_match(response: str, problem: "Problem") -> float:
    r"""1.0 if the ``\boxed{...}`` answer matches the reference."""
    try:
        predicted = _normalize(extract_boxed(response))
    except ValueError:
        return 0.0
    ref = _normalize(problem.reference)
    # Handle GSM8K-style references with ``#### <answer>``
    if "####" in ref:
        ref = _normalize(ref.split("####")[-1])
    return 1.0 if predicted == ref else 0.0


def boxed_format_bonus(response: str, problem: "Problem") -> float:
    r"""0.1 bonus if a ``\boxed{...}`` is present, -0.1 penalty otherwise."""
    try:
        extract_boxed(response)
        return 0.1
    except ValueError:
        return -0.1


def regex_match(pattern: str) -> RewardFunc:
    """Factory: returns a reward func that scores 1.0 if *pattern* matches the response."""
    compiled = re.compile(pattern, re.DOTALL)

    def _reward(response: str, problem: "Problem") -> float:
        return 1.0 if compiled.search(response) else 0.0

    _reward.__qualname__ = f"regex_match({pattern!r})"
    return _reward


def contains_reference(response: str, problem: "Problem") -> float:
    """1.0 if the reference string appears anywhere in the response (case-insensitive)."""
    if not problem.reference:
        return 0.0
    return 1.0 if problem.reference.lower() in response.lower() else 0.0


_ANSWER_TAG_RE = re.compile(r"\*\*ANSWER:\s*(.+?)\*\*", re.IGNORECASE)


def extract_answer_tag(text: str) -> str:
    """Return the contents of the last ``**ANSWER: ...**`` tag in *text*."""
    matches = _ANSWER_TAG_RE.findall(text)
    if not matches:
        raise ValueError("no **ANSWER: ...** found")
    return matches[-1].strip()


def answer_tag_match(response: str, problem: "Problem") -> float:
    """1.0 if the ``**ANSWER: ...**`` value matches the reference."""
    try:
        predicted = _normalize(extract_answer_tag(response))
    except ValueError:
        return 0.0
    ref = _normalize(problem.reference)
    return 1.0 if predicted == ref else 0.0


def answer_tag_format_bonus(response: str, problem: "Problem") -> float:
    """0.1 bonus if ``**ANSWER: ...**`` is present, -0.1 penalty otherwise."""
    try:
        extract_answer_tag(response)
        return 0.1
    except ValueError:
        return -0.1


def length_penalty(max_tokens: int = 512, *, penalty_per_token: float = 0.001) -> RewardFunc:
    """Small penalty proportional to response length, encouraging concise answers."""
    def _reward(response: str, problem: "Problem") -> float:
        n = len(response.split())
        return -penalty_per_token * max(0, n - max_tokens)

    _reward.__qualname__ = f"length_penalty(max={max_tokens})"
    return _reward


# ---------------------------------------------------------------------------
# Chemistry / SMILES reward functions
# ---------------------------------------------------------------------------

def _strip_latex(smiles: str) -> str:
    """Remove LaTeX wrappers like \\text{}, \\texttt{}, \\mathrm{} from SMILES."""
    s = smiles.strip()
    s = re.sub(r"\\(?:text|texttt|textbf|mathrm|mathtt)\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\,", "", s)
    return s.strip()


def _canonicalize_smiles(smiles: str) -> str | None:
    """Canonicalize a SMILES and sort multi-component reactants."""
    try:
        from rdkit import Chem
        s = _strip_latex(smiles)
        parts = s.split(".")
        canonical = []
        for part in parts:
            mol = Chem.MolFromSmiles(part.strip())
            if mol is None:
                return None
            canonical.append(Chem.MolToSmiles(mol))
        return ".".join(sorted(canonical))
    except Exception:
        return None


def smiles_match(response: str, problem: "Problem") -> float:
    r"""1.0 if the ``\boxed{...}`` SMILES matches the reference (canonicalized)."""
    try:
        predicted_raw = extract_boxed(response)
    except ValueError:
        return 0.0
    predicted_clean = _strip_latex(predicted_raw)
    ref_canon = _canonicalize_smiles(problem.reference)
    pred_canon = _canonicalize_smiles(predicted_clean)
    if ref_canon is None or pred_canon is None:
        # Fallback: stripped exact match
        return 1.0 if predicted_clean.strip() == problem.reference.strip() else 0.0
    return 1.0 if pred_canon == ref_canon else 0.0


# ---------------------------------------------------------------------------
# Combiner
# ---------------------------------------------------------------------------

def combined(*weighted_fns: tuple[float, RewardFunc]) -> RewardFunc:
    """Combine multiple reward functions with weights.

    Example::

        reward_fn = combined(
            (1.0, boxed_match),
            (0.1, boxed_format_bonus),
        )
    """
    def _reward(response: str, problem: "Problem") -> float:
        return sum(w * fn(response, problem) for w, fn in weighted_fns)

    names = [f"{w}*{fn.__qualname__}" for w, fn in weighted_fns]
    _reward.__qualname__ = f"combined({', '.join(names)})"
    return _reward
