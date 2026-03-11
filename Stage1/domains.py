"""
Domain configurations for TTT-Discover.

Each domain defines how to:
    1. Generate subproblems from a hard source problem
    2. Prompt the LLM to solve generated problems
    3. Extract and normalize answers from solutions
    4. Validate that an answer is well-formed

Adding a new domain = adding a new DomainConfig instance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class DomainConfig:
    """Everything TTT-Discover needs to know about a problem domain."""

    name: str

    # --- Prompt templates ---
    # {problem}, {batch_size}, {failed_solutions} are interpolated.
    generate_prompt: str

    # {problem} is interpolated.
    solve_prompt: str

    # {solution} is interpolated.  Used when regex extraction fails.
    extract_answer_prompt: str

    # --- Answer handling ---
    # Returns True if the answer string is valid for this domain.
    is_valid_answer: Callable[[str], bool] = lambda a: bool(a.strip())

    # Normalize an answer for comparison (e.g. canonical SMILES, strip whitespace).
    normalize_answer: Callable[[str], str] = lambda a: a.strip()


# ---------------------------------------------------------------------------
# Math domain (original)
# ---------------------------------------------------------------------------

_MATH_GENERATE_PROMPT = """\
You are designing training problems for test-time training on ONE hard source problem.
Your job is to create problems that preserve the same mathematical bottleneck as the
source problem, while being smaller, cleaner, and self-contained.

Your goal is NOT to create random "similar-looking" problems or parametric variants
(e.g. changing the number of variables). Your goal is to create problems whose solution
would train the model on a reusable subskill needed for the original problem.

## Source Problem

{problem}

## Attempted Solutions (may be incorrect)

Below are solution attempts on the source problem. They may contain errors — use them
ONLY to identify what mathematical techniques and subskills are being attempted, NOT
to determine the correct answer.

{failed_solutions}

## Instructions

1. Read the attempted solutions above and identify the DISTINCT mathematical techniques
   they use (e.g. Dirichlet series, CRT, Euler products, divisor sums, asymptotic
   estimates, Tauberian theorems, etc.)
2. For each technique, consider: what is a simpler, self-contained problem that would
   drill EXACTLY that technique?
3. Generate {batch_size} problems, each targeting a DIFFERENT technique or subskill.
   DO NOT generate multiple problems that only differ in a parameter (like the number
   of factors in a product). Each problem must be structurally distinct.

Each generated problem MUST satisfy all of the following:
- It has a SINGLE numerical final answer that is a DECIMAL NUMBER (not a symbolic
  expression like 1/π², 6/π², √2, ln 2, etc.). If the exact answer is irrational
  or a fraction, the problem must ask the solver to ROUND to 4 decimal places.
  For example, instead of "Find the value of C" where C = 6/π², write
  "Find the value of C, rounded to 4 decimal places" (answer: 0.6079).
- It is self-contained.
- It isolates one real bottleneck or subskill from the source problem.
- It is NOT a parametric variant of the source problem (e.g. changing ab+1=cde to
  ab+1=cdef is NOT acceptable — that tests the same skill at the same difficulty).
- It avoids fake complexity and decorative algebraic clutter.
- CRITICAL: The problem must be HARD — comparable to a research-level or competition
  math problem. A strong LLM should get it WRONG 20-40% of the time.
  DO NOT generate textbook exercises, definitions, or routine calculations like
  "compute ζ(2)", "count divisors of 60", "evaluate a standard limit", or
  "state a well-known asymptotic formula". These are TOO EASY and useless for training.
  The problem should require COMBINING techniques or applying them in a non-obvious way.

## Output Format

Return ONLY valid JSON, with no markdown fences and no extra text.
Return a JSON array of objects in exactly this shape:
[
  {{"problem": "..."}}
]
"""

_MATH_SOLVE_PROMPT = """\
## Problem

{problem}

When you are done, write your final numerical answer on the very last line \
in exactly this format (including the double asterisks):

**ANSWER: <your numerical answer here>**

Your answer MUST be a single decimal number (e.g. 0.6079, 42, 3.1416). \
Do NOT write symbolic expressions like 1/π², 6/π², √2, or ln 2. \
If the answer is irrational or a fraction, round to 4 decimal places.

Solve the problem above step by step.

"""

_MATH_EXTRACT_PROMPT = """\
Below is a student's solution to a math problem. What is the student's \
final answer? Reply with ONLY the answer value — nothing else.

Rules:
- If the answer is a number, return just the number (e.g. 42, 3/2, 0.75).
- If the answer is "infinity" or "does not exist" or similar, return \
  that phrase in lowercase (e.g. "infinity", "does not exist").
- No LaTeX, no units, no extra words.
- If there is no clear final answer, reply with exactly: NONE

## Solution

{solution}"""

_NUMERIC_RE = re.compile(r"^[+-]?\d+([.,]\d+)?(/\d+)?$")


def _math_is_valid(answer: str) -> bool:
    return bool(_NUMERIC_RE.match(answer))


def _math_normalize(answer: str) -> str:
    a = answer.strip().lower()
    a = re.sub(r"[\\${}]", "", a)
    a = re.sub(r"\s+", "", a)
    a = a.replace(",", ".")
    try:
        from fractions import Fraction
        if "/" in a and a.replace("/", "").replace("-", "").replace(".", "").isdigit():
            val = float(Fraction(a))
            a = f"{val:.10g}"
        elif a.replace(".", "").replace("-", "").isdigit():
            val = float(a)
            a = f"{val:.10g}"
    except (ValueError, ZeroDivisionError):
        pass
    return a


MATH = DomainConfig(
    name="math",
    generate_prompt=_MATH_GENERATE_PROMPT,
    solve_prompt=_MATH_SOLVE_PROMPT,
    extract_answer_prompt=_MATH_EXTRACT_PROMPT,
    is_valid_answer=_math_is_valid,
    normalize_answer=_math_normalize,
)


# ---------------------------------------------------------------------------
# Retrosynthesis domain
# ---------------------------------------------------------------------------

_RETRO_GENERATE_PROMPT = """\
You are designing training problems for test-time training on ONE hard retrosynthesis
problem. Your job is to create SIMPLER retrosynthesis problems that teach the model
the specific reaction types and disconnection strategies needed for the source problem.

## Source Problem

{problem}

## Attempted Solutions (may be incorrect)

Below are the model's attempts on the source problem. They may contain errors — use
them to identify which disconnection strategies the model is trying and WHERE it fails.

{failed_solutions}

## Instructions

1. Analyze the source product and identify ALL the key bonds that could be disconnected.
2. Identify what reaction types are relevant (e.g. Suzuki coupling, amide formation,
   N-alkylation, reductive amination, protection/deprotection, etc.)
3. Identify WHERE the model's attempts go wrong — what bonds does it incorrectly cut?
   What reactant types does it confuse?
4. Generate {batch_size} retrosynthesis problems that each target a DIFFERENT subskill:
   - Some should drill the CORRECT disconnection on simpler molecules
   - Some should be "decoy" problems where the model must NOT cut an obvious-looking
     bond (e.g. an amide bond when the actual reaction is N-alkylation)
   - Each problem should use different molecular scaffolds to avoid memorization

Each generated problem MUST:
- Be a retrosynthesis problem: "Given product SMILES X, predict the reactants"
- Have a SINGLE correct set of reactants in SMILES notation
- Include the product SMILES and expected reactant SMILES in the output
- Be SIMPLER than the source problem (fewer rings, fewer functional groups)
- Be self-contained and unambiguous — only ONE reasonable disconnection
- Target a difficulty where a strong LLM gets it wrong 20-40% of the time

## Output Format

Return ONLY valid JSON, with no markdown fences and no extra text.
Return a JSON array of objects in exactly this shape:
[
  {{"problem": "You are an expert organic chemist performing retrosynthetic analysis.\\n\\nGiven the target product molecule (SMILES): <PRODUCT_SMILES>\\n\\nPredict the set of reactants that can be combined to synthesize this product. Provide reactant SMILES separated by '.' for multiple reactants.\\n\\nThink step by step about which bonds were formed, what functional group transformations occurred, and what the starting materials must have been.\\n\\nPut your final reactant SMILES inside \\\\boxed{{}}."}}
]
"""

_RETRO_SOLVE_PROMPT = """\
## Problem

{problem}

Think step by step:
1. Draw out the product structure from the SMILES
2. Identify which bonds were likely formed in the synthesis
3. Determine what reaction type was used (Suzuki coupling, amide formation, etc.)
4. Work backwards to determine the starting materials
5. Write the reactant SMILES

Put your final reactant SMILES inside \\boxed{{}}.
If there are multiple reactants, separate them with '.' inside the boxed answer.

"""

_RETRO_EXTRACT_PROMPT = """\
Below is a chemist's retrosynthetic analysis. Extract ONLY the final reactant \
SMILES string(s) they propose. Reply with ONLY the SMILES — nothing else.

Rules:
- Return reactant SMILES separated by '.' if multiple reactants
- No text, no explanation, just the SMILES string(s)
- If there is no clear answer, reply with exactly: NONE

## Solution

{solution}"""


def _retro_is_valid(answer: str) -> bool:
    """Check if the answer looks like valid SMILES (contains atoms, no prose)."""
    answer = answer.strip()
    if not answer:
        return False
    # Must contain at least one atom-like character
    if not re.search(r"[A-Z][a-z]?", answer):
        return False
    # Must not contain spaces (prose, not SMILES)
    if " " in answer:
        return False
    # Rough SMILES check: contains only valid SMILES characters
    if re.match(r"^[A-Za-z0-9@+\-\[\]\(\)\\/#=:%.]+$", answer):
        return True
    return False


def _retro_normalize(answer: str) -> str:
    """Canonicalize SMILES using RDKit, sort multi-component reactants."""
    answer = answer.strip()
    # Remove \text{} wrappers
    answer = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)
    try:
        from rdkit import Chem
        parts = answer.split(".")
        canonical = []
        for part in parts:
            mol = Chem.MolFromSmiles(part.strip())
            if mol is None:
                return answer  # can't canonicalize, return as-is
            canonical.append(Chem.MolToSmiles(mol))
        return ".".join(sorted(canonical))
    except Exception:
        return answer


RETROSYNTHESIS = DomainConfig(
    name="retrosynthesis",
    generate_prompt=_RETRO_GENERATE_PROMPT,
    solve_prompt=_RETRO_SOLVE_PROMPT,
    extract_answer_prompt=_RETRO_EXTRACT_PROMPT,
    is_valid_answer=_retro_is_valid,
    normalize_answer=_retro_normalize,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DOMAINS: dict[str, DomainConfig] = {
    "math": MATH,
    "retrosynthesis": RETROSYNTHESIS,
}
