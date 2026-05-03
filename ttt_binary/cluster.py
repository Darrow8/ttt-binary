"""Answer clustering + accept/reject decision rule for Stage 3 calibration.

Pure functions, no LLM calls or I/O — easy to unit-test in isolation.

Convention: ALL final answers are numerical, rounded to DECIMAL_PLACES=4
decimal places, and emitted inside \\boxed{}. The generator and solver
prompts both enforce this, and clustering canonicalizes via float-parse +
round, so e.g. "0.5", "1/2", "0.5000" all collapse to "0.5000".

Two main exports:

  cluster_answers(answers) -> {canonical_form: count}
      Group answers by 4-decimal numeric equivalence. Anything that doesn't
      parse cleanly as a real number maps to UNPARSEABLE.

  decide(clusters, k_calibrate, band, ambiguity_threshold,
         max_unparseable=3) -> Decision
      REVISIONS.md decision rule:
        - p1 in band AND p2 < ambiguity_threshold -> ACCEPT
        - p1 in band AND p2 >= ambiguity_threshold -> REJECT_AMBIGUOUS
        - p1 > band[1] -> REJECT_TOO_EASY
        - p1 < band[0] -> REJECT_TOO_HARD_OR_AMBIGUOUS
        - too many unparseable -> REJECT_TOO_HARD_OR_AMBIGUOUS (overrides)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Literal


UNPARSEABLE = "__unparseable__"

# Number of decimal places used for clustering-equivalent comparisons.
DECIMAL_PLACES = 4


# ---------------------------------------------------------------------------
# Canonicalization (numeric-only)
# ---------------------------------------------------------------------------

_LATEX_WRAPPERS = re.compile(
    r"\\(?:text|textbf|texttt|mathrm|mathbf|mathtt|mathit|operatorname)\{([^}]*)\}"
)
_LATEX_NOPS = re.compile(
    r"\\(?:lfloor|rfloor|lceil|rceil|left|right|bigl|bigr|Bigl|Bigr|"
    r",|;|!|quad|qquad|displaystyle)"
)
_LATEX_RELATIONS = re.compile(r"\\(?:approx|sim|simeq|cong|neq|le|ge|leq|geq)")
_THOUSANDS_SEP = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_FRAC_RE = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")


def _strip_decorations(s: str) -> str:
    """Strip LaTeX wrappers, $...$, leading 'x = ', thousands separators,
    \\boxed{} layers, and trailing punctuation. Cheap text cleanup only."""
    s = s.strip()
    # Peel \boxed{...} (possibly nested or multiple).
    while True:
        m = _BOXED_RE.search(s)
        if not m:
            break
        s = s[:m.start()] + m.group(1) + s[m.end():]
    s = _LATEX_WRAPPERS.sub(r"\1", s)
    s = _LATEX_NOPS.sub(" ", s)
    s = _LATEX_RELATIONS.sub(" ", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    # \frac{a}{b} -> ((a)/(b))
    s = _FRAC_RE.sub(r"((\1)/(\2))", s)
    # Trim $...$.
    s = s.strip("$ ").strip()
    s = s.rstrip(".").rstrip(",").strip()
    if "=" in s:
        s = s.split("=")[-1].strip()
    s = _THOUSANDS_SEP.sub("", s)
    return s


def _to_float(s: str) -> float | None:
    """Parse *s* as a real number. Supports plain decimals, scientific
    notation, and simple 'a/b' fractions. Returns None on failure."""
    s = s.strip()
    if not s:
        return None
    # Handle a single-level a/b fraction first (must not be tuple-like).
    if "/" in s and s.count("/") == 1:
        a, b = s.split("/", 1)
        try:
            num = float(a.strip())
            den = float(b.strip())
            if den == 0:
                return None
            return num / den
        except ValueError:
            pass
    # Handle a/(b) or (a)/(b) introduced by the \frac rewrite.
    paren_frac = re.fullmatch(r"\(*\s*\(([^()]+)\)\s*/\s*\(([^()]+)\)\s*\)*", s)
    if paren_frac:
        try:
            num = float(paren_frac.group(1).strip())
            den = float(paren_frac.group(2).strip())
            if den == 0:
                return None
            return num / den
        except ValueError:
            pass
    try:
        return float(s)
    except ValueError:
        return None


def _format_rounded(x: float, decimal_places: int = DECIMAL_PLACES) -> str:
    """Format *x* rounded to *decimal_places* decimals, eliminating -0.0."""
    rounded = round(float(x), decimal_places)
    if rounded == 0.0:
        rounded = 0.0  # collapses -0.0 -> 0.0
    return f"{rounded:.{decimal_places}f}"


def _canonicalize(answer, *, decimal_places: int = DECIMAL_PLACES) -> str:
    """Return the canonical 4-decimal form for *answer*, or UNPARSEABLE.

    Equal canonicals iff the two answers agree to *decimal_places* decimals.
    The convention is numeric-only: anything that does not parse as a real
    number (including symbolic expressions in unbound variables, unparseable
    text, NaN/inf) returns UNPARSEABLE.
    """
    if answer is None:
        return UNPARSEABLE
    s = str(answer).strip()
    if not s:
        return UNPARSEABLE
    cleaned = _strip_decorations(s)
    if not cleaned:
        return UNPARSEABLE
    x = _to_float(cleaned)
    if x is None:
        return UNPARSEABLE
    # Reject NaN / inf — both sentinel and canonicalized-NaN-as-string would
    # mislead clustering.
    if x != x or x in (float("inf"), float("-inf")):
        return UNPARSEABLE
    return _format_rounded(x, decimal_places=decimal_places)


def cluster_answers(
    answers: Iterable, *, decimal_places: int = DECIMAL_PLACES,
) -> dict[str, int]:
    """Group answers by canonical-form (4-decimal numeric) equivalence.

    Returns a dict mapping canonical_form -> count (insertion order preserved).
    Unparseable answers bucket under :data:`UNPARSEABLE`.
    """
    out: dict[str, int] = {}
    for a in answers:
        key = _canonicalize(a, decimal_places=decimal_places)
        out[key] = out.get(key, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------

DecisionKind = Literal[
    "ACCEPT",
    "REJECT_AMBIGUOUS",
    "REJECT_TOO_EASY",
    "REJECT_TOO_HARD_OR_AMBIGUOUS",
]


@dataclass
class Decision:
    kind: DecisionKind
    consensus_answer: str | None
    p1: float
    p2: float
    n_unparseable: int
    n_total: int
    clusters: dict[str, int] = field(default_factory=dict)
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.kind == "ACCEPT"


def decide(
    clusters: dict[str, int],
    *,
    k_calibrate: int,
    band: tuple[float, float] = (0.4, 0.6),
    ambiguity_threshold: float = 0.2,
    max_unparseable: int = 3,
) -> Decision:
    """Apply the REVISIONS.md decision rule.

    Args:
        clusters: cluster_answers() output, including any UNPARSEABLE bucket.
        k_calibrate: total number of solver attempts (denominator for p1/p2).
        band: difficulty band (inclusive on both ends).
        ambiguity_threshold: max allowed second-cluster fraction.
        max_unparseable: if more than this many attempts are unparseable, the
            problem is treated as too noisy regardless of the parseable
            cluster shape.
    """
    n_unparseable = clusters.get(UNPARSEABLE, 0)
    parseable = {k: v for k, v in clusters.items() if k != UNPARSEABLE}
    # Sort largest first; ties broken by insertion order for stable cluster labels.
    ranked = sorted(parseable.items(), key=lambda kv: (-kv[1], list(parseable).index(kv[0])))

    # Too many unparseable answers — treat as ill-posed/too hard regardless.
    if n_unparseable > max_unparseable:
        consensus = ranked[0][0] if ranked else None
        p1 = ranked[0][1] / k_calibrate if ranked else 0.0
        p2 = ranked[1][1] / k_calibrate if len(ranked) >= 2 else 0.0
        return Decision(
            kind="REJECT_TOO_HARD_OR_AMBIGUOUS",
            consensus_answer=consensus,
            p1=p1, p2=p2,
            n_unparseable=n_unparseable, n_total=k_calibrate,
            clusters=clusters,
            reason=f"unparseable={n_unparseable}/{k_calibrate} > max_unparseable={max_unparseable}",
        )

    # No parseable answers at all.
    if not ranked:
        return Decision(
            kind="REJECT_TOO_HARD_OR_AMBIGUOUS",
            consensus_answer=None,
            p1=0.0, p2=0.0,
            n_unparseable=n_unparseable, n_total=k_calibrate,
            clusters=clusters,
            reason="no parseable answers",
        )

    consensus = ranked[0][0]
    p1 = ranked[0][1] / k_calibrate
    p2 = ranked[1][1] / k_calibrate if len(ranked) >= 2 else 0.0

    # Decision tree — order matters: easy/hard checks happen on p1 alone, the
    # ambiguity check is layered inside the in-band branch.
    if p1 > band[1]:
        return Decision(
            kind="REJECT_TOO_EASY",
            consensus_answer=consensus,
            p1=p1, p2=p2,
            n_unparseable=n_unparseable, n_total=k_calibrate,
            clusters=clusters,
            reason=f"p1={p1:.2f} > {band[1]}",
        )
    if p1 < band[0]:
        return Decision(
            kind="REJECT_TOO_HARD_OR_AMBIGUOUS",
            consensus_answer=consensus,
            p1=p1, p2=p2,
            n_unparseable=n_unparseable, n_total=k_calibrate,
            clusters=clusters,
            reason=f"p1={p1:.2f} < {band[0]}",
        )
    # p1 in band
    if p2 >= ambiguity_threshold:
        # Two competing answers near 50/50 — well-posedness in doubt.
        runner_up = ranked[1][0] if len(ranked) >= 2 else "?"
        return Decision(
            kind="REJECT_AMBIGUOUS",
            consensus_answer=consensus,
            p1=p1, p2=p2,
            n_unparseable=n_unparseable, n_total=k_calibrate,
            clusters=clusters,
            reason=(
                f"two competing answers ({consensus!r} and {runner_up!r}) "
                f"with p1={p1:.2f}, p2={p2:.2f} >= {ambiguity_threshold}"
            ),
        )
    return Decision(
        kind="ACCEPT",
        consensus_answer=consensus,
        p1=p1, p2=p2,
        n_unparseable=n_unparseable, n_total=k_calibrate,
        clusters=clusters,
        reason=f"p1={p1:.2f} in band, p2={p2:.2f} < {ambiguity_threshold}",
    )


# ---------------------------------------------------------------------------
# Feedback strings for the regen prompt
# ---------------------------------------------------------------------------

def regen_feedback(decision: Decision) -> str:
    """Concrete instruction for the next generation attempt, given an outcome."""
    if decision.kind == "REJECT_TOO_EASY":
        return (
            f"- The previous attempt was too easy (consensus rate "
            f"{decision.p1:.2f}). Make the chain DEEPER: increase parameter "
            "sizes, require an extra non-trivial reduction, remove hint-y "
            "framing, or use parameters that are not classical/famous values."
        )
    if decision.kind == "REJECT_AMBIGUOUS":
        return (
            f"- The previous problem produced two competing answers (one "
            f"appearing in {decision.p1:.0%} of solves and another in "
            f"{decision.p2:.0%}). The problem must have ONE well-defined "
            "answer. Re-read your phrasing for ambiguity (e.g. ordering, "
            "orientation, choice of representative, multiplicity) and "
            "produce a problem whose answer is uniquely determined by the "
            "stated conditions."
        )
    if decision.kind == "REJECT_TOO_HARD_OR_AMBIGUOUS":
        if decision.n_unparseable > 0:
            return (
                f"- The previous problem was likely ill-posed: "
                f"{decision.n_unparseable}/{decision.n_total} solver attempts "
                "produced no parseable final answer. Simplify the framing or "
                "remove sources of ambiguity; ensure the answer is a single "
                "real number rounded to 4 decimal places inside \\boxed{}."
            )
        return (
            f"- The previous attempt was too hard (top-cluster rate "
            f"{decision.p1:.2f}, attempts spread across many wrong answers). "
            "Reduce parameter sizes slightly, simplify the framing, or remove "
            "one source of bookkeeping complexity. Keep the chain of all "
            "listed skills intact."
        )
    return "- Aim for a difficulty where a strong solver gets it right roughly half the time."
