"""Extract and compare final answers from model outputs.

Mirrors the spirit of grpo-pipeline/pipeline/rewards.py but kept self-contained
so the new pipeline does not depend on the legacy reward module.
"""
from __future__ import annotations

import re


def extract_boxed(text: str) -> str | None:
    """Return the contents of the LAST \\boxed{...} in *text*, or None."""
    if not text:
        return None
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out: list[str] = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
            out.append(c)
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(c)
        else:
            out.append(c)
        i += 1
    if depth != 0:
        return None
    return "".join(out).strip()


_LATEX_WRAPPERS = re.compile(
    r"\\(?:text|textbf|texttt|mathrm|mathbf|mathtt|mathit|operatorname)\{([^}]*)\}"
)
_LATEX_NOPS = re.compile(
    r"\\(?:lfloor|rfloor|lceil|rceil|left|right|bigl|bigr|Bigl|Bigr|,|;|!|quad|qquad)"
)
_LATEX_RELATIONS = re.compile(r"\\(?:approx|sim|simeq|cong|neq|le|ge|leq|geq)")


def normalize_answer(s: str | None) -> str:
    """Aggressively normalize a candidate answer string for equality comparison."""
    if s is None:
        return ""
    s = _LATEX_WRAPPERS.sub(r"\1", s)
    s = _LATEX_NOPS.sub(" ", s)
    s = _LATEX_RELATIONS.sub(" ", s)
    if "=" in s:
        s = s.split("=")[-1]
    s = s.replace(",", "").replace(" ", "").strip().rstrip(".")
    s = s.strip("$")
    try:
        n = float(s)
        if n == int(n):
            return str(int(n))
        return f"{n:.10g}"
    except Exception:
        return s


def answers_match(predicted: str | None, expected: str | None) -> bool:
    if predicted is None or expected is None:
        return False
    return normalize_answer(predicted) == normalize_answer(expected)
