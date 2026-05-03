"""Extract and compare final answers from model outputs.

Mirrors the spirit of grpo-pipeline/pipeline/rewards.py but kept self-contained
so the new pipeline does not depend on the legacy reward module.
"""
from __future__ import annotations

import json
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


# ---------------------------------------------------------------------------
# Multi-part answer extraction
# ---------------------------------------------------------------------------

# Match an "ANSWERS:" sentinel line followed by a JSON object. Tolerates extra
# whitespace, optional bold markers, and code-fence wrapping.
_ANSWERS_LINE = re.compile(
    r"(?:^|\n)\s*\*?\*?ANSWERS\*?\*?\s*[:=]\s*(\{[^\n]*\})",
    re.IGNORECASE,
)


def extract_answers_multipart(text: str, labels: list[str]) -> dict[str, str | None]:
    """Extract per-part numerical answers from a critic response.

    The critic is asked to end its response with a single line of the form
        ANSWERS: {"a": "X.XXXX", "b": "Y.YYYY", ...}
    so we try that JSON form first. If that fails, we fall back to scanning
    for "Part (label) answer" markers near \\boxed{} expressions.

    Returns a dict mapping each label in *labels* to its parsed answer string,
    or None if that part's answer could not be located.
    """
    out: dict[str, str | None] = {label: None for label in labels}
    if not text:
        return out

    # --- Primary: ANSWERS: {...} JSON line ---
    m = _ANSWERS_LINE.search(text)
    if m:
        snippet = m.group(1)
        # The model occasionally wraps values in \boxed{} or trailing prose;
        # do a permissive parse, then a strict one if that fails.
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                for label in labels:
                    if label in parsed and parsed[label] is not None:
                        out[label] = str(parsed[label]).strip()
                if any(v is not None for v in out.values()):
                    return out
        except json.JSONDecodeError:
            pass

    # --- Fallback: per-part labeled boxed answers ---
    # Look for "Part (label)" / "Part label" / "(label)" headers and find the
    # nearest following \boxed{...}.
    for label in labels:
        if out[label] is not None:
            continue
        header_re = re.compile(
            rf"(?:Part\s*\(?\s*{re.escape(label)}\s*\)?|\(\s*{re.escape(label)}\s*\))"
            rf"[\s\S]{{0,400}}?\\boxed\{{([^{{}}]+)\}}",
            re.IGNORECASE,
        )
        hm = header_re.search(text)
        if hm:
            out[label] = hm.group(1).strip()

    return out
