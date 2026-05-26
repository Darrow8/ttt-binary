"""Boxed answer extraction and normalization."""

from __future__ import annotations


def extract_boxed_answer(text: str) -> str | None:
    """Extract the content of the last \\boxed{...} expression in text.

    Handles nested braces, surrounding whitespace, and math delimiters.
    Returns None if no valid boxed answer is found.
    """
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None

    start = idx + len("\\boxed{")
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth == 0:
                content = text[start:i].strip()
                content = _strip_math_delimiters(content)
                return content
            depth -= 1

    return None


def _strip_math_delimiters(s: str) -> str:
    """Remove surrounding $ or \\( \\) delimiters."""
    s = s.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    if s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()
    return s


def normalize_answer(s: str) -> str:
    """Normalize an answer string for exact comparison.

    Removes commas, spaces, trailing periods. Converts float-valued integers
    to int strings (e.g. "3264.0000" -> "3264"). Uses Decimal to avoid
    float precision loss on large integers (e.g. 2^53 + 1).
    """
    from decimal import Decimal, InvalidOperation

    s = s.replace(",", "").replace(" ", "").strip().rstrip(".")
    try:
        d = Decimal(s)
        if not d.is_finite():
            return s
        if d == d.to_integral_value():
            return str(int(d.to_integral_value()))
        return str(d.normalize())
    except InvalidOperation:
        pass
    return s


def answers_match(predicted: str, reference: str) -> bool:
    """Check if predicted and reference answers match after normalization."""
    return normalize_answer(predicted) == normalize_answer(reference)
