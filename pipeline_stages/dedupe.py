"""Deduplication of subproblems by normalized-text hash + k-gram Jaccard.

Used by Stage 1 (pre-solve, intra-run) and Stage 2 (cross-run aggregate).
Catches exact and near-exact duplicates. Does NOT catch semantic/conceptual
duplicates — see 2026-04-21-subproblem-dedupe-design.md for scope.
"""

from __future__ import annotations

import hashlib
import re

JACCARD_THRESHOLD = 0.85
# Controls the word k-gram size used by the exported `shingles()` function.
WORD_SHINGLE_SIZE = 5
# Controls the character k-gram size used internally by `DedupeIndex`.
CHAR_SHINGLE_SIZE = 5

_LATEX_SPACING = re.compile(r"\\[,;\s]|\\quad|\\qquad")
_WS = re.compile(r"\s+")
_WORD = re.compile(r"\w+")
# Strip spaces adjacent to math operators so "x + y" and "x+y" normalise the same.
_OP_SPACE = re.compile(r" (?=[+\-=*/^<>|&])|(?<=[+\-=*/^<>|&]) ")


def normalize_problem(text: str) -> str:
    """Lowercase, strip LaTeX spacing commands, collapse whitespace, and drop spaces immediately adjacent to math operators."""
    t = text.lower()
    # Replace LaTeX spacing commands with a regular space so that adjacent
    # word tokens remain separated (e.g. "Let\\,N(p)" → "let n(p)").
    t = _LATEX_SPACING.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    # Drop spaces that sit immediately beside math operators so that
    # "x + y = 1" and "x+y=1" share the same normalised form.
    t = _OP_SPACE.sub("", t)
    return t


def problem_hash(text: str) -> str:
    """Return the SHA-1 hexdigest of ``normalize_problem(text)``."""
    return hashlib.sha1(normalize_problem(text).encode("utf-8")).hexdigest()


def shingles(text: str, k: int = WORD_SHINGLE_SIZE) -> frozenset[str]:
    """Return a frozenset of word-level k-grams of *text*."""
    tokens = _WORD.findall(normalize_problem(text))
    if len(tokens) < k:
        return frozenset()
    return frozenset(
        " ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)
    )


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Return |a ∩ b| / |a ∪ b|, or 0.0 when both sets are empty.

    The 0.0 convention for two empty sets is intentional: an empty shingle set
    indicates text that is too short to shingle, so treating two such inputs as
    identical (Jaccard = 1.0) would cause incorrect duplicate detection.
    """
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _char_shingles(text: str, k: int = CHAR_SHINGLE_SIZE) -> frozenset[str]:
    """Return a frozenset of character-level k-grams of normalized *text*."""
    norm = normalize_problem(text)
    if len(norm) < k:
        return frozenset()
    return frozenset(norm[i : i + k] for i in range(len(norm) - k + 1))


class DedupeIndex:
    """Stateful index that reports whether a new problem is a duplicate.

    Uses two-level deduplication:
      1. Exact match via normalized SHA-1 hash.
      2. Near-duplicate match via character-level k-gram Jaccard similarity
         (threshold 0.85, which catches single-word differences in typical
         problem-length strings while keeping distinct problems separate).

    Character-level shingles (rather than word-level) give higher Jaccard
    for strings that differ by just a few words.

    Not thread-safe on its own; callers sharing an instance across threads
    must guard `add()` with an external lock.
    """

    def __init__(self, threshold: float = JACCARD_THRESHOLD):
        self._threshold = threshold
        self._hashes: set[str] = set()
        self._shingle_sets: list[frozenset[str]] = []
        self.n_kept = 0
        self.n_exact_dropped = 0
        self.n_fuzzy_dropped = 0

    def add(self, problem_text: str) -> bool:
        """Return True if added (unique), False if a duplicate.

        Empty/whitespace-only text is treated as a duplicate (returns False).
        """
        if not problem_text or not problem_text.strip():
            return False

        h = problem_hash(problem_text)
        if h in self._hashes:
            self.n_exact_dropped += 1
            return False

        sh = _char_shingles(problem_text)
        if sh:
            # _shingle_sets entries can be empty (e.g. very short text produced no char shingles); skip those to avoid spurious 0.0 comparisons.
            for existing in self._shingle_sets:
                if existing and jaccard(sh, existing) >= self._threshold:
                    self.n_fuzzy_dropped += 1
                    return False

        self._hashes.add(h)
        self._shingle_sets.append(sh)
        self.n_kept += 1
        return True
