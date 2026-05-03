"""Unit tests for ttt_binary.cluster (Stage 3 calibration helpers).

Covers:
- Numeric / symbolic / unparseable canonicalization
- Cluster counting
- The decide() decision rule and its edge cases (REVISIONS.md spec)
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from ttt_binary.cluster import (
    UNPARSEABLE,
    cluster_answers,
    decide,
    regen_feedback,
)


# ---------------------------------------------------------------------------
# cluster_answers — canonicalization
# ---------------------------------------------------------------------------

class TestClusterNumeric:
    def test_int_decimal_equivalence(self):
        c = cluster_answers(["3264", "3264.0", "3264.00", "3264.0000"])
        assert sum(c.values()) == 4
        assert len(c) == 1
        # Canonical form is 4-decimal.
        assert "3264.0000" in c

    def test_thousand_separator_equivalence(self):
        c = cluster_answers(["3,264", "3264", "3264.0000"])
        assert len(c) == 1

    def test_assignment_form(self):
        c = cluster_answers(["x = 866", "866", "y=866"])
        assert len(c) == 1

    def test_boxed_strip(self):
        c = cluster_answers([r"\boxed{866.0000}", "866", "866.0"])
        assert len(c) == 1

    def test_distinct_numbers_split(self):
        c = cluster_answers(["866", "3264", "866", "0"])
        assert sum(c.values()) == 4
        assert len(c) == 3
        # Largest is "866" with count 2 — canonical form is 866.0000
        max_key = max(c, key=lambda k: c[k])
        assert max_key == "866.0000" and c[max_key] == 2

    def test_canonical_clustering_fraction_decimal(self):
        # spec test: ["0.5", "1/2", "0.50"] -> one cluster
        c = cluster_answers(["0.5", "1/2", "0.50"])
        assert len(c) == 1, f"expected 1 cluster, got {c}"
        assert sum(c.values()) == 3
        assert "0.5000" in c

    def test_latex_frac_clusters_with_decimal(self):
        c = cluster_answers([r"\frac{1}{2}", "1/2", "0.5"])
        assert len(c) == 1, c

    def test_negative_numbers_with_neg_zero(self):
        c = cluster_answers(["-0.0001", "-0.00009", "0.0000", "-0.0000"])
        # First two round to -0.0001; last two collapse via -0.0 elimination
        assert "-0.0001" in c and "0.0000" in c
        assert c["-0.0001"] == 2 and c["0.0000"] == 2

    def test_rounding_within_4dp(self):
        # 0.12345 and 0.12348 differ in the 5th decimal — both round to 0.1235
        c = cluster_answers(["0.12345", "0.12348"])
        assert len(c) == 1
        # 0.12351 rounds up to 0.1235 too (by banker's rounding); 0.12361 → 0.1236
        c2 = cluster_answers(["0.1235", "0.12361"])
        assert len(c2) == 2

    def test_scientific_notation(self):
        c = cluster_answers(["1.5e2", "150", "150.0"])
        assert len(c) == 1


class TestClusterSymbolicRejected:
    """Symbolic and non-numeric answers should now be UNPARSEABLE."""

    def test_pure_symbolic(self):
        c = cluster_answers(["x^2 - 1", "(x-1)*(x+1)"])
        assert c == {UNPARSEABLE: 2}

    def test_constant_with_unbound_var(self):
        c = cluster_answers(["x + 0"])
        assert c == {UNPARSEABLE: 1}


class TestClusterUnparseable:
    def test_none_and_empty(self):
        c = cluster_answers([None, "", "   "])
        assert c == {UNPARSEABLE: 3}

    def test_mixed_unparseable_and_numeric(self):
        c = cluster_answers(["3264", None, "3264", ""])
        assert c.get(UNPARSEABLE) == 2
        assert c.get("3264.0000") == 2

    def test_nan_and_inf_unparseable(self):
        c = cluster_answers(["nan", "inf", "-inf", "NaN"])
        assert c == {UNPARSEABLE: 4}

    def test_pure_text_unparseable(self):
        c = cluster_answers(["the answer", "depends", "see above"])
        assert c == {UNPARSEABLE: 3}


# ---------------------------------------------------------------------------
# decide — pure-function decision rule
# ---------------------------------------------------------------------------

K = 10
BAND = (0.4, 0.6)
AMB = 0.2


def _from_counts(counts_for_label):
    """Helper: build a clusters dict from a dict-of-counts (no canonicalization)."""
    return dict(counts_for_label)


class TestDecideAccept:
    def test_accept_clean_majority(self):
        # spec test variant: {A: 5, B: 1, C: 1, D: 1, E: 1, F: 1}
        clusters = _from_counts({"A": 5, "B": 1, "C": 1, "D": 1, "E": 1, "F": 1})
        d = decide(clusters, k_calibrate=K, band=BAND, ambiguity_threshold=AMB)
        assert d.kind == "ACCEPT", d
        assert d.consensus_answer == "A"
        assert d.p1 == 0.5
        assert d.p2 == 0.1

    def test_accept_lower_band_edge(self):
        clusters = _from_counts({"A": 4, "B": 1, "C": 1, "D": 1, "E": 1, "F": 1, "G": 1})
        d = decide(clusters, k_calibrate=K, band=BAND, ambiguity_threshold=AMB)
        assert d.kind == "ACCEPT", d
        assert d.p1 == 0.4

    def test_accept_upper_band_edge(self):
        clusters = _from_counts({"A": 6, "B": 1, "C": 1, "D": 1, "E": 1})
        d = decide(clusters, k_calibrate=K, band=BAND, ambiguity_threshold=AMB)
        assert d.kind == "ACCEPT", d
        assert d.p1 == 0.6


class TestDecideRejectAmbiguous:
    def test_reject_ambiguous_44_2(self):
        clusters = _from_counts({"A": 4, "B": 4, "C": 2})
        d = decide(clusters, k_calibrate=K, band=BAND, ambiguity_threshold=AMB)
        assert d.kind == "REJECT_AMBIGUOUS", d
        assert d.p1 == 0.4 and d.p2 == 0.4

    def test_borderline_50_40_10(self):
        # spec test: {A: 5, B: 4, C: 1} -> reject_ambiguous (p2=0.4 >= 0.2)
        clusters = _from_counts({"A": 5, "B": 4, "C": 1})
        d = decide(clusters, k_calibrate=K, band=BAND, ambiguity_threshold=AMB)
        assert d.kind == "REJECT_AMBIGUOUS", d


class TestDecideRejectTooEasy:
    def test_reject_too_easy_82(self):
        # spec test: {A: 8, B: 2} -> reject_too_easy
        clusters = _from_counts({"A": 8, "B": 2})
        d = decide(clusters, k_calibrate=K, band=BAND, ambiguity_threshold=AMB)
        assert d.kind == "REJECT_TOO_EASY", d
        assert d.p1 == 0.8

    def test_reject_too_easy_all_agree(self):
        clusters = _from_counts({"A": 10})
        d = decide(clusters, k_calibrate=K, band=BAND, ambiguity_threshold=AMB)
        assert d.kind == "REJECT_TOO_EASY", d


class TestDecideRejectTooHard:
    def test_reject_too_hard_uniform(self):
        # spec test: {A: 2, B: 2, C: 2, D: 2, E: 2} -> too_hard_or_ambiguous
        clusters = _from_counts({"A": 2, "B": 2, "C": 2, "D": 2, "E": 2})
        d = decide(clusters, k_calibrate=K, band=BAND, ambiguity_threshold=AMB)
        assert d.kind == "REJECT_TOO_HARD_OR_AMBIGUOUS", d
        assert d.p1 == 0.2

    def test_reject_too_hard_all_distinct(self):
        # 10 distinct answers — p1 = 0.1
        clusters = _from_counts({chr(ord("A") + i): 1 for i in range(10)})
        d = decide(clusters, k_calibrate=K, band=BAND, ambiguity_threshold=AMB)
        assert d.kind == "REJECT_TOO_HARD_OR_AMBIGUOUS", d


class TestDecideUnparseable:
    def test_too_many_unparseable_overrides(self):
        # 4 unparseable + 5 of one + 1 of another. p1 would be 0.5 (in band)
        # and p2 would be 0.1 (under threshold) — would normally accept — but
        # the unparseable count > max_unparseable=3 rejects it.
        clusters = _from_counts({UNPARSEABLE: 4, "A": 5, "B": 1})
        d = decide(clusters, k_calibrate=K, band=BAND,
                   ambiguity_threshold=AMB, max_unparseable=3)
        assert d.kind == "REJECT_TOO_HARD_OR_AMBIGUOUS", d
        assert "unparseable" in d.reason

    def test_unparseable_under_threshold_does_not_override(self):
        clusters = _from_counts({UNPARSEABLE: 2, "A": 5, "B": 1, "C": 1, "D": 1})
        d = decide(clusters, k_calibrate=K, band=BAND,
                   ambiguity_threshold=AMB, max_unparseable=3)
        assert d.kind == "ACCEPT", d
        assert d.p1 == 0.5 and d.p2 == 0.1

    def test_all_unparseable(self):
        clusters = _from_counts({UNPARSEABLE: 10})
        d = decide(clusters, k_calibrate=K, band=BAND, ambiguity_threshold=AMB)
        assert d.kind == "REJECT_TOO_HARD_OR_AMBIGUOUS", d
        assert d.consensus_answer is None


class TestDecideStats:
    def test_record_carries_clusters(self):
        clusters = _from_counts({"A": 5, "B": 1, "C": 1, "D": 1, "E": 1, "F": 1})
        d = decide(clusters, k_calibrate=K, band=BAND, ambiguity_threshold=AMB)
        assert d.clusters == clusters
        assert d.n_total == K
        assert d.n_unparseable == 0


class TestRegenFeedback:
    def test_too_easy_feedback_mentions_deeper(self):
        clusters = _from_counts({"A": 8, "B": 2})
        d = decide(clusters, k_calibrate=K, band=BAND, ambiguity_threshold=AMB)
        fb = regen_feedback(d)
        assert "deeper" in fb.lower() or "DEEPER" in fb

    def test_ambiguous_feedback_mentions_competing(self):
        clusters = _from_counts({"A": 5, "B": 4, "C": 1})
        d = decide(clusters, k_calibrate=K, band=BAND, ambiguity_threshold=AMB)
        fb = regen_feedback(d)
        assert "competing" in fb.lower() or "well-defined" in fb.lower()

    def test_too_hard_feedback_mentions_simplify(self):
        clusters = _from_counts({"A": 2, "B": 2, "C": 2, "D": 2, "E": 2})
        d = decide(clusters, k_calibrate=K, band=BAND, ambiguity_threshold=AMB)
        fb = regen_feedback(d)
        assert "simplify" in fb.lower() or "tractable" in fb.lower() or "reduce" in fb.lower()
