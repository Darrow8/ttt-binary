"""Unit tests for pipeline_stages.dedupe."""

from __future__ import annotations

import pytest

from pipeline_stages.dedupe import (
    DedupeIndex,
    jaccard,
    normalize_problem,
    problem_hash,
    shingles,
)


class TestNormalize:
    def test_lowercases(self):
        assert normalize_problem("Hello World") == "hello world"

    def test_collapses_whitespace(self):
        assert normalize_problem("a   b\n\tc") == "a b c"

    def test_strips_latex_spacing(self):
        assert normalize_problem(r"x\,+\;y\ =\quad 1") == "x+y=1"

    def test_idempotent(self):
        once = normalize_problem(r"Consider \quad x \, + y")
        twice = normalize_problem(once)
        assert once == twice


class TestHash:
    def test_same_text_same_hash(self):
        assert problem_hash("hello") == problem_hash("hello")

    def test_case_differences_collapse(self):
        assert problem_hash("Hello") == problem_hash("hello")

    def test_whitespace_differences_collapse(self):
        assert problem_hash("a  b") == problem_hash("a b")

    def test_different_text_different_hash(self):
        assert problem_hash("foo") != problem_hash("bar")


class TestShingles:
    def test_produces_kgrams(self):
        out = shingles("one two three four five six", k=5)
        assert "one two three four five" in out
        assert "two three four five six" in out
        assert len(out) == 2

    def test_too_short_returns_empty(self):
        assert shingles("too short", k=5) == frozenset()


class TestJaccard:
    def test_identical_sets_return_1(self):
        s = frozenset({"a", "b"})
        assert jaccard(s, s) == 1.0

    def test_disjoint_sets_return_0(self):
        assert jaccard(frozenset({"a"}), frozenset({"b"})) == 0.0

    def test_both_empty_returns_0(self):
        assert jaccard(frozenset(), frozenset()) == 0.0


class TestDedupeIndex:
    def test_first_add_returns_true(self):
        idx = DedupeIndex()
        assert idx.add("Consider the polynomial f(x) = x^4 - 5.") is True

    def test_exact_duplicate_returns_false(self):
        idx = DedupeIndex()
        idx.add("Consider the polynomial f(x) = x^4 - 5.")
        assert idx.add("Consider the polynomial f(x) = x^4 - 5.") is False
        assert idx.n_exact_dropped == 1
        assert idx.n_fuzzy_dropped == 0

    def test_whitespace_duplicate_detected_as_exact(self):
        idx = DedupeIndex()
        idx.add("Consider the polynomial f(x) = x^4 - 5.")
        assert idx.add("Consider  the polynomial  f(x) = x^4 - 5.") is False
        assert idx.n_exact_dropped == 1

    def test_latex_spacing_duplicate_detected_as_exact(self):
        idx = DedupeIndex()
        idx.add(r"Let N(p) be the number of roots of f modulo p.")
        assert idx.add(r"Let\,N(p) be the number of roots of f modulo p.") is False
        assert idx.n_exact_dropped == 1

    def test_fuzzy_duplicate_detected(self):
        idx = DedupeIndex()
        a = (
            "Consider the polynomial f(x) = x^4 - 5 in Z[x]. For a prime p "
            "such that f modulo p is separable, let N(p) be the number of "
            "solutions. Compute the limit of the average N(p)."
        )
        b = (
            "Let us consider the polynomial f(x) = x^4 - 5 in Z[x]. For a "
            "prime p such that f modulo p is separable, let N(p) be the "
            "number of solutions. Compute the limit of the average N(p)."
        )
        assert idx.add(a) is True
        assert idx.add(b) is False
        assert idx.n_fuzzy_dropped == 1
        assert idx.n_exact_dropped == 0

    def test_distinct_problems_both_kept(self):
        idx = DedupeIndex()
        assert idx.add("Find the number of integer solutions to x + y = 10.") is True
        assert idx.add(
            "Compute the determinant of the 3x3 matrix with ones on the "
            "diagonal and zeros elsewhere."
        ) is True
        assert idx.n_kept == 2
        assert idx.n_exact_dropped == 0
        assert idx.n_fuzzy_dropped == 0

    def test_empty_text_rejected(self):
        idx = DedupeIndex()
        assert idx.add("") is False
        assert idx.n_kept == 0

    def test_stats_across_mixed_sequence(self):
        idx = DedupeIndex()
        idx.add("alpha beta gamma delta epsilon zeta")  # kept
        idx.add("ALPHA  beta gamma delta epsilon zeta")  # exact dup (normalize)
        idx.add("alpha beta gamma delta epsilon zeta eta")  # fuzzy dup (high overlap)
        idx.add("completely different problem here now")  # kept
        assert idx.n_kept == 2
        assert idx.n_exact_dropped == 1
        assert idx.n_fuzzy_dropped == 1


class TestStage2Integration:
    def test_aggregate_drops_fuzzy_duplicates(self, tmp_path, monkeypatch):
        """Stage 2 aggregate should drop near-duplicate problems across runs."""
        import json

        # Lay out a minimal fake runs/<id>/stage1/<ts>/keeps.json tree.
        runs_root = tmp_path / "runs" / "testid"
        s1 = runs_root / "stage1"
        (s1 / "run1").mkdir(parents=True)
        (s1 / "run2").mkdir(parents=True)

        dup_a = "Consider the polynomial f(x)=x^4-5 in Z[x]. Compute the density of separable primes."
        dup_b = "Let us consider the polynomial f(x)=x^4-5 in Z[x]. Compute the density of separable primes."
        unique = "Find the number of ways to tile a 2x10 rectangle with dominoes."

        (s1 / "run1" / "keeps.json").write_text(json.dumps({
            "source_problem": "src",
            "target_agreement_low": 0.6,
            "target_agreement_high": 0.8,
            "n_problems": 2,
            "problems": [
                {"problem": dup_a, "ground_truth_answer": "1", "agreement_rate": 0.7,
                 "all_answers": [], "all_solutions": [], "n_samples": 10},
                {"problem": unique, "ground_truth_answer": "89", "agreement_rate": 0.75,
                 "all_answers": [], "all_solutions": [], "n_samples": 10},
            ],
        }))
        (s1 / "run2" / "keeps.json").write_text(json.dumps({
            "source_problem": "src",
            "target_agreement_low": 0.6,
            "target_agreement_high": 0.8,
            "n_problems": 1,
            "problems": [
                {"problem": dup_b, "ground_truth_answer": "1", "agreement_rate": 0.7,
                 "all_answers": [], "all_solutions": [], "n_samples": 10},
            ],
        }))

        from pipeline_stages import stage2_aggregate
        monkeypatch.setattr(stage2_aggregate, "REPO_ROOT", tmp_path)

        summary = stage2_aggregate.aggregate_one("testid", include_skips=False)
        assert summary["n_problems"] == 2  # dup_a/dup_b collapsed
        assert summary["dedupe"]["n_kept"] == 2
        assert summary["dedupe"]["n_dropped_fuzzy"] + summary["dedupe"]["n_dropped_exact"] == 1
