"""Tests for boxed answer extraction and normalization."""

import pytest

from cookbook_grpo.parser import extract_boxed_answer, normalize_answer, answers_match


class TestExtractBoxedAnswer:
    def test_basic_integer(self):
        assert extract_boxed_answer("The answer is \\boxed{866}") == "866"

    def test_whitespace_inside(self):
        assert extract_boxed_answer("\\boxed{ 866 }") == "866"

    def test_last_box_selected(self):
        text = "First \\boxed{42}, then \\boxed{866}"
        assert extract_boxed_answer(text) == "866"

    def test_no_box(self):
        assert extract_boxed_answer("The answer is 866.") is None

    def test_malformed_box(self):
        assert extract_boxed_answer("\\boxed{866") is None

    def test_latex_fraction(self):
        assert extract_boxed_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"

    def test_nested_braces(self):
        assert extract_boxed_answer("\\boxed{\\frac{a}{b+{c}}}") == "\\frac{a}{b+{c}}"

    def test_decimal(self):
        assert extract_boxed_answer("\\boxed{42.0}") == "42.0"

    def test_expression_with_equals(self):
        # Should extract the full content including "100L = 866"
        assert extract_boxed_answer("\\boxed{100L = 866}") == "100L = 866"

    def test_empty_box(self):
        assert extract_boxed_answer("\\boxed{}") == ""

    def test_math_delimiters_stripped(self):
        assert extract_boxed_answer("\\boxed{$866$}") == "866"

    def test_no_boxed_keyword(self):
        assert extract_boxed_answer("boxed{866}") is None


class TestNormalizeAnswer:
    def test_integer(self):
        assert normalize_answer("866") == "866"

    def test_float_to_int(self):
        assert normalize_answer("3264.0000") == "3264"

    def test_commas_removed(self):
        assert normalize_answer("3,264") == "3264"

    def test_spaces_removed(self):
        assert normalize_answer(" 8 6 6 ") == "866"

    def test_trailing_period(self):
        assert normalize_answer("866.") == "866"

    def test_non_numeric(self):
        assert normalize_answer("abc") == "abc"


class TestAnswersMatch:
    def test_exact(self):
        assert answers_match("866", "866")

    def test_with_float_normalization(self):
        assert answers_match("866.0", "866")

    def test_with_commas(self):
        assert answers_match("3,264", "3264")

    def test_mismatch(self):
        assert not answers_match("865", "866")

    def test_non_numeric_match(self):
        assert answers_match("abc", "abc")

    def test_non_numeric_mismatch(self):
        assert not answers_match("abc", "def")
