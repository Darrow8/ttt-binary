"""Unit tests for Stage1.taxonomy_generation."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from Stage1.taxonomy_generation import (
    GENERATOR_MODEL,
    Skill,
)


def test_generator_model_hardcoded():
    """The model constant must be the 120b Vertex MaaS model, not overridable."""
    assert GENERATOR_MODEL == "openai/gpt-oss-120b-maas"


class TestSkill:
    def test_fields(self):
        s = Skill(
            name="Bezout intersection count",
            description="Counting degrees of intersection on projective varieties.",
            example_problem_hint="Compute the intersection number of two plane curves.",
        )
        assert s.name == "Bezout intersection count"
        assert asdict(s) == {
            "name": "Bezout intersection count",
            "description": "Counting degrees of intersection on projective varieties.",
            "example_problem_hint": "Compute the intersection number of two plane curves.",
        }
