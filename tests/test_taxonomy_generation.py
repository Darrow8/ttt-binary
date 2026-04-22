"""Unit tests for Stage1.taxonomy_generation."""

from __future__ import annotations

import json
from dataclasses import asdict
from unittest.mock import MagicMock

import pytest

from Stage1.taxonomy_generation import (
    DECOMPOSE_PROMPT,
    GENERATOR_MODEL,
    Skill,
    decompose_target,
    load_skills,
    save_skills,
)


def test_generator_model_hardcoded():
    """The model constant must be the 120b Vertex MaaS model, not overridable."""
    assert GENERATOR_MODEL == "openai/gpt-oss-120b-maas"


class TestSkill:
    def test_fields(self):
        s = Skill(
            name="Bezout intersection count",
            description="Counting degrees of intersection on projective varieties.",
        )
        assert s.name == "Bezout intersection count"
        assert asdict(s) == {
            "name": "Bezout intersection count",
            "description": "Counting degrees of intersection on projective varieties.",
        }

    def test_load_skills_tolerates_legacy_hint_field(self, tmp_path):
        """Old skills.json files with example_problem_hint must still load."""
        path = tmp_path / "skills.json"
        path.write_text(json.dumps({
            "skills": [
                {"name": f"S{i}", "description": f"D{i}",
                 "example_problem_hint": f"H{i} (legacy)"}
                for i in range(3)
            ]
        }))
        loaded = load_skills(str(path))
        assert loaded is not None and len(loaded) == 3
        assert loaded[0].name == "S0"
        assert loaded[0].description == "D0"
        # Old hint field is silently dropped.
        assert not hasattr(loaded[0], "example_problem_hint")


def _mock_client_returning(content: str) -> MagicMock:
    """Build a mock OpenAI client whose chat.completions.create returns `content`."""
    client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    client.chat.completions.create.return_value = resp
    return client


_VALID_SKILLS_JSON = json.dumps({
    "skills": [
        {"name": f"Skill {i}",
         "description": f"Description for skill {i}."}
        for i in range(10)
    ]
})


class TestDecompose:
    def test_valid_json_returns_ten_skills(self):
        client = _mock_client_returning(_VALID_SKILLS_JSON)
        skills = decompose_target(client, "the target problem text")
        assert len(skills) == 10
        assert all(isinstance(s, Skill) for s in skills)
        assert skills[0].name == "Skill 0"

    def test_uses_hardcoded_model(self):
        client = _mock_client_returning(_VALID_SKILLS_JSON)
        decompose_target(client, "target")
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == GENERATOR_MODEL

    def test_no_max_tokens_no_timeout(self):
        client = _mock_client_returning(_VALID_SKILLS_JSON)
        decompose_target(client, "target")
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert "max_tokens" not in call_kwargs
        assert "timeout" not in call_kwargs

    def test_includes_target_in_prompt(self):
        client = _mock_client_returning(_VALID_SKILLS_JSON)
        decompose_target(client, "UNIQUE TARGET STRING")
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        assert "UNIQUE TARGET STRING" in user_msg

    def test_parse_failure_retries_then_raises(self):
        client = _mock_client_returning("not json at all")
        with pytest.raises(ValueError, match="parse"):
            decompose_target(client, "target", max_retries=3)
        # 3 retries = 3 calls total
        assert client.chat.completions.create.call_count == 3

    def test_wrong_skill_count_raises(self):
        bad = json.dumps({"skills": [{"name": "only one", "description": "x"}]})
        client = _mock_client_returning(bad)
        with pytest.raises(ValueError, match="expected 10 skills"):
            decompose_target(client, "target", max_retries=1)


class TestSkillsPersistence:
    def test_round_trip(self, tmp_path):
        skills = [
            Skill(name=f"S{i}", description=f"D{i}")
            for i in range(10)
        ]
        path = tmp_path / "skills.json"
        save_skills(str(path), skills, target_path="target.txt",
                    target_hash="abc123", model=GENERATOR_MODEL)
        loaded = load_skills(str(path))
        assert loaded is not None
        assert len(loaded) == 10
        assert loaded[0].name == "S0"

    def test_load_missing_returns_none(self, tmp_path):
        assert load_skills(str(tmp_path / "nope.json")) is None


from Stage1.taxonomy_generation import (
    GENERATE_PROMPT,
    _parse_problem,
    generate_for_skill,
)

from Stage1.taxonomy_generation import TEMPERATURE


class TestParseProblem:
    def test_extracts_between_tags(self):
        raw = "<problem>\nFind the value of \\(x\\). Put your final answer inside \\boxed{}.\n</problem>"
        assert _parse_problem(raw) == "Find the value of \\(x\\). Put your final answer inside \\boxed{}."

    def test_returns_empty_when_no_tags(self):
        assert _parse_problem("no tags here") == ""

    def test_strips_surrounding_whitespace(self):
        raw = "\n\n<problem>   hi   </problem>\n"
        assert _parse_problem(raw) == "hi"


class TestGenerateForSkill:
    def test_stops_at_n_target_keeps(self, monkeypatch):
        # Mock: generator always returns a valid problem; solver always agrees at 0.70.
        from Stage1 import taxonomy_generation as tg

        gen_calls = {"n": 0}

        def fake_gen_candidate(client, target, skill, _temperature=TEMPERATURE):
            gen_calls["n"] += 1
            return f"Find n. Put your final answer inside \\boxed{{}}. Candidate {gen_calls['n']}."

        def fake_solve(client, model, problem_text, n_samples, pool=None):
            # Return (agreement, majority, all_answers, all_solutions)
            return (0.70, "42", ["42"] * n_samples, ["reasoning"] * n_samples)

        monkeypatch.setattr(tg, "_generate_one_candidate", fake_gen_candidate)
        monkeypatch.setattr(tg, "solve_and_check_agreement", fake_solve)

        skill = Skill("s", "d")
        keeps, skips, stats = generate_for_skill(
            client=MagicMock(),
            target="target",
            skill=skill,
            n_target=3,
            n_samples=5,
            max_candidates=50,
            agree_low=0.60,
            agree_high=0.80,
        )
        assert len(keeps) == 3
        assert stats["n_passed"] == 3
        assert stats["n_attempted"] == 3  # every candidate was a keep
        assert stats["status"] == "ok"

    def test_caps_at_max_candidates(self, monkeypatch):
        from Stage1 import taxonomy_generation as tg

        def fake_gen(client, target, skill, _temperature=TEMPERATURE):
            return "Find n. \\boxed{}"

        def fake_solve(client, model, problem_text, n_samples, pool=None):
            # All candidates fail the agreement window (too high -> skip)
            return (0.95, "42", ["42"] * n_samples, ["r"] * n_samples)

        monkeypatch.setattr(tg, "_generate_one_candidate", fake_gen)
        monkeypatch.setattr(tg, "solve_and_check_agreement", fake_solve)

        skill = Skill("s", "d")
        keeps, skips, stats = generate_for_skill(
            client=MagicMock(),
            target="target",
            skill=skill,
            n_target=10,
            n_samples=5,
            max_candidates=7,
            agree_low=0.60,
            agree_high=0.80,
        )
        assert len(keeps) == 0
        assert stats["n_attempted"] == 7
        assert stats["status"] == "capped"

    def test_rejects_non_numeric(self, monkeypatch):
        from Stage1 import taxonomy_generation as tg

        def fake_gen(client, target, skill, _temperature=TEMPERATURE):
            return "Find n. \\boxed{}"

        def fake_solve(client, model, problem_text, n_samples, pool=None):
            return (0.70, "does not exist", ["does not exist"] * n_samples, ["r"] * n_samples)

        monkeypatch.setattr(tg, "_generate_one_candidate", fake_gen)
        monkeypatch.setattr(tg, "solve_and_check_agreement", fake_solve)

        skill = Skill("s", "d")
        keeps, skips, stats = generate_for_skill(
            client=MagicMock(),
            target="target",
            skill=skill,
            n_target=5,
            n_samples=5,
            max_candidates=3,
            agree_low=0.60,
            agree_high=0.80,
        )
        assert len(keeps) == 0
        assert stats["status"] == "capped"


class TestBuildTaxonomyDataset:
    def test_end_to_end_writes_expected_files(self, tmp_path, monkeypatch):
        from Stage1 import taxonomy_generation as tg

        skills_payload = json.dumps({
            "skills": [
                {"name": f"Skill {i}", "description": f"desc {i}"}
                for i in range(10)
            ]
        })

        def fake_decompose(client, target, *, n_skills=10, max_retries=3):
            data = json.loads(skills_payload)
            return [Skill(**e) for e in data["skills"]]

        def fake_generate_for_skill(*, client, target, skill, n_target, n_samples, max_candidates, agree_low, agree_high, solve_pool=None):
            keeps = [
                {
                    "skill": skill.name,
                    "problem": f"Problem {i} for {skill.name}. \\boxed{{}}",
                    "ground_truth_answer": str(i),
                    "agreement_rate": 0.70,
                    "all_answers": [str(i)] * n_samples,
                    "all_solutions": [f"reasoning {i}"] * n_samples,
                    "n_samples": n_samples,
                }
                for i in range(n_target)
            ]
            stats = {
                "name": skill.name,
                "n_target": n_target,
                "n_passed": len(keeps),
                "n_attempted": n_target,
                "status": "ok",
            }
            return keeps, [], stats

        monkeypatch.setattr(tg, "decompose_target", fake_decompose)
        monkeypatch.setattr(tg, "generate_for_skill", fake_generate_for_skill)
        monkeypatch.setattr(tg, "get_client", lambda: (MagicMock(), GENERATOR_MODEL))

        out_dir = tmp_path / "run1"
        skills_path = tmp_path / "skills.json"

        tg.build_taxonomy_dataset(
            target_text="TARGET PROBLEM BODY",
            target_path="data/target-problems/fake.txt",
            out_dir=str(out_dir),
            skills_path=str(skills_path),
            n_skills=10,
            problems_per_skill=3,
            max_candidates_per_skill=20,
            n_samples=5,
            agree_low=0.60,
            agree_high=0.80,
        )

        # Files exist
        assert (out_dir / "keeps.json").exists()
        assert (out_dir / "skips.json").exists()
        assert (out_dir / "per_skill_stats.json").exists()
        assert skills_path.exists()

        keeps = json.load(open(out_dir / "keeps.json"))
        assert keeps["n_problems"] == 30
        assert keeps["generator_model"] == GENERATOR_MODEL
        assert keeps["solve_model"] == GENERATOR_MODEL
        assert all("skill" in p for p in keeps["problems"])

        stats = json.load(open(out_dir / "per_skill_stats.json"))
        assert stats["total_passed"] == 30
        assert stats["total_target"] == 30
        assert len(stats["skills"]) == 10

    def test_reuses_cached_skills(self, tmp_path, monkeypatch):
        """If skills.json exists, decompose_target is NOT called."""
        from Stage1 import taxonomy_generation as tg

        # Pre-seed skills.json
        skills = [Skill(f"S{i}", f"d{i}") for i in range(10)]
        skills_path = tmp_path / "skills.json"
        tg.save_skills(str(skills_path), skills,
                       target_path="data/target-problems/fake.txt",
                       target_hash=tg.target_text_hash("TARGET"),
                       model=GENERATOR_MODEL)

        decompose_was_called = {"flag": False}

        def fake_decompose(*a, **kw):
            decompose_was_called["flag"] = True
            raise AssertionError("should not have been called")

        def fake_generate_for_skill(*, skill, n_target, **_kw):
            return ([], [], {"name": skill.name, "n_target": n_target,
                             "n_passed": 0, "n_attempted": 0, "status": "capped"})

        monkeypatch.setattr(tg, "decompose_target", fake_decompose)
        monkeypatch.setattr(tg, "generate_for_skill", fake_generate_for_skill)
        monkeypatch.setattr(tg, "get_client", lambda: (MagicMock(), GENERATOR_MODEL))

        tg.build_taxonomy_dataset(
            target_text="TARGET",
            target_path="data/target-problems/fake.txt",
            out_dir=str(tmp_path / "run1"),
            skills_path=str(skills_path),
            n_skills=10,
            problems_per_skill=1,
            max_candidates_per_skill=1,
            n_samples=1,
            agree_low=0.60,
            agree_high=0.80,
        )

        assert decompose_was_called["flag"] is False
