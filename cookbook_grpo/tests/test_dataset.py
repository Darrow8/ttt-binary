"""Tests for dataset loading."""

import json
import tempfile
from pathlib import Path

import pytest

from cookbook_grpo.dataset import load_subproblems, SubproblemRecord


class TestLoadSubproblems:
    def _write_jsonl(self, rows: list[dict]) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for row in rows:
            f.write(json.dumps(row) + "\n")
        f.close()
        return f.name

    def test_valid_with_prompt_reference(self):
        path = self._write_jsonl([
            {"prompt": "What is 2+2?", "reference": "4"},
            {"prompt": "What is 3+3?", "reference": "6"},
        ])
        records = load_subproblems(path)
        assert len(records) == 2
        assert records[0].problem == "What is 2+2?"
        assert records[0].answer == "4"

    def test_valid_with_problem_answer(self):
        path = self._write_jsonl([
            {"problem": "What is 2+2?", "answer": "4"},
        ])
        records = load_subproblems(path)
        assert len(records) == 1
        assert records[0].problem == "What is 2+2?"
        assert records[0].answer == "4"

    def test_missing_prompt(self):
        path = self._write_jsonl([
            {"reference": "4"},
        ])
        with pytest.raises(ValueError, match="Missing 'prompt' or 'problem'"):
            load_subproblems(path)

    def test_missing_answer(self):
        path = self._write_jsonl([
            {"prompt": "What is 2+2?"},
        ])
        with pytest.raises(ValueError, match="Missing 'reference' or 'answer'"):
            load_subproblems(path)

    def test_malformed_json(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        f.write("not valid json\n")
        f.close()
        with pytest.raises(ValueError, match="Malformed JSON"):
            load_subproblems(f.name)

    def test_empty_lines_skipped(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        f.write('{"prompt": "Q1", "reference": "A1"}\n')
        f.write("\n")
        f.write('{"prompt": "Q2", "reference": "A2"}\n')
        f.close()
        records = load_subproblems(f.name)
        assert len(records) == 2

    def test_metadata_preserved(self):
        path = self._write_jsonl([
            {
                "prompt": "Q",
                "reference": "A",
                "id": "test_1",
                "agreement_rate": 0.7,
                "metadata": {"source": "test"},
            }
        ])
        records = load_subproblems(path)
        assert records[0].id == "test_1"
        assert records[0].agreement_rate == 0.7
        assert records[0].metadata == {"source": "test"}

    def test_numeric_answer_converted_to_string(self):
        path = self._write_jsonl([
            {"prompt": "Q", "answer": 42},
        ])
        records = load_subproblems(path)
        assert records[0].answer == "42"
