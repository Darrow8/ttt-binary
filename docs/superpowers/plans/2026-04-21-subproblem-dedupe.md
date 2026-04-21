# Subproblem Dedupe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the existing exact-string dedupe in Stage 1 and Stage 2 to a normalized-text-hash + 5-gram Jaccard check so near-duplicate subproblems (same problem restated with trivial wording/whitespace/LaTeX-spacing edits) are dropped before training.

**Architecture:** A single new module `pipeline_stages/dedupe.py` provides `DedupeIndex`, which exposes `add(text) -> bool` (returns `False` when the text duplicates something already in the index). Stage 1 and Stage 2 each construct their own `DedupeIndex` and replace the `set[str]` checks that already exist at known line numbers. No new pipeline stage, no changes to `run_pipeline.py`.

**Tech Stack:** Python stdlib only (`re`, `hashlib`). No new deps. `pytest` for unit tests (check if already used; add to `requirements.txt` if missing).

**Reference:** spec at `docs/superpowers/specs/2026-04-21-subproblem-dedupe-design.md`.

---

## File Structure

**Create:**
- `pipeline_stages/dedupe.py` — `DedupeIndex`, `normalize_problem`, `problem_hash`, `shingles`, `jaccard`. ~80 lines.
- `tests/__init__.py` — empty, makes `tests/` a package.
- `tests/test_dedupe.py` — unit tests for the module. ~100 lines.

**Modify:**
- `pipeline_stages/stage2_aggregate.py:49,72-75` plus add `--no-dedupe` flag.
- `Stage1/distinct_llm_prompting.py:660,669,750-752` (the resume-seed at 669 gets ported, not deleted), plus `use_dedupe` kwarg threaded through `build_dataset()` (line 618) and `run()` (line 918), plus `--no-dedupe` CLI flag in `main()`.

---

### Task 1: Create `pipeline_stages/dedupe.py` with tests

**Files:**
- Create: `pipeline_stages/dedupe.py`
- Create: `tests/__init__.py`
- Create: `tests/test_dedupe.py`

- [ ] **Step 1.1: Verify pytest is available**

Run: `python -c "import pytest; print(pytest.__version__)"`

If it fails, add `pytest>=7` to `requirements.txt` and run `pip install pytest`. Do not skip — the whole plan is TDD-driven.

- [ ] **Step 1.2: Create empty `tests/__init__.py`**

```python
```

(Literally empty file. Makes `tests/` importable as a package.)

- [ ] **Step 1.3: Write the failing tests**

Create `tests/test_dedupe.py`:

```python
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
```

- [ ] **Step 1.4: Run tests to confirm they all fail**

Run: `pytest tests/test_dedupe.py -v`

Expected: All tests fail with `ModuleNotFoundError: No module named 'pipeline_stages.dedupe'` (or similar). If any test passes, something is wrong — investigate before continuing.

- [ ] **Step 1.5: Create `pipeline_stages/dedupe.py`**

```python
"""Deduplication of subproblems by normalized-text hash + k-gram Jaccard.

Used by Stage 1 (pre-solve, intra-run) and Stage 2 (cross-run aggregate).
Catches exact and near-exact duplicates. Does NOT catch semantic/conceptual
duplicates — see 2026-04-21-subproblem-dedupe-design.md for scope.
"""

from __future__ import annotations

import hashlib
import re

JACCARD_THRESHOLD = 0.9
SHINGLE_SIZE = 5

_LATEX_SPACING = re.compile(r"\\[,;\s]|\\quad|\\qquad")
_WS = re.compile(r"\s+")
_WORD = re.compile(r"\w+")


def normalize_problem(text: str) -> str:
    """Lowercase, strip LaTeX spacing commands, collapse whitespace."""
    t = text.lower()
    t = _LATEX_SPACING.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    return t


def problem_hash(text: str) -> str:
    return hashlib.sha1(normalize_problem(text).encode("utf-8")).hexdigest()


def shingles(text: str, k: int = SHINGLE_SIZE) -> frozenset[str]:
    tokens = _WORD.findall(normalize_problem(text))
    if len(tokens) < k:
        return frozenset()
    return frozenset(
        " ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)
    )


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class DedupeIndex:
    """Stateful index that reports whether a new problem is a duplicate.

    Not thread-safe on its own; callers that use it from multiple threads
    must guard `add()` with a lock (Stage 1 already holds `seen_lock`).
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

        sh = shingles(problem_text)
        if sh:
            for existing in self._shingle_sets:
                if existing and jaccard(sh, existing) >= self._threshold:
                    self.n_fuzzy_dropped += 1
                    return False

        self._hashes.add(h)
        self._shingle_sets.append(sh)
        self.n_kept += 1
        return True
```

- [ ] **Step 1.6: Run tests to verify they pass**

Run: `pytest tests/test_dedupe.py -v`

Expected: all tests pass. If any fail, fix the implementation (not the tests) unless the test itself is wrong.

- [ ] **Step 1.7: Commit**

```bash
git add pipeline_stages/dedupe.py tests/__init__.py tests/test_dedupe.py
git commit -m "add dedupe module: normalized-hash + 5-gram Jaccard

New DedupeIndex wraps hash-set exact check plus Jaccard near-duplicate
check. Used next by Stage 1 and Stage 2 to replace their existing
exact-string dedupe sets. Design: docs/superpowers/specs/2026-04-21-subproblem-dedupe-design.md"
```

---

### Task 2: Wire `DedupeIndex` into Stage 2

**Files:**
- Modify: `pipeline_stages/stage2_aggregate.py`

- [ ] **Step 2.1: Re-read the current state**

Run: `sed -n '40,100p' pipeline_stages/stage2_aggregate.py`

Confirm the `seen: set[str] = set()` at line 49 and the `if text in seen:` block at 72-75 are unchanged from the spec's reference state.

- [ ] **Step 2.2: Write an integration test for Stage 2 dedupe**

Add to `tests/test_dedupe.py` at the bottom:

```python
class TestStage2Integration:
    def test_aggregate_drops_fuzzy_duplicates(self, tmp_path):
        """Stage 2 aggregate should drop near-duplicate problems across runs."""
        import json
        import sys

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

        # Point REPO_ROOT at tmp_path for this test.
        import importlib
        from pipeline_stages import stage2_aggregate
        monkeypatched = tmp_path
        stage2_aggregate.REPO_ROOT = monkeypatched

        summary = stage2_aggregate.aggregate_one("testid", include_skips=False)
        assert summary["n_problems"] == 2  # dup_a/dup_b collapsed
        assert summary["dedupe"]["n_kept"] == 2
        assert summary["dedupe"]["n_dropped_fuzzy"] + summary["dedupe"]["n_dropped_exact"] == 1
```

- [ ] **Step 2.3: Run the integration test to verify it fails**

Run: `pytest tests/test_dedupe.py::TestStage2Integration -v`

Expected: fails, likely on `summary["dedupe"]` missing or `n_problems == 3` instead of `2` (current exact-string dedupe doesn't catch fuzzy dups).

- [ ] **Step 2.4: Modify `pipeline_stages/stage2_aggregate.py`**

Apply three edits:

**(a) Add import** after line 20:

```python
from pipeline_stages.dedupe import DedupeIndex
```

**(b) Replace lines 49-85** (the `seen: set[str] = set()` loop) with:

```python
    dedupe_enabled = not no_dedupe
    dedupe = DedupeIndex() if dedupe_enabled else None
    seen_fallback: set[str] = set()
    aggregated: list[dict] = []
    per_run_counts: list[dict] = []
    source_problem: str | None = None

    for kp in keeps_files:
        try:
            with open(kp) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  [warn] skipping unreadable {kp}: {e}")
            continue

        if source_problem is None:
            source_problem = data.get("source_problem")

        run_kept = data.get("problems", [])
        added = 0
        dup = 0
        for p in run_kept:
            text = p.get("problem", "")
            if not text:
                continue
            if dedupe_enabled:
                if not dedupe.add(text):
                    dup += 1
                    continue
            else:
                if text in seen_fallback:
                    dup += 1
                    continue
                seen_fallback.add(text)
            aggregated.append(p)
            added += 1

        per_run_counts.append({
            "run": kp.parent.name,
            "kept_in_run": len(run_kept),
            "added_after_dedup": added,
            "dropped_as_duplicate": dup,
        })
        print(f"  {kp.parent.name}: {len(run_kept)} kept, +{added} new, {dup} dup")
```

**(c) Replace lines 87-96** (the `summary = {...}` dict and `_save_atomic`) with:

```python
    out_path = runs_root / "aggregated_keeps.json"
    summary = {
        "id": problem_id,
        "source_problem": source_problem,
        "n_runs": len(per_run_counts),
        "n_problems": len(aggregated),
        "per_run": per_run_counts,
        "problems": aggregated,
    }
    if dedupe_enabled:
        summary["dedupe"] = {
            "n_kept": dedupe.n_kept,
            "n_dropped_exact": dedupe.n_exact_dropped,
            "n_dropped_fuzzy": dedupe.n_fuzzy_dropped,
        }
        print(
            f"  dedupe: kept {dedupe.n_kept}, "
            f"dropped {dedupe.n_exact_dropped + dedupe.n_fuzzy_dropped} "
            f"(exact={dedupe.n_exact_dropped}, fuzzy={dedupe.n_fuzzy_dropped})"
        )
    _save_atomic(out_path, summary)
    print(f"\nWrote {out_path}  ({len(aggregated)} unique problems from {len(keeps_files)} runs)")
```

**(d) Change `aggregate_one` signature** (line 33) to accept `no_dedupe`:

```python
def aggregate_one(problem_id: str, *, include_skips: bool = False, no_dedupe: bool = False) -> dict:
```

**(e) Add `--no-dedupe` CLI arg in `main()`** (around line 128-133):

```python
    parser.add_argument("--no-dedupe", action="store_true",
                        help="Disable near-duplicate dedup (use exact-string only, for ablation)")
    args = parser.parse_args()
    aggregate_one(args.id, include_skips=args.include_skips, no_dedupe=args.no_dedupe)
```

- [ ] **Step 2.5: Run integration test to verify it passes**

Run: `pytest tests/test_dedupe.py -v`

Expected: all tests pass including `TestStage2Integration::test_aggregate_drops_fuzzy_duplicates`.

- [ ] **Step 2.6: Smoke test on real data**

Run: `python pipeline_stages/stage2_aggregate.py --id conics`

Compare the new `runs/conics/aggregated_keeps.json` with a backup of the pre-change version (make the backup first!). The new one should have `dedupe` block and should have the same or fewer `n_problems` (any decrease = near-duplicates caught that the old exact-string check missed).

Back up first:
```bash
cp runs/conics/aggregated_keeps.json runs/conics/aggregated_keeps.json.pre-dedupe
```

Then run, then diff-check:
```bash
python -c "
import json
old = json.load(open('runs/conics/aggregated_keeps.json.pre-dedupe'))
new = json.load(open('runs/conics/aggregated_keeps.json'))
print(f'old n_problems: {old[\"n_problems\"]}')
print(f'new n_problems: {new[\"n_problems\"]}')
print(f'new dedupe block: {new.get(\"dedupe\")}')
"
```

If the numbers look reasonable (old ≥ new, new has a `dedupe` block), continue. If n_problems jumped up or weirdness appeared, stop and investigate.

- [ ] **Step 2.7: Commit**

```bash
git add pipeline_stages/stage2_aggregate.py tests/test_dedupe.py
git commit -m "stage2: use DedupeIndex for near-duplicate aggregation

Replaces exact-string seen-set with DedupeIndex (normalized-hash +
Jaccard). Adds dedupe stats to aggregated_keeps.json and a --no-dedupe
flag for ablation runs."
```

---

### Task 3: Wire `DedupeIndex` into Stage 1

**Files:**
- Modify: `Stage1/distinct_llm_prompting.py`

- [ ] **Step 3.1: Re-read the current state**

Run: `sed -n '615,760p' Stage1/distinct_llm_prompting.py`

Confirm lines 660 (`seen_problems: set[str] = set()`), 669 (resume pre-seed `seen_problems.add(entry.problem)`), and 750-752 (the pre-solve check) match the spec reference state.

- [ ] **Step 3.2: Add `use_dedupe` param to `build_dataset` signature**

Edit line 618-633 (signature). Add `use_dedupe: bool = True` before the closing `) -> Dataset:`:

```python
def build_dataset(
    client: OpenAI,
    model: str,
    hard_problem: str,
    n_target: int = 100,
    n_samples_per_problem: int = 10,
    target_agreement_low: float = 0.60,
    target_agreement_high: float = 0.80,
    output_path: str | None = None,
    max_workers: int = 16,
    gen_workers: int = 8,
    failed_solutions: list[str] | None = None,
    solve_client: OpenAI | None = None,
    solve_model: str | None = None,
    quality_threshold: int | None = None,
    use_dedupe: bool = True,
) -> Dataset:
```

- [ ] **Step 3.3: Replace the dedupe state init and resume pre-seed**

Find line 660:

```python
    seen_problems: set[str] = set()
```

Replace with:

```python
    from pipeline_stages.dedupe import DedupeIndex
    dedupe = DedupeIndex() if use_dedupe else None
    seen_problems: set[str] = set()  # used only when use_dedupe=False
```

Find lines 666-669 (resume pre-seed loop):

```python
            for p in existing.get("problems", []):
                entry = GeneratedProblem(**p)
                dataset.problems.append(entry)
                seen_problems.add(entry.problem)
```

Replace with:

```python
            for p in existing.get("problems", []):
                entry = GeneratedProblem(**p)
                dataset.problems.append(entry)
                if use_dedupe:
                    dedupe.add(entry.problem)
                else:
                    seen_problems.add(entry.problem)
```

- [ ] **Step 3.4: Replace the pre-solve dedupe check**

Find lines 748-752:

```python
        problem_text = candidates[0]["problem"]
        with seen_lock:
            if problem_text in seen_problems:
                return {"kind": "duplicate", "candidate_num": cn, "gen_time": gen_time}
            seen_problems.add(problem_text)
```

Replace with:

```python
        problem_text = candidates[0]["problem"]
        with seen_lock:
            if use_dedupe:
                if not dedupe.add(problem_text):
                    return {"kind": "duplicate", "candidate_num": cn, "gen_time": gen_time}
            else:
                if problem_text in seen_problems:
                    return {"kind": "duplicate", "candidate_num": cn, "gen_time": gen_time}
                seen_problems.add(problem_text)
```

- [ ] **Step 3.5: Add dedupe summary print at end of `build_dataset`**

Find where `build_dataset` returns `dataset` (after the gen/eval loop completes — search for `_flush()` near the end of the function or `return dataset`). Immediately before `return dataset`, add:

```python
    if use_dedupe and dedupe is not None:
        print(
            f"  dedupe: kept {dedupe.n_kept}, "
            f"dropped {dedupe.n_exact_dropped + dedupe.n_fuzzy_dropped} "
            f"(exact={dedupe.n_exact_dropped}, fuzzy={dedupe.n_fuzzy_dropped})"
        )
```

- [ ] **Step 3.6: Thread `use_dedupe` through `run()`**

Edit `run()` signature at line 918-934. Add `use_dedupe: bool = True` before closing `) -> Dataset:`:

```python
def run(
    problem: str,
    *,
    n_problems: int = 100,
    n_samples: int = 10,
    agree_low: float = 0.60,
    agree_high: float = 0.80,
    output: str | None = None,
    model: str | None = None,
    max_workers: int = 16,
    gen_workers: int = 8,
    failed_solutions: list[str] | None = None,
    use_tinker: bool = False,
    tinker_checkpoint: str | None = None,
    tinker_checkpoint_step: int = 50,
    quality_threshold: int | None = None,
    use_dedupe: bool = True,
) -> Dataset:
```

Then, inside `run()`, find the call to `build_dataset(...)` (search for `build_dataset(` inside `run`). Add `use_dedupe=use_dedupe,` to the kwargs list.

- [ ] **Step 3.7: Add CLI flag and plumb it**

In `main()` at around line 1102-1111 (near the `--output` arg), add a new argument before `args = parser.parse_args()`:

```python
    parser.add_argument("--no-dedupe", action="store_true",
                        help="Disable near-duplicate dedup (use exact-string only, for ablation)")
```

Then at the `run(...)` call at line 1142-1155, add `use_dedupe=not args.no_dedupe,` as a kwarg.

- [ ] **Step 3.8: Verify module still imports**

Run: `python -c "from Stage1 import distinct_llm_prompting"`

Expected: no error. If ImportError on `pipeline_stages.dedupe`, check the `sys.path`. If `Stage1/distinct_llm_prompting.py` is run directly (not as a module), the existing `_REPO_ROOT` logic near the top should already add the repo root to the path — confirm this works.

- [ ] **Step 3.9: Verify CLI still parses**

Run: `python Stage1/distinct_llm_prompting.py --help`

Expected: help text includes `--no-dedupe`. No stack trace.

- [ ] **Step 3.10: Run existing tests to confirm nothing regressed**

Run: `pytest tests/ -v`

Expected: all tests from Task 1 and Task 2 still pass.

- [ ] **Step 3.11: Commit**

```bash
git add Stage1/distinct_llm_prompting.py
git commit -m "stage1: use DedupeIndex for pre-solve candidate dedup

Replaces exact-string seen_problems set with DedupeIndex in build_dataset.
Threads use_dedupe through run() and adds a --no-dedupe CLI flag for
ablation. Resume pre-seeding now also goes through DedupeIndex so
resumed runs get the upgraded near-duplicate check."
```

---

### Task 4: End-to-end sanity check

**Files:** none modified.

- [ ] **Step 4.1: Full test suite**

Run: `pytest tests/ -v`

Expected: every test green.

- [ ] **Step 4.2: Dry-run Stage 1 (small)**

Run (only if you have the API keys locally):
```bash
python Stage1/distinct_llm_prompting.py \
    --problem-path data/target-problems/conics.txt \
    --runs-subdir dedup-smoke \
    --n-problems 3 \
    --n-samples 2 \
    --gen-workers 2 \
    --max-workers 4
```

Expected: summary line ending with something like `dedupe: kept N, dropped M (exact=E, fuzzy=F)`. No crashes. Output dir `runs/dedup-smoke/stage1/<ts>/keeps.json` exists.

If you don't have API access here, skip this step and rely on the integration test at Step 2.3.

- [ ] **Step 4.3: Dry-run Stage 2 on the smoke-run output**

Run:
```bash
python pipeline_stages/stage2_aggregate.py --id dedup-smoke
```

Expected: `runs/dedup-smoke/aggregated_keeps.json` exists and contains a top-level `dedupe` block.

- [ ] **Step 4.4: Final commit (if any cleanup)**

If everything passed cleanly, no commit needed — Tasks 1-3 are already committed individually.

---

## Self-Review Notes

**Spec coverage:**
- Scope (exact + near-exact text only) → Task 1 enforces via hash + Jaccard only, no LLM/embedding.
- `pipeline_stages/dedupe.py` module surface (normalize, hash, shingles, jaccard, DedupeIndex) → Task 1 Step 1.5.
- Stage 1 pre-solve integration → Task 3.
- Stage 2 aggregate integration → Task 2.
- No changes to `run_pipeline.py` → confirmed, no tasks touch it.
- `--no-dedupe` flag → Task 2 Step 2.4e, Task 3 Step 3.7.
- Per-stage summary log line → Task 2 Step 2.4c, Task 3 Step 3.5.
- `dedupe` block in `aggregated_keeps.json` → Task 2 Step 2.4c.
- Unit tests → Task 1 Step 1.3.
- Integration sanity-check on `runs/conics/*/keeps.json` → Task 2 Step 2.6.

**Placeholder scan:** none left. All code steps contain actual code.

**Type / name consistency:**
- `DedupeIndex.add` returns `bool` everywhere.
- `n_kept`, `n_exact_dropped`, `n_fuzzy_dropped` used consistently in tests, module, Stage 1, Stage 2.
- `use_dedupe` kwarg consistent across `build_dataset`, `run`, CLI plumbing.
- Stage 2 `aggregate_one` keyword is `no_dedupe` (matches the CLI flag and inverts only at call site) — kept that name deliberately to mirror `--no-dedupe`.
