# Subproblem Dedupe — Design

## Motivation

The subproblem generation pipeline (Stage 1) produces near-duplicate problems.
Example observed in `runs/conics/stage1/…/keeps.json`: rows 33, 35, and 36 are
the "polynomial x^4−5 separable primes density" problem stated essentially
verbatim three times in one aggregate. Duplicates inflate the effective weight
of a single mode in GRPO training, reduce diversity, and waste LLM solve
compute (each candidate costs ~10 solve calls at 16k tokens).

Both stages already dedupe on exact-string equality:

- `Stage1/distinct_llm_prompting.py:660,669,750-752` — `seen_problems: set[str]`
  checked before dispatching a candidate to solve.
- `pipeline_stages/stage2_aggregate.py:49,72-75` — `seen: set[str]` during
  aggregation of all `keeps.json` files for an id.

These miss near-duplicates (trivial wording edits, whitespace, LaTeX spacing).
This spec upgrades both dedupe spots to a shared, smarter check.

## Scope

**In scope:**
- Exact and near-exact textual duplicate detection (case A: "near-duplicate
  wording, same numbers, same answer").
- Replacing the existing exact-string checks in Stage 1 and Stage 2.

**Explicitly deferred:**
- Same-task-different-numbers duplicates (case B from brainstorming).
- Semantic / concept-level duplicates (case C from brainstorming).
- Any use of LLMs or embedding models for similarity.
- Dedupe at Stage 3, 3c, or 4 — if an item survives 1+2 dedupe, downstream
  stages already see only unique problems.

## Approach

Two-check dedupe, deterministic, no external API calls:

1. **Normalized-text SHA-1 hash** — catches exact and near-exact duplicates
   (whitespace, case, trivial LaTeX spacing differences). O(1) per candidate.
2. **5-gram word-shingle Jaccard similarity ≥ 0.9** — catches trivial wording
   edits (e.g. "Consider the polynomial…" vs "Let me consider the polynomial…").
   O(n) per candidate against the running index.

Threshold `0.9` and shingle size `5` are module-level constants (not CLI
flags) — tune in code if needed.

**Tie-breaking:** when a duplicate is found, the candidate already in the
index wins; the new candidate is dropped. This preserves insertion order and
is stable.

**Dedupe key:** problem text only. Not `(problem, answer)`. Simpler; false
positives from genuinely-different problems with near-identical wording are
vanishingly unlikely in this domain.

## Components

### New: `pipeline_stages/dedupe.py`

Single module, no external deps beyond `re`, `hashlib`.

```python
JACCARD_THRESHOLD = 0.9
SHINGLE_SIZE = 5

def normalize_problem(text: str) -> str:
    """Lowercase, strip LaTeX spacing commands, collapse whitespace."""

def problem_hash(text: str) -> str:
    """SHA-1 of normalize_problem(text)."""

def shingles(text: str, k: int = SHINGLE_SIZE) -> frozenset[str]:
    """Word-level k-gram shingles over normalized text."""

def jaccard(a: frozenset, b: frozenset) -> float: ...

class DedupeIndex:
    def __init__(self, threshold: float = JACCARD_THRESHOLD): ...
    def add(self, problem_text: str) -> bool:
        """Return True if added (unique), False if duplicate of something
        already in the index."""
    @property
    def n_exact_dropped(self) -> int: ...
    @property
    def n_fuzzy_dropped(self) -> int: ...
    @property
    def n_kept(self) -> int: ...
```

Implementation notes:

- `add()` first checks the hash set (O(1)). On miss, computes shingles and
  compares against each existing shingle set (O(n) where n is current
  `n_kept`). Stage 1 and Stage 2 both operate on O(100) problems, so O(n²)
  total is fine.
- Empty text returns `False` from `add()` (treat as already-present so it's
  skipped).
- Problems with too few tokens to form any 5-gram fall back to hash-only.

### Stage 1 integration (`Stage1/distinct_llm_prompting.py`)

- Replace `seen_problems: set[str] = set()` at line 660 with
  `dedupe = DedupeIndex()` (plus an `if dedupe_enabled:` guard — see CLI
  flag below).
- Replace the pre-solve check at lines 750-752 with
  `if not dedupe.add(problem_text): return {"kind": "duplicate", ...}`.
- Remove the unreachable `seen_problems.add(entry.problem)` loop at 669 — it
  was pre-seeding the set from something; audit during implementation and
  either port to `DedupeIndex.add()` or delete if redundant.
- Add `--no-dedupe` flag (defaults to dedupe ON). When off, fall back to the
  existing exact-string `set` behavior so historical runs can be reproduced.

### Stage 2 integration (`pipeline_stages/stage2_aggregate.py`)

- Replace `seen: set[str] = set()` at line 49 with `dedupe = DedupeIndex()`.
- Replace the check at lines 72-75 with `if not dedupe.add(text): dup += 1;
  continue`.
- Add to the `summary` dict written to `aggregated_keeps.json`:
  ```json
  {
    "dedupe": {
      "n_kept": ...,
      "n_dropped_exact": ...,
      "n_dropped_fuzzy": ...
    }
  }
  ```
- Add `--no-dedupe` flag, same semantics as Stage 1.

### No changes to

- `run_pipeline.py` — no new subcommands, no new flags. Dedupe is a
  transparent improvement to Stage 1 and Stage 2, both of which already had
  (weaker) dedupe.
- Stages 3, 3c, 4, 5 — they see unique inputs by construction.

## Testing

Unit tests for `dedupe.py` in `tests/test_dedupe.py`:

- `normalize_problem` idempotent.
- Exact duplicate detected (returns `False` on second add).
- Whitespace-only difference detected as duplicate.
- LaTeX `\,` vs `\ ` normalized.
- High-Jaccard pair ("Consider the polynomial f(x)=x^4-5…" vs "Let us
  consider the polynomial f(x)=x^4-5…") detected as duplicate.
- Low-Jaccard pair (two genuinely different problems sharing few words) kept.
- Stats (`n_exact_dropped`, `n_fuzzy_dropped`, `n_kept`) correct across a
  mixed sequence.

Integration sanity-check: re-run Stage 2 on the existing
`runs/conics/stage1/*/keeps.json` and verify the new `aggregated_keeps.json`
has fewer problems than the exact-string version, with the polynomial
x^4−5 case collapsing to one copy.

## Logging

Stage 1 and Stage 2 each print one summary line at completion:

```
dedupe: kept 47, dropped 8 (exact=5, fuzzy=3)
```

Stage 2 additionally writes the `dedupe` block into `aggregated_keeps.json`.

## Rollout

- Default ON for both stages.
- `--no-dedupe` on both CLIs for ablation runs (useful for the paper — "here
  is what happens if you skip the dedupe step").
- No data migration needed. Existing `aggregated_keeps.json` files simply
  reflect the old exact-string dedupe; rerunning Stage 2 overwrites with the
  new behavior.

## Risk

- **False positives** (dropping legitimately-distinct problems): low in this
  domain — the generator temperature and prompt produce textually distinct
  wording for distinct problems. Jaccard threshold 0.9 requires near-total
  token overlap. If we see unexpected drops in practice, lower the threshold
  first before defaulting off.
- **Performance**: O(n²) per-stage, n ≤ a few hundred. Negligible vs. the
  LLM calls.
- **Backwards compatibility**: the `--no-dedupe` flag preserves the ability
  to run the old exact-string behavior for reproducibility.
