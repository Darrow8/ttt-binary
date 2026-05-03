# Task: Update Stage 3 calibration logic to use consensus-based answer agreement

We're changing how the difficulty calibration loop in Stage 3 of the pipeline works. The old version compared solver attempts against a generator-provided "expected answer." The new version uses agreement among independent solver attempts as the signal — we don't trust any single source of ground truth, only consensus among many attempts.

## What to change

In `pipeline/stage3_generate_subproblems.py`, replace the difficulty calibration logic with the rule below.

### New calibration procedure

For each generated subproblem candidate:

1. Send the problem text (no skill labels) to the solver model K=10 times at temperature 0.7. Collect 10 final answers.

2. Cluster the 10 answers by equivalence:
   - For numeric answers: exact equality after canonicalization (handle floats, fractions, scientific notation consistently).
   - For symbolic answers: canonical form comparison (use sympy `simplify` and `equals` for expressions, or string-normalize after parsing).
   - Implement this as a helper function `cluster_answers(answers: list) -> dict[canonical_form, count]`.

3. Compute:
   - `p1` = size of largest cluster / 10
   - `p2` = size of second-largest cluster / 10 (0 if only one cluster exists)
   - `consensus_answer` = the canonical form of the largest cluster

4. Apply the decision rule:

   ```
   if p1 in [0.4, 0.6] and p2 < 0.2:
       ACCEPT — store consensus_answer as the expected answer
   elif p1 in [0.4, 0.6] and p2 >= 0.2:
       REJECT_AMBIGUOUS — regenerate with feedback "your problem produced
       two competing answers (X and Y, each appearing roughly equally).
       Make sure the problem has one well-defined answer."
   elif p1 > 0.6:
       REJECT_TOO_EASY — regenerate harder
   elif p1 < 0.4:
       REJECT_TOO_HARD_OR_AMBIGUOUS — regenerate easier or simpler
   ```

5. Cap regeneration attempts at 5 per skill triple. If still no acceptance, mark the triple as failed and log the rejection reasons across all 5 attempts.

### Configuration

Add these to the config:

```yaml
K_calibrate: 10
difficulty_band: [0.4, 0.6]
ambiguity_threshold: 0.2  # max allowed second-cluster fraction
max_regen_attempts: 5
```

Make `ambiguity_threshold` configurable since we may need to tune it (start at 0.2, may tighten to 0.15 or loosen to 0.3 based on rejection rates).

### Data model changes

The accepted subproblem record changes shape:

```json
{
  "problem_text": "...",
  "skills_used": ["a", "b", "c"],
  "skill_chain_rationale": "...",
  "consensus_answer": "the answer the largest cluster agreed on",
  "generator_proposed_answer": "what the generator originally claimed",
  "p1": 0.5,
  "p2": 0.1,
  "regeneration_attempts": 2,
  "all_answer_clusters": {"42": 5, "43": 1, "100": 1, "...": "..."}
}
```

Note: `consensus_answer` replaces `expected_answer` as the training reward target. Keep `generator_proposed_answer` as a separate field for diagnostic purposes — we want to log cases where the generator was wrong about its own problem and the solver consensus corrected it.

### Logging requirements

For every triple processed, log:
- Outcome (accepted / failed)
- Per-iteration rejection reason (too_easy / too_hard / ambiguous / accepted)
- Final p1, p2 values for the accepted version (if any)
- Whether `consensus_answer == generator_proposed_answer` (boolean)
- The full cluster distribution

Aggregate stats to surface at end of run:
- % of triples accepted
- % of triples rejected at each cap-out reason
- % of accepted problems where generator's proposed answer disagreed with consensus
- Distribution of p1 across accepted problems (should center around 0.5)

### Things NOT to change

- Stage 1 (skill generation) is unchanged.
- Stage 2 (combination enumeration) is unchanged.
- The generator prompt is unchanged in structure — it can still propose an expected answer, but that answer is no longer authoritative.
- Stage 4 (solving for training data) should now use `consensus_answer` as the verification target, not `generator_proposed_answer`.

## Edge cases to handle

1. **All 10 attempts produce different answers.** p1 = 0.1, p2 = 0.1. Falls under p1 < 0.4, regenerate. Don't crash on the "no clusters of size > 1" case.

2. **All 10 attempts agree.** p1 = 1.0, p2 = 0. Way too easy, regenerate harder.

3. **Solver model returns malformed output (no parseable answer).** Treat as a separate cluster of "unparseable" or drop from the cluster computation. If more than 3/10 attempts are unparseable, the problem is probably ill-posed — treat as too hard / regenerate.

4. **Numeric answers that are equal but formatted differently** (e.g., `1/2` vs `0.5` vs `0.50`). The canonicalization step in `cluster_answers` must handle this. Use sympy's `Rational` or `Float` parsing where possible, falling back to string normalization.

5. **Symbolic answers where two attempts produce equivalent expressions in different forms** (e.g., `x^2 - 1` vs `(x-1)(x+1)`). Use `sympy.simplify(a - b) == 0` for equivalence. This is more expensive than string matching, so cache.

## Tests to write

In `tests/test_stage3_calibration.py`:

- `test_accept_clean_majority`: clusters {A: 5, B: 1, C: 1, D: 1, E: 2} → accept (p1=0.5, p2=0.2 — wait, that's borderline; use {A: 5, B: 1, C: 1, D: 1, E: 1, F: 1} instead → accept).
- `test_reject_ambiguous`: clusters {A: 4, B: 4, C: 2} → reject_ambiguous.
- `test_reject_too_easy`: clusters {A: 8, B: 2} → reject_too_easy.
- `test_reject_too_hard`: clusters {A: 2, B: 2, C: 2, D: 2, E: 2} → reject_too_hard_or_ambiguous.
- `test_borderline_50_40_10`: clusters {A: 5, B: 4, C: 1} → reject_ambiguous (p1 in band, p2 = 0.4 ≥ 0.2).
- `test_canonical_clustering_numeric`: ["0.5", "1/2", "0.50"] should cluster into one group.
- `test_canonical_clustering_symbolic`: ["x^2 - 1", "(x-1)*(x+1)"] should cluster into one group.
- `test_unparseable_handling`: 4 unparseable + 6 valid (5 of one, 1 of another) should reject (too noisy).
- `test_regen_cap`: after 5 failed regenerations, return failure with full log.

## Suggested implementation order

1. Write `cluster_answers` first with full test coverage. This is the trickiest piece.
2. Wire up the decision rule as a pure function `decide(clusters: dict) -> Decision`. Test in isolation.
3. Wire both into the existing Stage 3 loop, replacing the old expected-answer comparison.
4. Update Stage 4 to use `consensus_answer` from the new record format.
5. Run the pipeline on a small set of triples (say 10 of 120) to sanity-check rejection rates before doing the full run.

Don't run the full 120-triple pipeline until the small-set test looks reasonable — rejection rates outside 30-70% probably mean a bug in the calibration, not a problem with the threshold.
