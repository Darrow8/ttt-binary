# TTT-Binary Pipeline: Skill-Chained Subproblem Generation (v1)

## Goal

Given a hard math problem (e.g., a FrontierMath Tier 3 problem like the 3264 conics problem), generate a training corpus of subproblems that, when used for verifiable-reward RL, teaches the model to compose skills in ways that transfer back to the original problem and to held-out problems of similar difficulty.

The pipeline must work without access to the gold answer or gold solution of the target problem. Only the problem statement is allowed as input. This constraint exists because the method needs to generalize to open problems where no answer is available.

## Pipeline overview

```
problem statement
      ↓
[1] generate 10 candidate skills
      ↓
[2] enumerate all C(10, 3) = 120 unordered 3-skill combinations
      ↓
[3] for each combination, generate one subproblem at the 40-60% difficulty band
      ↓
[4] solve subproblems with strong model → training data
      ↓
[5] GRPO/DAPO training on target model
      ↓
[6] inference on original problem + held-out problems
```

## Design choices for v1

- **X = 10 skills.** Fixed. Ablation over X comes in v2.
- **M = 3 skills per chain.** Fixed. No mixture distribution.
- **Exhaustive enumeration.** Every C(10, 3) = 120 combination gets exactly one subproblem. No random sampling — we cover the full combinatorial space at this M.
- **Difficulty target: 40-60% solve rate.** Each subproblem should be hard enough that a strong model gets it right roughly half the time. This is the band where RL signal is densest (advantages are non-trivial, rewards aren't sparse).

## Stage 1: Skill generation

**Input:** problem statement (string).

**Output:** list of 10 skills. Each skill is a structured object:
```json
{
  "name": "short skill name",
  "description": "1-2 sentence description of what the skill does",
  "preconditions": "what must be true to apply this skill",
  "postconditions": "what is true after applying this skill",
  "example": "minimal worked example of the skill in isolation"
}
```

**Implementation:** prompt a strong generator model (Claude/GPT-5) to produce 10 skills relevant to solving the problem. The generator sees the problem statement but NOT any gold solution. Skills should be primitive enough to be reusable across problems in the same domain.

## Stage 2: Enumerate combinations

**Output:** list of 120 unordered 3-skill combinations, i.e. all subsets of size 3 from the 10 skills.

```python
from itertools import combinations
all_chains = list(combinations(skills, 3))  # 120 tuples
```

For each combination, also decide on a chain *order* — the dependency direction in which the three skills will be applied. Default: let the subproblem generator decide the order based on what makes sense for that triple (some orderings may be infeasible).

## Stage 3: Subproblem generation with consensus difficulty filter

**Goal:** for each of the 120 combinations, produce exactly one subproblem whose difficulty falls in the 40-60% solve-rate band.

**Per-combination loop:**

1. **Generate.** Prompt the generator to produce a subproblem whose solution chains the 3 skills in dependency order. The output of skill_i must be an input to skill_{i+1}. The subproblem must have a single verifiable final answer.

2. **Calibrate difficulty.** Sample K=10 solution attempts from a strong model at moderate temperature (~0.7). Check each attempt's final answer against the generator's claimed expected answer. Compute consensus solve rate = (correct attempts) / K.

3. **Accept or regenerate.**
   - If solve rate ∈ [0.4, 0.6]: accept this subproblem.
   - If solve rate > 0.6: subproblem is too easy. Regenerate with instruction to make it harder (deeper composition, less hint-y framing, harder intermediate values).
   - If solve rate < 0.4: subproblem is too hard or possibly ill-posed. Regenerate with instruction to make it more tractable.
   - Cap regeneration attempts at 5 per combination. If still no acceptable subproblem after 5 tries, log the combination as failed and move on.

4. **Record.** Save the accepted subproblem with its skill triple, ordering, expected answer, and observed solve rate.

**Output:** ~120 subproblems (some combinations may fail; that's fine, log them).

```json
{
  "problem_text": "the subproblem as it will be shown to the trained model",
  "skills_used": ["skill_a", "skill_b", "skill_c"],
  "skill_chain_rationale": "how the three skills compose",
  "expected_answer": "verifiable final answer",
  "consensus_solve_rate": 0.5,
  "regeneration_attempts": 2
}
```

**Critical:** the `problem_text` must NOT mention skill names or list the skills. The trained model has to recognize which skills apply on its own.

**Why consensus, not single-shot?** Solve rate from a single attempt is noisy. K=10 gives you ±15% accuracy on the rate, which is enough to discriminate the 40-60% band from "always solves" or "never solves" without burning excessive compute. If compute is tight, K=5 is acceptable but the band gets fuzzier.

## Stage 4: Solve subproblems for training data

**Purpose:** generate solution traces (chain-of-thought + final answer) for the accepted subproblems.

**Implementation:** for each accepted subproblem, generate K_solve=16 solution attempts with a strong model. Verify final answer against expected. Keep correct solutions as training data. Discard incorrect ones.

This gives roughly (120 subproblems) × (16 attempts) × (~50% correct) ≈ ~960 training traces. If that's too few, bump K_solve.

Note: the K=10 attempts from Stage 3 can be reused here — no need to regenerate.

## Stage 5: Training

Standard GRPO or DAPO on the target 120B model. Reward = verifiable correctness on final answer. Model sees only `problem_text`. Use existing TTT-Binary training infrastructure.

## Stage 6: Evaluation

1. **Target problem** (e.g., conics). Solve rate over many samples.
2. **Held-out FrontierMath Tier 3 problems.** Pick 3-5 problems not used to generate the skills. Run Stages 1-5 independently for each, then evaluate. Report mean solve rate. **This is the key generalization metric.**
3. **Baseline.** Same target model without TTT-Binary training, on the same problems.

## Configuration

```yaml
# v1 defaults
X: 10                     # skill pool size
M: 3                      # chain length (fixed)
enumerate_all: true       # exhaustive over C(X, M)
difficulty_band: [0.4, 0.6]
K_calibrate: 10           # attempts for difficulty calibration
K_solve: 16               # attempts for training data generation
max_regen_attempts: 5     # per combination
generator_model: claude-opus-4-7   # or GPT-5
critic_model: gpt-5                # different family from generator
target_model: <your-120b>
```

## What to log

- The 10 skills (Stage 1)
- For each of 120 combinations: accepted/failed, regeneration attempts, final solve rate
- Distribution of consensus solve rates across accepted subproblems (should be tight in [0.4, 0.6])
- Stage 4 correct-solution yield per subproblem
- Final eval solve rates on target + held-out problems

## Things the pipeline must NOT do

- Use the gold answer or gold solution of any target problem at any stage before evaluation.
- Provide the skill list to the model being trained at any stage.
- Hardcode skills for specific problems.
- Validate skill coverage by checking against a gold solution. The held-out transfer eval in Stage 6 is the validation.

## v2 extensions (don't implement yet)

After v1 works end-to-end:
- Ablate X ∈ {10, 20, 50}
- Mixed M distribution {2: 0.30, 3: 0.50, 4: 0.20}
- Skill self-consistency check
- Repeat-allowed chains (same skill twice in a chain)

## Open questions to resolve while implementing

- Exact prompt for skill generation (Stage 1). Iterate on a few example problems.
- Exact prompt for subproblem generation that enforces genuine chaining (Stage 3). Biggest failure mode: decorative skill mentions where the problem could be solved with fewer than 3 skills.
- How to phrase the "make it harder" / "make it easier" feedback during regeneration. Specific instructions beat vague ones — e.g., "increase the modular arithmetic prime from 7 to a larger prime" vs. "make it harder."
- What to do with failed combinations (>5 regen attempts). For v1: just log and skip. For v2: maybe fall back to M=2 for those combinations.

## Repository structure

```
ttt_binary/
├── pipeline/
│   ├── stage1_generate_skills.py
│   ├── stage3_generate_subproblems.py    # includes calibration loop
│   ├── stage4_solve.py
│   └── run_pipeline.py                    # orchestrates 1-4
├── training/
│   └── grpo_train.py                      # Stage 5, existing code
├── eval/
│   ├── eval_target.py
│   └── eval_heldout.py                    # Stage 6
├── configs/
│   └── v1.yaml
└── data/
    ├── skills/
    ├── subproblems/
    └── solutions/
```
