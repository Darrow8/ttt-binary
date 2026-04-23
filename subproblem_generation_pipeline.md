# Subproblem Generation Pipeline

A domain-agnostic pipeline for decomposing hard binary problems (math, physics, biology — anything with a checkable right/wrong answer) into atomic skill-level subproblems, without requiring access to the parent problem's correct answer.

## Motivation

The naive pipeline `problem → 10 skills → 15 subproblems each → dedupe` fails when the parent problem has a **high-confidence wrong attractor** — a plausible-looking answer that a model would produce by pattern-matching on surface features, even though the correct solution requires fundamentally different techniques.

Example failure: a problem about "conics tangent to 5 conics" pattern-matches to the classical count of 3264, but the actual question (average over 𝔽_p-points of a cover) requires Chebotarev density and harmonic numbers. Decomposing the wrong interpretation produces subproblems that test irrelevant skills.

**Fix**: insert structural analysis and interpretation enumeration *before* skill decomposition, and add an adversarial coverage critic *after*. None of this requires knowing the correct answer.

## Pipeline Overview

1. **Structural feature extraction** — analyze the problem statement, not its content
2. **Interpretation enumeration** — produce 3+ distinct readings, including the surface one
3. **Skill decomposition** — conditioned on features + interpretations
4. **Adversarial coverage critic** — check whether skills cover every problem clause
5. **Subproblem generation** — per skill, conditioned on context
6. **Dedupe** — existing step

---

## Step 1: Structural Feature Extraction

**Purpose**: force the model to attend to problem-statement structure before committing to any technique.

```
You will be given a problem with a definite correct answer (numerical, categorical, or otherwise checkable). Do NOT attempt to solve the problem. Your task is to analyze the problem statement itself.

Answer the following questions. For each, quote the specific words or phrases from the problem statement that support your answer.

1. WHAT IS THE ANSWER, AS AN OBJECT?
Describe the nature of what the answer must be — a count, a probability, a rate, a ratio, a physical quantity with specific units, a yes/no, a parameter value, a dimensionless number, etc. Do not guess the value; characterize the type.

2. WHAT OPERATIONS ARE APPLIED TO PRODUCE THE ANSWER?
The problem may request a raw quantity, or it may apply transformations: an average, a limit, a specific rounded form, a ratio to a reference, an asymptotic leading coefficient, a difference from a baseline, a probability of a specific event. List every transformation the problem statement applies to the underlying quantity of interest.

3. WHAT AGGREGATION OR INDEXING IS PRESENT?
Is the answer about a single object, or is it aggregated over a family of objects? If aggregated, over what family, and with what weighting (uniform, measure-weighted, conditional, limiting)? If there is no aggregation, note that explicitly.

4. WHAT REGIME, LIMIT, OR BOUNDARY IS SPECIFIED?
The problem may specify an asymptotic regime, a range of parameters, boundary conditions, an idealization (e.g., "assume infinite population," "in the large-N limit," "for sufficiently large p"), or a specific scale of interest. What constraints on the regime does the problem impose? What is being held fixed and what is being varied?

5. WHAT NAMED OBJECTS, SYSTEMS, OR MODELS APPEAR?
If the problem names a specific theorem, equation, system, model, organism, process, or construction, each named object brings a body of theory with it. List each named object and briefly note what theory or assumptions it invokes.

6. WHAT IS THE SIMPLEST RELATED PROBLEM, AND HOW DOES THIS ONE DEVIATE FROM IT?
Write down the simplest, most direct question someone could ask about the same subject matter. Then identify how the actual problem statement deviates from that simple phrasing. Every deviation is intentional; each one is a signal about what techniques the problem requires.

7. WHAT FEATURES OF THE PROBLEM STATEMENT SEEM SURPRISING OR EASY TO OVERLOOK?
Identify phrases, conditions, or qualifiers that a quick reader might gloss over but that constrain the answer. Unusual word choices, precise quantifiers, specific phrasings — anything that would not appear in a generic version of the problem.

For each answer, quote the relevant portion of the problem statement verbatim.

Problem: [PROBLEM]
```

**Notes**: Question 6 is the highest-leverage question. It catches attractor problems by forcing the model to articulate why the problem isn't phrased in the simpler way a pattern-matched answer would suggest.

---

## Step 2: Interpretation Enumeration

**Purpose**: force multiple readings so the pipeline isn't committed to the model's first pattern-match.

```
You will be given a problem and a structural analysis of that problem's statement. Do NOT attempt to solve the problem yet. Your task is to produce THREE distinct interpretations of what the problem is literally asking for.

Each interpretation must include:
- A precise statement of what the answer IS, as a specific object produced by a specific procedure
- The body of theory, technique, or knowledge that would be used to compute that object
- A justification grounded in specific features of the problem statement, citing clauses from the structural analysis

Constraints on your interpretations:
- They must differ in what body of theory or technique they invoke. If two interpretations would be solved by the same technique, they are the same interpretation and should be merged.
- One interpretation should be the most obvious surface reading — the interpretation that pattern-matching on the problem's subject matter would suggest.
- At least one of the other interpretations must take seriously the features identified in question 6 and question 7 of the structural analysis (the deviation from the simplest phrasing, and the easily-overlooked details). These features are there for a reason; a correct interpretation should explain why they are there.
- Do not hedge. Each interpretation should commit to a definite answer type and technique. If you are uncertain which interpretation is right, that uncertainty is captured by having multiple interpretations, not by hedging within each.

Problem: [PROBLEM]

Structural analysis: [OUTPUT OF STEP 1]

Output a JSON list of three interpretations, each with:
- answer_type: what the answer IS under this interpretation
- technique: the body of theory that would compute it
- justification: which features of the problem statement support this interpretation
- ignored_features: which features of the problem statement this interpretation does not fully account for (be honest; if an interpretation perfectly accounts for everything, say so)
```

**Notes**: The `ignored_features` field is the key honesty lever. A surface-read interpretation that claims to account for every feature is suspicious; that lie is detectable at step 4.

---

## Step 3: Skill Decomposition

**Purpose**: produce a union skill set that covers all interpretations and all structural features.

```
You will be given a problem, a structural analysis of it, and several interpretations of what it is asking. Your task is to produce a union skill set — a list of atomic skills such that a solver who has mastered all of them would be equipped to solve the problem, regardless of which interpretation turns out to be correct.

Requirements:

1. For EACH interpretation, list the skills a solver would need to execute that interpretation. Label each skill with which interpretation(s) it serves.

2. For EACH feature identified in the structural analysis (especially from questions 4, 5, 6, and 7), verify that at least one skill addresses that feature. If no skill addresses it, add a skill that does.

3. Skills must be ATOMIC: each skill should be a specific, nameable technique, theorem, identity, or body of knowledge. Prefer "knowledge of [specific theorem or technique]" over "algebraic geometry" or "population genetics."

4. Skills must be LOAD-BEARING: if removing the skill would prevent the solver from completing the interpretation, it belongs. If the interpretation would work without it, it does not.

5. Skills should be stated as PREREQUISITES, not as problem-specific observations. "Knowledge of the harmonic sum identity for expected cycle counts in random permutations" is a skill. "The answer involves a sum of reciprocals" is not.

6. Do not bias toward any one interpretation. The skill set should cover all three interpretations, even if you suspect one is more likely correct than the others. The purpose is to hedge against the model's own pattern-matching.

Problem: [PROBLEM]
Structural analysis: [OUTPUT OF STEP 1]
Interpretations: [OUTPUT OF STEP 2]

Output a JSON list of skills, each with:
- name: short identifier
- description: 1-2 sentences
- serves_interpretations: which interpretations this skill supports
- addresses_features: which structural features this skill addresses (if any)
- role: one of {prerequisite_fact, computational_technique, named_theorem, domain_identity}
```

---

## Step 4: Adversarial Coverage Critic

**Purpose**: check whether the skill set accounts for every clause of the problem statement, without knowing the correct answer.

```
Below is a problem and a decomposition of it into atomic skills. Your task is to assess whether the skill set is adequate, by checking coverage.

You will NOT attempt to solve the problem. You will only check whether the skill set addresses every part of the problem statement.

Procedure:

1. Read the problem statement and segment it into clauses (sentences or meaningful sub-sentences).

2. For each clause, identify whether any skill in the list addresses the techniques required by that clause. Quote the clause and either cite the covering skill(s) or mark it as UNCOVERED.

3. For each skill, identify which clause(s) of the problem statement it addresses. If a skill addresses no clause, mark it as UNUSED.

4. Perform the stripping test: mentally remove every clause of the problem that is covered by some skill. What remains? If what remains still contains essential content from the original problem (a transformation, a limit, an aggregation, a qualifier), then the skill set has a gap. Describe what is left over.

5. Perform the inverse stripping test: if you had only the skills listed, with no other knowledge, could you in principle reconstruct what kind of problem these skills are meant to solve? If the skills suggest a different problem than the one given, the skill set is misaligned with the actual problem.

Problem: [PROBLEM]
Skills: [OUTPUT OF STEP 3]

Output:
- uncovered_clauses: problem clauses with no covering skill
- unused_skills: skills addressing no clause
- stripping_test_residue: what remains of the problem after removing covered clauses
- alignment_check: does the skill set suggest a different problem than the one given?
- recommendation: "skills adequate" or "skills need revision, see [specific gaps]"
```

**Notes**: If the critic flags `skills need revision`, loop back to step 3 with the critic's output appended to the context. Cap at 2 revision loops to avoid runaway compute.

---

## Step 5: Subproblem Generation

**Purpose**: generate subproblems that test each skill in a way that mirrors its use in the parent problem, one skill at a time.

```
You are generating a subproblem to test mastery of one specific skill. The subproblem will be used to verify whether a solver has the skill in hand.

You have access to:
- The parent problem
- The skill to be tested
- Which interpretation(s) of the parent problem use this skill
- Which feature(s) of the parent problem this skill addresses

Constraints on the subproblem:

1. RELEVANCE: The subproblem should test this skill in a way that mirrors how the parent problem uses it. If the parent problem applies the skill in a specific regime, at a specific scale, or to a specific kind of object, the subproblem should do the same. Do not generate subproblems that test the skill in scenarios the parent problem does not touch.

2. ISOLATION: The subproblem should test this skill and ONLY this skill. A solver who has mastered this skill but lacks the other skills needed for the parent problem should still be able to solve the subproblem.

3. CHECKABILITY: The subproblem must have a definite, short-form answer that can be verified automatically (a number, a yes/no, a specific named object, a closed-form expression).

4. NON-TRIVIALITY: The subproblem should require actual application of the skill, not just recall of a definition. If the skill is "knowledge of theorem X," the subproblem should require using theorem X to compute something, not just stating theorem X.

5. HONESTY ABOUT DIFFICULTY: The subproblem should be at a difficulty where a solver who knows the skill would solve it reliably but a solver who doesn't would fail reliably. Avoid subproblems where the answer can be guessed, pattern-matched, or brute-forced without the skill.

Parent problem: [PARENT]
Skill: [SKILL]
Interpretations using this skill: [INTERPRETATIONS]
Features addressed: [FEATURES]

Output:
- problem_statement: the subproblem as it would be presented to a solver
- answer: the correct answer in a checkable form
- solution_sketch: a brief explanation of how the skill is applied
- why_relevant: a sentence explaining how this subproblem mirrors the parent problem's use of this skill
- failure_mode: what mistake a solver who lacks this skill would make
```

**Notes**: Run this N times per skill (e.g., N=15 as in your current pipeline) to produce the subproblem bank for that skill. The `why_relevant` and `failure_mode` fields let downstream filtering discard subproblems whose stated relevance is weak.

---

## Step 6: Dedupe

Unchanged from existing pipeline. Apply to the union of all subproblems produced in step 5.

---

## Why This Generalizes

None of the prompts mention domain-specific vocabulary. They refer only to:

- **Structural features** (aggregation, regime, named objects, deviations from simple phrasings) — exist in every domain with well-defined problems
- **Interpretations** (what the answer is as an object, what technique computes it) — a category of analysis that works across math, physics, and biology
- **Skills** (atomic, prerequisite, load-bearing knowledge) — the notion applies to any knowledge domain
- **Coverage** (does the skill set account for every problem clause) — a purely logical check

**What doesn't generalize**: the specific structural features that matter in each domain. Math problems have limits, quantifiers, named categories; physics problems have regimes, boundary conditions, symmetries, named equations; biology problems have demographic parameters, selection regimes, named models. The meta-question "identify structural features that constrain the answer" is domain-agnostic and strong enough on its own — a capable model will surface the domain-specific features when asked.

---

## Handling the No-Answer-Access Constraint

This pipeline is designed to work without knowing the correct answer. The validation signal comes from:

1. **Multiple interpretations** at step 2 — if all interpretations converge on the same technique family, that's evidence the problem is well-posed and its pattern-match is correct. If they diverge, the pipeline must hedge by covering all interpretations in the skill set.

2. **Coverage checking** at step 4 — a skill set that leaves problem clauses uncovered is inadequate, regardless of whether those skills would produce the "obvious" answer. This catches the case where the pipeline has committed to a surface-read interpretation and missed essential structural features.

3. **Agreement rate filtering** at the subproblem level (your existing step) — subproblems that fail to discriminate between skilled and unskilled solvers get filtered out, regardless of what the parent problem's correct answer is.

The pipeline never needs ground truth on the parent problem. It only needs the problem statement and a model capable of careful structural analysis.

---

## Failure Modes to Watch For

- **Step 1 question 6 produces a trivial "deviation"**: if the model claims the only deviation from the simple phrasing is cosmetic, it probably hasn't engaged with the problem. Re-prompt with explicit examples of meaningful deviations.

- **Step 2 interpretations collapse to one technique**: if all three interpretations invoke the same body of theory, the model is pattern-matching. Re-prompt with "produce interpretations that disagree about what the answer is, not just about how to compute it."

- **Step 4 critic always returns "skills adequate"**: the critic is being sycophantic. Prompt it with specific examples of what a gap looks like, and require it to quote the specific clause and specific skill mapping for each covered clause.

- **Step 5 subproblems drift from parent context**: if a subproblem's `why_relevant` field reads as post-hoc rationalization, the subproblem probably isn't relevant. Filter out subproblems whose stated relevance doesn't ground in specific features of the parent.

---

## Implementation Notes

- Steps 1-4 are each a single LLM call. Total added cost per parent problem: ~4 calls before subproblem generation starts.
- Step 3 can be looped with step 4 feedback for up to 2 revision rounds.
- Step 5 is per-skill, run N times per skill, just as in your current pipeline.
- The full pipeline adds ~4-10 calls per parent problem on top of whatever step 5 requires. Given typical step-5 cost of 150+ calls (10 skills × 15 subproblems), the overhead is small (<10%) for a substantial improvement in skill relevance.
- All prompt outputs should be JSON where indicated; use a validator to enforce schema and retry on malformed output.
