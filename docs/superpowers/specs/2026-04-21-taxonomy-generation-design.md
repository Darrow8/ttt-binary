# Taxonomy-First Subproblem Generation — Design

## Motivation

The existing Stage 1 generator (`Stage1/distinct_llm_prompting.py`) prompts the
model to generate subproblems "similar to" the hard target. It operates on a
single prompt mode, which tends to concentrate output in a few popular
structural templates (e.g. separable-primes-density variants, conic tangency
counts). Diversity is shallow — the textual dedupe in the `dedupe` branch
catches lexical repeats, but many surviving problems are structural
near-duplicates that test the same underlying reasoning.

For GRPO training, lack of reasoning diversity is the direct cause of
`problems/skipped_frac` climbing to 0.44 in `c-100-run-originalbatch-2/iaw5abzo`
(44% of late-training batches had zero advantage — every completion in the
group got identical reward, meaning the problems weren't distinguishing).

Hypothesis: decomposing the target into explicit reasoning skills and
generating subproblems *per skill* produces broader coverage of the target's
component skills, lower degenerate-group rate during training, and better
downstream solve-rate on the target.

## Scope

**In scope (v1):**
- A new Stage 1 variant script, `Stage1/taxonomy_generation.py`, that runs
  alongside the existing `distinct_llm_prompting.py` without modifying it.
- Two-phase generation: (1) decompose target into 10 skills, (2) generate
  10 agreement-window-passing subproblems per skill.
- Skills cached to disk for reproducibility.
- Existing Stage 2/3/3c/4/5/train/eval untouched.

**Explicitly out of scope (v1, deferred):**
- Textual or semantic dedupe inside this pipeline. The taxonomy structure
  diversifies by construction — we defer dedupe until we measure whether
  it's needed.
- Adaptive agreement window per skill.
- Skill-drop + redecomposition on repeated failure.
- Mixing generator models.
- Parallelizing across skills (skills run sequentially in v1).

**Intentionally parallel to existing pipeline:**
This pipeline does not replace `distinct_llm_prompting.py`. Both scripts
coexist. The paper ablation is "run both on the same target, compare
downstream solve-rate."

## Approach

Two phases, both using the same hard-coded generator model (gpt-oss-120b-maas).

### Phase 1: decomposition (once per target, cached)

Single LLM call. Prompt asks for exactly 10 distinct reasoning skills with
names, descriptions, and example-problem hints. JSON output. Result written
atomically to `runs/<id>/skills.json`. If the file exists on a subsequent run,
reuse it.

### Phase 2: per-skill generation loop

For each of the 10 skills, sequentially:

1. Generate a candidate subproblem targeted at the skill (not the hard
   target) using the skill's `name`, `description`, and `example_problem_hint`
   as prompt context.
2. Solve the candidate `n_samples=10` times. Compute agreement rate.
3. Keep the candidate iff: `0.60 <= agreement <= 0.80` AND majority answer
   is non-empty AND numeric.
4. Repeat until the skill has 10 keeps, OR `max_candidates_per_skill=100`
   attempts have been made.
5. Move to the next skill.

Per-skill generation pipelines `gen_workers` candidates concurrently (matching
the existing Stage 1 pattern). Skills themselves are processed sequentially
for implementation simplicity — parallelizing them is a v2 optimization.

### Non-negotiable constraints

1. **Model hardcoded to `openai/gpt-oss-120b-maas`.** Module-level constant,
   no CLI override, used for all three call types (decompose, generate,
   solve-for-agreement). Recorded in every output artifact so provenance is
   auditable.
2. **No `max_tokens`.** Chat completions calls pass `model + messages +
   temperature` only. The model runs to its own EOS token. Matches the
   existing `distinct_llm_prompting.py` behavior.
3. **No client-side timeouts.** The OpenAI client is constructed without
   `timeout=`. Completion calls run without `timeout=`. Long Vertex
   responses are permitted. Solve calls still get a `tenacity` retry wrapper
   (matching existing `_solve_one`) on *exceptions*, not timeouts.
4. **LaTeX content, not ASCII math.** Every subproblem statement stored in
   `keeps.json` / `skips.json` must be written in LaTeX. Every reasoning
   trace in `all_solutions` is likewise LaTeX, which follows automatically
   because the solver prompt instructs the solve model to produce LaTeX
   output ending in `\boxed{...}`. The generator prompt enforces this on
   the problem-statement side; the solve prompt (reused from
   `distinct_llm_prompting.py`) already enforces it on the solver side.
   No separate post-processing to convert ASCII → LaTeX.

## Components

### New: `Stage1/taxonomy_generation.py`

Single script, ~300 lines. Reuses existing infrastructure via direct imports
from `distinct_llm_prompting.py`:
- `_get_vertex_access_token`, `_build_vertex_base_url`, `get_client`
- `solve_and_check_agreement`, `_solve_one`
- `_is_numeric_answer`, `normalize_answer`, `extract_answer`
- `Dataset`, `GeneratedProblem` dataclasses
- `_save_atomic`

Net-new code:
- Module-level `GENERATOR_MODEL = "openai/gpt-oss-120b-maas"` constant.
- `decompose_target(client, target_problem) -> list[Skill]` — one LLM call,
  JSON output, up to 3 retries on parse failure.
- `Skill` dataclass: `name`, `description`, `example_problem_hint`.
- `generate_for_skill(client, target, skill, n_target, n_samples, max_candidates, gen_workers)`
  — per-skill generate-until loop.
- `build_taxonomy_dataset(...)` — orchestrator: decompose, loop skills,
  write outputs.
- `main()` — argparse CLI.

### Prompts

**Decomposition prompt** (system message + user message):

```
You are designing a curriculum to help a student learn to solve a hard
target problem by mastering its component reasoning skills first.

Target problem:
{target}

Decompose this target into EXACTLY 10 distinct reasoning skills. Each skill:
- Must be a component of the target — a specific reasoning step or tool,
  not a rephrasing of the whole problem.
- Must be DISTINCT from the others: no two skills should test the same
  underlying reasoning.
- Must be testable in a self-contained subproblem that can be solved
  without requiring the other skills.
- Should fall roughly in difficulty order, prerequisite to advanced.

Respond with JSON only, no prose, exactly this shape:
{
  "skills": [
    {
      "name": "Short skill name (3–10 words)",
      "description": "1–2 sentences explaining what the skill is.",
      "example_problem_hint": "One sentence sketching what a problem testing this skill looks like."
    },
    ... 10 entries total
  ]
}
```

**Per-skill generation prompt:**

```
You are designing one subproblem to help a student practice a specific
reasoning skill.

The end goal is mastery of this hard target problem (for context only —
do NOT generate a variant of the target):

{target}

Skill to test:
Name: {skill.name}
Description: {skill.description}
Example hint: {skill.example_problem_hint}

Requirements:
- The subproblem tests THIS SKILL SPECIFICALLY, in isolation.
- A student who has mastered only this skill should be able to solve it.
  The problem must not rely on the other 9 skills from the taxonomy.
- The answer MUST be a single number (integer or decimal). If a decimal,
  ask the solver to round to 4 decimal places.
- State the problem in 3–10 sentences.
- ALL math must be written in LaTeX using `\(...\)` for inline expressions
  and `\[...\]` for display expressions. No ASCII math ("x^2", "sqrt(5)",
  "sum from i=1 to n", etc.) — use proper LaTeX instead.
- The problem statement MUST end with an instruction to the solver:
  "Put your final answer inside \boxed{}." (so the answer can be
  extracted from the solver's reply).

Output format:
Begin your response with `<problem>` on its own line, then the full
problem statement, then `</problem>` on its own line. Nothing else before,
between, or after — the tags are used for parsing, not for the student.
```

### Output layout

```
runs/<id>/
├── skills.json                    ← decomposition cache; persists across runs
└── stage1_taxonomy/
    └── <YYYYMMDD_HHMMSS>_<pid>/
        ├── keeps.json             ← 100 (target) kept subproblems, each tagged with "skill"
        ├── skips.json             ← candidates that failed the agreement window or were non-numeric
        └── per_skill_stats.json   ← {skill_name: {n_passed, n_attempted, status}}
```

**`skills.json` schema:**
```json
{
  "target_problem_path": "data/target-problems/conics.txt",
  "target_problem_hash": "sha1 hex",
  "generator_model": "openai/gpt-oss-120b-maas",
  "decomposed_at": "ISO-8601 UTC",
  "skills": [
    {
      "name": "Bézout intersection count",
      "description": "...",
      "example_problem_hint": "..."
    },
    ... 10 entries
  ]
}
```

**`keeps.json` schema** — extends the existing Stage 1 schema with a `skill`
field per problem so downstream stages remain compatible:
```json
{
  "source_problem": "...",
  "target_agreement_low": 0.60,
  "target_agreement_high": 0.80,
  "n_problems": 100,
  "generator_model": "openai/gpt-oss-120b-maas",
  "solve_model": "openai/gpt-oss-120b-maas",
  "problems": [
    {
      "skill": "Bézout intersection count",
      "problem": "...",
      "ground_truth_answer": "...",
      "agreement_rate": 0.7,
      "all_answers": [...],
      "all_solutions": [...],
      "n_samples": 10
    },
    ...
  ]
}
```

**`per_skill_stats.json` schema:**
```json
{
  "skills": [
    {"name": "...", "n_target": 10, "n_passed": 10, "n_attempted": 28, "status": "ok"},
    {"name": "...", "n_target": 10, "n_passed": 6, "n_attempted": 100, "status": "capped"},
    ...
  ],
  "total_passed": 94,
  "total_attempted": 478,
  "total_target": 100
}
```

### CLI

```
python Stage1/taxonomy_generation.py \
  --problem-path data/target-problems/conics.txt \
  --runs-subdir conics-tangent-5 \
  --failed-solutions runs/conics-tangent-5/base_attempts.json \
  --n-skills 10 \
  --problems-per-skill 10 \
  --max-candidates-per-skill 100 \
  --n-samples 10 \
  --agree-low 0.60 \
  --agree-high 0.80 \
  --gen-workers 4 \
  --max-workers 16 \
  [--output <dir>]
```

All numeric knobs are CLI-tunable. `--model` is intentionally omitted; model
is hardcoded. `--failed-solutions` is accepted for CLI symmetry with the
existing Stage 1 script but is NOT used by the generator prompt in v1 —
skill name/description/hint are the only generator context beyond the
target (see "Why no failed solutions" below).

### Resume behavior

1. If `runs/<id>/skills.json` exists, read it. Skip Phase 1.
2. If no `--output` override, create a new timestamped run directory. Do
   not attempt to resume partial per-skill progress across runs in v1 —
   each invocation runs Phase 2 from scratch. Deferred to v2 if needed.

### Why no failed-solutions context in v1

The existing Stage 1 uses `failed_solutions` (top-K most-common wrong-answer
traces from Stage 0) as context for the generator. In taxonomy-first,
skill name + description + example hint already focus the generator
narrowly. Adding failed-solution noise is likely to pull the generator
back toward variants of the target problem, defeating the
skill-in-isolation goal. We can A/B this later; v1 skips it.

## Testing

Unit tests in `tests/test_taxonomy_generation.py`:

- `decompose_target` returns 10 parsed `Skill` objects from a mocked JSON
  response.
- JSON parse failure triggers retry up to 3 times, then raises.
- `generate_for_skill` stops at `n_target` keeps.
- `generate_for_skill` stops at `max_candidates` attempts if not enough pass.
- Per-skill stats are correctly aggregated.
- `keeps.json` contains a `skill` field on every problem.
- `skills.json` is reused if present (no second decomposition call).

Not tested: the actual LLM calls. Tests mock the OpenAI client (return
canned responses). Integration smoke test is manual: run against
`conics-tangent-5` with `--problems-per-skill 2 --max-candidates-per-skill
6` as a cheap check.

## Logging

Console output during a run:

```
=== Taxonomy decomposition ===
Model: openai/gpt-oss-120b-maas
Target: data/target-problems/conics.txt (731 chars)
Decomposing into 10 skills...
  10 skills written to runs/conics-tangent-5/skills.json

=== Per-skill generation ===
[1/10] Bézout intersection count
  attempt 1: KEEP (agreement 0.70)
  attempt 2: skip (too easy, agreement 0.90)
  attempt 3: KEEP (agreement 0.60)
  ...
  done: 10/10 passed after 28 attempts

[2/10] ...
```

Final summary:

```
======================================================================
  Taxonomy dataset complete: 94/100 (6 shortfall from 1 capped skill)
  Skills ok: 9/10
  Total attempted: 478
======================================================================
```

## Rollout

- Ships as a new script. Existing users of `distinct_llm_prompting.py`
  unaffected.
- Eventual `run_pipeline.py stage1-taxonomy` subcommand is a follow-up PR
  (not v1 — keeps this PR small).
- Paper ablation: run both scripts for 3–5 hard problems. Compare:
  - Downstream solve rate at matching training checkpoints
  - `problems/skipped_frac` and `advantage/std` in wandb
  - Final subproblem pool structure (flat vs. skill-stratified)

## Risk

- **Decomposition quality.** If gpt-oss-120b produces poor taxonomies
  (overlapping skills, trivial skills, off-target skills), the whole run
  suffers. Mitigated by: explicit diversity requirement in prompt, cached
  output for human inspection before launching the expensive phase 2.
- **Unbounded runtime.** No timeouts + no max_tokens + up to 100 attempts
  per skill × 10 skills = worst case 1000 generate+solve cycles. At
  reasonable rates this is hours, not days. Still worth monitoring. Ship
  with `nohup` / `tee` guidance in the recipe.
- **Cost.** Per-skill generation is ~25% pass rate → ~400 gen + 4000 solve
  calls for 100 keeps. Tens of dollars to ~$100 per target on
  gpt-oss-120b-maas. Acceptable for a paper run; not suitable for sweeps
  without further work.
- **Divergence from main pipeline.** This script duplicates the
  agreement-check loop logic rather than refactoring `distinct_llm_prompting.py`
  to be pluggable. Acceptable for v1 (the two pipelines differ enough that
  a shared abstraction is premature). If both prove valuable long-term, a
  v2 refactor can factor out the common engine.
