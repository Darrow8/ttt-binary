"""Stage 3 — generate one subproblem per C(X, M) skill combination, with a
consensus difficulty-calibration loop AND a generator/critic answer-agreement
cross-check.

For each unordered M-tuple of skills:
  1. Ask the generator to produce a subproblem whose solution chains the M
     skills, with a single verifiable answer AND per-skill rationale.
  2. Sample K_calibrate attempts from the critic (a different family from
     the generator). Compute (a) solve-rate against the generator's claimed
     answer, (b) the modal critic answer and its consensus strength.
  3. Accept iff: critic mode == generator's claim AND solve-rate ∈ band.
     Reject as "answer_disagreement" if the critic mode disagrees with the
     generator's claim — this catches generator hallucinations cheaply.
  4. Out-of-band → regenerate with a directional hint. Cap at max_regen_attempts.

Output: data/subproblems/<problem_id>.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import threading
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ttt_binary.answer_extract import extract_boxed
from ttt_binary.cluster import (
    Decision,
    UNPARSEABLE,
    cluster_answers,
    decide,
    regen_feedback,
)
from ttt_binary.llm import call_anthropic, call_openai, parse_json_loose


GEN_SYSTEM = """\
You write math subproblems that exercise a specified set of reasoning skills. \
You are given a list of skills, each with a description, preconditions, \
postconditions, and an isolated example. Your job: produce one subproblem \
whose intended solution genuinely requires composing ALL the listed skills — \
each skill's postcondition feeds the next skill's preconditions.

The subproblem MUST have a single numerical final answer rounded to exactly \
4 decimal places, expressed inside \\boxed{} (e.g. \\boxed{866.0000}, \
\\boxed{0.5000}, \\boxed{-2.3457}). You must compute that answer correctly."""


GEN_PROMPT_TEMPLATE = """\
Produce ONE subproblem that requires composing the {m} skills below in a \
chain. The output of one skill must be the input of the next. The reader \
must NOT be told which skills to use — the problem statement is plain math.

HARD CONSTRAINTS:
- The subproblem must have a single, fully-determined NUMERICAL final answer \
  (a real number). No symbolic expressions, no fractions like "p/q" without \
  simplifying to a decimal, no "depends on parameter" answers.
- The expected answer must be a real number, ROUNDED TO 4 DECIMAL PLACES, \
  reported inside \\boxed{} in the form \\boxed{X.XXXX} \
  (e.g. \\boxed{866.0000}, \\boxed{0.5000}, \\boxed{-2.3457}). Always include \
  trailing zeros to fill all 4 decimal places.
- The problem statement should explicitly tell the reader: "Round your final \
  answer to 4 decimal places and place it inside \\boxed{}."
- Do NOT mention skill names, the word "skill", or any meta-language about \
  the technique inside problem_text.
- Do NOT include the answer inside problem_text.
- The chain MUST GENUINELY REQUIRE all {m} skills. A solver who skips any \
  one of the {m} skills must be unable to reach the answer. If you can solve \
  it without one of the listed skills, the subproblem is wrong — redraft.
- Vary parameters so different subproblems produce different numerical \
  answers. Avoid parameter values that match famous classical results.
- The expected_answer you emit MUST be the answer you would compute by \
  actually solving the problem step-by-step. Before responding, work the \
  problem and confirm the value.
{difficulty_hint}

Skills (use ALL of them):
{skills_block}

Respond as a single JSON object (no prose around it) with this exact shape:
{
  "problem_text": "the full problem as a self-contained math statement, ending with the rounding-and-boxing instruction",
  "per_skill_role": {
    "<skill_name_1>": "<one sentence: which step in the solution invokes this skill, what its input is, what it produces>",
    "<skill_name_2>": "...",
    "<skill_name_3>": "..."
  },
  "skill_chain_rationale": "1-3 sentences describing the dependency order of the chain (which skill's output feeds which skill's input)",
  "expected_answer": "the verifiable final answer as a number rounded to 4 decimal places, e.g. 866.0000"
}

After drafting, verify: (a) every one of the {m} skill names appears as a key \
in per_skill_role; (b) removing any one skill from your solution would make \
the problem unsolvable; (c) you have actually computed expected_answer rather \
than guessing; (d) expected_answer is a single real number with 4 decimal \
places of precision.
"""


SOLVE_SYSTEM = """\
You are a careful and rigorous math student. Solve the problem step by step, \
showing all important intermediate work. Compute a numerical final answer \
and round it to exactly 4 decimal places. Place ONLY that rounded number \
inside \\boxed{} at the end (e.g. \\boxed{866.0000}, \\boxed{0.5000}, \
\\boxed{-2.3457}). Always include trailing zeros to fill all 4 decimals."""


def _format_skill(s: dict) -> str:
    return (
        f"- {s['name']}\n"
        f"    description:    {s['description']}\n"
        f"    preconditions:  {s['preconditions']}\n"
        f"    postconditions: {s['postconditions']}\n"
        f"    example:        {s['example']}"
    )


def _hint_from_decision(prev: Decision | None) -> str:
    if prev is None:
        return "- Aim for a difficulty where a strong solver gets it right roughly half the time."
    return regen_feedback(prev)


def _generate_one(
    skills_in_combo: list[dict],
    *,
    prev_decision: Decision | None,
    generator_model: str,
    temperature: float,
) -> dict:
    skills_block = "\n".join(_format_skill(s) for s in skills_in_combo)
    # Use plain .replace() rather than .format() because the template body
    # contains literal LaTeX braces (\boxed{X.XXXX}, etc) that would otherwise
    # be misinterpreted as positional placeholders.
    prompt = (
        GEN_PROMPT_TEMPLATE
        .replace("{m}", str(len(skills_in_combo)))
        .replace("{difficulty_hint}", _hint_from_decision(prev_decision))
        .replace("{skills_block}", skills_block)
    )
    text = call_anthropic(
        prompt,
        model=generator_model,
        system=GEN_SYSTEM,
        temperature=temperature,
    )
    obj = parse_json_loose(text)
    for k in ("problem_text", "skill_chain_rationale", "expected_answer", "per_skill_role"):
        if k not in obj:
            raise ValueError(f"generator missing field {k}: keys={list(obj)}")
    psr = obj["per_skill_role"]
    if not isinstance(psr, dict):
        raise ValueError(f"per_skill_role must be a dict, got {type(psr).__name__}")
    expected_names = {s["name"] for s in skills_in_combo}
    rationale_names = {str(k) for k in psr.keys()}
    missing = expected_names - rationale_names
    if missing:
        raise ValueError(
            f"per_skill_role missing entries for: {sorted(missing)}; "
            f"got keys={sorted(rationale_names)} (decorative-skill failure)"
        )
    return obj


def _calibrate(
    problem_text: str,
    *,
    k: int,
    critic_model: str,
    parallel: int = 8,
) -> list[dict]:
    """Run K critic solves at temperature 0.7 in parallel.

    Returns a list of per-attempt records:
        {ok: bool, predicted: str|None, text: str, error?: str}

    Calibration NO LONGER compares against any "expected_answer" — that
    comparison happened in the previous (mode==claim) variant. The new
    decision rule (REVISIONS.md) clusters the predictions and treats the
    largest cluster as the authoritative answer.
    """
    def one_attempt(_i: int) -> dict:
        try:
            text = call_openai(
                problem_text
                + "\n\nRound your final answer to 4 decimal places and place"
                  " it inside \\boxed{} (e.g. \\boxed{866.0000}).",
                model=critic_model,
                system=SOLVE_SYSTEM,
                temperature=0.7,
            )
        except Exception as e:
            return {"ok": False, "error": str(e)[:200], "predicted": None, "text": ""}
        predicted = extract_boxed(text)
        return {"ok": True, "text": text, "predicted": predicted}

    attempts: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=parallel) as pool:
        for r in pool.map(one_attempt, range(k)):
            attempts.append(r)
    return attempts


def _process_combo(
    combo_idx: int,
    skills_in_combo: list[dict],
    *,
    band: tuple[float, float],
    ambiguity_threshold: float,
    k_calibrate: int,
    max_regen: int,
    generator_model: str,
    critic_model: str,
    temperature: float,
    max_unparseable: int = 3,
    write_attempt=None,  # callable(record: dict, accepted: bool) -> None
) -> dict:
    """Generate→calibrate→cluster→decide→regen loop for one skill combination.

    Outputs the new (consensus-based) record schema from REVISIONS.md:
        consensus_answer (training target), generator_proposed_answer,
        p1, p2, all_answer_clusters, per_iteration log, calibration_attempts.

    Status values:
        "accepted"
        "REJECT_AMBIGUOUS" / "REJECT_TOO_EASY" / "REJECT_TOO_HARD_OR_AMBIGUOUS"
        "errored_out"
    """
    skill_names = [s["name"] for s in skills_in_combo]
    last_decision: Decision | None = None
    last_record: dict | None = None
    per_iteration: list[dict] = []
    n_transient_errors = 0
    attempt = 0

    while attempt <= max_regen:
        try:
            gen = _generate_one(
                skills_in_combo,
                prev_decision=last_decision,
                generator_model=generator_model,
                temperature=temperature,
            )
            cal_attempts = _calibrate(
                gen["problem_text"],
                k=k_calibrate,
                critic_model=critic_model,
            )
        except Exception as e:
            n_transient_errors += 1
            print(f"  [combo {combo_idx}] error attempt {attempt}: {e}", flush=True)
            per_iteration.append({
                "attempt": attempt,
                "kind": "ERROR",
                "reason": f"{type(e).__name__}: {e}"[:300],
            })
            time.sleep(1)
            if n_transient_errors >= 3:
                rec = {
                    "combo_idx": combo_idx,
                    "skills_used": skill_names,
                    "status": "errored_out",
                    "regeneration_attempts": attempt,
                    "n_transient_errors": n_transient_errors,
                    "per_iteration": per_iteration,
                    "error": f"{type(e).__name__}: {e}"[:300],
                }
                if write_attempt is not None:
                    write_attempt(rec, accepted=False)
                return rec
            continue

        # Cluster the K predictions and apply the decision rule.
        predictions = [a.get("predicted") for a in cal_attempts]
        clusters = cluster_answers(predictions)
        decision = decide(
            clusters,
            k_calibrate=k_calibrate,
            band=band,
            ambiguity_threshold=ambiguity_threshold,
            max_unparseable=max_unparseable,
        )
        last_decision = decision

        gen_claim = str(gen["expected_answer"])
        # Canonicalize the generator's claim with the same machinery so the
        # comparison to consensus is on equal footing.
        from ttt_binary.cluster import _canonicalize  # local import to avoid leaking
        gen_claim_canon = _canonicalize(gen_claim)
        consensus_matches_generator = (
            decision.consensus_answer is not None
            and decision.consensus_answer == gen_claim_canon
        )

        per_iteration.append({
            "attempt": attempt,
            "kind": decision.kind,
            "consensus_answer": decision.consensus_answer,
            "generator_proposed_answer": gen_claim,
            "consensus_matches_generator": consensus_matches_generator,
            "p1": decision.p1,
            "p2": decision.p2,
            "n_unparseable": decision.n_unparseable,
            "reason": decision.reason,
        })

        record = {
            "combo_idx": combo_idx,
            "skills_used": skill_names,
            "per_skill_role": gen.get("per_skill_role"),
            "skill_chain_rationale": gen.get("skill_chain_rationale"),
            "problem_text": gen["problem_text"],
            "consensus_answer": decision.consensus_answer,
            "generator_proposed_answer": gen_claim,
            "consensus_matches_generator": consensus_matches_generator,
            "p1": decision.p1,
            "p2": decision.p2,
            "n_unparseable": decision.n_unparseable,
            "all_answer_clusters": dict(decision.clusters),
            "k_calibrate": k_calibrate,
            "regeneration_attempts": attempt,
            "per_iteration": list(per_iteration),
            "calibration_attempts": [
                {
                    "ok": a.get("ok"),
                    "predicted": a.get("predicted"),
                    "error": a.get("error"),
                    "text": a.get("text"),
                }
                for a in cal_attempts
            ],
        }
        last_record = record

        if decision.kind == "ACCEPT":
            record["status"] = "accepted"
            print(
                f"  [combo {combo_idx}] accepted "
                f"(p1={decision.p1:.2f}, p2={decision.p2:.2f}, "
                f"consensus={decision.consensus_answer!r}, "
                f"gen_match={consensus_matches_generator}, "
                f"attempts={attempt+1})",
                flush=True,
            )
            if write_attempt is not None:
                write_attempt(record, accepted=True)
            return record

        # Rejected — log immediately, then regen.
        print(
            f"  [combo {combo_idx}] {decision.kind} "
            f"(p1={decision.p1:.2f}, p2={decision.p2:.2f}); "
            f"regen {attempt+1}/{max_regen}",
            flush=True,
        )
        if write_attempt is not None:
            # Mark this rejected attempt with its decision kind so resume
            # logic can tell mid-run rejects apart from final cap-outs.
            rec_for_skip = dict(record, status=decision.kind)
            write_attempt(rec_for_skip, accepted=False)
        attempt += 1

    # Fell out of the regen loop without acceptance. Use the LAST decision
    # kind as the status so the cap-out reason is informative. Also mark
    # cap_out=True so resume can distinguish this from a mid-run skip.
    if last_record is None:
        rec = {
            "combo_idx": combo_idx,
            "skills_used": skill_names,
            "status": "errored_out",
            "regeneration_attempts": attempt,
            "per_iteration": per_iteration,
            "cap_out": True,
        }
        if write_attempt is not None:
            write_attempt(rec, accepted=False)
        return rec
    last_record["status"] = last_decision.kind if last_decision else "errored_out"
    last_record["cap_out"] = True
    if write_attempt is not None:
        write_attempt(last_record, accepted=False)
    return last_record


def _aggregate_stats(results: list[dict]) -> dict:
    """Run-end summary stats from REVISIONS.md §"Logging requirements"."""
    n = len(results)
    accepted = [r for r in results if r.get("status") == "accepted"]
    fail_reasons = Counter(
        r.get("status") for r in results if r.get("status") != "accepted"
    )
    n_generator_disagrees = sum(
        1 for r in accepted if r.get("consensus_matches_generator") is False
    )
    p1_values = [r.get("p1") for r in accepted if isinstance(r.get("p1"), (int, float))]
    p1_hist_bins = [0.40, 0.45, 0.50, 0.55, 0.60]
    p1_hist = {f"<= {b:.2f}": 0 for b in p1_hist_bins}
    for v in p1_values:
        for b in p1_hist_bins:
            if v <= b:
                p1_hist[f"<= {b:.2f}"] += 1
                break
    p1_mean = (sum(p1_values) / len(p1_values)) if p1_values else None
    return {
        "n_total": n,
        "n_accepted": len(accepted),
        "pct_accepted": (len(accepted) / n) if n else 0.0,
        "fail_counts": dict(fail_reasons),
        "fail_pcts": {k: v / n for k, v in fail_reasons.items()} if n else {},
        "n_accepted_with_generator_mismatch": n_generator_disagrees,
        "pct_accepted_with_generator_mismatch": (
            (n_generator_disagrees / len(accepted)) if accepted else 0.0
        ),
        "p1_mean_accepted": p1_mean,
        "p1_histogram_accepted": p1_hist,
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path or not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def generate_subproblems(
    problem_id: str,
    skills: list[dict],
    *,
    m: int,
    band: tuple[float, float],
    ambiguity_threshold: float,
    k_calibrate: int,
    max_regen: int,
    generator_model: str,
    critic_model: str,
    temperature: float,
    workers: int,
    out_path: Path,
    keeps_path: Path | None = None,
    skips_path: Path | None = None,
    max_unparseable: int = 3,
    max_combos: int | None = None,
) -> list[dict]:
    """For each C(X, M) combination: generate -> calibrate -> decide -> regen.

    Per-attempt writes:
        keeps_path  -- one JSONL line per ACCEPTED combo (final accepted record)
        skips_path  -- one JSONL line per REJECTED attempt (mid-run AND cap-out)

    Both files are appended to as work happens, so partial progress survives a
    crash. The final consolidated <id>.json is still written at the end.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if keeps_path is None:
        keeps_path = out_path.parent / f"{problem_id}.keeps.jsonl"
    if skips_path is None:
        skips_path = out_path.parent / f"{problem_id}.skips.jsonl"

    combos = list(combinations(range(len(skills)), m))
    if max_combos is not None and max_combos < len(combos):
        combos = combos[:max_combos]
        print(f"smoke mode: capped at {max_combos} of "
              f"C({len(skills)},{m}) combinations", flush=True)
    print(f"enumerating {len(combos)} combinations of {m} skills from {len(skills)}",
          flush=True)

    # Resume support: a combo is "done" if it has a keep, OR a skip record
    # with cap_out=True, OR a skip record with errored_out status.
    keep_records = _read_jsonl(keeps_path)
    skip_records = _read_jsonl(skips_path)
    accepted_idx: dict[int, dict] = {r["combo_idx"]: r for r in keep_records
                                     if "combo_idx" in r}
    capped_idx: dict[int, dict] = {}
    for r in skip_records:
        if "combo_idx" not in r:
            continue
        if r.get("cap_out") is True or r.get("status") == "errored_out":
            capped_idx[r["combo_idx"]] = r
    done_ids = set(accepted_idx) | set(capped_idx)
    if done_ids:
        print(
            f"  resume: {len(accepted_idx)} accepted in {keeps_path.name}, "
            f"{len(capped_idx)} cap-outs in {skips_path.name} "
            f"-> skipping {len(done_ids)} combos",
            flush=True,
        )

    todo = [(i, c) for i, c in enumerate(combos) if i not in done_ids]

    write_lock = threading.Lock()

    def write_attempt(rec: dict, *, accepted: bool) -> None:
        path = keeps_path if accepted else skips_path
        line = json.dumps(rec) + "\n"
        with write_lock:
            with path.open("a") as f:
                f.write(line)
                f.flush()

    def task(item):
        i, idx_tuple = item
        return _process_combo(
            i,
            [skills[j] for j in idx_tuple],
            band=band,
            ambiguity_threshold=ambiguity_threshold,
            k_calibrate=k_calibrate,
            max_regen=max_regen,
            generator_model=generator_model,
            critic_model=critic_model,
            temperature=temperature,
            max_unparseable=max_unparseable,
            write_attempt=write_attempt,
        )

    new_results: list[dict] = []
    if workers <= 1:
        for item in todo:
            new_results.append(task(item))
    else:
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(task, item) for item in todo]
            for fut in cf.as_completed(futs):
                new_results.append(fut.result())

    # Combine resumed + new results for the rolled-up summary.
    all_results: list[dict] = []
    all_results.extend(accepted_idx.values())
    # For capped-out, drop ones we just re-ran (shouldn't happen since they
    # were in done_ids) and prefer the new record otherwise.
    new_idx = {r.get("combo_idx") for r in new_results}
    for cidx, rec in capped_idx.items():
        if cidx not in new_idx and cidx not in accepted_idx:
            all_results.append(rec)
    all_results.extend(new_results)

    all_results.sort(key=lambda r: r.get("combo_idx", -1))
    accepted = [r for r in all_results if r.get("status") == "accepted"]
    failed = [r for r in all_results if r.get("status") != "accepted"]
    stats = _aggregate_stats(all_results)

    out_path.write_text(json.dumps({
        "problem_id": problem_id,
        "m": m,
        "n_skills": len(skills),
        "n_combinations": len(combos),
        "n_accepted": len(accepted),
        "n_failed": len(failed),
        "band": band,
        "ambiguity_threshold": ambiguity_threshold,
        "k_calibrate": k_calibrate,
        "max_regen": max_regen,
        "keeps_file": str(keeps_path),
        "skips_file": str(skips_path),
        "stats": stats,
        "subproblems": accepted,
        "failed": failed,
    }, indent=2))
    print(f"wrote {out_path} (accepted={len(accepted)}, failed={len(failed)})")
    print(f"  per-generation logs: {keeps_path}, {skips_path}")
    print("aggregate stats:", json.dumps(stats, indent=2))
    return accepted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem-id", required=True)
    ap.add_argument("--skills-file", required=True,
                    help="Path to skills JSON from Stage 1")
    ap.add_argument("--m", type=int, default=3, help="chain length")
    ap.add_argument("--band-lo", type=float, default=0.4)
    ap.add_argument("--band-hi", type=float, default=0.6)
    ap.add_argument("--ambiguity-threshold", type=float, default=0.2,
                    help="max allowed second-cluster fraction p2 (REVISIONS.md)")
    ap.add_argument("--k-calibrate", type=int, default=10)
    ap.add_argument("--max-regen", type=int, default=5)
    ap.add_argument("--max-unparseable", type=int, default=3,
                    help="reject if more than this many critic attempts return no parseable answer")
    ap.add_argument("--max-combos", type=int, default=None,
                    help="process at most N of the C(X,M) combinations (smoke-test mode)")
    ap.add_argument("--generator-model", default="openai/gpt-oss-120b-maas")
    ap.add_argument("--critic-model", default="openai/gpt-oss-120b-maas")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default="ttt_binary/data/subproblems")
    ap.add_argument("--keeps-file", default=None,
                    help="JSONL path appended on each ACCEPTED combo (default: <out-dir>/<id>.keeps.jsonl)")
    ap.add_argument("--skips-file", default=None,
                    help="JSONL path appended on each REJECTED attempt (default: <out-dir>/<id>.skips.jsonl)")
    args = ap.parse_args()

    skills_obj = json.loads(Path(args.skills_file).read_text())
    skills = skills_obj["skills"]
    out_path = Path(args.out_dir) / f"{args.problem_id}.json"
    keeps = Path(args.keeps_file) if args.keeps_file else None
    skips = Path(args.skips_file) if args.skips_file else None
    generate_subproblems(
        problem_id=args.problem_id,
        skills=skills,
        m=args.m,
        band=(args.band_lo, args.band_hi),
        ambiguity_threshold=args.ambiguity_threshold,
        k_calibrate=args.k_calibrate,
        max_regen=args.max_regen,
        generator_model=args.generator_model,
        critic_model=args.critic_model,
        temperature=args.temperature,
        workers=args.workers,
        out_path=out_path,
        keeps_path=keeps,
        skips_path=skips,
        max_unparseable=args.max_unparseable,
        max_combos=args.max_combos,
    )


if __name__ == "__main__":
    main()
