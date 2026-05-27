"""Prepare multipart-pipeline output for the existing GRPO training infrastructure.

Bridges the new multi-part schema (parts[] with per-part consensus) to the
single-prompt/single-reference format that grpo-pipeline expects.

Strategy: render the full multi-part problem as one prompt; use the FINAL
part's consensus answer as the reference. The existing boxed_match reward
extracts the last \\boxed{} in the model's response, which corresponds to
the final part's answer — and getting the final answer right typically
requires getting earlier parts right (cumulative dependency), so binary
boxed_match is a reasonable proxy for the multi-part k/m reward.

Usage:
  python scripts/prep_multipart_for_grpo.py \\
      --keeps ttt_binary/data/subproblems/conics-multipart-full.keeps.jsonl \\
      --skips ttt_binary/data/subproblems/conics-multipart-full.skips.jsonl \\
      --band 0.25,0.75 --max-unparseable 5 --ambiguity-threshold 0.20 \\
      --out runs/conics-multipart-band25-75/subproblems.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _well_posed(rec: dict, *, ambiguity_threshold: float, max_unparseable: int) -> bool:
    parts = rec.get("parts") or []
    if not parts:
        return False
    for p in parts:
        if p.get("consensus_answer") is None:
            return False
        if p.get("n_unparseable", 0) > max_unparseable:
            return False
        if p.get("p2", 0.0) >= ambiguity_threshold:
            return False
    return True


def _render_prompt(parts: list[dict]) -> str:
    """Render a multipart problem as a single training prompt.

    Each part's text is preserved verbatim. We append a unified instruction
    asking the model to solve all parts in order and end with the FINAL
    part's answer in \\boxed{} so that boxed_match scores against the last
    consensus answer."""
    lines: list[str] = []
    for p in parts:
        lines.append(f"Part ({p['label']}). {p['text']}")
    lines.append(
        "\nSolve all parts in order, showing all important reasoning. "
        "Each part's intermediate answer must be computed before the next. "
        "After completing every part, state the FINAL part's answer in "
        "\\boxed{X.XXXX} as the very last expression of your response."
    )
    return "\n\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keeps", required=True)
    ap.add_argument("--skips", required=True)
    ap.add_argument("--band", default="0.25,0.75",
                    help="Comma-separated 'lo,hi' for r_bar acceptance")
    ap.add_argument("--ambiguity-threshold", type=float, default=0.20)
    ap.add_argument("--max-unparseable", type=int, default=5)
    ap.add_argument("--out", required=True,
                    help="Output JSONL (one {prompt, reference, meta} per accepted combo)")
    args = ap.parse_args()

    band_lo, band_hi = (float(x) for x in args.band.split(","))
    keeps = [json.loads(l) for l in Path(args.keeps).read_text().splitlines() if l.strip()]
    skips = [json.loads(l) for l in Path(args.skips).read_text().splitlines() if l.strip()]
    all_records = keeps + skips
    print(f"Loaded {len(keeps)} keeps + {len(skips)} skips = {len(all_records)} records")

    # For each combo, find the BEST attempt that satisfies the candidate band
    # AND is well-posed. "Best" = closest to band center (0.5).
    by_combo: dict[int, list[dict]] = defaultdict(list)
    for r in all_records:
        c = r.get("combo_idx")
        if c is None:
            continue
        rb = r.get("r_bar")
        if not isinstance(rb, (int, float)):
            continue
        if not (band_lo <= rb <= band_hi):
            continue
        if not _well_posed(r, ambiguity_threshold=args.ambiguity_threshold,
                           max_unparseable=args.max_unparseable):
            continue
        by_combo[c].append(r)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    rewards_preview: list[float] = []
    with out_path.open("w") as f:
        for combo_idx in sorted(by_combo):
            candidates = by_combo[combo_idx]
            best = min(candidates, key=lambda r: abs((r.get("r_bar") or 0.5) - 0.5))
            parts = best["parts"]
            final_part = parts[-1]
            reference = str(final_part["consensus_answer"])
            prompt = _render_prompt(parts)

            row = {
                "prompt": prompt,
                "reference": reference,
                # Metadata for diagnostics — not used by load_problems' core fields
                "meta": {
                    "combo_idx": combo_idx,
                    "skills_used": best.get("skills_used"),
                    "r_bar": best.get("r_bar"),
                    "per_part_consensus": [
                        {"label": p["label"], "answer": p["consensus_answer"]}
                        for p in parts
                    ],
                    "per_part_p1": [
                        {"label": p["label"], "p1": p["p1"]} for p in parts
                    ],
                    "regeneration_attempts": best.get("regeneration_attempts"),
                    "source_status": best.get("status"),
                },
            }
            f.write(json.dumps(row) + "\n")
            n_written += 1
            rewards_preview.append(best["r_bar"])

    print(f"\nBand: [{band_lo}, {band_hi}]")
    print(f"Per-part filters: ambiguity<{args.ambiguity_threshold}, "
          f"unparseable<={args.max_unparseable}")
    print(f"Wrote {n_written} prompts to {out_path}")
    if rewards_preview:
        rewards_preview.sort()
        print(f"r_bar of selected: min={rewards_preview[0]:.2f} "
              f"max={rewards_preview[-1]:.2f} "
              f"median={rewards_preview[len(rewards_preview)//2]:.2f}")


if __name__ == "__main__":
    main()
