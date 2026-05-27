"""Retrofit different acceptance bands on existing keeps/skips data.

Useful for the band-ablation table in the paper: rather than re-running the
full pipeline for each candidate band, take the existing per-attempt records
and re-apply the per-part well-posedness checks plus a candidate band on
r_bar. Count how many combos would have at least one accept-able attempt
under each band.

Per-attempt records carry parts[i].{p1, p2, n_unparseable, consensus_answer}
and a top-level r_bar, so we can re-decide without re-running any LLM calls.

Usage:
  python scripts/band_retrofit.py \\
      --keeps ttt_binary/data/subproblems/conics-multipart-full.keeps.jsonl \\
      --skips ttt_binary/data/subproblems/conics-multipart-full.skips.jsonl \\
      --bands "0.40,0.60 0.35,0.65 0.30,0.70 0.25,0.75 0.20,0.80" \\
      --ambiguity-threshold 0.20 --max-unparseable 5
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def _record_well_posed(rec: dict, *, ambiguity_threshold: float,
                       max_unparseable: int) -> tuple[bool, str | None]:
    """Apply per-part well-posedness checks (band-independent).

    Returns (is_well_posed, reason_if_not). Mirrors decide_multipart's
    per-part gate logic exactly."""
    parts = rec.get("parts") or []
    if not parts:
        return False, "no parts"
    for p in parts:
        if p.get("consensus_answer") is None:
            return False, f"part {p.get('label')!r}: no consensus"
        if p.get("n_unparseable", 0) > max_unparseable:
            return False, (f"part {p.get('label')!r}: unparseable="
                           f"{p['n_unparseable']} > {max_unparseable}")
        if p.get("p2", 0.0) >= ambiguity_threshold:
            return False, (f"part {p.get('label')!r}: p2={p['p2']:.2f}"
                           f" >= {ambiguity_threshold}")
    return True, None


def _r_bar(rec: dict) -> float | None:
    return rec.get("r_bar") if isinstance(rec.get("r_bar"), (int, float)) else None


def retrofit(records: list[dict], *, band_lo: float, band_hi: float,
             ambiguity_threshold: float, max_unparseable: int) -> dict:
    """Re-decide every record under a candidate band. Returns aggregate stats."""
    accepts = 0
    in_band_well_posed = 0  # would accept
    in_band_ill_posed = 0
    too_easy = 0
    too_hard = 0
    rbars_by_combo: dict[int, list[float]] = defaultdict(list)
    accepted_combos: set[int] = set()

    for rec in records:
        cidx = rec.get("combo_idx")
        rb = _r_bar(rec)
        if cidx is None or rb is None:
            continue
        rbars_by_combo[cidx].append(rb)

        well_posed, _ = _record_well_posed(
            rec, ambiguity_threshold=ambiguity_threshold,
            max_unparseable=max_unparseable,
        )
        in_band = band_lo <= rb <= band_hi

        if in_band and well_posed:
            accepts += 1
            in_band_well_posed += 1
            accepted_combos.add(cidx)
        elif in_band and not well_posed:
            in_band_ill_posed += 1
        elif rb > band_hi:
            too_easy += 1
        elif rb < band_lo:
            too_hard += 1

    return {
        "band": (band_lo, band_hi),
        "n_records": sum(len(v) for v in rbars_by_combo.values()),
        "n_combos": len(rbars_by_combo),
        "n_accepts_total_attempts": accepts,
        "n_combos_with_at_least_one_accept": len(accepted_combos),
        "n_too_easy_attempts": too_easy,
        "n_too_hard_attempts": too_hard,
        "n_in_band_but_ill_posed_attempts": in_band_ill_posed,
        "rbars_per_combo_best": {
            c: min(rs, key=lambda r: abs(r - 0.5))
            for c, rs in rbars_by_combo.items()
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keeps", required=True)
    ap.add_argument("--skips", required=True)
    ap.add_argument("--bands", default="0.40,0.60 0.35,0.65 0.30,0.70 0.25,0.75 0.20,0.80",
                    help="Space-separated 'lo,hi' pairs")
    ap.add_argument("--ambiguity-threshold", type=float, default=0.20)
    ap.add_argument("--max-unparseable", type=int, default=5)
    args = ap.parse_args()

    keeps = [json.loads(l) for l in Path(args.keeps).read_text().splitlines() if l.strip()]
    skips = [json.loads(l) for l in Path(args.skips).read_text().splitlines() if l.strip()]
    records = keeps + skips
    print(f"Loaded {len(keeps)} keeps + {len(skips)} skips = {len(records)} records")

    bands = []
    for tok in args.bands.split():
        lo, hi = tok.split(",")
        bands.append((float(lo), float(hi)))

    print(f"\nApplied per-part filters: ambiguity_threshold={args.ambiguity_threshold}, "
          f"max_unparseable={args.max_unparseable}")
    print()
    print(f"{'BAND':<14} {'COMBOS':<10} {'ACCEPT_RATE':<14} "
          f"{'TOTAL_ATTEMPTS':<16} {'IN_BAND_ILLPOSED':<18} "
          f"{'TOO_EASY':<10} {'TOO_HARD':<10}")
    for lo, hi in bands:
        s = retrofit(records, band_lo=lo, band_hi=hi,
                     ambiguity_threshold=args.ambiguity_threshold,
                     max_unparseable=args.max_unparseable)
        nc = s["n_combos"]
        na = s["n_combos_with_at_least_one_accept"]
        rate = na / nc if nc else 0.0
        print(f"[{lo:.2f},{hi:.2f}]   {na}/{nc:<8} {rate:>7.0%}        "
              f"{s['n_accepts_total_attempts']:<16} "
              f"{s['n_in_band_but_ill_posed_attempts']:<18} "
              f"{s['n_too_easy_attempts']:<10} {s['n_too_hard_attempts']:<10}")

    # Show how many combos move IN to band as we widen
    print()
    print("Combo r_bar (best-attempt-closest-to-0.5) distribution:")
    base = retrofit(records, band_lo=0.0, band_hi=1.0,
                    ambiguity_threshold=args.ambiguity_threshold,
                    max_unparseable=args.max_unparseable)
    rbars_best = list(base["rbars_per_combo_best"].values())
    rbars_best.sort()
    if rbars_best:
        print(f"  n_combos={len(rbars_best)} mean={statistics.mean(rbars_best):.2f} "
              f"median={statistics.median(rbars_best):.2f}")
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        labels = [f"[{bins[i]:.1f},{bins[i+1]:.1f})" for i in range(len(bins)-1)]
        counts = [0] * (len(bins) - 1)
        for r in rbars_best:
            for i in range(len(bins) - 1):
                if bins[i] <= r < bins[i+1]:
                    counts[i] += 1
                    break
        print("  Histogram of best-r_bar per combo:")
        for lab, ct in zip(labels, counts):
            bar = '#' * ct
            print(f"    {lab:<12} {ct:3d}  {bar}")


if __name__ == "__main__":
    main()
