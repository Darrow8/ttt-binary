"""Near-duplicate removal for a keeps.json produced by taxonomy_generation.py.

Embeds every kept subproblem with a modern sentence-transformer, computes
pairwise cosine similarity, and greedy-drops any item whose similarity to a
previously-kept item exceeds the threshold. Preserves the input JSON shape
but writes to a separate output path so the input is never destroyed.

Two dedup modes:
  - global (default): dedupe across all subproblems regardless of skill.
  - per-skill: dedupe within each skill independently. Use when near-
    duplicates ACROSS different skills should be kept (different skills
    testing adjacent techniques may naturally produce similar-looking
    subproblems and that's fine -- they test different things).

Requires sentence-transformers:
    pip install sentence-transformers

Default embedding model is BAAI/bge-large-en-v1.5. Falls back to
all-mpnet-base-v2 if bge isn't cached and you pass --fast.

Usage:
    python -m Stage1.dedupe_subproblems \\
        --input  runs/conics-tangent-5/stage1_taxonomy/LATEST/keeps.json \\
        --output runs/conics-tangent-5/stage1_taxonomy/LATEST/keeps_deduped.json \\
        --threshold 0.88
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict


DEFAULT_THRESHOLD = 0.88
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
FAST_MODEL = "sentence-transformers/all-mpnet-base-v2"


def _load_st(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(
            "ERROR: sentence-transformers is not installed. Install with:\n"
            "  pip install sentence-transformers\n"
            f"(underlying error: {e})",
            file=sys.stderr,
        )
        sys.exit(2)
    print(f"loading embedding model: {model_name}", flush=True)
    return SentenceTransformer(model_name)


def _embed(model, texts: list[str]):
    import numpy as np
    embs = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embs, dtype=np.float32)


def _greedy_dedupe(embs, threshold: float) -> list[int]:
    """Return indices of items to KEEP. First-seen wins.

    embs is assumed L2-normalized, so cosine == dot product.
    """
    import numpy as np
    n = embs.shape[0]
    keep_mask = np.ones(n, dtype=bool)
    kept_indices: list[int] = []
    for i in range(n):
        if not keep_mask[i]:
            continue
        kept_indices.append(i)
        if i + 1 >= n:
            break
        sims = embs[i + 1:] @ embs[i]
        dup_local = np.where(sims >= threshold)[0]
        for j_local in dup_local:
            keep_mask[i + 1 + j_local] = False
    return kept_indices


def dedupe(
    *,
    input_path: str,
    output_path: str,
    threshold: float,
    model_name: str,
    per_skill: bool,
) -> dict:
    with open(input_path) as f:
        data = json.load(f)
    problems = data.get("problems", [])
    if not problems:
        print("input has no problems; writing passthrough", file=sys.stderr)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return {"input": len(problems), "kept": 0, "dropped": 0}

    model = _load_st(model_name)

    if per_skill:
        by_skill: dict[str, list[int]] = defaultdict(list)
        for i, p in enumerate(problems):
            by_skill[p.get("skill", "__unknown__")].append(i)
        keep_set: set[int] = set()
        for skill, idxs in by_skill.items():
            texts = [problems[i]["problem"] for i in idxs]
            if len(texts) == 1:
                keep_set.add(idxs[0])
                continue
            print(f"[{skill}] embedding {len(texts)}", flush=True)
            embs = _embed(model, texts)
            local_keep = _greedy_dedupe(embs, threshold)
            for lk in local_keep:
                keep_set.add(idxs[lk])
            print(f"[{skill}] kept {len(local_keep)}/{len(texts)}", flush=True)
        kept_idx = sorted(keep_set)
    else:
        texts = [p["problem"] for p in problems]
        print(f"embedding {len(texts)} subproblems (global dedupe)", flush=True)
        embs = _embed(model, texts)
        kept_idx = _greedy_dedupe(embs, threshold)

    kept_problems = [problems[i] for i in kept_idx]
    out = {**data, "problems": kept_problems, "n_problems": len(kept_problems)}
    out["_dedupe"] = {
        "source": os.path.abspath(input_path),
        "model": model_name,
        "threshold": threshold,
        "mode": "per_skill" if per_skill else "global",
        "n_in": len(problems),
        "n_out": len(kept_problems),
        "n_dropped": len(problems) - len(kept_problems),
    }
    with open(output_path + ".tmp", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    os.replace(output_path + ".tmp", output_path)
    return {
        "input": len(problems),
        "kept": len(kept_problems),
        "dropped": len(problems) - len(kept_problems),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="path to keeps.json")
    p.add_argument("--output", required=True, help="path for deduped JSON")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="cosine similarity threshold (default 0.88)")
    p.add_argument("--per-skill", action="store_true",
                   help="dedupe within each skill independently (default: global)")
    p.add_argument("--fast", action="store_true",
                   help=f"use smaller/faster model {FAST_MODEL} instead of {DEFAULT_MODEL}")
    p.add_argument("--model", default=None,
                   help="override embedding model name (takes precedence over --fast)")
    args = p.parse_args()

    model_name = args.model or (FAST_MODEL if args.fast else DEFAULT_MODEL)
    stats = dedupe(
        input_path=args.input,
        output_path=args.output,
        threshold=args.threshold,
        model_name=model_name,
        per_skill=args.per_skill,
    )
    print(
        f"\nwrote {args.output}\n"
        f"  in  : {stats['input']}\n"
        f"  kept: {stats['kept']}\n"
        f"  drop: {stats['dropped']}\n"
        f"  threshold: {args.threshold}\n"
        f"  model: {model_name}\n"
        f"  mode : {'per_skill' if args.per_skill else 'global'}"
    )


if __name__ == "__main__":
    main()
