"""
run_pipeline.py — top-level driver for the multi-problem TTT pipeline.

Subcommands map 1:1 onto the per-stage scripts under pipeline_stages/, plus
parallelization across hard-problem ids via a process pool.

Usage examples::

    python run_pipeline.py stage0 --ids all
    python run_pipeline.py stage0 --ids conics-tangent-5,ord-density --workers 2
    python run_pipeline.py stage1 --ids all --workers 4 --gen-workers 8
    python run_pipeline.py stage2 --ids all
    python run_pipeline.py stage3 --ids all
    python run_pipeline.py stage4 --ids all --target 50
    python run_pipeline.py stage5 --ids all
    python run_pipeline.py stage5-shared --ids id1,id2,id3
    python run_pipeline.py train --id conics-tangent-5 --epochs 50
    python run_pipeline.py eval --id conics-tangent-5 --adapter base --n 500

`--ids all` reads ids from problems/hard_problems.jsonl. `--workers K` spawns
K subprocesses to run independent ids in parallel. The intra-stage parallelism
(e.g. Stage 1 gen-workers, Stage 3 judge workers) is unaffected.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PROBLEM_SET = REPO_ROOT / "problems" / "hard_problems.jsonl"
STAGES_DIR = REPO_ROOT / "pipeline_stages"
GRPO_DIR = REPO_ROOT / "grpo-pipeline"


def _all_ids() -> list[str]:
    if not PROBLEM_SET.exists():
        raise FileNotFoundError(f"Missing {PROBLEM_SET}")
    ids: list[str] = []
    with open(PROBLEM_SET) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ids.append(row["id"])
    return ids


def _resolve_ids(arg: str) -> list[str]:
    if arg == "all":
        return _all_ids()
    return [s.strip() for s in arg.split(",") if s.strip()]


def _run_subproc(cmd: list[str], *, label: str) -> tuple[str, int]:
    print(f"\n>>> [{label}] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return label, result.returncode


def _fan_out(commands: list[tuple[str, list[str]]], workers: int) -> int:
    """Run (label, cmd) pairs with up to `workers` concurrent subprocesses.
    Returns the count of failed jobs."""
    failures = 0
    if workers <= 1:
        for label, cmd in commands:
            _, rc = _run_subproc(cmd, label=label)
            if rc != 0:
                failures += 1
                print(f"  [error] {label} exited {rc}")
        return failures

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_run_subproc, cmd, label=label): label for label, cmd in commands}
        for f in concurrent.futures.as_completed(futs):
            label, rc = f.result()
            if rc != 0:
                failures += 1
                print(f"  [error] {label} exited {rc}")
    return failures


# ── Stage commands ──────────────────────────────────────────────────────────

def cmd_stage0(args):
    ids = _resolve_ids(args.ids)
    cmds = [
        (f"stage0:{pid}", [
            sys.executable, str(STAGES_DIR / "stage0_collect_attempts.py"),
            "--id", pid, "--n", str(args.n), "--workers", str(args.attempt_workers),
        ])
        for pid in ids
    ]
    return _fan_out(cmds, args.workers)


def cmd_stage1(args):
    ids = _resolve_ids(args.ids)
    cmds = [
        (f"stage1:{pid}", [
            sys.executable, str(REPO_ROOT / "Stage1" / "distinct_llm_prompting.py"),
            "--id", pid,
            "--n-problems", str(args.n_problems),
            "--n-samples", str(args.n_samples),
            "--gen-workers", str(args.gen_workers),
            "--max-workers", str(args.max_workers),
        ])
        for pid in ids
    ]
    return _fan_out(cmds, args.workers)


def cmd_stage2(args):
    ids = _resolve_ids(args.ids)
    cmds = [
        (f"stage2:{pid}", [
            sys.executable, str(STAGES_DIR / "stage2_aggregate.py"),
            "--id", pid,
        ] + (["--include-skips"] if args.include_skips else []))
        for pid in ids
    ]
    return _fan_out(cmds, args.workers)


def cmd_stage3(args):
    ids = _resolve_ids(args.ids)
    cmds = [
        (f"stage3:{pid}", [
            sys.executable, str(STAGES_DIR / "stage3_filter.py"),
            "--id", pid,
            "--judge-model", args.judge_model,
            "--workers", str(args.judge_workers),
        ] + (["--skip-judge"] if args.skip_judge else []))
        for pid in ids
    ]
    return _fan_out(cmds, args.workers)


def cmd_stage4(args):
    ids = _resolve_ids(args.ids)
    cmds = []
    for pid in ids:
        cmd = [
            sys.executable, str(STAGES_DIR / "stage4_select.py"),
            "--id", pid,
            "--target", str(args.target),
            "--batch-size", str(args.batch_size),
            "--judge", args.judge,
        ]
        if args.judge_model:
            cmd += ["--judge-model", args.judge_model]
        if args.judge_checkpoint:
            cmd += ["--judge-checkpoint", args.judge_checkpoint]
        cmds.append((f"stage4:{pid}", cmd))
    return _fan_out(cmds, args.workers)


def cmd_stage5(args):
    ids = _resolve_ids(args.ids)
    cmds = [
        (f"stage5:{pid}", [
            sys.executable, str(STAGES_DIR / "stage5_export.py"),
            "--id", pid,
        ])
        for pid in ids
    ]
    return _fan_out(cmds, args.workers)


def cmd_stage5_shared(args):
    ids = _resolve_ids(args.ids)
    cmd = [
        sys.executable, str(STAGES_DIR / "stage5_export.py"),
        "--shared", "--ids", ",".join(ids),
    ]
    return _fan_out([("stage5-shared", cmd)], 1)


def cmd_train(args):
    """Run Stage 6 train_one for one id (or shared). NOT batched across ids by
    default to limit Tinker quota surprises — pass --ids and --workers for
    cross-id parallelism deliberately."""
    if args.id and args.ids:
        raise SystemExit("Pass --id <id> OR --ids <ids>, not both.")
    if args.id:
        ids = [args.id]
    elif args.ids:
        ids = _resolve_ids(args.ids)
    else:
        raise SystemExit("Pass --id <id> or --ids <ids>")

    cmds = []
    for pid in ids:
        cmd = [
            sys.executable, "-m", "pipeline.train_one",
            "--id", pid,
            "--epochs", str(args.epochs),
        ]
        if args.resume_from:
            cmd += ["--resume-from", args.resume_from]
        if args.batch_size is not None:
            cmd += ["--batch-size", str(args.batch_size)]
        if args.group_size is not None:
            cmd += ["--group-size", str(args.group_size)]
        if args.max_tokens is not None:
            cmd += ["--max-tokens", str(args.max_tokens)]
        if args.save_every is not None:
            cmd += ["--save-every", str(args.save_every)]
        cmds.append((f"train:{pid}", cmd))

    # train_one expects to be invoked from grpo-pipeline/
    if args.workers <= 1:
        failures = 0
        for label, cmd in cmds:
            print(f"\n>>> [{label}] {' '.join(cmd)}")
            rc = subprocess.run(cmd, cwd=str(GRPO_DIR)).returncode
            if rc != 0:
                failures += 1
                print(f"  [error] {label} exited {rc}")
        return failures

    def _run_in_grpo(label_cmd):
        label, cmd = label_cmd
        print(f"\n>>> [{label}] {' '.join(cmd)}")
        return label, subprocess.run(cmd, cwd=str(GRPO_DIR)).returncode

    failures = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_run_in_grpo, lc) for lc in cmds]
        for f in concurrent.futures.as_completed(futs):
            label, rc = f.result()
            if rc != 0:
                failures += 1
                print(f"  [error] {label} exited {rc}")
    return failures


def cmd_eval(args):
    if args.id and args.ids:
        raise SystemExit("Pass --id <id> OR --ids <ids>, not both.")
    if args.id:
        ids = [args.id]
    elif args.ids:
        ids = _resolve_ids(args.ids)
    else:
        raise SystemExit("Pass --id <id> or --ids <ids>")

    cmds = []
    for pid in ids:
        cmd = [
            sys.executable, str(STAGES_DIR / "stage7_eval.py"),
            "--id", pid,
            "--adapter", args.adapter,
            "--n", str(args.n),
            "--workers", str(args.eval_workers),
        ]
        if args.adapter_checkpoint:
            cmd += ["--adapter-checkpoint", args.adapter_checkpoint]
        if args.save_solutions:
            cmd += ["--save-solutions"]
        cmds.append((f"eval:{pid}:{args.adapter}", cmd))
    return _fan_out(cmds, args.workers)


# ── CLI ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TTT-binary multi-problem pipeline driver")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_ids(p):
        p.add_argument("--ids", type=str, default="all",
                       help="'all' or comma-separated ids (default: all)")
        p.add_argument("--workers", type=int, default=4,
                       help="Cross-id parallelism (process pool, default 4)")

    s0 = sub.add_parser("stage0", help="Collect base-model attempts (N=100)")
    add_ids(s0)
    s0.add_argument("--n", type=int, default=100)
    s0.add_argument("--attempt-workers", type=int, default=16,
                    help="Threads inside each Stage 0 invocation")
    s0.set_defaults(func=cmd_stage0)

    s1 = sub.add_parser("stage1", help="Generate subproblems (parallelized)")
    add_ids(s1)
    s1.add_argument("--n-problems", type=int, default=20)
    s1.add_argument("--n-samples", type=int, default=10)
    s1.add_argument("--gen-workers", type=int, default=8,
                    help="Concurrent gen+eval candidates within one id")
    s1.add_argument("--max-workers", type=int, default=16)
    s1.set_defaults(func=cmd_stage1)

    s2 = sub.add_parser("stage2", help="Aggregate Stage 1 keeps")
    add_ids(s2)
    s2.add_argument("--include-skips", action="store_true")
    s2.set_defaults(func=cmd_stage2)

    s3 = sub.add_parser("stage3", help="Filter (rounding + LLM judge)")
    add_ids(s3)
    s3.add_argument("--judge-model", type=str, default="openai/gpt-oss-20b-maas")
    s3.add_argument("--judge-workers", type=int, default=16)
    s3.add_argument("--skip-judge", action="store_true")
    s3.set_defaults(func=cmd_stage3)

    s4 = sub.add_parser("stage4", help="LLM-judge selection")
    add_ids(s4)
    s4.add_argument("--target", type=int, default=50)
    s4.add_argument("--batch-size", type=int, default=20)
    s4.add_argument("--judge", type=str, default="base", choices=["base", "tinker"])
    s4.add_argument("--judge-model", type=str, default=None)
    s4.add_argument("--judge-checkpoint", type=str, default=None)
    s4.set_defaults(func=cmd_stage4)

    s5 = sub.add_parser("stage5", help="Export per-id GRPO JSONL")
    add_ids(s5)
    s5.set_defaults(func=cmd_stage5)

    s5s = sub.add_parser("stage5-shared", help="Export shared-adapter union JSONL")
    s5s.add_argument("--ids", type=str, required=True,
                     help="Comma-separated ids (or 'all')")
    s5s.set_defaults(func=cmd_stage5_shared)

    tr = sub.add_parser("train", help="Stage 6 GRPO train per-id or shared")
    tr.add_argument("--id", type=str, default=None)
    tr.add_argument("--ids", type=str, default=None)
    tr.add_argument("--workers", type=int, default=1,
                    help="Cross-id training parallelism (DEFAULT 1 — costs Tinker quota)")
    tr.add_argument("--epochs", type=int, default=50)
    tr.add_argument("--resume-from", type=str, default=None)
    tr.add_argument("--batch-size", type=int, default=None,
                    help="Override train_one batch_size (default 25)")
    tr.add_argument("--group-size", type=int, default=None,
                    help="Override train_one group_size (default 16)")
    tr.add_argument("--max-tokens", type=int, default=None,
                    help="Override train_one max_tokens (default 16384)")
    tr.add_argument("--save-every", type=int, default=None,
                    help="Override train_one save_every (default 5)")
    tr.set_defaults(func=cmd_train)

    ev = sub.add_parser("eval", help="Stage 7 N=500 eval")
    ev.add_argument("--id", type=str, default=None)
    ev.add_argument("--ids", type=str, default=None)
    ev.add_argument("--workers", type=int, default=2,
                    help="Cross-id eval parallelism (default 2)")
    ev.add_argument("--adapter", type=str, required=True,
                    help="'base' or per-id adapter name (e.g. 'shared', 'conics-tangent-5')")
    ev.add_argument("--n", type=int, default=500)
    ev.add_argument("--eval-workers", type=int, default=16,
                    help="Threads inside each eval invocation")
    ev.add_argument("--adapter-checkpoint", type=str, default=None)
    ev.add_argument("--save-solutions", action="store_true")
    ev.set_defaults(func=cmd_eval)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    failures = args.func(args)
    if failures:
        print(f"\n{failures} subprocess(es) failed.")
        sys.exit(1)
    print("\nDone.")


if __name__ == "__main__":
    main()
