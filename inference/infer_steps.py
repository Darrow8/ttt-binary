"""Sample the tinker model at training steps 20, 40, 60, 80, 100 and save results.

Usage:
    # Auto-detect the UUID from the latest training checkpoint:
    python inference/infer_steps.py

    # Explicitly supply the training-run UUID:
    python inference/infer_steps.py --run-uuid <UUID>

    # Override run name (default: subproblems-run) or sample count:
    python inference/infer_steps.py --run-name subproblems-run --n-samples 50

    # Use a custom set of steps:
    python inference/infer_steps.py --steps 20 40 60 80 100
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Re-use helpers from sibling infer.py
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from infer import run_local, _get_tinker_service, DEFAULT_PROBLEM  # noqa: E402

DEFAULT_STEPS = [20, 40, 60, 80, 100]
DEFAULT_RUN_NAME = "subproblems-run"
DEFAULT_N_SAMPLES = 500


def _resolve_run_uuid(service, run_name: str) -> str:
    """Return the UUID for *run_name* by inspecting saved checkpoints."""
    rest = service.create_rest_client()
    response = rest.list_user_checkpoints(limit=200).result()
    for ckpt in response.checkpoints:
        path: str = ckpt.tinker_path or ""
        # path looks like: tinker://<uuid>:train:0/weights/<run_name>.ckpt-<step>
        if f"/{run_name}.ckpt-" in path:
            # Extract the UUID portion
            # e.g. "tinker://abc123:train:0/weights/..."  →  "abc123"
            after_scheme = path.removeprefix("tinker://")
            uuid = after_scheme.split(":")[0]
            return uuid
    sys.exit(
        f"Could not find any checkpoint for run '{run_name}'. "
        "Pass --run-uuid explicitly or check that training has been run."
    )


def _checkpoint_path(uuid: str, run_name: str, step: int) -> str:
    return f"tinker://{uuid}:train:0/weights/{step:06d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local tinker inference for each of the specified training steps.",
    )
    parser.add_argument(
        "--run-uuid", type=str, default=None,
        help="UUID of the training run (auto-detected if omitted)",
    )
    parser.add_argument(
        "--run-name", type=str, default=DEFAULT_RUN_NAME,
        help=f"Training run name used in checkpoint labels (default: {DEFAULT_RUN_NAME})",
    )
    parser.add_argument(
        "--steps", type=int, nargs="+", default=DEFAULT_STEPS,
        help=f"Training steps to evaluate (default: {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--n-samples", type=int, default=DEFAULT_N_SAMPLES,
        help=f"Number of samples per step (default: {DEFAULT_N_SAMPLES})",
    )
    args = parser.parse_args()

    service = _get_tinker_service()

    run_uuid = args.run_uuid or _resolve_run_uuid(service, args.run_name)
    print(f"Run UUID:  {run_uuid}")
    print(f"Run name:  {args.run_name}")
    print(f"Steps:     {args.steps}")
    print(f"Samples:   {args.n_samples}")
    print()

    for step in args.steps:
        ckpt = _checkpoint_path(run_uuid, args.run_name, step)
        print(f"\n{'='*60}")
        print(f"  Step {step:>4d}  →  {ckpt}")
        print(f"{'='*60}\n")
        run_local(DEFAULT_PROBLEM, args.n_samples, checkpoint=ckpt)

    print("\nAll steps complete.")


if __name__ == "__main__":
    main()
