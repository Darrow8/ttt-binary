"""Taxonomy-first subproblem generation.

Stage 1 variant that:
  1. Decomposes a hard target problem into 10 distinct reasoning skills.
  2. Generates 10 agreement-window-passing subproblems per skill.

Coexists with Stage1/distinct_llm_prompting.py; does not replace it.

Non-negotiable constraints (see design spec):
- Model hardcoded to openai/gpt-oss-120b-maas for all three call types.
- No max_tokens on any completion call.
- No client-side timeouts.
- All problem text + reasoning traces in LaTeX.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

# Ensure repo root is on sys.path so this script can import sibling packages
# (Stage1 and downstream stages) regardless of how it was invoked.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Non-negotiable constants — do NOT parameterize on CLI.
# ---------------------------------------------------------------------------

GENERATOR_MODEL = "openai/gpt-oss-120b-maas"
TEMPERATURE = 0.7

# ---------------------------------------------------------------------------
# Defaults for CLI-tunable knobs.
# ---------------------------------------------------------------------------

N_SKILLS_DEFAULT = 10
PROBLEMS_PER_SKILL_DEFAULT = 10
MAX_CANDIDATES_PER_SKILL_DEFAULT = 100
N_SAMPLES_DEFAULT = 10
AGREE_LOW_DEFAULT = 0.60
AGREE_HIGH_DEFAULT = 0.80
GEN_WORKERS_DEFAULT = 4
MAX_WORKERS_DEFAULT = 16


@dataclass
class Skill:
    name: str
    description: str
    example_problem_hint: str
