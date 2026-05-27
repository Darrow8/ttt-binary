#!/usr/bin/env bash
# End-to-end v4 pipeline: generate subproblems → prep for grpo → train.
# Skills are pre-written at ttt_binary/data/skills/conics-multipart-v4-corrected.json
# so stage 1 is skipped.

set -euo pipefail

cd /home/ubuntu/ttt-binary

PID="conics-multipart-v4-corrected"
RUN_NAME="conics-multipart-v4-corrected-band25-75"
SUBPROBLEMS_OUT="runs/${RUN_NAME}/subproblems.jsonl"
LOG_DIR="runs/${RUN_NAME}/grpo"
PIPELINE_LOG_DIR="logs/v4-pipeline"

mkdir -p "runs/${RUN_NAME}" "${LOG_DIR}" "${PIPELINE_LOG_DIR}"

echo "=========================================="
echo "  v4 pipeline launched at $(date -Iseconds)"
echo "  problem_id:  ${PID}"
echo "  run_name:    ${RUN_NAME}"
echo "=========================================="

# ---------- Stage 3 + 4: generate subproblems and solutions ----------
echo
echo "[$(date -Iseconds)] >>> stage 3+4: generate subproblems via run_pipeline.py"
python -m ttt_binary.pipeline.run_pipeline \
    --problem-id "${PID}" \
    --problem-file data/target-problems/conics.txt \
    --skip-stage1 \
    2>&1 | tee "${PIPELINE_LOG_DIR}/01_subproblem_gen.log"

# ---------- Prep for GRPO: render multipart -> single prompt+reference ----------
echo
echo "[$(date -Iseconds)] >>> prep multipart for GRPO (band 0.25-0.75)"
python scripts/prep_multipart_for_grpo.py \
    --keeps "ttt_binary/data/subproblems/${PID}.keeps.jsonl" \
    --skips "ttt_binary/data/subproblems/${PID}.skips.jsonl" \
    --band 0.25,0.75 \
    --max-unparseable 5 \
    --ambiguity-threshold 0.20 \
    --out "${SUBPROBLEMS_OUT}" \
    2>&1 | tee "${PIPELINE_LOG_DIR}/02_prep_grpo.log"

# Sanity: make sure we have something to train on
N_TRAIN=$(wc -l < "${SUBPROBLEMS_OUT}")
echo
echo "[$(date -Iseconds)] >>> ${N_TRAIN} subproblems prepared at ${SUBPROBLEMS_OUT}"
if [ "${N_TRAIN}" -lt 5 ]; then
    echo "ERROR: too few subproblems (${N_TRAIN}) to train. Aborting before Tinker spend." >&2
    exit 1
fi

# ---------- GRPO training ----------
echo
echo "[$(date -Iseconds)] >>> launching GRPO training (50 epochs, max_tokens=30000)"
cd grpo-pipeline
python -m pipeline.train_one \
    --id "${RUN_NAME}" \
    --subproblems "../${SUBPROBLEMS_OUT}" \
    --log-dir "../${LOG_DIR}" \
    --epochs 50 \
    --max-tokens 30000 \
    2>&1 | tee "../${PIPELINE_LOG_DIR}/03_grpo_train.log"

echo
echo "[$(date -Iseconds)] >>> v4 pipeline complete"
