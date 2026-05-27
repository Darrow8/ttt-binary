#!/usr/bin/env bash
# Infer the hard conics problem on 4 GRPO checkpoints, sequentially.
#
# 1) original-best   step 40   correct=0.734
# 2) original-final  step 100
# 3) weighted-best   step 80   correct=0.884
# 4) weighted-final  step 100
#
# Each run produces a timestamped dir under runs/local_inference/, which we
# rename to a labelled dir afterwards.

set -uo pipefail
cd /home/ubuntu/ttt-binary
export PATH=/home/ubuntu/miniconda3/bin:$PATH

ORIG=tinker://0434ecac-ef87-5cd5-87e9-06df5afe7b2c:train:0/weights
WEIGHTED=tinker://492728a2-5794-5f38-be25-55a66daf3184:train:0/weights

declare -a LABELS=(orig-best-step40 orig-final weighted-best-step80 weighted-final)
declare -a CKPTS=("${ORIG}/000040" "${ORIG}/final" "${WEIGHTED}/000080" "${WEIGHTED}/final")

mkdir -p runs/infer_compare
SUMMARY=runs/infer_compare/summary_$(date +%Y%m%d_%H%M%S).txt
: > "$SUMMARY"

for i in 0 1 2 3; do
  label=${LABELS[$i]}
  ckpt=${CKPTS[$i]}
  echo "===== [$((i+1))/4] $label ====="
  echo "ckpt: $ckpt"
  pre=$(ls -1 runs/local_inference 2>/dev/null | sort | tail -1)
  python inference/infer.py --local --checkpoint "$ckpt" --n-samples 100
  rc=$?
  post=$(ls -1 runs/local_inference 2>/dev/null | sort | tail -1)
  if [[ -n "$post" && "$post" != "$pre" ]]; then
    target="runs/infer_compare/${label}"
    rm -rf "$target"
    mv "runs/local_inference/$post" "$target"
    echo "saved -> $target"
    {
      echo "----- $label (rc=$rc) -----"
      python -c "import json,sys; d=json.load(open('${target}/results.json')); s=d.get('summary',{}); print('majority:',s.get('majority_answer'),' agreement:',s.get('agreement_rate'),' n_valid:',s.get('n_valid_answers'),' n_empty:',s.get('n_empty')); print('distribution:',s.get('answer_distribution'))"
    } >> "$SUMMARY"
  else
    echo "WARN: no new output dir found for $label (rc=$rc)"
    echo "----- $label (rc=$rc) FAILED: no output -----" >> "$SUMMARY"
  fi
  echo
done

echo "=== ALL DONE ==="
cat "$SUMMARY"
