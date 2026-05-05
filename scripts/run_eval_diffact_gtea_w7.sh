#!/usr/bin/env bash
# GTEA × DiffAct softmax → SKTR (all folds). Intended for: tmux attach -t paper_gtea_sktr
set -euo pipefail
cd /home/dsi/eli-bogdanov/sktr_for_long_traces
OUT="results/paper_diffact_gtea_w7_topm1_topk3"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
python -u eval_diffact_sktr_fold1_paper.py \
  --datasets gtea \
  --all-folds \
  --workers 7 \
  --inner-parallel \
  --top-m 1 \
  --candidate-top-k 3 \
  --data-root "${DATA_ROOT:-/home/dsi/eli-bogdanov/data/data}" \
  --out-dir "$OUT" \
  >> "${OUT}/run.log" 2>&1
echo "EXIT:$?" >> "${OUT}/run.log"
