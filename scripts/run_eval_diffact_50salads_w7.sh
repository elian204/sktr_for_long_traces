#!/usr/bin/env bash
# 50 Salads × DiffAct softmax → SKTR (all 5 folds). Run via tmux (see project tmux rule).
# Memo: /home quota was tight during Breakfast exports — watch `df -h $HOME` while this runs.
set -euo pipefail
cd /home/dsi/eli-bogdanov/sktr_for_long_traces
OUT="results/paper_diffact_50salads_w7_topm1_topk3"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
python -u eval_diffact_sktr_fold1_paper.py \
  --datasets 50salads \
  --all-folds \
  --workers 7 \
  --inner-parallel \
  --top-m 1 \
  --candidate-top-k 3 \
  --data-root "${DATA_ROOT:-/home/dsi/eli-bogdanov/data/data}" \
  --out-dir "$OUT" \
  >> "${OUT}/run.log" 2>&1
echo "EXIT:$?" >> "${OUT}/run.log"
