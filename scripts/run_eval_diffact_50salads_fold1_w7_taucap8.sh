#!/usr/bin/env bash
# 50 Salads fold 1 x DiffAct softmax -> SKTR with canonical bounded conformance settings.
# Intended for tmux: tmux new -s paper_50salads_fold1_taucap8 scripts/run_eval_diffact_50salads_fold1_w7_taucap8.sh
set -euo pipefail

cd /home/dsi/eli-bogdanov/sktr_for_long_traces

# Keep the historical output path so existing partial results remain resumable.
OUT="results/paper_diffact_50salads_fold1_w7_taucap8_topm1_topk3_chunk11_nobeam"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1

{
  echo "START:$(date -Is)"
  python -u eval_diffact_sktr_fold1_paper.py \
    --datasets 50salads \
    --fold 1 \
    --workers 7 \
    --inner-parallel \
    --top-m 1 \
    --candidate-top-k 3 \
    --chunk-size 11 \
    --restrict-log-moves \
    --restrict-model-moves-to-tau \
    --max-consecutive-tau-moves 8 \
    --progress-log-interval-chunks 25 \
    --enabled-cache-size 100000 \
    --sktr-log-level INFO \
    --data-root "${DATA_ROOT:-/home/dsi/eli-bogdanov/data/data}" \
    --out-dir "$OUT"
  echo "EXIT:$?"
  echo "END:$(date -Is)"
} >> "${OUT}/run.log" 2>&1
