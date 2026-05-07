#!/usr/bin/env bash
# 50 Salads fold 1 x DiffAct softmax -> SKTR switch-penalty grid.
# Intended for tmux:
#   tmux new -s paper_50salads_fold1_beam100_switch_grid scripts/run_eval_diffact_50salads_fold1_w7_taucap8_beam100_switch_grid.sh
set -euo pipefail

cd /home/dsi/eli-bogdanov/sktr_for_long_traces

export PYTHONUNBUFFERED=1

for PENALTY in 0.05 0.10 0.25 0.50; do
  PENALTY_LABEL="${PENALTY/./p}"
  OUT="results/paper_diffact_50salads_fold1_w7_taucap8_beam100_switch${PENALTY_LABEL}_topm1_topk3_chunk11"
  mkdir -p "$OUT"

  {
    echo "START:$(date -Is)"
    echo "SWITCH_PENALTY:${PENALTY}"
    python -u eval_diffact_sktr_fold1_paper.py \
      --datasets 50salads \
      --fold 1 \
      --workers 7 \
      --inner-parallel \
      --top-m 1 \
      --candidate-top-k 3 \
      --chunk-size 11 \
      --conformance-switch-penalty-weight "$PENALTY" \
      --restrict-model-moves-to-tau \
      --max-consecutive-tau-moves 8 \
      --dijkstra-beam-width 100 \
      --progress-log-interval-chunks 25 \
      --enabled-cache-size 100000 \
      --sktr-log-level INFO \
      --data-root "${DATA_ROOT:-/home/dsi/eli-bogdanov/data/data}" \
      --out-dir "$OUT"
    echo "EXIT:$?"
    echo "END:$(date -Is)"
  } >> "${OUT}/run.log" 2>&1
done
