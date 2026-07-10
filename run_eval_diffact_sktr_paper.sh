#!/usr/bin/env bash
# Canonical SKTR eval used by the full-data DiffAct experiments.
# Run after baselines/DiffAct/export_softmax.py has populated results/<ds>/softmax_fold{k}/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export DATA_ROOT="${DATA_ROOT:-${HOME}/data/data}"

exec python -u eval_diffact_sktr_fold1_paper.py \
  --data-root "${DATA_ROOT}" \
  --chunk-size 11 \
  --prob-threshold 1e-6 \
  --model-move-cost 1.0 \
  --state-mode topm \
  --top-m 1 \
  --candidate-top-k 3 \
  --candidate-top-p 1.0 \
  --candidate-min-k 1 \
  --conformance-switch-penalty-weight 1.0 \
  --restrict-log-moves \
  --restrict-model-moves-to-tau \
  --max-consecutive-tau-moves 8 \
  --enabled-cache-size 100000 \
  "$@"
