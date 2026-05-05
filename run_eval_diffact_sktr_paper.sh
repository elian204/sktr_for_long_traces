#!/usr/bin/env bash
# SKTR eval on DiffAct softmax bundles (paper-style search: top-m=1, candidate top-k=3).
# Run after baselines/DiffAct/export_softmax.py has populated results/<ds>/softmax_fold{k}/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export DATA_ROOT="${DATA_ROOT:-${HOME}/data/data}"

exec python -u eval_diffact_sktr_fold1_paper.py \
  --data-root "${DATA_ROOT}" \
  --state-mode topm \
  --top-m 1 \
  --candidate-top-k 3 \
  "$@"
