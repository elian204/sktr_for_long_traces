# Low-data DiffAct + Petri-net experiments

This directory contains the reproducible low-data pipeline for creating nested
training subsets, training and exporting DiffAct models, applying Petri-net
postprocessing, and aggregating results.

## Workflow

Create and verify deterministic manifests first:

```bash
python scripts/low_data_petri_diffact/create_low_data_splits.py \
  --dataset 50salads \
  --fold 1 \
  --experiment-dir results/low_data_petri_diffact_ablation

python scripts/low_data_petri_diffact/verify_low_data_manifests.py \
  --experiment-dir results/low_data_petri_diffact_ablation
```

Preview the full pipeline without launching training:

```bash
python scripts/low_data_petri_diffact/run_low_data_pipeline.py \
  --experiment-dir results/low_data_petri_diffact_ablation \
  --max-runs 1
```

Add `--execute` only after reviewing the generated
`pipeline_commands.sh`. Long GPU runs should be launched in a named detached
tmux session.

## Postprocessing methods

- `petri_conformance` is the canonical SKTR path used by the full-data DiffAct
  experiments and is the orchestrator default.
- `petri_transition_viterbi` is a separate scalable baseline for experiments
  where exact conformance is too slow.

The conformance preset uses chunk size 11, top-M state mode with M=1,
candidate top-K filtering with K=3, restricted log and model moves, and a
maximum of eight consecutive tau moves. Dataset-specific conditioning
parameters are selected in `run_petri_postprocessing_low_data.py`.

## Outputs

Experiment manifests, checkpoints, exported probabilities, commands, metrics,
and reports are written beneath `--experiment-dir`. The repository ignores
`results/`, model files, and logs so generated artifacts are not accidentally
committed. Case outputs include `postprocess_seconds` when exact per-case timing
is available.
