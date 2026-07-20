# GTEA decoder-boundary-loss pilot

This pilot asks whether DiffAct's dominant GTEA boundary mistiming can be
reduced by strengthening its existing derived decoder-boundary loss. It is a
base-model intervention, not a Petri postprocessor.

## Locked first-wave design

- Dataset: GTEA official fold 1 (21 train videos, 7 untouched test videos).
- Training seed: 0.
- Fresh trajectories for `decoder_boundary_loss` in `{0.1, 0.3, 0.5, 1.0}`.
- `0.5` is the pre-registered primary treatment; the other non-baseline weights
  are exploratory.
- The four trajectories differ in exactly that loss weight. In particular,
  `boundary_smooth=1`, `soft_label=1.4`, `cond_types`, purge-3 postprocessing,
  epochs, checkpoint cadence, learning rate, and all architecture fields remain
  fixed.
- Checkpoint readouts: epochs 200, 1000, 2000, 5000, and 10000.
- Diffusion inference seeds: 0, 1, and 2 at every checkpoint. Seed 0 preserves
  the historical exporter contract (`video_seed = video_index`); seeds 1 and 2
  use `video_seed = inference_seed * 1_000_000 + video_index`.
- Physical GPU 3 only. The generated launcher exits if GPU 3 is occupied and
  never falls back to GPUs 0--2.
- Training-video order is copied from the locked low-data `frac_100` manifest,
  not from the differently ordered official split file. The generated manifest
  and the actual DiffAct train bundle are byte-checked against their locked
  counterparts before study creation succeeds.

The official test set is never used for checkpoint or hyperparameter selection.
The `0.5` primary value is fixed before launch. The seven-video fold-1 result is
a pilot, not a final multi-fold claim.

## Baseline reproduction gate

The `0.1` baseline is retrained inside this harness. Before any treatment is
started, its epoch-10000, inference-seed-0 export is compared with the locked
D100/E10000 export from the epoch-scarcity study. Both pre-purge argmax and
official purge-3 streams must satisfy:

- absolute difference no greater than 0.5 points for Accuracy, Edit, and
  F1@10/25/50; and
- frame-prediction disagreement no greater than 1%.

The serial queue uses `set -e`: a failed reconciliation stops before 0.3, 0.5,
or 1.0 is trained. This tolerance is a comparability guard, not a statistical
equivalence claim.

Failure diagnosis starts by comparing epoch-200 checkpoints: early divergence
points to a data/order/config pipeline mismatch, whereas agreement at epoch 200
followed by later divergence points more strongly to accumulated numeric drift.

## Readouts

For both raw pre-purge argmax and official purge-3 predictions:

- Accuracy, Edit, and F1@10/25/50;
- class-agnostic and transition-aware boundary F1 at +/-5 and +/-10 frames;
- signed and absolute boundary offsets under one-to-one matching within 50
  frames;
- non-background predicted/GT segment-count ratio;
- false predicted segments of length at most 3 frames;
- frame-level fixed/broke ledger against the in-harness 0.1 trajectory; and
- mean, standard deviation, minimum, and maximum over inference seeds.

The primary decision is made at epoch 10000 on the official purge-3 stream,
using the mean over inference seeds. The ladder advances to a class-specific
onset head only if weight 0.5, relative to 0.1:

1. improves class-agnostic boundary F1@10;
2. improves both Edit and F1@25;
3. has non-negative Accuracy delta;
4. does not increase segment-count ratio by more than 0.02; and
5. reduces median absolute class-agnostic boundary offset.

Failure does not prove all onset modeling is futile, but it argues against
scaling simple loss reweighting. A second training seed on `{0.1, 0.5}` would be
a separately generated immutable follow-up after this gate, not an unrecorded
extension of the first-wave directory.

## Prepare, review, and launch

Preparation writes an immutable study but starts no jobs:

```bash
python scripts/gtea_boundary_weight_pilot/prepare_study.py \
  --study-dir /data1/eli-bogdanov/sktr_runs/gtea_boundary_weight_fold1_seed0_v1
```

Review `study_metadata.json`, `tasks.json`, and all four generated configs before
launch. Then, while GPU 3 is free:

```bash
/data1/eli-bogdanov/sktr_runs/gtea_boundary_weight_fold1_seed0_v1/launch_tmux.sh
/data1/eli-bogdanov/sktr_runs/gtea_boundary_weight_fold1_seed0_v1/status.sh
```

The single detached tmux queue trains the baseline, executes the reproduction
gate, and only then runs the three treatments sequentially. Training resumes
from each trajectory's own `latest.pt`; completed exports are hash-validated
before they can be skipped.

After all tasks complete:

```bash
/data1/eli-bogdanov/sktr_runs/gtea_boundary_weight_fold1_seed0_v1/analyze.sh
```

Generated study metadata are immutable. Source, config, manifest, or protocol
changes require a new versioned study directory.
