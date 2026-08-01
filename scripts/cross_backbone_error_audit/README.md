# Cross-backbone error audit

## Paper hypothesis

As sequence modeling improves from MS-TCN++ to ASFormer to DiffAct, residual
errors shift away from illegal transitions and fragmentation and toward
procedurally legal, visually confusable substitutions. The predicted consequence
is that untimed symbolic postprocessing has progressively less error mass it can
reach.

This package is the paper-spine audit. It reads existing per-frame probabilities;
it does not modify any prior study or retrain a model without a separate reviewed
Phase-B study.

## Phase A: inventory and comparability (this review package)

The inventory covers 26 legacy cells:

- MS-TCN++ and ASFormer;
- GTEA folds 1–4, Breakfast folds 1–4, and 50Salads folds 1–5.

A cell is `USABLE` only when all of the following pass:

1. its official test bundle has unique cases and complete video-index coverage;
2. its stored class mapping equals the official dataset mapping;
3. every test case has a finite `classes × time` probability matrix in `[0,1]`;
4. every frame distribution sums to one within `1e-4`;
5. the probability length equals official GT after the model's official sample
   rate (50Salads uses rate 2; the others use rate 1); and
6. the final fold checkpoint exists and is hash-locked.

Published-row reconciliation uses the papers' arithmetic mean of the official
fold metrics for Acc, Edit, and F1@10/25/50. Pooled-case metrics are retained as
an audit readout but never decide comparability. `PASS` requires
every absolute delta ≤2.0 percentage points and mean absolute delta ≤1.0.
`PASS_WITH_NOTES` requires every delta ≤3.0 and mean ≤1.5. Anything worse is a
`FAIL` and requires retraining or exclusion. These thresholds were fixed before
the study computed metrics.

The original paper rows are:

| Backbone | Dataset | Acc | Edit | F1@10 | F1@25 | F1@50 |
|---|---:|---:|---:|---:|---:|---:|
| MS-TCN++ | GTEA | 80.1 | 83.5 | 88.8 | 85.7 | 76.0 |
| MS-TCN++ | 50Salads | 83.7 | 74.3 | 80.7 | 78.5 | 70.1 |
| MS-TCN++ | Breakfast | 67.6 | 65.6 | 64.1 | 58.6 | 45.9 |
| ASFormer | GTEA | 79.7 | 84.6 | 90.1 | 88.8 | 79.2 |
| ASFormer | 50Salads | 85.6 | 79.6 | 85.1 | 83.4 | 76.0 |
| ASFormer | Breakfast | 73.5 | 75.0 | 76.0 | 70.6 | 57.4 |

Primary sources: MS-TCN++ TPAMI paper/table 16 and ASFormer BMVC paper/table 7.

## Phase B: fill only verified gaps

Phase B is empty when all 26 cells are usable and no dataset-level paper
reconciliation fails. Otherwise, only the missing or failed cells may be
retrained with the official repository config. A separate immutable study must
estimate GPU-hours and receive Fable review before launch.

The Phase-A planning check found 17 failed cells. Median adjacent-checkpoint
cadence in the historical repositories implies approximately 408.2 active
GPU-hours. With four independent lanes, the slowest historical cell implies an
ideal lower bound of about 113.1 wall-hours; operationally budget 5–6 days.
Historical checkpoint directories for those cells occupy about 25.2 GiB. This
is a planning estimate observed on 2026-08-01, not a scientific result; Phase B
must refresh it before launch. All new checkpoints must live under
`/data1/eli-bogdanov/sktr_runs` because the home filesystem lacks safe headroom.

## Phase C: pre-registered audit sweep (not run in Phase A)

The hypothesis and definitions below are fixed before Phase C opens any audit
result.

- `boundary_timing`: a wrong frame within ±25 frames of a GT transition whose
  prediction equals either adjacent GT label.
- `present_nonadjacent`: not boundary timing, and the wrong predicted label
  occurs elsewhere in that video's GT.
- `absent_label_confusion`: not boundary timing, and the predicted label is
  absent from that video's GT. These three buckets are mutually exclusive and
  partition all wrong frames.
- `wrong_label_span_mass`: error frames inside predicted segments whose strict
  GT-majority label differs from the predicted label.
- `long_substitution_share`: error frames in a contiguous wrong span of at least
  100 frames whose modal predicted label occupies at least 90% of the span.
- `illegal_transition_rate`: unseen edges in the collapsed predicted trace,
  relative to the fold-pure train-GT directly-follows graph. Start/end edges are
  reported separately from internal edges.
- `over_segmentation_ratio`: predicted non-background segment count divided by
  GT non-background segment count.
- `candidate_rank`: the GT class rank in the probability matrix at each wrong
  frame; report top-2/3/5 coverage, median, and p90.

For every headline metric and taxonomy share, Phase C reports:

1. the conventional paper aggregation;
2. a frame-weighted or segment-count-weighted aggregation as applicable; and
3. an unweighted per-video macro aggregation with fold spread.

DiffAct uses the already hash-locked official release exports from the completed
GTEA, Breakfast, and 50Salads studies; it is not regenerated. Phase C produces
the model-generation table, per-dataset findings, and a releasable manifest and
script suite only after Phase-A review.

## Governance

- Phase A is CPU-only and read-only over historical repositories.
- The legacy repositories may be dirty; the study records Git HEAD, status,
  tracked-diff digest, and hashes every consumed probability/checkpoint/source.
- No sealed study is opened or modified by Phase A.
- The generated study explicitly sets `phase_c_launch_allowed=false` and
  `gpu_training_allowed=false`.
- Any source or protocol change requires a new immutable study directory.

## Commands

```bash
/usr/bin/python scripts/cross_backbone_error_audit/prepare_study.py \
  --study-dir /data1/eli-bogdanov/sktr_runs/cross_backbone_error_audit_phase_a_review_v1

/data1/eli-bogdanov/sktr_runs/cross_backbone_error_audit_phase_a_review_v1/run_phase_a.sh
```
