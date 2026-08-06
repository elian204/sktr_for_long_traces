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

### Phase B Option 0: author checkpoints first

Before retraining any failed cell, Phase B inventories the original authors'
released checkpoints. The official ASFormer README publishes one archive with
all 13 GTEA, 50Salads, and Breakfast fold checkpoints. The archive is frozen at
SHA-256 `7b255d8cefb90012b192aedef6f10366474acc291e3988e759a0aae3dadf5909`;
the four Breakfast checkpoints are re-exported with the untouched author code
at Git commit `e1bbe4f3ed083748f91467c51a63ac2a8b9277ad`.

The official MS-TCN++ repository currently has no releases, tags, or checkpoint
link in its README. Its linked Zenodo record contains the 30 GB feature archive,
not model weights. These negative inventory claims are saved as hash-locked
GitHub API/README snapshots; an inference study must not silently substitute a
third-party checkpoint.

Option 0 permits author-checkpoint inference only. It explicitly forbids
training and Phase C. After reconciliation it emits the residual training bill,
which requires separate explicit approval before any training launcher exists.

### Phase B residual MS-TCN++ retraining

Fable approved the residual 13 official-config trainings on 2026-08-01 after
Option 0 reproduced author-ASFormer Breakfast to a maximum absolute paper-row
delta of 0.036 percentage points and verified that no official MS-TCN++ weights
are available. The production matrix is MS-TCN++ × GTEA folds 1–4, 50Salads
folds 1–5, and Breakfast folds 1–4. Every fold uses the untouched authors'
configuration: 100 epochs, batch size 1, learning rate 5e-4, 64 feature maps,
11 prediction-generation layers, 10 refinement layers, and 3 refinements.

The study copies the clean official source at Git `f423a9e` into an isolated
runtime per fold, hash-locks every feature/GT/split input, and exports the final
stage probabilities. That Git revision contains one stray `MS_TCB` token before
`MS_TCN2.__init__`, so the runtime applies an exact one-line syntax-only repair;
the architecture, loss, optimizer, and training loop are unchanged. Four queues
use two consecutive free-GPU checks and never preempt another process. Phase C
stays closed until the resulting paper reconciliation receives a separate
review.

### Phase B Step 0: validation-only checkpoint selection

The epoch-100 reconciliation failure is treated as a scientific protocol issue,
not a cosmetic discrepancy. All 100 checkpoints exist for each of the 13
MS-TCN++ cells. Breakfast degrades sharply late in training while GTEA and
50Salads are comparatively stable; the authors have also confirmed that their
published MS-TCN++ rows selected the best epoch on the test set. We do not
repeat that test-selected protocol.

Before any probe runs, this package fixes a deterministic validation-only rule.
Within each official training fold, cases are ranked by
`sha256(cross-backbone-step0-carve-v1|dataset|fold|case_id)` and the first
`ceil(0.15*N)` form the carve. Original bundle order is retained inside both
partitions. Existing checkpoints are evaluated on the carve at epochs 5, 10,
..., 95, 100 plus 96–99. The checkpoint composite is the arithmetic mean of
Edit, F1@10, and F1@25, with earlier epoch winning an exact tie.

A Breakfast fold is informative only if its composite peaks no later than epoch
85 and drops by at least 5 percentage points by epoch 100. If all four
Breakfast folds are informative, Branch A selects each of all 13 cells by that
carved-validation composite using the existing checkpoints. This branch is
explicitly labeled as a seen-video diagnostic because carve videos participated
in those trainings. Otherwise Branch B retrains all 13 cells on train-minus-
carve and performs genuine held-out validation selection with the official
configuration otherwise unchanged. No test prediction or metric can enter
either branch decision or checkpoint selection.

Separately, the same checkpoint grid is evaluated on the official test folds as
a descriptive trajectory exhibit. It lives in a separate namespace, carries
checkpoint hashes, and is firewalled from the selection finalizer. Published
numbers remain the reconciliation reference; the trajectory only explains the
observed deviation.

The future Phase-C loader must hard-error on any input under
`~/cross_backbone_pred_cache/`. It accepts only digest-recorded Phase-B selected
exports and author-release exports. Phase C additionally pre-registers absolute
error rates per GT segment and per minute, Breakfast checkpoint sensitivity at
selected/epoch-100/epoch-30 for MS-TCN++ and ASFormer, and the disclosure that
ASFormer/DiffAct use author-released checkpoints while MS-TCN++ is selected by
our validation protocol. ASFormer's selected sensitivity reference is its
author-release epoch-120 checkpoint.

The staged Step-0 study is review-only: all inference, conditional training,
selected-export, and Phase-C permissions default to false. V1/V2 repair studies
remain closed and immutable.

## Phase C: pre-registered cross-backbone audit sweep

R6 accepted the selected-checkpoint reconciliation and authorized **staging**, not
execution. The review package therefore keeps `phase_c_allowed=false`,
`asformer_materialization_allowed=false`, and `audit_execution_allowed=false`.
It contains no approval digest, launches no process, and is not committed before
the digest review. All definitions below are frozen before the audit sees a
Phase-C result.

### Headline matrix and source asymmetry

The headline matrix is MS-TCN++ → ASFormer → DiffAct on all official folds of
GTEA, 50Salads, and Breakfast:

- MS-TCN++ uses the Phase-B checkpoints selected only on the genuine carved
  training validation set. These are our official-config retrainings on
  train-minus-carve.
- ASFormer uses the authors' released epoch-120 checkpoints. The already
  accepted Option-0 Breakfast exports are consumed directly; the missing GTEA
  and 50Salads exports are inference-only materializations from the same
  hash-locked official archive.
- DiffAct uses the existing official-release probability matrices and official
  postprocessed predictions. The audit records their disagreement with
  probability argmax rather than silently replacing the official prediction.

This asymmetry is disclosed in every report. Local ASFormer Breakfast epoch-30
and epoch-100 checkpoints are descriptive sensitivity arms only and are loaded
with the separately hash-locked local source that produced their registered
attention-mask buffers. They never enter the headline generation test. The
original full-train epoch-100 MS-TCN++
exports for GTEA and 50Salads are likewise a pre-registered robustness
comparator, not a replacement for the selected primary.

### Exclusive error taxonomy

The four buckets use this fixed precedence and partition every wrong frame:

1. `boundary_offset`: a wrong frame within ±25 analysis frames of a GT
   transition whose predicted label is one of the two labels adjacent to that
   transition. Widths 10 and 50 are reported as orthogonal sensitivity
   readouts.
2. `fragmentation`: a remaining wrong frame in an internal predicted island of
   at most 25 frames, fully contained within one GT segment and bounded on both
   sides by that GT label.
3. `illegal_order`: a remaining wrong frame in the destination predicted
   segment of an illegal start or internal transition in the fold-pure training
   DFG, or in an illegal final segment.
4. `legal_substitution`: every remaining wrong frame. This bucket is split
   descriptively into predicted labels that occur elsewhere in the video's GT
   (`present_nonadjacent`) and labels absent from that video's GT.

The DFG is discovered separately for each official fold from full **training GT
only**, after applying that dataset's evaluation sample rate and collapsing
runs. Its legal starts, internal directly-follows edges, and legal ends are all
checked. The per-case invariant
`boundary_offset + fragmentation + illegal_order + legal_substitution = all
wrong frames` is a hard failure, not a report warning.

Orthogonal readouts reproduce the original taxonomy harness: wrong-label span
mass, long homogeneous substitutions (error span length ≥100 and modal
prediction ≥90%), illegal internal/start/end counts, over-segmentation ratio,
and the GT class's best rank under probability ties (top-2/3/5, median, p90).

### Aggregation and hypothesis test

Every fold and dataset table places two conventions side by side:

- `frame_weighted`: frame or segment numerators and denominators are pooled
  before division; accuracy and segmental F1 use their conventional pooled
  definitions, while Edit retains the conventional unweighted video mean.
- `per_video_macro`: the unweighted mean of each per-video readout. Error-share
  means exclude zero-error videos and explicitly report that denominator.

For each exclusive bucket, both tables include counts and shares plus absolute
rates in frames and contiguous bucket spans per GT segment and per minute. Time
uses the fixed 15-fps analysis convention. Per-video, per-fold, and pooled
tables remain available so fold spread is visible.

The primary model-generation hypothesis has three endpoints across the 13
official dataset-fold pairs: fragmentation frames/min decreases, illegal-order
frames/min decreases, and legal-substitution share increases from MS-TCN++ to
DiffAct. Each endpoint uses a paired, one-sided exact sign test on DiffAct minus
MS-TCN++; Holm correction covers the three endpoints. Support requires all
three adjusted p-values ≤0.05 and all three pooled directions to agree. The
three-backbone monotonic rows and per-dataset directions are descriptive and
remain visible whether or not the primary rule passes.

Breakfast checkpoint sensitivity recomputes the complete taxonomy for
selected/epoch-100/epoch-30 MS-TCN++ and ASFormer (where ASFormer's selected arm
is the author epoch-120 model). GTEA and 50Salads separately repeat the taxonomy
share comparison with the original full-train epoch-100 MS-TCN++ exports.

### Provenance and releasable package

The input manifest admits only digest-recorded Phase-B selected exports,
Option-0 ASFormer artifacts, the official ASFormer archive/source, official
DiffAct exports, and official data assets. Any path resolving under
`~/cross_backbone_pred_cache/` hard-fails, including through a symlink. The
staged `release/audit_suite/` contains one copy of each source/test file plus the
frozen config, manifest, case index, analysis-arm table, and a release digest.
After approval and execution, the results package will add per-case/fold/dataset
tables, the centerpiece generation table and hypothesis test, checkpoint
sensitivity and robustness tables, per-dataset findings, and an output digest.

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

/usr/bin/python scripts/cross_backbone_error_audit/prepare_phase_b_training.py \
  --study-dir /data1/eli-bogdanov/sktr_runs/cross_backbone_phase_b_mstcn2_v1 \
  --authorize-training --fable-approval-digest APPROVAL_DIGEST
```
