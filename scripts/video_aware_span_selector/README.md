# Breakfast four-fold video-aware selector scale-up

This directory implements the leakage-safe scale-up of the successful
Breakfast fold-1 localization pilot. The question is whether a learned
selector can consistently identify DiffAct's wrong predicted spans across all
four official outer folds, and whether those out-of-fold predictions provide a
sound corpus for the next deployable repair-head experiment.

This is a localization study, not a repair system. Ground-truth oracle
corrections are reported only as upper bounds on what a future repair head
could gain.

## Locked design

- Breakfast official outer folds 1–4, training seed 0.
- Three subject-disjoint inner folds inside each outer-training split.
- One DiffAct trajectory per inner fold, trained for 1,000 epochs on 26
  participants and exported on the held-out 13 participants.
- Fold 1's three completed trajectories are imported from the reviewed v1
  study. Their checkpoint and every held-out export artifact are hash-verified;
  they are not retrained.
- Folds 2–4 require nine new trajectories in total.
- Each outer-fold selector is trained only on that fold's concatenated OOF
  predictions, then evaluated once on the untouched official outer-test split.
- Duration and rarity statistics for an OOF video use only its matching
  inner-training manifest. Outer-test features use only the corresponding
  official outer-training split.
- Results are reported per outer fold and after pooling the four disjoint
  official outer-test splits.

The four outer-training splits contain 1,460, 1,261, 1,279, and 1,136 videos.
Consequently, the repair corpus contains 5,136 OOF case records—not an
approximate 5,800. The official outer-test splits contain 252, 451, 433, and
576 videos and partition all 1,712 Breakfast videos.

## Selector variants

The pre-registered primary, `base`, uses exactly these 15
prediction-derived numeric features:

1. `uncertainty_mean`
2. `uncertainty_q90`
3. `entropy_mean`
4. `top1_uncertainty_mean`
5. `margin_mean`
6. `pred_probability_mean`
7. `official_override_gap_mean`
8. `duration_raw_abs_z`
9. `duration_norm_abs_z`
10. `segment_log_length`
11. `segment_fraction`
12. `video_progress_mid`
13. `class_frame_rarity`
14. `class_segment_rarity`
15. `neighbor_class_rarity`

`task__*` and `camera__*` one-hot columns are deliberately excluded because
filename metadata may be unavailable at deployment. The runtime analysis fails
closed if either prefix enters the primary feature list.

`base_plus_metadata` is an analysis-only ablation that adds those task and
camera columns.

`base_plus_shape` is a pre-specified exploratory, analysis-only comparison. It
adds five features computed entirely from the raw probability matrix and the
pre-purge argmax stream:

1. `confidence_slope`: least-squares slope of the framewise top-1 probability
   against normalized position within the segment;
2. `edge_vs_core_margin`: mean p1−p2 margin in the outer approximately 10% at
   both ends minus its mean in the remaining core;
3. `flicker_rate`: fraction of adjacent raw-argmax frame pairs that disagree
   inside the segment;
4. `runner_up_gap`: mean p1−p2 over the segment; and
5. `runner_up_consistency`: fraction of frames whose second-ranked label equals
   the label ranked second by segment-mean probability.

`runner_up_gap` is included exactly as requested, but is mathematically
identical to the existing base feature `margin_mean`. The implementation
asserts their equality. Thus this comparison has four genuinely new signals,
not five.

No DFG or Petri variants are run: both were rejected by the fold-1 pilot.

All variants use the same fixed histogram gradient boosting regressor and
complete-span ranking rule. Readouts are computed at matched frame budgets of
0.5%, 1%, 2%, 5%, and 10%:

- error recall and precision;
- long-substitution recall;
- oracle Accuracy gain; and
- oracle F1@25 gain.

The saved repair corpus has one row per predicted OOF segment, including the
15 deployable features, cross-fitted selector score, error targets, and the
ground-truth majority correction label. Each row also stores the top five
candidate activities ranked by segment-mean raw probability. Every rank has
the class ID, activity label, and mean probability; the corpus additionally
records the ground-truth majority label's full probability rank and whether it
is inside the candidate pool.

This candidate pool is a design constraint, not a guarantee. In the measured
outer-fold-1 5%-budget spans, ground truth was rank 2 for only 26.3% of flagged
wrong frames, within the top 3 for 39.0%, and within the top 5 for 62.3%;
median rank was 4. At segment level, the runner-up matched the ground-truth
majority label in only 18.8% of majority-wrong flagged segments. Therefore a
future repair head must consider the full top-five pool and abstain by default
when the desired label is outside it—approximately 37.7% in this diagnostic.

The corpus is a training artifact for a future repair head; none of its
ground-truth targets are used to fit or score the selector that generated the
row. The exact candidate, target, score, and feature contract is frozen in
`repair_corpus_schema.json`; analysis verifies its recorded digest before
writing the corpus.

## Pre-registered success rule

At the 5% frame budget, using official DiffAct predictions and the `base`
selector:

- every outer fold must have at least 85% error precision;
- every outer fold must have at least 15% error recall;
- the maximum minus minimum recall across folds must be at most 8 percentage
  points;
- pooled precision must be at least 85%; and
- pooled recall must be at least 18%.

Passing all five checks green-lights development of the deployable repair
head. Failure does not invalidate the descriptive localization results, but it
does block automatic advancement.

## Exploratory shape-feature rule

`base_plus_shape` must earn retention independently. At the same official
5% frame budget, compared with `base`, it must satisfy all four checks:

- pooled error-recall gain of at least 0.5 percentage points;
- positive error-recall gain in at least three of four official outer folds;
- non-negative difference in pooled oracle Accuracy gain; and
- non-negative difference in pooled oracle F1@25 gain.

Passing retains the shape features for repair-head localization. Failure drops
them and leaves the frozen 15-feature base unchanged. This secondary decision
cannot alter the primary four-fold selector gate.

## Prepare the immutable review study

```bash
python scripts/video_aware_span_selector/prepare_study.py \
  --study-dir /tmp/breakfast_selector_allfolds_v2_final_review_20260723
```

Generation creates manifests, DiffAct views/configs, fold-1 import records,
GPU waiters, queues, and immutable metadata. It never launches training.
Fable should review the staged directory before the branch is committed and
before a production study is generated.

## Production sequence after review

Commit and push the clean branch first, then generate a new immutable
production directory:

```bash
python scripts/video_aware_span_selector/prepare_study.py \
  --study-dir /data1/eli-bogdanov/sktr_runs/breakfast_video_selector_oof_allfolds_seed0_v2
```

The production launcher creates `logs/` defensively, starts one named detached
tmux waiter for each of GPUs 0–2, and requires two consecutive free occupancy
checks before a queue begins. It never falls back to another GPU.

```bash
/data1/eli-bogdanov/sktr_runs/breakfast_video_selector_oof_allfolds_seed0_v2/launch_tmux.sh
/data1/eli-bogdanov/sktr_runs/breakfast_video_selector_oof_allfolds_seed0_v2/status.sh
```

Each GPU queue trains one inner fold for outer folds 2, 3, and 4 sequentially.
Fold 1 remains imported. Completed new tasks are accepted only after the final
epoch-1000 checkpoint and every raw/canonical/official held-out export pass
validation and are hashed.

## Analyze

After all twelve tasks—three imported and nine newly trained—verify:

```bash
/data1/eli-bogdanov/sktr_runs/breakfast_video_selector_oof_allfolds_seed0_v2/analyze.sh
```

Analysis writes per-fold and pooled matched-budget tables, model/feature
leakage audits, the pre-registered decision, and the reusable OOF segment
corpus. Oracle corrections remain diagnostic and must not be presented as
deployable predictions.
