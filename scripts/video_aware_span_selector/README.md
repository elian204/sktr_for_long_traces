# Breakfast video-aware span selector pilot

This directory implements the leakage-safe follow-up to the Petri recovery
diagnostics.  It tests whether a learned selector can localize DiffAct's wrong
segments, and whether route-conditioned process features add anything after
ordinary video, uncertainty, class, and duration features are available.

The first pilot is deliberately narrow:

- Breakfast official outer fold 1, seed 0.
- Three subject-disjoint inner folds over the 39 outer-training participants.
- Three fresh DiffAct trajectories.  Each trajectory trains on 26 participants
  and exports predictions for the held-out 13 participants.
- The concatenated exports are out-of-fold (OOF) predictions for every one of
  the 1,460 outer-training videos.  They are used to train the selector.
- The existing release DiffAct export is used only for the 252 official outer
  test videos.  It was trained on the complete outer-training fold.
- Every duration, directly-follows, and Petri feature for an OOF video is learned
  only from that video's inner-training split.  Outer-test features use the
  complete outer-training split.

The fixed selector comparison is:

1. `base`: raw probability/confidence, duration, position, task, camera, and
   predicted-neighbour class features;
2. `base_plus_dfg`: base plus route-conditioned directly-follows features;
3. `base_plus_prefix_petri`: base plus exact route-conditioned prefix-conformance
   features.

All three use the same pre-specified histogram gradient boosting regressor and
the same complete-span ranking rule.  Results are reported at matched frame
budgets of 0.5%, 1%, 2%, 5%, and 10%.  Oracle relabeling with ground truth is a
diagnostic ceiling, not a deployable repair system.

## Create the immutable study

```bash
python scripts/video_aware_span_selector/prepare_study.py \
  --study-dir /data1/eli-bogdanov/sktr_runs/breakfast_video_selector_oof_outer1_seed0_v1
```

This writes metadata, manifests, DiffAct dataset views, configs, and detached
tmux launch wrappers.  It does not start training.

## Launch and monitor

Only launch after the selected GPUs are free:

```bash
/data1/eli-bogdanov/sktr_runs/breakfast_video_selector_oof_outer1_seed0_v1/launch_tmux.sh
/data1/eli-bogdanov/sktr_runs/breakfast_video_selector_oof_outer1_seed0_v1/status.sh
```

Each task is resumable through DiffAct's `latest.pt`.  A completed task must
contain its final epoch-1000 checkpoint and a verified held-out raw/canonical
probability export before it is marked complete.

## Analyze

```bash
/data1/eli-bogdanov/sktr_runs/breakfast_video_selector_oof_outer1_seed0_v1/analyze.sh
```

The analysis first cross-fits the selector across the three OOF folds, then
fits on all OOF segments and evaluates once on official outer fold 1.  The
pilot criterion for a process feature at the 5% frame budget is:

- outer-test error-recall improvement over `base` of at least 0.5 percentage
  points;
- positive improvement in at least two of three internal OOF folds; and
- non-negative oracle-correction changes in both Accuracy and F1@25.

Passing this one-fold pilot is evidence to scale the experiment, not a final
Breakfast result.  A final claim requires all four official outer folds.

