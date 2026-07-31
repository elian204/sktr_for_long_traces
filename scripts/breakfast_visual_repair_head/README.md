# Breakfast visual repair-head capstone

This CPU-only study asks whether the passed, frozen Breakfast span selector can
be paired with a simple visual classifier to produce **realized** improvements
over official DiffAct predictions on all four untouched outer test folds. It is
the Track B capstone: ground truth is used only for OOF repair-head training and
final evaluation, never for test-span selection, confidence gating, or proposal
selection.

Prototype lineage is descriptive context, not confirmatory evidence: v1
(output-only) was harmful; v2 (plus priors) gained about 0.1 Accuracy point; v3
(visual mean+std pooling) gained about 1.15 points on fold 1; v4's additional
upgrades did not beat v3's simple core. This study freezes that v3 core and
tests whether it generalizes across all official folds.

## Pre-registered primary

For each outer fold independently:

1. Train on that fold's official-mode OOF segments only. The target is the
   ground-truth-majority activity for the complete predicted segment.
2. Load the canonical I3D feature array for each video. Orientation is inferred
   solely from the unique axis of length 2048; `(2048, T)` is used directly and
   `(T, 2048)` is transposed. Missing, ambiguous, or time-misaligned arrays are
   hard failures.
3. Mean- and population-standard-deviation-pool each of the 2048 feature
   channels over the full predicted segment, producing 4096 inputs.
4. Fit an unweighted `StandardScaler` with sklearn defaults, followed by
   `LogisticRegression(C=1.0, max_iter=2000)` with all other sklearn defaults,
   matching the prototype behind the +1.15-point fold-1 reference. The logistic
   loss is weighted by segment length. This is the multinomial `lbfgs` path
   under the pinned sklearn environment; no metadata, process, selector,
   probability, or shape feature enters the head. Any `ConvergenceWarning` from
   any of the four fits aborts the study before results are accepted; it is not
   treated as a logged warning or a usable fit.
5. On that fold's untouched outer-test split, reproduce the frozen `base_score`
   selector's exact 5% frame budget. Ranking uses the original deterministic
   tie rule and selects whole predicted segments plus at most one centered
   cutoff span. Only the selected frames—not an unbudgeted remainder of a
   partially selected segment—are eligible for repair.
6. Propose the unrestricted repair-head top-1 class. Relabel an eligible span
   only when its unmodified multinomial probability is at least **tau=0.5** and
   the proposal differs from DiffAct's current segment label. Otherwise
   abstain.

The primary pooled result concatenates these four independently deployed fold
predictions; it does not rerank test segments globally across folds.

## Frozen inputs and integrity

The study consumes the passed selector study's:

- `segment_scores.csv` (frozen outer-test `base_score` and segment contract);
- `repair_training_corpus_segments.csv` (fold-specific official OOF rows);
- passed decision, schema, baseline, and budget-audit artifacts; and
- all 1,712 canonical Breakfast I3D arrays, mapping, and framewise ground truth.

Preparation records a SHA-256 for every file and the shape/orientation contract
for every I3D array. Execution re-hashes all inputs before fitting. It also
reconciles reconstructed official DiffAct metrics and the exact 5% selector
mask against the passed selector study before evaluating repairs.

## Pre-registered realized-gain gates

All checks apply to the unrestricted, tau=0.5 primary; oracle results cannot
satisfy them:

- pooled Accuracy gain at least +0.5 percentage points;
- positive Accuracy gain in at least three of four folds;
- pooled Edit gain non-negative;
- pooled F1@25 gain non-negative;
- no individual video loses more than 5 Accuracy points; and
- no single video's positive net-correct-frame contribution exceeds 50% of
  the pooled net gain.

The last check is defined on frame-count contribution, not mean per-video
Accuracy. A non-positive pooled net gain fails it automatically.

## Analysis-only readouts

- tau sensitivity at `{0.3, 0.4, ..., 0.9}`;
- a top-5-restricted proposal: choose the highest repair-head probability among
  the segment's frozen DiffAct top-five candidates, without renormalizing, then
  apply the same threshold and abstention rule;
- helped/hurt/lateral and fixed/broken-frame ledgers;
- realized gains split by wrong-majority versus right-majority flagged spans;
- per-class contributions, with `cut_fruit` and `peel_fruit` explicitly visible;
- per-video gains and the concentration/worst-video guards; and
- complete Acc/Edit/F1@10/25/50 tables per fold and pooled.

Sensitivity and top-five ablations cannot replace the primary result.

## Extensions (blocked unless the primary passes)

Only a passing primary unlocks two separate GPU studies: DiffAct-encoder feature
pooling and multi-sample diffusion proposals. Neither is implemented or launched
by this CPU study.

## Review staging

Create a non-launched review study for Fable:

```bash
python scripts/breakfast_visual_repair_head/prepare_study.py \
  --study-dir /home/dsi/eli-bogdanov/breakfast_visual_repair_head_review_v1
```

After review, commit/push first, generate a fresh clean-tree production study
under `/data1/eli-bogdanov/sktr_runs/`, and run its `run.sh`. The generated
directory is immutable with respect to protocol and inputs; only declared
runtime outputs are added.
