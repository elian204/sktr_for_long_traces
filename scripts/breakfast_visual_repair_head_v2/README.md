# Breakfast visual repair head v2: harm mitigation and envelope optimization

This CPU-only nested study asks whether the v1 visual repair head can retain
realized gains while eliminating its rare catastrophic per-video failures. The
protocol has a hard two-stage contamination boundary:

1. **OOF screening** may read only each outer fold's OOF corpus, matching I3D
   arrays, and ground truth. It cannot open `segment_scores.csv` or any outer
   result from v1. It freezes one configuration independently for each outer
   fold.
2. **Outer evaluation** is generated as a separate review study. Its runner
   requires a reviewed frozen-config digest and an explicit approval record
   before it can open any outer-test row. Each outer fold is evaluated once.

The 15 v1 harmed outer-test videos are never read, identified, or listed during
development. OOF per-video outputs use one-way hashed identifiers. Fold-specific
selection is necessary: pooling OOF evidence across outer folds would let a
video affect its own fold's configuration through another fold's training set.

## Frozen base model

The visual head remains the approved v1 prototype:

- full-segment I3D channel mean + population standard deviation (4096 inputs);
- unweighted default `StandardScaler`;
- length-weighted `LogisticRegression(C=1.0, max_iter=2000)`;
- hard abort on every `ConvergenceWarning`; and
- free-choice top-1 proposal.

For OOF evaluation, the repair head is cross-fitted: for held inner fold `h`,
it trains only on the other two subject-disjoint inner folds. The selector's
already-cross-fitted `base_score` ranks spans. A 5%, 8%, or 10% budget is exact
within each outer fold, with whole spans plus at most one deterministic centered
cutoff.

## OOF-only selection procedure

All selections below occur independently within each outer fold.

### 1. Harm forensics

Reproduce the v1 rule (`tau=0.5`, 5%, plain logistic, no mitigation) on OOF and
record aggregate/quantile harm plus a hashed per-video distribution.

### 2. Harm rule at fixed tau=0.5 and 5%

Screen exactly:

- `video_cap`, `X in {1,2,3,5}` percent: among threshold-eligible selected
  spans in a video, relabel highest-head-confidence spans first, breaking ties
  by the frozen selector tie key. At most one final span is deterministically
  center-clipped to fill the cap.
- `incumbent_margin`, `delta in {0.05,0.10,0.20,0.30}`: require
  `p(proposal)-p(incumbent) >= delta` in addition to tau.
- `large_span_guard`, `Y in {2,5,10,20}` percent: when the full predicted span
  exceeds Y% of its video, require `min(0.9, tau+0.2)` confidence.

Eligible candidates have zero OOF videos below -5 Accuracy points. Select the
largest OOF Accuracy gain, then F1@25 gain, Edit gain, worst-video delta, the
listed rule order, and finally the lower numeric parameter. If no rule
is safe, outer evaluation is blocked.

### 3. Tau

With the rule frozen, choose tau from `{0.3,0.5}` at 5% using the same safety
constraint and tie order. The full `{0.3,...,0.9}` sweep is descriptive only.

### 4. Model screen

At the chosen rule/tau and 5%, compare:

- plain logistic;
- isotonic-calibrated logistic. Calibration probabilities are produced by
  swapping the two available training inner folds, fitting one on each and
  predicting the other; per-class length-weighted isotonic maps are then fit,
  the logistic is refit on both folds, and calibrated probabilities are
  renormalized; and
- a 50/50 probability average of the plain logistic and an unweighted MLP with
  `(128,)` hidden units, Adam, `max_iter=300`, early stopping, and seed 0. MLP
  weighting is unweighted because sklearn 1.4's MLP fit has no sample-weight
  contract.

A non-plain model is promoted only when it remains harm-safe and improves OOF
Accuracy by at least +0.1 point over plain. If both qualify, use the same metric
tie order. Otherwise plain is frozen. Model convergence warnings are failures.

### 5. Budget envelope

With rule, tau, and model frozen, evaluate `{5,8,10}` percent. Five percent
remains primary. A larger harm-safe budget becomes one secondary outer readout
only if it beats 5% by at least +0.15 OOF Accuracy point; choose the best gain,
then the smaller budget.

The resulting fold-specific configuration JSON contains no video identities
and is self-hashed. No outer runner accepts a different digest.

## Single-shot outer gates

The 5% primary uses the same v1 gates: pooled Delta Acc at least +0.5; positive
Acc in at least 3/4 folds; pooled Delta Edit and F1@25 non-negative; no video
below -5 Acc points; and no video supplies more than 50% of pooled net gain.
The optional larger budget is secondary and cannot rescue a failed primary.

## Operations

After tests and a clean commit, prepare and run the OOF-only study. This is
allowed before outer review:

```bash
python scripts/breakfast_visual_repair_head_v2/prepare_study.py \
  --study-dir /data1/eli-bogdanov/sktr_runs/breakfast_visual_repair_head_oof_screen_v2
/data1/eli-bogdanov/sktr_runs/breakfast_visual_repair_head_oof_screen_v2/run_oof.sh
```

Then prepare—but do not approve or run—the outer review study:

```bash
python scripts/breakfast_visual_repair_head_v2/prepare_outer_review.py \
  --oof-study /data1/eli-bogdanov/sktr_runs/breakfast_visual_repair_head_oof_screen_v2 \
  --study-dir /home/dsi/eli-bogdanov/breakfast_visual_repair_head_v2_outer_review_v1
```

Fable reviews the frozen configuration and hashes. Only after explicit approval
may `record_outer_approval.py` create the exact-digest approval record and
`run_outer.sh` execute the one outer evaluation.
