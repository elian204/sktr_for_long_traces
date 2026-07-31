# Breakfast visual repair head v3: ensemble-agreement abstention

This CPU-only study is the **final outer attempt for the Breakfast repair
line**. It is justified by a mechanism that neither v1 nor v2 tested:
epistemic uncertainty estimated from agreement among independently fitted
repair heads. It is not another confidence-threshold optimization. If the
pre-registered outer evaluation fails, this repair line closes permanently;
there will be no v4 threshold or rule sweep.

The protocol keeps the v2 contamination boundary:

1. OOF screening may read only each outer fold's OOF repair corpus, matching
   I3D arrays, and ground truth. Outer-test artifacts remain sealed.
2. A pooled OOF mechanism gate must pass before fold-specific configuration
   selection is allowed.
3. Outer evaluation is generated separately and requires Fable's explicit
   approval of the exact frozen-config, source, and input digests. Each outer
   fold is evaluated once.

No v1/v2 outer-test outcome, harmed-video identity, or outer-test row may be
used in design or selection. OOF outputs contain only one-way video hashes.
Selection is independent per outer fold so that no video can influence the
configuration used when it is an outer-test example.

## Pre-registered seven-member ensemble

Every OOF prediction is cross-fitted by held inner fold. Every outer model is
fitted on that fold's complete OOF corpus. The seven members are exactly:

1. one full-data, length-weighted `LogisticRegression(C=1.0,
   max_iter=2000)`;
2. three case-bootstrap versions of that logistic, with bootstrap seeds
   `{101,202,303}`; and
3. three unweighted `MLPClassifier` models with one 256-unit hidden layer and
   seeds `{0,1,2}` (`solver=adam`, `max_iter=300`, early stopping,
   `n_iter_no_change=20`).

Each bootstrap samples training videos with replacement and includes every
segment of a sampled video once per draw. All scalers and heads are fitted
independently. Every convergence warning is a hard failure. The features stay
the approved full-segment I3D channel mean plus population standard deviation
(4096 dimensions); no metadata, Petri, or selector-shape features are added.

For a segment, each member independently proposes its probability argmax. The
ensemble consensus is the modal proposed label; ties are resolved by the
largest seven-member mean probability and then the lower class ID. A repair is
eligible only if at least `k` members propose the consensus label, its mean
probability across all seven members is at least `tau`, and it differs from the
incumbent. `k` is restricted to `{4,5,6,7}` and `tau` to `{0.3,0.5}`.

## Hard OOF mechanism gate

Before configuration selection, reproduce the v2 reference action set at the
5% frame budget: the full-data logistic, `tau=0.5`, free-choice proposal, and
no harm rule. The reference must contain exactly the previously registered 13
harmful accepted spans; disagreement aborts as protocol drift.

For every `(k,tau)` pair, apply the actual deployment gate on each reference
span. The action is counted as vetoed only when the ensemble would abstain:
there is no qualifying consensus label different from the incumbent. A
different qualifying consensus proposal is therefore *not* credited as a
veto. Support for the original v2 proposal is retained as a diagnostic. Report
separately:

- harmful-action veto rate;
- helpful-action veto rate;
- their ratio; and
- harmful-action veto rate within OOF tail videos below -5 Accuracy points.

The mechanism gate passes only if at least one pre-specified pair has
`harmful_veto_rate / helpful_veto_rate >= 2.0`. A zero helpful-veto rate with a
positive harmful-veto rate is treated as infinite selectivity; zero divided by
zero fails. If no pair passes, screening stops, outer evaluation is blocked,
and the repair line closes without opening outer data.

## Fold-specific OOF selection after a passing mechanism gate

Within each outer fold, screen all `(k,tau)` pairs at the primary 5% selector
budget with these optional overlays:

- `none`;
- per-video relabel caps `X in {1,2,3,5}%`;
- incumbent margins `delta in {0.05,0.10,0.20,0.30}`; and
- large-span guards `Y in {2,5,10,20}%`, using the v2 `tau+0.2` higher bar
  capped at 0.9.

Only configurations with zero OOF videos below -5 Accuracy points are
eligible. Choose by OOF Accuracy gain, F1@25 gain, Edit gain, worst-video
delta, then simpler/more aggressive deployment: no overlay before an overlay,
lower `k`, lower `tau`, listed overlay order, and lower numeric parameter.
This explicitly tests whether agreement itself removes the need for v2's
conservative fold-specific rules.

Five percent remains primary. With the chosen configuration, `{8,10}%` are
OOF-only secondary envelope candidates under the same safety constraint; one
is frozen only if it beats 5% by at least +0.15 Accuracy point. A secondary
readout can never rescue the primary outer verdict.

## Final single-shot outer gates

The primary uses the unchanged six v1/v2 gates:

- pooled Delta Accuracy at least +0.5 point;
- positive Delta Accuracy in at least 3/4 folds;
- pooled Delta Edit non-negative;
- pooled Delta F1@25 non-negative;
- no video below -5 Accuracy points; and
- no video supplies more than 50% of pooled net correct-frame gain.

Failure of any gate closes the repair line permanently.

## Operations

After tests and a clean commit, prepare and run only the OOF phase:

```bash
python scripts/breakfast_visual_repair_head_v3/prepare_study.py \
  --study-dir /data1/eli-bogdanov/sktr_runs/breakfast_visual_repair_head_agreement_oof_v3
/data1/eli-bogdanov/sktr_runs/breakfast_visual_repair_head_agreement_oof_v3/run_oof.sh
```

If and only if the mechanism gate and all fold screens pass, stage the outer
review without approving or launching it:

```bash
python scripts/breakfast_visual_repair_head_v3/prepare_outer_review.py \
  --oof-study /data1/eli-bogdanov/sktr_runs/breakfast_visual_repair_head_agreement_oof_v3 \
  --study-dir /home/dsi/eli-bogdanov/breakfast_visual_repair_head_v3_outer_review_v1
```

Fable must review the exact digests. Only then may
`record_outer_approval.py` write the approval record and `run_outer.sh`
perform the final single-shot outer evaluation.
