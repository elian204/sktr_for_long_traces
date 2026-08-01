# Independent-acceptance verifier campaign

## Framing and governance

The closed repair line failed at acceptance, not average gains — every
acceptance mechanism so far consulted the proposing model's own confidence. This
campaign fields the first independent acceptance mechanism (a trained pairwise
verifier). New mechanism class, fresh pre-registration, declared attempt budget:
ONE sealed outer evaluation for the entire campaign, same six gates, Fable review
at every stage boundary.

The six sealed-outer gates remain unchanged:

1. pooled ΔAcc ≥ +0.5 percentage points;
2. ΔAcc > 0 in at least 3/4 outer folds;
3. pooled ΔEdit ≥ 0;
4. pooled ΔF1@25 ≥ 0;
5. no video loses more than 5 accuracy points; and
6. no single video supplies more than 50% of the pooled gain.

No V0/V1/V2 choice may inspect outer-test data. There is one V3 sealed-outer
attempt for the entire campaign, only if V1 passes its nested-OOF gates.

## V0 — OOF candidate corpus (this review package)

V0 uses only the four-fold selector study's dedicated official-OOF repair
corpus. It deliberately does not open the mixed-scope `segment_scores.csv`.
Within each outer fold, the frozen `base_score` selects exactly 5% of OOF frames,
using the previously locked deterministic tie key and centered partial cutoff.

Each selected span receives the deduplicated union of:

- the incumbent official DiffAct label;
- the top 3 labels from the existing nested-OOF plain visual logistic head; and
- the top 5 labels from the existing segment-mean DiffAct probability pool.

For every candidate label, V0 records:

- whether it matches the selected span's deterministic GT-modal label;
- whether that modal label is a unique strict majority;
- incumbent and candidate correct-frame counts;
- net-frame effect and helpful/harmful/lateral status; and
- source membership, rank, and probability.

The schema includes reserved nullable fields for the optional V2 inpainting
cluster medoids. V0 is descriptive/training-corpus construction; its oracle
candidate-availability readout is not realized repair performance.

## V1 — temporal pairwise verifier (blocked pending V0 review)

Each candidate will receive the span's unpooled I3D features resampled to 48
temporal bins plus 16-bin context on each side, projected 2048→256. The frozen
architecture choice is a small dilated TCN: it has a direct locality/multiscale
inductive bias for the temporal comparison and is lighter than a transformer for
the available OOF corpus. Incumbent/candidate label embeddings are concatenated;
the output is P(candidate better than incumbent).

Architecture and threshold selection must be nested: tune on two inner folds,
evaluate on the third, rotate, and never tune and gate on the same pooled OOF.
Hard negatives are high-confidence wrong proposals. A class-conditioned
nearest-exemplar baseline with soft-DTW over unpooled features is mandatory.

V1 passes only if nested-OOF pooled ΔAcc ≥ +0.75, all three evaluations are
positive (or two positive and none below −0.1), zero OOF videos fall below −5pp,
helpful:harmful changed frames ≥3:1, and pooled ΔEdit/ΔF1@25 are non-negative.
Failure closes the campaign without any outer evaluation.

## V2 — corrected masked sampling as candidate generation only (blocked)

V2 may run in parallel with V1 after V0 review. It uses the pre-registered
three-region mask, correct diffusion noising schedule, shared context-noise
trajectory, restart grid, `k=15` sequential samples, and segmental-edit medoid
candidates. It must first pass exterior-invariance, empty-mask-identity, and
seeded-replay validity tests.

The OOF best-of-k kill bar is +1.4 Acc. Samples join V0 only at ≥+1.75 Acc,
correct candidates for ≥35% of flagged wrong mass, and ≥+0.5 incremental oracle
over visual candidates. No consensus acceptor is built; V1 remains the acceptor.

## V3 — one sealed outer attempt (blocked)

V3 exists only if V1 passes. Candidate sources, verifier, and threshold are then
frozen and hash-locked for one Fable-approved sealed evaluation. Whatever lands
is the final campaign record.

## Commands

```bash
/usr/bin/python scripts/independent_acceptance_verifier/prepare_v0.py \
  --study-dir /data1/eli-bogdanov/sktr_runs/independent_acceptance_verifier_v0_review_v1

/data1/eli-bogdanov/sktr_runs/independent_acceptance_verifier_v0_review_v1/run_v0.sh
```
