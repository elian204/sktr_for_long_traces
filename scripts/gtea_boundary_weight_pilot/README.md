# GTEA multi-fold confirmation of boundary-loss weight 1.0

This immutable follow-up asks whether the fold-1 signal for
`decoder_boundary_loss=1.0` generalizes across all four official GTEA folds.
It is a minimal paired confirmation, not another weight search.

## Locked design

- New trainings: folds 2, 3, and 4 × weights `{0.1, 1.0}` × training seeds
  `{0, 1}` = 12 trajectories.
- Every one of the nine completed fold-1 v2 tasks is imported from
  `gtea_boundary_weight_fold1_v2` and protected by checkpoint/export hashes.
- The primary grid on every fold is the paired `{0.1, 1.0}` × `{0, 1}` set.
  Fold-1 exploratory weights are preserved only as imported curve context.
- Checkpoints are fixed at epochs 200, 1000, 2000, 5000, and 10000.
- Diffusion inference seeds are 0, 1, and 2 at every checkpoint.
- Both raw pre-purge argmax and official purge-3 predictions are evaluated.
- There is no reconciliation gate. Cross-training-seed metric and
  frame-disagreement readouts quantify the noise floor per fold and weight.

For every fold, the generated DiffAct train bundle is byte-identical to the
actual `frac_100` bundle in the locked low-data study. Its order is also
identical to that fold's locked `train_cases_frac_100.txt` manifest and its set
is checked against the official train split. The official test split remains
untouched and disjoint.

## Pre-registered primary decision

The primary comparison is weight 1.0 versus the paired weight 0.1 baseline at
epoch 10000, post-purge, averaged over two training seeds and three inference
seeds. The same six v2 checks are reported separately for each fold and pooled
equally over folds 1–4:

- Accuracy delta is non-negative.
- Edit delta is positive.
- F1@25 delta is positive.
- Class-agnostic boundary F1 at ±10 frames improves.
- Segment-count-ratio delta is no greater than +0.02.
- Class-agnostic mean absolute boundary offset decreases.

The primary claim passes only if all six pooled checks pass and the Edit and
F1@25 deltas are each positive in at least three of four folds. Mean absolute
offset is retained from v2 because the earlier median statistic was
tie-degenerate at zero.

Passing records scientific support for considering the separately reviewed
class-specific onset-head launch. Failing does not launch that study; it
requires a new explicit decision.

## GPU queues and non-preemption

The 12 new trainings are Latin-rotated into four independent physical-GPU
queues with exactly three trajectories each. `launch_tmux.sh` arms one named
tmux waiter per GPU. A lane starts only after `nvidia-smi` succeeds and reports
zero compute processes on two consecutive checks 60 seconds apart. A busy lane
keeps waiting; it never kills, preempts, or shares with the running Breakfast
selector study, and there is no automatic GPU fallback.

GPU 3 can therefore start first when free. GPUs 0–2 join automatically only
after their current selector lanes (and any other compute processes) leave.

## Review and immutability

`prepare_study.py` creates configs, manifests, imports, queues, waiters, status
tools, and provenance, but launches nothing. Generate a review study with:

```bash
python scripts/gtea_boundary_weight_pilot/prepare_study.py \
  --study-dir /tmp/gtea_boundary_weight_multifold_review
```

Fable's spot-check is required before committing, generating the clean-tree
production study, or running `launch_tmux.sh`. Once production starts, do not
edit this worktree or its referenced DiffAct source; generate a new version for
any change.
