# Phase 2: epoch-scarcity DiffAct + SKTR study

This study treats optimization maturity as a separate axis from labeled-data
scarcity. It imports the five pre-specified checkpoints from each completed
GTEA D100 trajectory in the immutable data-scarcity study; it never trains from
or modifies those checkpoints.

This is intentionally the same seed-0 D100 trajectory used by the Phase 1 D100
endpoint, not an independent retraining replicate. Phase 2 must therefore be
reported as a learning-trajectory analysis underlying that endpoint.

## Approved first wave

- Dataset: GTEA, official folds 1-4, trajectory seed 0.
- Checkpoint epoch indices: 200, 1000, 2000, 5000, 10000.
- Epoch indices are zero-based. `completed_epochs = epoch_index + 1`.
- `schedule_fraction_pct = 100 * epoch_index / final_epoch_index`; use the
  zero-based epoch index consistently on plots.
- Primary condition: `official_full_train_petri`, discovered from the complete
  official training fold.
- Ceiling condition: `oracle_test_fold_petri`, discovered from official test
  ground truth and always tagged as oracle/leaky.
- Decoder: canonical `petri_conformance` configuration (chunk 11, top-M 1,
  top-K 3, restricted log/model moves, tau cap 8).
- No data-fraction axis, nested Petri condition, or D100+P25 control.
- 50 Salads remains gated on its exact-conformance runtime pilot.

## Sequencing remembered

1. GTEA data scarcity.
2. GTEA epoch scarcity using its D100 trajectories.
3. 50 Salads exact-conformance runtime gate.
4. 50 Salads data scarcity if the gate passes.
5. 50 Salads epoch scarcity using its D100 trajectories.

Seed-0 results are descriptive. Cross-dataset and robust initialization claims
wait for 50 Salads and additional seeds.

## Task graph per fold

1. Import and hash the five checkpoints after the source D100 task completes.
2. Export raw, canonical, and official DiffAct streams for each checkpoint.
3. Discover the full-train Petri context once and decode all five checkpoints.
4. Discover the oracle Petri context once and decode all five checkpoints.

The two decode tasks depend on all five exports. The generated first-wave GTEA
matrix therefore has 32 tasks: four imports, twenty exports, four full-train
decode-all tasks, and four oracle decode-all tasks.

Each completed export records SHA-256 values for every raw, canonical,
official, mapping, and ground-truth artifact. Those hashes are recomputed both
when an export is skipped and before a Petri decode begins.

## Checkpoint and step provenance

Every imported checkpoint records its source path, copied path, SHA-256,
zero-based epoch index, completed epoch count, schedule fraction, training item
presentations, trainer step, and optimizer update count. Here `trainer_step` is
DiffAct's post-loop next-presentation counter, not the number of completed
presentations. DiffAct accumulates
gradients over `batch_size` training items, so for a checkpoint epoch `e`:

```
training_item_presentations = (e + 1) * n_train_cases
trainer_step = 1 + training_item_presentations
optimizer_updates = training_item_presentations // batch_size
```

The final imported checkpoint is accepted only after the source trajectory has
a valid `train_complete.json` and its official-train manifest exactly matches
the source study's `train_pool_cases.txt`.

## Analysis outputs

The pre-specified primary endpoint remains the per-video paired delta in
F1@25: SKTR full-train Petri minus official DiffAct. Per-video values are
nested within official fold, and seed-0 uncertainty is descriptive only.

Aggregation also writes explicitly secondary fold-global TAS curves. These use
the conventional fold-level global segmental calculation needed to compare
absolute F1/Edit/accuracy values with published DiffAct-style tables. They do
not replace the per-video paired primary endpoint, and their rows are marked as
secondary rather than primary. Oracle curves remain separate under both metric
conventions.

## Development and launch safety

This package is developed in a separate Git worktree. It must not modify the
main worktree or the running data-scarcity study. Generation is configuration
only; importing, exporting, decoding, and tmux launch require separate explicit
commands after the source D100 trajectories complete.

The four generated queues are shared, non-exclusive GPU queues. Each GTEA fold
is assigned to one physical GPU, so its five exports are serial on that GPU;
the four folds run concurrently. The subsequent Petri tasks are CPU tasks but
remain in the same queue to preserve simple dependency order.

The separate worktree does not duplicate the independently cloned DiffAct
repository, so pass its existing checkout explicitly:

```bash
python scripts/epoch_scarcity_petri_diffact/generate_epoch_scarcity_study.py \
  --study-dir /data1/eli-bogdanov/sktr_runs/epoch_scarcity_gtea_seed0_v1 \
  --study-id epoch_scarcity_gtea_seed0_v1 \
  --source-study-dir /data1/eli-bogdanov/sktr_runs/low_data_decoding_study_gtea_seed0_bounds_v2 \
  --diffact-root /home/dsi/eli-bogdanov/sktr_for_long_traces/baselines/DiffAct \
  --gpus 0 1 2 3
```

After generation, review `study_metadata.json`, run `prepare_study.sh`, and
check `study_status.sh`. Do not run `tmux_commands.sh` until all four source
D100 task states are `complete` with return code 0. Operationally, wait for the
entire Phase 1 GTEA study to finish before launching Phase 2 so the four shared
GPUs are not contended. Every real task also
recomputes the recorded source digest and refuses to run if the code changed.

Status revalidates imported checkpoint hashes and provenance, complete export
bundle hashes, and every epoch output in each completed decode grid; it does
not trust completion booleans alone. Aggregation refuses to run until every
task state is complete with return code zero and both the per-case and
fold-global inputs cover the exact fold × checkpoint × condition × method ×
metric grid.
