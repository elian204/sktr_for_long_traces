# GTEA boundary-weight replication v2

This immutable follow-up asks whether the exploratory `decoder_boundary_loss=1.0`
improvement from v1 survives a second training seed and where the local
dose-response peaks.

## Locked design

- Dataset: GTEA official fold 1.
- Training seeds: 0 and 1.
- Primary grid: `decoder_boundary_loss ∈ {0.1, 0.75, 1.0, 1.5}`.
- Seed-0 weights 0.1 and 1.0 are imported from v1, not retrained.
- Seed-0 weight 0.5 is also imported as an exploratory curve reference; it is
  outside the v2 primary grid and has no seed-1 counterpart.
- Net new training trajectories: six.
- Checkpoints: epochs 200, 1000, 2000, 5000, and 10000.
- Diffusion inference seeds: 0, 1, and 2 at every checkpoint.
- Both raw pre-purge argmax and official purge-3 predictions are evaluated.
- Physical GPU 3 only, sequentially, with no automatic fallback.

The generated train manifest and actual DiffAct train bundle must be
byte-identical to the locked D100 inputs. The required bundle SHA-256 is
`fe02e8f838bf30f2a050d67b8986ecc88fa07c18e9f87d39b3733c459a6bfa03`.
All imported checkpoints and all exported inference artifacts are hashed into
per-task `import_complete.json` manifests and revalidated before analysis.

## Why there is no reproduction gate

V1's `protocol_amendment_1.json` records a back-to-back determinism probe:
nominally identical 201-epoch trainings produced different checkpoint hashes.
The v1 locked-versus-retrained gate therefore measured benign run-to-run
numeric variation rather than a protocol mismatch. V2 does not pretend that
training is bitwise reproducible.

Instead, V2 reports seed-0 versus seed-1 metric deltas and prediction-frame
disagreement for every primary-grid weight, checkpoint, inference seed, and
prediction stream. This is the per-weight noise floor and the replication
readout.

## Pre-registered primary

The primary comparison is weight 1.0 versus the paired 0.1 baseline at epoch
10000, post-purge, averaged over both training seeds and all three inference
seeds. It passes only if all six checks pass:

- Accuracy delta is non-negative.
- Edit delta is positive.
- F1@25 delta is positive.
- Class-agnostic boundary F1 at ±10 frames improves.
- Segment-count-ratio delta is no greater than +0.02.
- Class-agnostic mean absolute boundary offset decreases.

V1 used median absolute boundary offset. That statistic tied at exactly zero
and was therefore decision-degenerate. Before any v2 launch, the rule is
deliberately replaced by mean absolute offset, which remains sensitive to
distributed timing improvements.

Weights 0.75 and 1.5 are exploratory. If the primary passes, the next ladder
step is a class-specific Gaussian onset head. If it fails, the config-only rung
closes and the onset-head decision uses the combined evidence rather than
advancing automatically.

## Operations and immutability

`prepare_study.py` creates a reviewable study but launches nothing. It:

1. validates the D100 bundle order and v1 protocol amendment;
2. writes all configs and tasks;
3. records and verifies imported v1 artifact hashes;
4. creates `logs/` before generating the launcher;
5. writes a serial fail-closed GPU-3 queue and detached-tmux launcher.

Production must be generated from a clean, committed branch. The launcher
checks that GPU 3 has no compute process, creates `logs/` defensively, and
refuses to fall back to another GPU.

Typical review generation:

```bash
python scripts/gtea_boundary_weight_pilot/prepare_study.py \
  --study-dir /tmp/gtea_boundary_weight_replication_v2_review
```

Nothing should be launched until the generated configs, import hashes, bundle
identity, primary rule, and queue have been independently reviewed.
