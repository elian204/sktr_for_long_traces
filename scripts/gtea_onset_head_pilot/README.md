# GTEA class-specific Gaussian onset-head pilot

This is a **screening study** for ladder step 3: add class identity to
DiffAct's boundary supervision without changing its activity vocabulary or
replacing any existing loss. It uses only GTEA fold 1 and training seed 0, so
it cannot support an advancing claim by itself. Implementation and review
staging are allowed now; launch is explicitly gated on the multi-fold
weight-1.0 confirmation and Fable's architecture review.

## Architecture

The original DiffAct encoder and diffusion activity decoder are intact. A
single auxiliary `Conv1d(..., num_activity_classes, kernel_size=1)` is attached
to the encoder feature tensor already consumed by the decoder. It predicts one
sigmoid onset curve per activity class and is constructed after all original
modules, so it does not perturb initialization of those modules.

The target has shape `class × time`. At each ground-truth segment start,
including frame 0, only the starting activity's channel receives an impulse.
The preceding class is never marked. Impulses are Gaussian-smoothed with
`sigma=1`, frozen equal to GTEA's `boundary_smooth`.

Original frame labels, encoder/decoder activity heads, and all six existing
DiffAct losses remain unchanged. Joint training adds
`class_specific_onset_loss`, a sigmoid BCE loss whose scalar weight is the
only new experimental knob. Its gradient also reaches the encoder, making the
onset supervision capable of improving video features rather than serving as a
detached diagnostic.

An optional Gaussian-bump-region upweighting is implemented as
`1 + (positive_weight - 1) * target`. It is frozen at `1.0` (off) for this
pilot and is not part of the sweep.

## Staged experiment

- GTEA official fold 1, training seed 0.
- Existing `decoder_boundary_loss=1.0` is frozen as the activity baseline.
- Onset loss weights: `{0.0, 0.1, 0.3, 0.5}`.
- `0.0` is an in-architecture onset-disabled baseline.
- `0.3` is pre-registered primary; `0.1` and `0.5` are exploratory.
- Fixed checkpoints: 200, 1000, 2000, 5000, and 10000.
- Three diffusion inference seeds per checkpoint.
- Raw pre-purge and official purge-3 activity metrics use the v2 readout set.

Retraining the `0.0` baseline is deliberate. A checkpoint from the old model
cannot be loaded strictly into an architecture with a new head, and a fresh
paired baseline avoids hiding any architecture/RNG difference behind a
partial-load exception.

Every export contains the normal activity files plus
`{video_index}_onset.npy` (`class × full-frame time`). Onset curves are an
encoder readout and must be byte-identical across the three diffusion
inference seeds for a fixed checkpoint. The analyzer fails closed if they are
not. It reports correct-class local-peak signed displacement, mean/median/p90
absolute displacement, and class-specific onset-peak F1 at ±5 and ±10 frames.
These curves are also the artifact intended for Track-B selector features.

## Staged decision rule

At epoch 10000 post-purge, onset weight 0.3 versus 0.0 must have non-negative
Accuracy, positive Edit and F1@25, segment-count-ratio delta no greater than
`+0.02`, and lower correct-class mean absolute onset-peak displacement.
This rule evaluates the pilot only; it does not override the upstream launch
gate.

A positive screen is not a confirmed result. Before any claim advances, run a
separate immutable replication with `{onset loss 0.0, best screened onset
weight}` at training seed 1. The best screened weight is selected only among
weights that pass the same locked activity and mechanistic checks; rank those
weights by mean ΔF1@25, then mean ΔEdit, then prefer the smaller weight as a
deterministic tie-break. Because 0.3 remains the pre-registered screening
primary, the screen is called positive only if 0.3 itself passes; exploratory
weights may determine the replication treatment only after that primary gate
passes. No onset-head claim or ladder advancement occurs before the seed-1
paired replication is completed and reported.

## Hard launch gate

`launch_tmux.sh` cannot start training unless the Task-1 multi-fold
`decision.json` exists and contains
`"onset_head_launch_gate_passes": true`. After that dependency passes, the
GPU-3 queue still requires two consecutive process-free checks and never
preempts another process. Fable approval is an additional human gate before
the launcher is invoked.

Review staging:

```bash
python scripts/gtea_onset_head_pilot/prepare_study.py \
  --study-dir /tmp/gtea_onset_head_review
```

The generator launches nothing. Production must be regenerated from clean,
committed SKTR and DiffAct onset branches after both reviews pass.
