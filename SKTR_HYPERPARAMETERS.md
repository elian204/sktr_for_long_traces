# SKTR hyperparameters — what to use

**Short version: don't pass any decode flags.** The defaults in every entry point
now come from `CANONICAL_DECODE` in [`sktr_hparams.py`](sktr_hparams.py), which is
the single source of truth. If you pass nothing, you get the right config.

```bash
# Breakfast / ASFormer, fold 1 — canonical config, nothing to remember
python3 kfold_learning_curve_experiment.py -d breakfast -m asformer --folds 1
```

`alpha` and `strategy` also default correctly per (dataset, backbone) — see the
table below. You only need flags to *deviate*, and deviating is what you should
be careful about.

---

## The canonical decode config

| Parameter | Value | Notes |
|---|---|---|
| `candidate_top_k` | **3** | **Cost-critical.** Max candidate labels per timestamp. |
| `candidate_top_p` | 1.0 | |
| `candidate_min_k` | 1 | |
| `candidate_source` | `conditioned` | |
| `conditioning_state_mode` | `topm` | |
| `conditioning_top_m` | **1** | **Inert unless `state_mode='topm'`.** |
| `max_hist_len` | 3 | |
| `chunk_size` | 11 | |
| `prob_threshold` | 1e-6 | |

### Why `candidate_top_k` deserves respect

Runtime and memory scale steeply with it, because the candidate set compounds
across chunks. This is not a knob to raise casually:

> On 2026-08-21 a Breakfast costing run was launched with `candidate_top_k=15,
> top_m=3` — 5× the candidate labels and 3× the conditioning states. It needed
> **>25 min/video** and grew to **45 GB across 3 workers, climbing ~180 GB/hour**,
> on a node with no swap left. It was killed and redone at `3/1`.
>
> Nothing warned, because at the time those *were* `kfold_learning_curve_experiment.py`'s
> argparse defaults. That is the bug this document and `sktr_hparams.py` exist to prevent.

## Per-dataset alpha / strategy

| Dataset | Backbone | alpha | strategy |
|---|---|---|---|
| 50Salads | ASFormer | 0.3 | `unigram_super_heavy` |
| 50Salads | MS-TCN2 | 0.3 | `trigram_heavy` |
| 50Salads | DiffAct | 0.3 | `unigram_super_heavy` |
| GTEA | ASFormer | 0.95 | `trigram_heavy` |
| GTEA | MS-TCN2 | 0.95 | `trigram_heavy` |
| GTEA | DiffAct | 0.95 | `trigram_heavy` |
| Breakfast | ASFormer | 0.7 | `trigram_heavy` |
| Breakfast | MS-TCN2 | 0.7 | `trigram_heavy` |
| Breakfast | DiffAct | 0.7 | `trigram_heavy` |

Fallback for any pair not listed: `alpha=0.9, trigram_heavy`.

Strategy weights are `[unigram, bigram, trigram]`: `trigram_heavy = [0.1, 0.15, 0.75]`,
`unigram_super_heavy = [0.75, 0.15, 0.1]`, `bigram_heavy = [0.15, 0.75, 0.1]`,
`balanced = [0.33, 0.34, 0.33]`.

---

## Traps

**1. The library defaults do NOT match the canonical config.**
Calling `incremental_softmax_recovery()` directly from
`src/incremental_softmax_recovery.py` gives you:

| Parameter | Library default | Canonical |
|---|---|---|
| `candidate_top_k` | `None` — **unbounded search** | 3 |
| `conditioning_top_m` | 3 | 1 |
| `conditioning_state_mode` | `exact` | `topm` |

These are left alone deliberately, for backwards compatibility with existing
callers. **Always pass `CANONICAL_DECODE` explicitly** when calling the library
directly. The repo's entry points already do.

**2. `conditioning_top_m` is silently inert under `state_mode='exact'`.**
Setting `top_m` while `state_mode='exact'` does nothing at all — no error, no
warning, just an unrestricted state set and a much slower run.

**3. `n_indices=1e9` in the logs is not a bug.** It is deliberate frame
retention (`# Large n_indices to keep all frames`), not an unbounded search.

---

## How to check a run used the right config

Every run now prints an effective-config banner at startup and **loudly flags any
deviation**. A canonical run ends with `-> matches canonical config`; a deviating
one prints `<== NON-CANONICAL` plus a warning block.

To audit a run after the fact, `experiment_config.json` now records the full
decode config — `top_m`, `state_mode`, `candidate_top_p`, `candidate_min_k`,
`candidate_source` and `max_hist_len` used to be omitted, which made older runs
un-auditable on exactly the parameters that matter most.

```python
import json
from sktr_hparams import deviations_from_canonical
cfg = json.load(open('results/.../experiment_config.json'))
print(deviations_from_canonical(cfg) or 'canonical')
```

---

## Provenance

`CANONICAL_DECODE` is not invented. A scan of every `*config*.json` under the SKTR
run trees found these values in **115 recorded runs**, uniform across all three
datasets (GTEA n=91, Breakfast n=12, 50Salads n=12). The only deviations on disk
are the 2026-08-21 mis-launch described above and an explicitly-named
`profiling/bounded_topk1/` experiment.

Independently, `eval_diffact_sktr_fold1_paper.py` — the paper evaluation entry
point — already carried these values as its defaults. It was
`kfold_learning_curve_experiment.py` that had drifted.

## Known caveats on the alpha values

- **Breakfast `alpha=0.7`** comes from a hyperparameter search with a known
  validation-set defect (the validation split contains official test videos).
  See `evidence_lock_v1/results/B10_FINDING.md`. The value is kept here because
  it is what the published numbers were produced with — it is documented, not endorsed.
- **GTEA**: the table says `0.95`, and the outputs reproducing the published table
  were produced at `0.95`. Some write-ups declare `0.9`. Treat `0.95` as what ran.
