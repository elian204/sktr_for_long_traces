"""Canonical SKTR decode hyperparameters — the single source of truth.

Why this module exists
----------------------
On 2026-08-21 a Breakfast costing run was launched using
``kfold_learning_curve_experiment.py``'s own argparse defaults
(``candidate_top_k=15``, ``top_m=3``) instead of the configuration that every
real SKTR run in this project has used (``candidate_top_k=3``, ``top_m=1``).

That is 5x the candidate labels per timestamp and 3x the conditioning states,
and the cost compounds across chunks. The run needed >25 min/video with
unbounded memory growth (45 GB across 3 workers and climbing ~180 GB/hour)
and had to be killed and redone. Nothing warned that the config was unusual,
because the defaults *were* the wrong values.

The defaults in every entry point now come from ``CANONICAL_DECODE`` below.
Import from here instead of retyping numbers.

Provenance of CANONICAL_DECODE
------------------------------
Empirical: a scan of every ``*config*.json`` under the SKTR run trees found
these values in 115 recorded runs, uniform across all three datasets
(gtea n=91, breakfast n=12, 50salads n=12). The only deviations on disk are
the run described above and an explicitly-named ``profiling/bounded_topk1/``
experiment.

Cross-check: ``eval_diffact_sktr_fold1_paper.py`` — the paper evaluation
entry point — already carried these values as its defaults. It was
``kfold_learning_curve_experiment.py`` that had drifted.
"""

# =============================================================================
# CANONICAL DECODE CONFIG  — change only with a recorded reason
# =============================================================================

CANONICAL_DECODE = {
    # Candidate-label restriction per timestamp. THE expensive knob:
    # cost scales steeply with top_k because it compounds across chunks.
    'candidate_top_k': 3,
    'candidate_top_p': 1.0,
    'candidate_min_k': 1,
    'candidate_source': 'conditioned',

    # Conditioning-state restriction. `conditioning_top_m` is INERT unless
    # `conditioning_state_mode == 'topm'` — setting top_m under 'exact'
    # silently does nothing.
    'conditioning_state_mode': 'topm',
    'conditioning_top_m': 1,

    # History / chunking / pruning
    'max_hist_len': 3,
    'chunk_size': 11,
    'prob_threshold': 1e-6,
}

# WARNING — the library defaults in src/incremental_softmax_recovery.py do NOT
# match CANONICAL_DECODE. Calling incremental_softmax_recovery() directly gives
# candidate_top_k=None (UNBOUNDED search), conditioning_top_m=3 and
# conditioning_state_mode='exact'. Those library defaults are deliberately left
# alone for backwards compatibility with existing callers; always pass
# CANONICAL_DECODE explicitly. Entry points in this repo already do.

HP_STRATEGIES = {
    'trigram_heavy': [0.1, 0.15, 0.75],
    'unigram_super_heavy': [0.75, 0.15, 0.1],
    'bigram_heavy': [0.15, 0.75, 0.1],
    'balanced': [0.33, 0.34, 0.33],
}

# Dataset and model-specific alpha/strategy.
# Based on hyperparameter search results (lowest avg_rank = best).
DATASET_HP_DEFAULTS = {
    # 50 Salads ASFormer: alpha=0.3, unigram_super_heavy (avg_rank 7.33)
    ('50salads', 'asformer'): {'alpha': 0.3, 'strategy': 'unigram_super_heavy'},
    # 50 Salads MS-TCN2: alpha=0.3, trigram_heavy (avg_rank 3.33)
    ('50salads', 'mstcn2'): {'alpha': 0.3, 'strategy': 'trigram_heavy'},
    # GTEA ASFormer: alpha=0.95, trigram_heavy (avg_rank 6.0)
    ('gtea', 'asformer'): {'alpha': 0.95, 'strategy': 'trigram_heavy'},
    # GTEA MS-TCN2: alpha=0.95, trigram_heavy (avg_rank 2.0)
    ('gtea', 'mstcn2'): {'alpha': 0.95, 'strategy': 'trigram_heavy'},
    # Breakfast (HP search): alpha=0.7, trigram_heavy
    # NOTE: the Breakfast HP search that produced 0.7 has a known validation-set
    # defect (see evidence_lock_v1/results/B10_FINDING.md). The value is kept
    # because it is what the paper's numbers were produced with.
    ('breakfast', 'asformer'): {'alpha': 0.7, 'strategy': 'trigram_heavy'},
    ('breakfast', 'mstcn2'): {'alpha': 0.7, 'strategy': 'trigram_heavy'},
    # DiffAct: reuse ASFormer defaults until a dedicated HP sweep is run
    ('50salads', 'diffact'): {'alpha': 0.3, 'strategy': 'unigram_super_heavy'},
    ('gtea', 'diffact'): {'alpha': 0.95, 'strategy': 'trigram_heavy'},
    ('breakfast', 'diffact'): {'alpha': 0.7, 'strategy': 'trigram_heavy'},
}

# Fallback when a (dataset, model) pair is not in the table above.
FALLBACK_HP = {'alpha': 0.9, 'strategy': 'trigram_heavy'}


def get_dataset_hp_defaults(dataset: str, model: str) -> dict:
    """Get default alpha/strategy for a dataset/model combination."""
    return DATASET_HP_DEFAULTS.get((dataset.lower(), model.lower()), dict(FALLBACK_HP))


def deviations_from_canonical(effective: dict) -> dict:
    """Return {param: (used, canonical)} for every deviation from CANONICAL_DECODE.

    `effective` may use either the argparse names (``top_m``, ``state_mode``) or
    the decoder names (``conditioning_top_m``, ``conditioning_state_mode``).
    """
    aliases = {
        'conditioning_top_m': ('conditioning_top_m', 'top_m'),
        'conditioning_state_mode': ('conditioning_state_mode', 'state_mode'),
    }
    out = {}
    for key, canon in CANONICAL_DECODE.items():
        names = aliases.get(key, (key,))
        used = next((effective[n] for n in names if n in effective), None)
        if used is not None and used != canon:
            out[key] = (used, canon)
    return out


def format_banner(effective: dict) -> str:
    """Human-readable effective-config banner, loudly flagging any deviation."""
    dev = deviations_from_canonical(effective)
    lines = ['=' * 70, 'SKTR DECODE CONFIG', '=' * 70]
    for key, canon in CANONICAL_DECODE.items():
        names = (key, key.replace('conditioning_', ''))
        used = next((effective[n] for n in names if n in effective), '(not set)')
        mark = f'   <== NON-CANONICAL (standard: {canon})' if key in dev else ''
        lines.append(f'  {key:26} = {used}{mark}')
    if dev:
        lines += [
            '-' * 70,
            f'*** WARNING: {len(dev)} parameter(s) deviate from the canonical config. ***',
            '*** Cost scales steeply with candidate_top_k. If this is not      ***',
            '*** deliberate, stop and re-launch. See SKTR_HYPERPARAMETERS.md.  ***',
        ]
    else:
        lines.append('  -> matches canonical config (SKTR_HYPERPARAMETERS.md)')
    lines.append('=' * 70)
    return '\n'.join(lines)


def source_state(repo_root=None) -> dict:
    """Capture git head + hashes of dirty files so a run's numbers are
    reconstructable without remembering what the tree looked like that day.

    Recorded into experiment_config.json by the entry points. Never raises —
    provenance capture must not be able to kill a run.
    """
    import hashlib
    import os
    import subprocess
    root = repo_root or os.path.dirname(os.path.abspath(__file__))

    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, cwd=root, capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except Exception:
            return '<unavailable>'

    def digest(path):
        try:
            with open(path, 'rb') as fh:
                return hashlib.sha256(fh.read()).hexdigest()[:16]
        except Exception:
            return '<unreadable>'

    dirty = {}
    try:
        for line in sh('git status --porcelain').splitlines():
            rel = line[3:].strip().strip('"')
            full = os.path.join(root, rel)
            if os.path.isdir(full):
                for base, _, files in os.walk(full):
                    for fn in sorted(files):
                        if fn.endswith('.pyc') or '/.git/' in base:
                            continue
                        p = os.path.join(base, fn)
                        dirty[os.path.relpath(p, root)] = digest(p)
            elif os.path.exists(full):
                dirty[rel] = digest(full)
    except Exception:
        pass

    return {
        'git_head': sh('git rev-parse HEAD'),
        'git_branch': sh('git branch --show-current'),
        'git_remote': sh('git remote get-url origin'),
        'dirty_files': dirty,
        'clean_tree': not dirty,
    }
