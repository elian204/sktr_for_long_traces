#!/usr/bin/env python3
"""
K-Fold Cross-Validation Learning Curve Experiment

This script evaluates sample efficiency using stratified cross-validation with learning curves,
following standard evaluation protocols from TAS literature (Farha 2019, Yi 2021).

Dataset configurations:
- 50 Salads: 5-fold CV, k in {1, 5, 10, 20, 30, 40}
- Breakfast: 4-fold CV, k in {1, 10, 50, 100, 200}
- GTEA: 4-fold CV, k in {1, 5, 10, 15, 20}

Usage:
    python kfold_learning_curve_experiment.py -d 50salads -m asformer
    python kfold_learning_curve_experiment.py -d breakfast --folds 1,2 --k-values 1,10,50
"""

from src.cv_utils import (
    build_video_to_case_mapping,
    get_dataset_cv_config,
    get_unique_representatives,
    load_fold_case_ids,
    stratified_sample_k_traces,
)
from src.evaluation import compute_sktr_vs_argmax_metrics
from src.incremental_softmax_recovery import incremental_softmax_recovery
from src.utils import (
    prepare_df, prepare_df_from_model, linear_prob_combiner,
    get_variant_info
)
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import logging
import time
import argparse
import hashlib
import json
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for terminal/tmux

# =============================================================================
# SETUP
# =============================================================================

# Determine workspace root relative to this script's location
workspace_root = str(Path(__file__).resolve().parent)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
for mod in ['src.classes', 'src.incremental_softmax_recovery', 'src.utils',
            'src.conformance_checking', 'src.data_processing', 'src.petri_model', 'src.calibration']:
    logging.getLogger(mod).setLevel(logging.DEBUG)
for mod in ['graphviz', 'matplotlib', 'PIL']:
    logging.getLogger(mod).setLevel(logging.WARNING)

# =============================================================================
# HYPERPARAMETER CONFIGURATIONS
# =============================================================================

# Hyperparameters are defined ONCE in sktr_hparams.py (single source of truth).
# Do not redefine them here. See SKTR_HYPERPARAMETERS.md for what to use and why.
from sktr_hparams import (
    CANONICAL_DECODE,
    HP_STRATEGIES,
    DATASET_HP_DEFAULTS,
    get_dataset_hp_defaults,
    format_banner,
    source_state,
)

# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='K-fold cross-validation learning curve experiment.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-d', '--dataset',
        type=str,
        choices=['50salads', 'gtea', 'breakfast'],
        required=True,
        help='Dataset to run experiments on'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        choices=['asformer', 'mstcn2', 'diffact'],
        default='asformer',
        help='Model source for softmax predictions'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=20,
        help='Number of inner workers per experiment (used with --inner-parallel)'
    )
    parser.add_argument(
        '-p', '--parallel',
        type=int,
        default=1,
        help='Number of (fold, k) experiments to run in parallel'
    )
    parser.add_argument(
        '--inner-parallel',
        action='store_true',
        help='Enable dataset-level parallelization inside each experiment'
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    # Fixed hyperparameters (defaults are dataset/model-specific if not provided)
    parser.add_argument(
        '--alpha',
        type=float,
        default=None,
        help='Conditioning alpha (interpolation between prior and observation). '
             'If not specified, uses dataset/model-specific optimized defaults.'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        choices=list(HP_STRATEGIES.keys()),
        default=None,
        help='Interpolation strategy for history weighting. '
             'If not specified, uses dataset/model-specific optimized defaults.'
    )
    # Fold/k control
    parser.add_argument(
        '--folds',
        type=str,
        default=None,
        help='Comma-separated fold numbers to run (e.g., "1,2,3"). Default: all folds.'
    )
    parser.add_argument(
        '--k-values',
        type=str,
        default=None,
        help='Comma-separated k values (e.g., "1,5,10"). Default: dataset-specific values.'
    )
    # Conformance checking parameters
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Write per-video checkpoints and reuse completed ones on restart. '
             'STRONGLY RECOMMENDED for long runs: without it a crash loses the whole run.'
    )
    parser.add_argument(
        '--allow-source-drift',
        action='store_true',
        help='Reuse checkpoints whose manifest records a different git state. Only when the '
             'code change is provably irrelevant to decode output.'
    )
    parser.add_argument(
        '--test-cases',
        type=str,
        default=None,
        help='Comma-separated test case IDs to decode (overrides --max-test-traces). '
             'Use for designed samples where WHICH videos matter, not just how many.'
    )
    parser.add_argument(
        '--state-mode',
        type=str,
        choices=['exact', 'topm'],
        default=CANONICAL_DECODE['conditioning_state_mode'],
        help='Conditioning state mode: exact (full history match) or topm (top-m states). Canonical: topm.'
    )
    parser.add_argument(
        '--top-m',
        type=int,
        default=CANONICAL_DECODE['conditioning_top_m'],
        help='Number of top conditioning states (canonical: 1). Inert unless --state-mode=topm.'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=CANONICAL_DECODE['chunk_size'],
        help='Chunk size for conformance checking'
    )
    parser.add_argument(
        '--prob-threshold',
        type=float,
        default=CANONICAL_DECODE['prob_threshold'],
        help='Minimum probability threshold for pruning'
    )
    parser.add_argument(
        '--use-calibration',
        action='store_true',
        help='Enable temperature calibration for test softmax'
    )
    parser.add_argument(
        '--model-move-cost',
        type=float,
        default=1.0,
        help='Cost for labeled model moves'
    )
    parser.add_argument(
        '--candidate-top-k',
        type=int,
        default=CANONICAL_DECODE['candidate_top_k'],
        help='Max candidate labels per timestamp (canonical: 3). COST-CRITICAL: runtime and memory scale steeply with this.'
    )
    parser.add_argument(
        '--candidate-top-p',
        type=float,
        default=CANONICAL_DECODE['candidate_top_p'],
        help='Cumulative probability cutoff for candidate labels (top-p)'
    )
    parser.add_argument(
        '--candidate-min-k',
        type=int,
        default=CANONICAL_DECODE['candidate_min_k'],
        help='Minimum candidate labels per timestamp'
    )
    parser.add_argument(
        '--restrict-log-moves',
        action='store_true',
        help='Restrict log moves to only top-1 + parent\'s last label'
    )
    parser.add_argument(
        '--restrict-model-moves-to-tau',
        action='store_true',
        help='Restrict model moves to only tau (silent) transitions'
    )
    parser.add_argument(
        '--no-save-models',
        action='store_true',
        help='Disable saving Petri net visualizations'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=True,
        help='Skip (fold, k) combinations that already have results (default: enabled)'
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_false',
        dest='skip_existing',
        help='Disable skipping of existing (fold, k) results'
    )
    parser.add_argument(
        '--max-test-traces',
        type=int,
        default=None,
        help='Limit number of test traces (for quick testing)'
    )
    parser.add_argument(
        '--unique-only',
        action='store_true',
        help='Use only unique (collapsed) traces for training and testing. '
             'Picks one representative per variant, reducing computation for datasets '
             'with many duplicate traces (e.g., breakfast).'
    )
    return parser.parse_args()


def load_dataset(dataset_name: str, model_source: str):
    """Load dataset from specified source."""
    print(f"Loading {dataset_name} from {model_source}...")
    df, softmax_lst = prepare_df_from_model(dataset_name, model_source)
    print(f"  Loaded {len(softmax_lst)} cases, {len(df)} events")
    return df, softmax_lst


def resolve_background_label(dataset_name: str) -> Optional[str]:
    """Return None to use auto background resolution."""
    return None


def build_base_config(args, dataset_parallelization: bool, dataset_parallelization_context: Optional[str]) -> dict:
    """Build base configuration for experiments."""
    return {
        'n_train_traces': None, 'n_test_traces': None,
        'train_cases': None, 'test_cases': None,
        'ensure_train_variant_diversity': False,  # We handle stratification ourselves
        'ensure_test_variant_diversity': False,
        'use_same_traces_for_train_test': False,
        'allow_train_cases_in_test': False,  # Proper CV: no overlap
        'compute_marking_transition_map': False, 'sequential_sampling': False,
        # Large n_indices to keep all frames
        'n_indices': 10**9, 'n_per_run': None, 'independent_sampling': True,
        'prob_threshold': args.prob_threshold,
        'chunk_size': args.chunk_size,
        'conformance_switch_penalty_weight': 1.0,
        'merge_mismatched_boundaries': False,
        'conditioning_combine_fn': linear_prob_combiner,
        'max_hist_len': 3, 'conditioning_n_prev_labels': 3, 'use_collapsed_runs': True,
        'cost_function': 'linear',
        'model_move_cost': args.model_move_cost,
        'log_move_cost': 1.0,
        'tau_move_cost': 1e-6, 'non_sync_penalty': 1.0,
        'use_calibration': args.use_calibration, 'temp_bounds': (1.0, 10.0), 'temperature': None,
        'verbose': True, 'log_level': logging.INFO, 'round_precision': 2,
        'random_seed': args.seed,
        'save_model_path': None,
        'save_model': False,
        'parallel_processing': False,
        'dataset_parallelization': dataset_parallelization,
        'dataset_parallelization_context': dataset_parallelization_context,
        'max_workers': args.workers if dataset_parallelization else 1,
        # Conditioning history mode
        'conditioning_state_mode': args.state_mode,
        'conditioning_top_m': args.top_m,
        # Bound branching factor
        'candidate_top_p': args.candidate_top_p,
        'candidate_top_k': args.candidate_top_k,
        'candidate_min_k': args.candidate_min_k,
        'candidate_source': 'conditioned',
        'candidate_apply_to_sync': True,
        # Search space restrictions
        'restrict_log_moves': args.restrict_log_moves,
        'restrict_model_moves_to_tau': args.restrict_model_moves_to_tau,
    }


def export_identity(dataset_name: str, model_source: str) -> dict:
    """Identity of the softmax export feeding a run.

    R2: expected_len and case-id cannot detect CHANGED logits at the same length,
    so export identity must be compared, not merely recorded. Digest is over the
    sorted (relative path, size, mtime_ns) of every file under the export root --
    cheap, and sensitive to replacement or in-place modification.
    """
    home = Path.home()
    src = model_source.lower()
    if src == 'asformer':
        root = home / 'ASFormer' / 'results' / dataset_name
    elif src == 'mstcn2':
        root = home / 'MS-TCN2' / 'results' / dataset_name
    else:
        root = Path(__file__).resolve().parent / 'baselines' / 'DiffAct' / 'results' / dataset_name
    if not root.exists():
        return {'export_root': str(root), 'exists': False, 'digest': None}
    entries = []
    for f in sorted(root.rglob('*')):
        if f.is_file():
            st = f.stat()
            entries.append((str(f.relative_to(root)), st.st_size, st.st_mtime_ns))
    h = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return {'export_root': str(root), 'exists': True, 'n_files': len(entries), 'digest': h}

def run_single_fold_k_experiment(
    df: pd.DataFrame,
    softmax_lst: List[np.ndarray],
    train_case_ids: List[str],
    test_case_ids: List[str],
    alpha: float,
    weights: List[float],
    base_config: dict,
    fold: int,
    k: int,
    dataset_name: str,
    results_dir: Path,
    save_models: bool = False,
    resume: bool = False,
    allow_source_drift: bool = False,
) -> Dict[str, Any]:
    """
    Run a single experiment for one fold and one k value.

    Returns dict with fold, k, and all metrics.
    """
    print(
        f"    Running: fold={fold}, k={k}, n_train={len(train_case_ids)}, n_test={len(test_case_ids)}")

    save_model = save_models
    save_model_path = None
    if save_model:
        save_model_path = str(results_dir / f'petri_net_fold{fold}_k{k}')

    cfg = base_config.copy()
    # Per-video checkpoint directory is CONTENT-ADDRESSED on the effective decode
    # config. The library's checkpoint loader validates only column presence and
    # row count -- NOT the configuration that produced the rows. Without this, a
    # rerun at a different alpha/top_k/train-set would silently reuse the previous
    # config's per-video outputs and yield a wrong-but-plausible table.
    ckpt_dir = None
    if resume:
        sig = {k_: cfg.get(k_) for k_ in (
            'conditioning_alpha', 'conditioning_interpolation_weights',
            'candidate_top_k', 'candidate_top_p', 'candidate_min_k',
            'conditioning_state_mode', 'conditioning_top_m', 'chunk_size',
            'prob_threshold', 'max_hist_len', 'model_move_cost',
            'restrict_log_moves', 'restrict_model_moves_to_tau',
            # added after external review: all of these change decode output
            'random_seed', 'n_indices', 'n_per_run', 'sequential_sampling',
            'independent_sampling', 'cost_function', 'log_move_cost', 'tau_move_cost',
            'round_precision', 'conformance_switch_penalty_weight',
            'conditioning_n_prev_labels', 'candidate_source', 'candidate_apply_to_sync',
            'max_consecutive_tau_moves', 'merge_mismatched_boundaries',
            'use_calibration', 'temperature', 'non_sync_penalty',
            'compute_marking_transition_map', 'adaptive_chunk_sizing', 'max_chunk_size')}
        sig['alpha'] = alpha
        sig['weights'] = list(weights)
        # ORDER matters: _extract_cases orders data by this list, so do NOT sort.
        sig['train_cases'] = [str(c) for c in train_case_ids]   # determines the discovered net
        sig['export_identity'] = export_identity(dataset_name, model_source)   # R2
        digest = hashlib.sha256(
            json.dumps(sig, sort_keys=True, default=str).encode()).hexdigest()[:32]
        ckpt_dir = str(results_dir / f'case_outputs_fold{fold}_k{k}_{digest}')
        # Manifest sidecar: the hash isolates, this makes the isolation auditable
        # and catches a digest collision instead of silently reusing another config.
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        mf = Path(ckpt_dir) / 'checkpoint_manifest.json'
        cur_src = source_state()
        if mf.exists():
            prev = json.loads(mf.read_text())
            if prev.get('signature') != sig:
                raise SystemExit(
                    f'Checkpoint digest collision at {ckpt_dir}: manifest config differs from '
                    'the current run. Refusing to reuse. Delete the directory to recompute.')
            # R3: recording provenance without comparing it is not invalidation.
            prev_src = prev.get('source_state') or {}
            if (prev_src.get('git_head') != cur_src.get('git_head')
                    or prev_src.get('dirty_files') != cur_src.get('dirty_files')):
                if not allow_source_drift:
                    raise SystemExit(
                        f'Checkpoint source-state mismatch at {ckpt_dir}: checkpoints were produced '
                        f"at git {prev_src.get('git_head','?')[:12]} with a different working tree "
                        f"than the current {cur_src.get('git_head','?')[:12]}. Delete the directory "
                        'to recompute, or pass --allow-source-drift if the change is provably '
                        'irrelevant to decode output.')
        else:
            # R4: a manifest-less but non-empty directory would let stale checkpoints
            # be adopted by a freshly written manifest.
            existing = [q for q in Path(ckpt_dir).iterdir() if q.name != 'checkpoint_manifest.json']
            if existing:
                raise SystemExit(
                    f'Checkpoint directory {ckpt_dir} holds {len(existing)} file(s) but no manifest. '
                    'Refusing to adopt unattributed checkpoints. Delete the directory to recompute.')
            mf.write_text(json.dumps({'digest': digest, 'signature': sig, 'fold': fold, 'k': k,
                                      'source_state': cur_src}, indent=2, default=str))
    cfg.update({
        'case_output_dir': ckpt_dir,
        'resume_case_outputs': bool(resume),
        'conditioning_alpha': alpha,
        'conditioning_interpolation_weights': weights,
        'train_cases': train_case_ids,
        'n_train_traces': len(train_case_ids),
        'test_cases': test_case_ids,
        'n_test_traces': len(test_case_ids),
        'save_model': save_model,
        'save_model_path': save_model_path,
    })

    # Time the recovery process (try/except wraps entire computation including metrics)
    start_time = time.time()
    try:
        results_df, _, _ = incremental_softmax_recovery(
            df=df, softmax_lst=softmax_lst, **cfg)
        end_time = time.time()

        total_time = end_time - start_time
        n_test = results_df['case:concept:name'].nunique()
        avg_time_per_trace = total_time / n_test if n_test > 0 else 0

        # Save per-fold-k results
        csv_path = results_dir / f'fold{fold}_k{k}_results.csv'
        results_df.to_csv(csv_path, index=False)

        # Compute metrics
        metrics = compute_sktr_vs_argmax_metrics(
            str(csv_path),
            case_col='case:concept:name',
            sktr_pred_col='sktr_activity',
            argmax_pred_col='argmax_activity',
            gt_col='ground_truth',
            background=resolve_background_label(dataset_name),
            dataset_name=dataset_name,
        )

        print(f"    -> SKTR: acc={metrics['sktr']['acc']:.2f}, edit={metrics['sktr']['edit']:.2f}, "
              f"f1@25={metrics['sktr']['f1@25']:.2f} | Time: {total_time:.1f}s")

        return {
            'fold': fold,
            'k': k,
            'n_train': len(train_case_ids),
            'n_test': len(test_case_ids),
            'sktr_acc': metrics['sktr']['acc'],
            'sktr_edit': metrics['sktr']['edit'],
            'sktr_f1@10': metrics['sktr']['f1@10'],
            'sktr_f1@25': metrics['sktr']['f1@25'],
            'sktr_f1@50': metrics['sktr']['f1@50'],
            'argmax_acc': metrics['argmax']['acc'],
            'argmax_edit': metrics['argmax']['edit'],
            'argmax_f1@10': metrics['argmax']['f1@10'],
            'argmax_f1@25': metrics['argmax']['f1@25'],
            'argmax_f1@50': metrics['argmax']['f1@50'],
            'total_time_sec': round(total_time, 2),
            'avg_time_per_trace_sec': round(avg_time_per_trace, 3),
        }
    except Exception as e:
        elapsed = time.time() - start_time
        error_result = {
            'fold': fold,
            'k': k,
            'n_train': len(train_case_ids),
            'n_test': len(test_case_ids),
            'error': str(e),
            'error_type': type(e).__name__,
            'traceback': traceback.format_exc(),
            'total_time_sec': round(elapsed, 2),
            'avg_time_per_trace_sec': None,
            'sktr_acc': None,
            'sktr_edit': None,
            'sktr_f1@10': None,
            'sktr_f1@25': None,
            'sktr_f1@50': None,
            'argmax_acc': None,
            'argmax_edit': None,
            'argmax_f1@10': None,
            'argmax_f1@25': None,
            'argmax_f1@50': None,
        }
        print(f"    ERROR: {e}", file=sys.stderr, flush=True)
        print(error_result['traceback'], file=sys.stderr, flush=True)
        save_fold_k_error(results_dir, fold, k, error_result)
        return error_result


def aggregate_results(all_results: List[Dict]) -> pd.DataFrame:
    """
    Aggregate results across folds for each k value.

    Returns DataFrame with mean and std for each metric.
    """
    results_df = pd.DataFrame(all_results)

    # Filter out error rows
    valid_results = results_df[results_df['sktr_acc'].notna()]

    if len(valid_results) == 0:
        return pd.DataFrame()

    # Metrics to aggregate
    metrics = ['sktr_acc', 'sktr_edit', 'sktr_f1@10', 'sktr_f1@25', 'sktr_f1@50',
               'argmax_acc', 'argmax_edit', 'argmax_f1@10', 'argmax_f1@25', 'argmax_f1@50',
               'total_time_sec', 'avg_time_per_trace_sec']

    # Group by k and compute mean/std
    agg_dict = {}
    for metric in metrics:
        agg_dict[metric] = ['mean', 'std', 'count']

    aggregated = valid_results.groupby('k').agg(agg_dict).reset_index()

    # Flatten column names
    new_cols = ['k']
    for metric in metrics:
        new_cols.extend([f'{metric}_mean', f'{metric}_std', f'{metric}_n'])
    aggregated.columns = new_cols

    return aggregated


def plot_learning_curves(aggregated_df: pd.DataFrame, results_dir: Path,
                         dataset_name: str, model_source: str):
    """Create learning curve visualization."""
    if len(aggregated_df) == 0:
        print("No data to plot.")
        return

    sns.set_theme(style='whitegrid', context='notebook', palette='deep')
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    metrics_config = [
        ('acc', 'Accuracy'),
        ('edit', 'Edit Score'),
        ('f1@10', 'F1@10'),
        ('f1@25', 'F1@25'),
        ('f1@50', 'F1@50'),
    ]

    method_styles = {
        'sktr': {'color': '#1f77b4', 'marker': 'o', 'label': 'Ours', 'linestyle': '-'},
        'argmax': {'color': '#ff7f0e', 'marker': 's', 'label': 'Argmax', 'linestyle': '--'},
    }

    x = aggregated_df['k'].values

    legend_handles = None
    legend_labels = None
    for ax_idx, (metric_suffix, metric_label) in enumerate(metrics_config):
        ax = axes[ax_idx]

        for method, style in method_styles.items():
            mean_col = f'{method}_{metric_suffix}_mean'
            std_col = f'{method}_{metric_suffix}_std'

            if mean_col in aggregated_df.columns:
                y_mean = aggregated_df[mean_col].values
                y_std = aggregated_df[std_col].values

                ax.plot(x, y_mean, **{k: v for k, v in style.items() if k != 'label'},
                        label=style['label'])
                ax.fill_between(x, y_mean - y_std, y_mean +
                                y_std, alpha=0.2, color=style['color'])

        ax.set_xlabel('Number of Training Traces (k)')
        ax.set_ylabel(metric_label)
        ax.set_title(f'{metric_label} vs Training Set Size')
        ax.grid(True, alpha=0.3)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        ax.legend().remove()

    # Last subplot: legend only
    ax = axes[5]
    ax.axis('off')
    if legend_handles:
        ax.legend(
            legend_handles,
            legend_labels,
            loc='center',
            title='Method',
            frameon=True,
            fancybox=True,
            framealpha=1.0,
        )

    plt.suptitle(
        f'Learning Curve: {dataset_name} ({model_source})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = results_dir / 'learning_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved learning curve plot to {plot_path}")


_WARNED_MISSING_META = False


def check_existing_fold_k(results_dir: Path, fold: int, k: int, config: dict) -> bool:
    """
    Check if result for (fold, k) already exists with matching config.

    Returns True only if the result file exists AND the config matches.
    This prevents silently mixing results from different hyperparameter runs.
    """
    csv_path = results_dir / f'fold{fold}_k{k}_results.csv'
    if not csv_path.exists():
        return False

    # Check if config matches
    meta_path = results_dir / f'fold{fold}_k{k}_meta.json'
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                saved_config = json.load(f)
            # Check all hyperparameters that affect results
            keys_to_check = [
                'alpha', 'strategy', 'seed', 'chunk_size',
                'restrict_log_moves', 'restrict_model_moves_to_tau',
                'candidate_top_k', 'candidate_top_p', 'candidate_min_k',
                'prob_threshold', 'model_move_cost', 'state_mode', 'top_m',
                'unique_only', 'max_test_traces', 'test_cases',
            ]
            for key in keys_to_check:
                if saved_config.get(key) != config.get(key):
                    return False  # Config mismatch, rerun needed
        except (json.JSONDecodeError, KeyError):
            return False  # Can't verify, rerun to be safe
    else:
        # No metadata file; assume existing to avoid recompute
        global _WARNED_MISSING_META
        if not _WARNED_MISSING_META:
            print(
                "WARNING: Missing meta files for some results; skipping based on CSV existence.")
            _WARNED_MISSING_META = True
        return True

    return True


def save_fold_k_meta(results_dir: Path, fold: int, k: int, config: dict):
    """Save metadata for a fold/k result to enable config validation on resume."""
    meta_path = results_dir / f'fold{fold}_k{k}_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(config, f, indent=2)


def save_fold_k_error(results_dir: Path, fold: int, k: int, error_result: dict):
    """Save error details for a fold/k result."""
    error_path = results_dir / f'fold{fold}_k{k}_error.json'
    with open(error_path, 'w') as f:
        json.dump(error_result, f, indent=2)


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    print("=" * 70)
    print("K-FOLD CROSS-VALIDATION LEARNING CURVE EXPERIMENT")
    print("=" * 70)

    # Get CV configuration
    cv_config = get_dataset_cv_config(args.dataset)
    n_folds = cv_config['n_folds']
    default_k_values = cv_config['k_values']

    # Parse fold numbers
    if args.folds:
        fold_numbers = [int(f) for f in args.folds.split(',')]
    else:
        fold_numbers = list(range(1, n_folds + 1))

    # Parse k values
    if args.k_values:
        k_values = [int(k) for k in args.k_values.split(',')]
    else:
        k_values = default_k_values

    # Get hyperparameters - use dataset/model-specific defaults if not specified
    hp_defaults = get_dataset_hp_defaults(args.dataset, args.model)
    alpha = args.alpha if args.alpha is not None else hp_defaults['alpha']
    strategy = args.strategy if args.strategy is not None else hp_defaults['strategy']
    weights = HP_STRATEGIES[strategy]

    # Print whether defaults were used
    using_default_alpha = args.alpha is None
    using_default_strategy = args.strategy is None

    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model source: {args.model}")
    print(f"  Folds: {fold_numbers} (of {n_folds} total)")
    print(f"  k values: {k_values}")
    alpha_note = " (dataset default)" if using_default_alpha else " (user-specified)"
    strategy_note = " (dataset default)" if using_default_strategy else " (user-specified)"
    print(f"  Alpha: {alpha}{alpha_note}")
    print(f"  Strategy: {strategy} = {weights}{strategy_note}")
    print(f"  Seed: {args.seed}")

    # Parallelization config
    dataset_parallelization = args.inner_parallel
    max_workers = args.workers if dataset_parallelization else 1
    dataset_parallelization_context = "spawn" if dataset_parallelization else None
    n_parallel = args.parallel

    print(f"  Parallel experiments: {n_parallel}")
    print(
        f"  Outer backend: {'threads' if dataset_parallelization else 'processes'}")
    print(f"  Inner parallelization: {dataset_parallelization}")
    print(f"  Inner workers: {max_workers}")
    if dataset_parallelization:
        print(f"  Inner start method: {dataset_parallelization_context}")
        print(f"  Total processes: ~{n_parallel * max_workers} (inner)")
    else:
        print(f"  Total processes: ~{n_parallel}")
    if args.restrict_log_moves:
        print(f"  Restrict log moves: enabled")
    if args.restrict_model_moves_to_tau:
        print(f"  Restrict model moves to tau: enabled")

    # Setup results directory
    results_dir = Path(workspace_root) / 'results' / \
        args.dataset / 'kfold_learning_curve' / args.model
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Results directory: {results_dir}")

    # Effective-config banner: makes a non-canonical run announce itself
    # instead of silently costing an order of magnitude more compute.
    print(format_banner({
        'candidate_top_k': args.candidate_top_k,
        'candidate_top_p': args.candidate_top_p,
        'candidate_min_k': args.candidate_min_k,
        'candidate_source': 'conditioned',
        'state_mode': args.state_mode,
        'top_m': args.top_m,
        'max_hist_len': 3,
        'chunk_size': args.chunk_size,
        'prob_threshold': args.prob_threshold,
    }), flush=True)

    # Save experiment config
    config_path = results_dir / 'experiment_config.json'
    config = {
        'dataset': args.dataset,
        'model_source': args.model,
        'n_folds': n_folds,
        'fold_numbers': fold_numbers,
        'k_values': k_values,
        'alpha': alpha,
        'strategy': strategy,
        'weights': weights,
        'seed': args.seed,
        'workers': args.workers,
        'parallel': n_parallel,
        'inner_parallel': dataset_parallelization,
        'unique_only': args.unique_only,
        'max_test_traces': args.max_test_traces,
        'test_cases': args.test_cases,
        'resume': args.resume,
        'chunk_size': args.chunk_size,
        'prob_threshold': args.prob_threshold,
        'candidate_top_k': args.candidate_top_k,
        'candidate_top_p': args.candidate_top_p,
        'candidate_min_k': args.candidate_min_k,
        'candidate_source': 'conditioned',
        'state_mode': args.state_mode,
        'top_m': args.top_m,
        'max_hist_len': 3,
        'restrict_log_moves': args.restrict_log_moves,
        'restrict_model_moves_to_tau': args.restrict_model_moves_to_tau,
        # Provenance: makes the run reconstructable from the artifact alone.
        # NOTE: this dict is written to JSON only. The decoder kwargs dict is a
        # DIFFERENT dict earlier in this file -- adding keys there passes them
        # straight into incremental_softmax_recovery() and raises TypeError.
        'source_state': source_state(),
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Load dataset
    print("\n" + "-" * 70)
    print("STEP 1: Loading Dataset")
    print("-" * 70)
    df, softmax_lst = load_dataset(args.dataset, args.model)

    # Build video-to-case mapping
    print("\n" + "-" * 70)
    print("STEP 2: Building Video-to-Case Mapping")
    print("-" * 70)
    video_to_case = build_video_to_case_mapping(
        args.dataset, model_source=args.model)
    print(f"  Mapped {len(video_to_case)} video names to case IDs")

    # Get variant info for stratified sampling
    print("\n" + "-" * 70)
    print("STEP 3: Analyzing Variants")
    print("-" * 70)
    variant_df = get_variant_info(df, use_collapsed=True)
    print(f"  Found {len(variant_df)} unique variants")

    # Build base config
    base_config = build_base_config(
        args, dataset_parallelization, dataset_parallelization_context)

    # Main experiment loop
    print("\n" + "-" * 70)
    print("STEP 4: Running Experiments")
    print("-" * 70)

    # Build config dict for skip-existing validation (all params that affect results)
    run_config = {
        'test_cases': args.test_cases,
        'alpha': alpha,
        'strategy': strategy,
        'seed': args.seed,
        'chunk_size': args.chunk_size,
        'restrict_log_moves': args.restrict_log_moves,
        'restrict_model_moves_to_tau': args.restrict_model_moves_to_tau,
        'candidate_top_k': args.candidate_top_k,
        'candidate_top_p': args.candidate_top_p,
        'candidate_min_k': args.candidate_min_k,
        'prob_threshold': args.prob_threshold,
        'model_move_cost': args.model_move_cost,
        'state_mode': args.state_mode,
        'top_m': args.top_m,
        'unique_only': args.unique_only,
        'max_test_traces': args.max_test_traces,
    }

    # Pre-build all jobs
    all_results = []
    jobs_to_run = []

    for fold in fold_numbers:
        # Load fold split
        fold_case_ids = load_fold_case_ids(args.dataset, fold, video_to_case)
        train_pool = fold_case_ids['train']
        test_case_ids = fold_case_ids['test']

        # Apply unique-only filter (one representative per variant)
        if args.unique_only:
            orig_train = len(train_pool)
            orig_test = len(test_case_ids)
            train_pool = get_unique_representatives(
                train_pool, variant_df, seed=args.seed + fold)
            test_case_ids = get_unique_representatives(
                test_case_ids, variant_df, seed=args.seed + fold)
            print(
                f"  Fold {fold}: unique filter applied: train {orig_train}->{len(train_pool)}, test {orig_test}->{len(test_case_ids)}")

        # Explicit case selection takes precedence: a designed sample cares WHICH
        # videos are decoded, not just how many. Fails loudly on unknown ids
        # rather than silently decoding a smaller set than the design specifies.
        if args.test_cases:
            want = [c.strip() for c in args.test_cases.split(',') if c.strip()]
            available = {str(c) for c in test_case_ids}
            missing = [c for c in want if c not in available]
            if missing:
                raise SystemExit(
                    f"--test-cases: {len(missing)} id(s) not in fold {fold} test set: {missing[:10]}"
                    f" (fold test set has {len(available)} cases)")
            by_id = {str(c): c for c in test_case_ids}
            test_case_ids = [by_id[c] for c in want]
            print(f"  Fold {fold}: train_pool={len(train_pool)}, "
                  f"test={len(test_case_ids)} (EXPLICIT SELECTION)")
        elif args.max_test_traces is not None and len(test_case_ids) > args.max_test_traces:
            test_case_ids = test_case_ids[:args.max_test_traces]
            print(
                f"  Fold {fold}: train_pool={len(train_pool)}, test={len(test_case_ids)} (LIMITED)")
        else:
            print(
                f"  Fold {fold}: train_pool={len(train_pool)}, test={len(test_case_ids)}")

        for k in k_values:
            if k > len(train_pool):
                print(
                    f"    SKIPPED: fold={fold}, k={k} exceeds train pool size ({len(train_pool)})")
                continue

            # Check for existing result with matching config
            if args.skip_existing and check_existing_fold_k(results_dir, fold, k, run_config):
                print(f"    SKIPPED: fold={fold}, k={k} result already exists")
                # Load existing result
                csv_path = results_dir / f'fold{fold}_k{k}_results.csv'
                metrics = compute_sktr_vs_argmax_metrics(
                    str(csv_path),
                    case_col='case:concept:name',
                    sktr_pred_col='sktr_activity',
                    argmax_pred_col='argmax_activity',
                    gt_col='ground_truth',
                    background=resolve_background_label(args.dataset),
                    dataset_name=args.dataset,
                )
                result = {
                    'fold': fold, 'k': k, 'n_train': k, 'n_test': len(test_case_ids),
                    'sktr_acc': metrics['sktr']['acc'], 'sktr_edit': metrics['sktr']['edit'],
                    'sktr_f1@10': metrics['sktr']['f1@10'], 'sktr_f1@25': metrics['sktr']['f1@25'],
                    'sktr_f1@50': metrics['sktr']['f1@50'],
                    'argmax_acc': metrics['argmax']['acc'], 'argmax_edit': metrics['argmax']['edit'],
                    'argmax_f1@10': metrics['argmax']['f1@10'], 'argmax_f1@25': metrics['argmax']['f1@25'],
                    'argmax_f1@50': metrics['argmax']['f1@50'],
                    'total_time_sec': None, 'avg_time_per_trace_sec': None,
                }
                all_results.append(result)
                continue

            # Stratified sample k traces from training pool
            # Seed is deterministic per (base_seed, fold, k) for reproducibility
            sample_seed = args.seed + fold * 1000 + k
            sampled_train_ids = stratified_sample_k_traces(
                train_pool, k, variant_df, seed=sample_seed
            )

            # Add job to run
            jobs_to_run.append({
                'fold': fold,
                'k': k,
                'train_case_ids': sampled_train_ids,
                'test_case_ids': test_case_ids,
            })

    total_jobs = len(jobs_to_run)
    print(f"\n  Total jobs to run: {total_jobs}")
    print(f"  Running with {min(n_parallel, total_jobs)} parallel jobs")

    if total_jobs > 0:
        # Run in parallel using joblib (thread backend when inner parallelization is enabled)
        parallel_kwargs = {'n_jobs': min(
            n_parallel, total_jobs), 'verbose': 10}
        if dataset_parallelization:
            parallel_kwargs['backend'] = 'threading'

        raw_results = Parallel(**parallel_kwargs)(
            delayed(run_single_fold_k_experiment)(
                df=df,
                softmax_lst=softmax_lst,
                train_case_ids=job['train_case_ids'],
                test_case_ids=job['test_case_ids'],
                alpha=alpha,
                weights=weights,
                base_config=base_config,
                fold=job['fold'],
                k=job['k'],
                dataset_name=args.dataset,
                results_dir=results_dir,
                save_models=not args.no_save_models,
                resume=args.resume,
                allow_source_drift=args.allow_source_drift,
            )
            for job in jobs_to_run
        )

        # Process results
        for job, result in zip(jobs_to_run, raw_results):
            if result is not None:
                all_results.append(result)
                # Save metadata for config validation on resume
                save_fold_k_meta(
                    results_dir, job['fold'], job['k'], run_config)
                if result.get('sktr_acc') is not None and result.get('sktr_edit') is not None:
                    print(f"  fold={job['fold']}, k={job['k']} -> "
                          f"acc={result['sktr_acc']:.1f}, edit={result['sktr_edit']:.1f}")
                else:
                    print(
                        f"  fold={job['fold']}, k={job['k']} -> ERROR: {result.get('error')}")

        # Save intermediate results after parallel batch completes
        if all_results:
            interim_df = pd.DataFrame(all_results)
            interim_df.to_csv(results_dir / 'all_results.csv', index=False)

    # Final results
    print("\n" + "-" * 70)
    print("STEP 5: Aggregating Results")
    print("-" * 70)

    # Save all results
    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv(results_dir / 'all_results.csv', index=False)
    print(f"  Saved all results to {results_dir / 'all_results.csv'}")

    # Aggregate across folds
    aggregated_df = aggregate_results(all_results)
    if len(aggregated_df) > 0:
        aggregated_df.to_csv(
            results_dir / 'aggregated_results.csv', index=False)
        print(
            f"  Saved aggregated results to {results_dir / 'aggregated_results.csv'}")

        # Print summary
        print("\n" + "=" * 70)
        print("AGGREGATED RESULTS (mean +/- std across folds)")
        print("=" * 70)
        for _, row in aggregated_df.iterrows():
            k = int(row['k'])
            n = int(row['sktr_acc_n'])
            print(f"\nk={k} (n={n} folds):")
            print(f"  SKTR:   acc={row['sktr_acc_mean']:.2f}+/-{row['sktr_acc_std']:.2f}  "
                  f"edit={row['sktr_edit_mean']:.2f}+/-{row['sktr_edit_std']:.2f}  "
                  f"f1@25={row['sktr_f1@25_mean']:.2f}+/-{row['sktr_f1@25_std']:.2f}")
            print(f"  Argmax: acc={row['argmax_acc_mean']:.2f}+/-{row['argmax_acc_std']:.2f}  "
                  f"edit={row['argmax_edit_mean']:.2f}+/-{row['argmax_edit_std']:.2f}  "
                  f"f1@25={row['argmax_f1@25_mean']:.2f}+/-{row['argmax_f1@25_std']:.2f}")

        # Plot learning curves
        plot_learning_curves(aggregated_df, results_dir,
                             args.dataset, args.model)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {results_dir}")


if __name__ == '__main__':
    main()
