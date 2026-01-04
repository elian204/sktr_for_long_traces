#!/usr/bin/env python3
"""
Unified Variant Experiment

This script runs experiments on video activity segmentation datasets (50salads, gtea, breakfast)
by controlling which trace variants are used for training and testing.

Supports loading softmax predictions from:
- ASFormer model
- MS-TCN2 model
- Original pickle files

Structure:
1. Setup & Data Loading - Load dataset and analyze variants
2. Hyperparameter Search - Fixed train/test split, sweep over alpha and interpolation strategies
3. Final Experiment - Fixed hyperparameters, sweep over number of training variants

Usage:
    python variant_experiment.py

Configuration:
    Edit the CONFIGURATION section below to select dataset, model source, and experiment parameters.
"""

import math
import os
import sys
import logging
import time
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for terminal/tmux
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

# =============================================================================
# SETUP
# =============================================================================

# Setup workspace path
workspace_root = '/home/dsi/eli-bogdanov/sktr_for_long_traces'
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from src.utils import (
    prepare_df, prepare_df_from_model, linear_prob_combiner,
    get_variant_info, get_cases_for_variants, select_variants_for_experiment
)
from src.incremental_softmax_recovery import incremental_softmax_recovery
from src.evaluation import compute_sktr_vs_argmax_metrics

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
# COMMAND-LINE ARGUMENTS
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run variant experiments on video activity segmentation datasets.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-d', '--dataset',
        type=str,
        choices=['50salads', 'gtea', 'breakfast'],
        default='50salads',
        help='Dataset to run experiments on'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        choices=['asformer', 'mstcn2', 'original'],
        default='asformer',
        help='Model source for softmax predictions'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=20,
        help='Number of parallel workers for dataset processing'
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--no-save-models',
        action='store_true',
        help='Disable saving Petri net visualizations'
    )
    parser.add_argument(
        '--skip-hp-search',
        action='store_true',
        help='Skip hyperparameter search and use default hyperparameters'
    )
    parser.add_argument(
        '--state-mode',
        type=str,
        choices=['exact', 'topm'],
        default='topm',
        help='Conditioning state mode: exact (full history match) or topm (top-m states)'
    )
    parser.add_argument(
        '--top-m',
        type=int,
        default=3,
        help='Number of top states to consider when state-mode=topm'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=11,
        help='Chunk size for conformance checking'
    )
    parser.add_argument(
        '--prob-threshold',
        type=float,
        default=1e-6,
        help='Minimum probability threshold for pruning'
    )
    parser.add_argument(
        '--candidate-top-k',
        type=int,
        default=15,
        help='Max candidate labels per timestamp (top-K)'
    )
    parser.add_argument(
        '--candidate-top-p',
        type=float,
        default=0.9,
        help='Cumulative probability cutoff for candidate labels (top-p)'
    )
    parser.add_argument(
        '--candidate-min-k',
        type=int,
        default=1,
        help='Minimum candidate labels per timestamp'
    )
    parser.add_argument(
        '-p', '--parallel-runs',
        type=int,
        default=1,
        help='Number of hyperparameter combinations to run in parallel (default: 1 = sequential)'
    )
    parser.add_argument(
        '--unique-train-variants',
        action='store_true',
        help='Train on unique variants only (one representative video per variant). '
             'For Breakfast: 267 unique variants instead of 1712 videos.'
    )
    parser.add_argument(
        '--unique-test-variants',
        action='store_true',
        help='Test on unique variants only (one representative video per variant). '
             'Useful for faster hyperparameter search on large datasets like Breakfast.'
    )
    return parser.parse_args()


# =============================================================================
# CONFIGURATION
# =============================================================================

# Parse command-line arguments (only when running as script)
if __name__ == '__main__':
    args = parse_args()
    DATASET_NAME = args.dataset
    MODEL_SOURCE = args.model
    RANDOM_SEED = args.seed
    N_DATASET_WORKERS = args.workers
    SAVE_PROCESS_MODELS = not args.no_save_models
    SKIP_HP_SEARCH = args.skip_hp_search
    CONDITIONING_STATE_MODE = args.state_mode
    CONDITIONING_TOP_M = args.top_m
    CHUNK_SIZE = args.chunk_size
    N_PARALLEL_RUNS = args.parallel_runs
    PROB_THRESHOLD = args.prob_threshold
    CANDIDATE_TOP_K = args.candidate_top_k
    CANDIDATE_TOP_P = args.candidate_top_p
    CANDIDATE_MIN_K = args.candidate_min_k
    UNIQUE_TRAIN_VARIANTS = args.unique_train_variants
    UNIQUE_TEST_VARIANTS = args.unique_test_variants
else:
    # Default values when imported as module
    DATASET_NAME = '50salads'
    MODEL_SOURCE = 'asformer'
    RANDOM_SEED = 42
    N_DATASET_WORKERS = 20
    SAVE_PROCESS_MODELS = True
    SKIP_HP_SEARCH = False
    CONDITIONING_STATE_MODE = 'topm'
    CONDITIONING_TOP_M = 3
    CHUNK_SIZE = 11
    N_PARALLEL_RUNS = 1
    PROB_THRESHOLD = 1e-6
    CANDIDATE_TOP_K = 15
    CANDIDATE_TOP_P = 0.9
    CANDIDATE_MIN_K = 1
    UNIQUE_TRAIN_VARIANTS = False
    UNIQUE_TEST_VARIANTS = False

# --- Parallelization ---
DATASET_PARALLELIZATION = True

# --- Hyperparameter Search Configuration ---
HP_ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]
HP_STRATEGIES = {
    # Top 2 strategies based on 50salads and GTEA HP search results:
    # - trigram_heavy: best on GTEA (rank 13.57), 3rd on 50salads (rank 14.26)
    # - unigram_super_heavy: best on 50salads (rank 13.31), 2nd on GTEA (rank 14.31)
    'trigram_heavy': [0.1, 0.15, 0.75],
    'unigram_super_heavy': [0.75, 0.15, 0.1],
}

# Default hyperparameters (used when --skip-hp-search is set)
DEFAULT_ALPHA = 0.5
DEFAULT_STRATEGY = 'trigram_heavy'
DEFAULT_WEIGHTS = HP_STRATEGIES[DEFAULT_STRATEGY]

# --- Training Sweep Configuration ---
# Predefined sweep ranges per dataset (based on unique variants in ground truth)
# GTEA: 28 videos, 28 unique variants (1 trace per variant)
# 50salads: 50 videos, 50 unique variants (1 trace per variant)
# Breakfast: 1712 videos, 267 unique variants (~6.4 traces per variant)
TRAIN_VARIANT_SWEEP = {
    '50salads': [1, 5, 10, 20, 30, 40, 50],    # 50 unique variants
    'gtea': [1, 5, 10, 15, 20, 28],             # 28 unique variants
    'breakfast': [1, 5, 10, 25, 50, 100, 150, 200, 267],  # 267 unique variants
}

# Datasets with multiple traces per variant need special handling:
# - Training: Use ONE representative trace per variant (for model building)
# - Testing: Predict on ALL traces belonging to selected variants
# For datasets where each trace is unique (GTEA, 50salads), this is equivalent to trace-based selection.
USE_VARIANT_BASED_SELECTION = {
    '50salads': False,   # Each trace is unique, no special handling needed
    'gtea': False,       # Each trace is unique, no special handling needed
    'breakfast': True,   # Multiple traces per variant, use variant-based selection
}

# Variant selection mode when using variant-based selection
# Options: "random" or "frequency"
VARIANT_SELECTION_MODE = "random"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_dataset(dataset_name: str, model_source: str):
    """Load dataset from specified source."""
    # Warn about potentially outdated data for breakfast with original source
    if dataset_name == 'breakfast' and model_source == 'original':
        print("WARNING: Using 'original' model source with breakfast dataset.")
        print("         The original pickle-based data may be outdated.")
        print("         Consider using 'asformer' or 'mstcn2' instead.")
    print(f"Loading {dataset_name} from {model_source}...")

    if model_source == 'original':
        result = prepare_df(dataset_name)
        df, softmax_lst = result[:2]
    else:
        df, softmax_lst = prepare_df_from_model(dataset_name, model_source)

    print(f"  Loaded {len(softmax_lst)} cases, {len(df)} events")
    return df, softmax_lst


def build_base_config() -> dict:
    """Build base configuration for experiments."""
    return {
        'n_train_traces': None, 'n_test_traces': None,
        'train_cases': None, 'test_cases': None,
        'ensure_train_variant_diversity': True,  # Key for variant-based selection
        'ensure_test_variant_diversity': True,
        'use_same_traces_for_train_test': False, 'allow_train_cases_in_test': True,
        'compute_marking_transition_map': False, 'sequential_sampling': True,
        'n_indices': None, 'n_per_run': 10000, 'independent_sampling': True,
        'prob_threshold': PROB_THRESHOLD, 'chunk_size': CHUNK_SIZE, 'conformance_switch_penalty_weight': 1.0,
        'merge_mismatched_boundaries': False, 'conditioning_combine_fn': linear_prob_combiner,
        'max_hist_len': 3, 'conditioning_n_prev_labels': 3, 'use_collapsed_runs': True,
        'cost_function': 'linear', 'model_move_cost': 1.0, 'log_move_cost': 1.0,
        'tau_move_cost': 0.0, 'non_sync_penalty': 1.0,
        'use_calibration': True, 'temp_bounds': (1.0, 10.0), 'temperature': None,
        'verbose': True, 'log_level': logging.INFO, 'round_precision': 2,
        'random_seed': RANDOM_SEED,
        'save_model_path': None,
        'save_model': False,
        'parallel_processing': False,
        'dataset_parallelization': DATASET_PARALLELIZATION,
        'max_workers': N_DATASET_WORKERS,
        # Conditioning history mode
        'conditioning_state_mode': CONDITIONING_STATE_MODE,
        'conditioning_top_m': CONDITIONING_TOP_M,
        # Bound branching factor
        'candidate_top_p': CANDIDATE_TOP_P,
        'candidate_top_k': CANDIDATE_TOP_K,
        'candidate_min_k': CANDIDATE_MIN_K,
        'candidate_source': 'observed',
        'candidate_apply_to_sync': True,
    }


def resolve_background_label(dataset_name: str) -> Optional[str]:
    """Return None to use auto background resolution."""
    return None


def build_result_filename(dataset_name: str, model_source: str, prefix: str,
                          alpha: float, strategy: str,
                          unique_train: bool = False, unique_test: bool = False) -> str:
    """Build a result filename (unique flags intentionally do not affect naming)."""
    return f"{dataset_name}_{model_source}_{prefix}_alpha_{alpha}_weights_{strategy}.csv"


def _result_meta_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".meta.json")


def check_existing_result(results_dir: Path, dataset_name: str, model_source: str,
                          prefix: str, alpha: float, strategy: str,
                          unique_train: bool = False, unique_test: bool = False):
    """Check if result CSV exists and load metrics from it."""
    filename = build_result_filename(dataset_name, model_source, prefix, alpha, strategy,
                                     unique_train, unique_test)
    csv_path = results_dir / filename
    if not csv_path.exists():
        return None
    meta_path = _result_meta_path(csv_path)
    if meta_path.exists():
        try:
            with meta_path.open('r') as f:
                meta = json.load(f)
            meta_unique_train = bool(meta.get('unique_train_variants', False))
            meta_unique_test = bool(meta.get('unique_test_variants', False))
        except Exception as e:
            print(f"  Warning: Could not read metadata {meta_path}: {e}")
            return None
        if meta_unique_train != unique_train or meta_unique_test != unique_test:
            return None
    else:
        if unique_train or unique_test:
            return None

    try:
        metrics = compute_sktr_vs_argmax_metrics(
            str(csv_path),
            case_col='case:concept:name',
            sktr_pred_col='sktr_activity',
            argmax_pred_col='argmax_activity',
            gt_col='ground_truth',
            background=resolve_background_label(dataset_name),
            dataset_name=dataset_name,
        )
        return metrics
    except Exception as e:
        print(f"  Warning: Could not load existing result from {csv_path}: {e}")
        return None


def run_single_experiment(n_train_variants, train_cases, test_cases, alpha, strategy, weights,
                          idx, total, df, softmax_lst, base_cfg, results_dir,
                          prefix, dataset_name, model_source, skip_existing=True,
                          save_models=True, unique_train=False, unique_test=False):
    """Run a single experiment and return metrics including timing info.

    Parameters
    ----------
    n_train_variants : int
        Number of training variants (for display/logging purposes).
    train_cases : List[str] or None
        Explicit list of training case IDs. If provided, these are used directly.
        For variant-based selection, this contains one representative per variant.
    test_cases : List[str] or None
        Explicit list of test case IDs. If None, uses all available cases.
        For variant-based selection, this contains all cases for selected variants.
    unique_train : bool
        Whether unique train variants mode is enabled (for filename).
    unique_test : bool
        Whether unique test variants mode is enabled (for filename).
    """
    n_train_display = len(train_cases) if train_cases else n_train_variants
    n_test_display = len(test_cases) if test_cases else 'all'
    print(f"[{idx}/{total}] n_variants={n_train_variants}, n_train={n_train_display}, n_test={n_test_display}, alpha={alpha}, strategy={strategy}")

    # Check for existing result to enable resume
    if skip_existing:
        existing_metrics = check_existing_result(
            results_dir, dataset_name, model_source, prefix, alpha, strategy,
            unique_train, unique_test)
        if existing_metrics is not None:
            print(f"  -> SKIPPED (already exists)")
            return {
                'n_train_variants': n_train_variants,
                'n_train_traces': n_train_display,
                'n_test_cases': n_test_display,
                'alpha': alpha, 'strategy': strategy,
                'sktr_acc': existing_metrics['sktr']['acc'], 'sktr_edit': existing_metrics['sktr']['edit'],
                'sktr_f1@10': existing_metrics['sktr']['f1@10'], 'sktr_f1@25': existing_metrics['sktr']['f1@25'],
                'sktr_f1@50': existing_metrics['sktr']['f1@50'],
                'argmax_acc': existing_metrics['argmax']['acc'], 'argmax_edit': existing_metrics['argmax']['edit'],
                'argmax_f1@10': existing_metrics['argmax']['f1@10'], 'argmax_f1@25': existing_metrics['argmax']['f1@25'],
                'argmax_f1@50': existing_metrics['argmax']['f1@50'],
                'total_time_sec': None, 'avg_time_per_trace_sec': None,
            }

    save_model = SAVE_PROCESS_MODELS and save_models
    save_model_path = None
    if save_model:
        save_model_path = str(results_dir / f'petri_net_v{n_train_variants}')

    cfg = base_cfg.copy()
    cfg.update({
        'conditioning_alpha': alpha,
        'conditioning_interpolation_weights': weights,
        'train_cases': train_cases,
        'n_train_traces': len(train_cases) if train_cases else n_train_variants,
        'test_cases': test_cases,
        'n_test_traces': len(test_cases) if test_cases else None,
        'save_model': save_model,
        'save_model_path': save_model_path,
    })

    # Time the recovery process
    start_time = time.time()
    results_df, _, _ = incremental_softmax_recovery(
        df=df, softmax_lst=softmax_lst, **cfg)
    end_time = time.time()

    total_time = end_time - start_time
    n_test = results_df['case:concept:name'].nunique()
    avg_time_per_trace = total_time / n_test if n_test > 0 else 0

    filename = build_result_filename(dataset_name, model_source, prefix, alpha, strategy,
                                      unique_train, unique_test)
    csv_path = results_dir / filename
    results_df.to_csv(csv_path, index=False)
    meta_path = _result_meta_path(csv_path)
    with meta_path.open('w') as f:
        json.dump({
            'unique_train_variants': unique_train,
            'unique_test_variants': unique_test,
        }, f, indent=2, sort_keys=True)

    metrics = compute_sktr_vs_argmax_metrics(
        str(csv_path),
        case_col='case:concept:name',
        sktr_pred_col='sktr_activity',
        argmax_pred_col='argmax_activity',
        gt_col='ground_truth',
        background=resolve_background_label(dataset_name),
        dataset_name=dataset_name,
    )

    print(f"  -> Time: {total_time:.1f}s total, {avg_time_per_trace:.2f}s/trace ({n_test} traces)")

    return {
        'n_train_variants': n_train_variants,
        'n_train_traces': n_train_display,
        'n_test_cases': n_test_display if isinstance(n_test_display, int) else n_test,
        'alpha': alpha, 'strategy': strategy,
        'sktr_acc': metrics['sktr']['acc'], 'sktr_edit': metrics['sktr']['edit'],
        'sktr_f1@10': metrics['sktr']['f1@10'], 'sktr_f1@25': metrics['sktr']['f1@25'],
        'sktr_f1@50': metrics['sktr']['f1@50'],
        'argmax_acc': metrics['argmax']['acc'], 'argmax_edit': metrics['argmax']['edit'],
        'argmax_f1@10': metrics['argmax']['f1@10'], 'argmax_f1@25': metrics['argmax']['f1@25'],
        'argmax_f1@50': metrics['argmax']['f1@50'],
        'total_time_sec': round(total_time, 2), 'avg_time_per_trace_sec': round(avg_time_per_trace, 3),
    }


def plot_sweep_results(sweep_summary_df: pd.DataFrame, results_dir: Path,
                       dataset_name: str, model_source: str):
    """Create visualization for sweep results."""
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
        'sktr': {'color': '#1f77b4', 'marker': 'o', 'label': 'SKTR', 'linestyle': '-'},
        'argmax': {'color': '#ff7f0e', 'marker': 's', 'label': 'Argmax', 'linestyle': '--'},
    }

    # Collect all plot columns for y-axis scaling
    plot_cols = []
    for metric_suffix, _ in metrics_config:
        for method in method_styles:
            col_name = f'{method}_{metric_suffix}'
            if col_name in sweep_summary_df.columns:
                plot_cols.append(col_name)

    # Compute y-axis limits (floor min to 10, ceil max to 10, min top of 80)
    y_limits = None
    y_ticks = None
    if plot_cols:
        y_min = sweep_summary_df[plot_cols].min().min()
        y_max = sweep_summary_df[plot_cols].max().max()
        y_lower = math.floor(y_min / 10) * 10
        y_max = max(y_max, 80)
        tick_start = min(y_lower, 50)  # Start from 50 or lower if data goes below
        tick_end = math.ceil(y_max / 10) * 10
        y_limits = (tick_start, tick_end)
        y_ticks = list(range(int(tick_start), int(tick_end) + 1, 10))

    # Determine x-axis column (prefer n_train_variants, fallback to n_train_traces)
    x_col = 'n_train_variants' if 'n_train_variants' in sweep_summary_df.columns else 'n_train_traces'

    for idx, (metric_suffix, title) in enumerate(metrics_config):
        ax = axes[idx]
        for method, style in method_styles.items():
            col_name = f'{method}_{metric_suffix}'
            if col_name in sweep_summary_df.columns:
                sns.lineplot(
                    x=sweep_summary_df[x_col],
                    y=sweep_summary_df[col_name],
                    ax=ax,
                    linewidth=2.5,
                    markersize=9,
                    **style,
                )
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Variants')
        ax.set_ylabel('Score')
        ax.set_xticks(sweep_summary_df[x_col].unique())
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        if y_ticks is not None:
            ax.set_yticks(y_ticks)
        ax.legend().remove()

    # Legend in last subplot
    ax_legend = axes[5]
    ax_legend.axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    ax_legend.legend(
        handles, labels,
        loc='center', title='Method',
        fontsize=14, title_fontsize=16,
        frameon=True, fancybox=True, shadow=True,
    )

    plt.suptitle(
        f'Performance vs. Training Variants ({dataset_name} - {model_source})',
        fontsize=18
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
    plot_path = results_dir / f'{dataset_name}_{model_source}_sweep_plots.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.close()


def write_experiment_config(results_dir: Path, config: Dict[str, Any], filename: str) -> None:
    """Write experiment configuration as JSON in the results directory."""
    results_dir.mkdir(parents=True, exist_ok=True)
    config_path = results_dir / filename
    with config_path.open('w') as f:
        json.dump(config, f, indent=2, sort_keys=True)
    print(f"Saved config: {config_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print(f"\nWorkspace: {workspace_root}")
    print(f"Running in tmux: {'TMUX' in os.environ}")
    print("=" * 70)
    print("Configuration:")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Model Source: {MODEL_SOURCE}")
    print(f"  State Mode: {CONDITIONING_STATE_MODE} (top_m={CONDITIONING_TOP_M})")
    print(f"  Chunk Size: {CHUNK_SIZE}")
    print(f"  Prob Threshold: {PROB_THRESHOLD}")
    print(f"  Candidate Top-K: {CANDIDATE_TOP_K}")
    print(f"  Candidate Top-P: {CANDIDATE_TOP_P}")
    print(f"  Candidate Min-K: {CANDIDATE_MIN_K}")
    print(f"  Workers: {N_DATASET_WORKERS}")
    print(f"  Parallel HP runs: {N_PARALLEL_RUNS}")
    print(f"  Skip HP Search: {SKIP_HP_SEARCH}")
    print(f"  Unique Train Variants: {UNIQUE_TRAIN_VARIANTS}")
    print(f"  Unique Test Variants: {UNIQUE_TEST_VARIANTS}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Load Dataset
    # -------------------------------------------------------------------------
    df, softmax_lst = load_dataset(DATASET_NAME, MODEL_SOURCE)

    print(f"\nDataset Statistics:")
    print(f"  Events: {len(df):,}")
    print(f"  Cases: {df['case:concept:name'].nunique()}")
    print(f"  Activities: {df['concept:name'].nunique()}")

    # -------------------------------------------------------------------------
    # Analyze Variants
    # -------------------------------------------------------------------------
    variant_df = get_variant_info(df, use_collapsed=True)
    n_unique_variants = len(variant_df)
    print(f"  Unique variants: {n_unique_variants}")

    # Setup results directory
    results_dir = Path(workspace_root) / 'results' / DATASET_NAME / 'variant_experiment' / MODEL_SOURCE
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults directory: {results_dir}")

    # Get all case IDs
    all_case_ids = df['case:concept:name'].unique().astype(str).tolist()

    # Build base config
    base_config = build_base_config()
    print("Base config ready.")

    # Check if we need variant-based selection for this dataset
    use_variant_selection = USE_VARIANT_BASED_SELECTION.get(DATASET_NAME, False)
    n_jobs = N_PARALLEL_RUNS if N_PARALLEL_RUNS is not None else -1

    # Warn if unique variant flags are set but not applicable
    if (UNIQUE_TRAIN_VARIANTS or UNIQUE_TEST_VARIANTS) and not use_variant_selection:
        print(f"\nWarning: --unique-train/test-variants has no effect for {DATASET_NAME}")
        print(f"  (Each trace is already unique in this dataset)")
    else:
        if UNIQUE_TRAIN_VARIANTS:
            print(f"\nUnique train variants mode: Training on {n_unique_variants} representative videos only")
        if UNIQUE_TEST_VARIANTS:
            print(f"\nUnique test variants mode: Testing on {n_unique_variants} representative videos only")

    experiment_config = {
        'dataset': DATASET_NAME,
        'model_source': MODEL_SOURCE,
        'random_seed': RANDOM_SEED,
        'variant_selection_mode': VARIANT_SELECTION_MODE,
        'use_variant_based_selection': use_variant_selection,
        'unique_train_variants': UNIQUE_TRAIN_VARIANTS,
        'unique_test_variants': UNIQUE_TEST_VARIANTS,
        'n_unique_variants': n_unique_variants,
        'state_mode': CONDITIONING_STATE_MODE,
        'top_m': CONDITIONING_TOP_M,
        'chunk_size': CHUNK_SIZE,
        'prob_threshold': PROB_THRESHOLD,
        'candidate_top_k': CANDIDATE_TOP_K,
        'candidate_top_p': CANDIDATE_TOP_P,
        'candidate_min_k': CANDIDATE_MIN_K,
        'workers': N_DATASET_WORKERS,
        'skip_hp_search': SKIP_HP_SEARCH,
        'train_variant_sweep': TRAIN_VARIANT_SWEEP.get(DATASET_NAME, list(range(1, 11))),
        'n_parallel_runs': N_PARALLEL_RUNS,
        'evaluation_background': resolve_background_label(DATASET_NAME),
    }

    # =========================================================================
    # Part A: Hyperparameter Search (optional)
    # =========================================================================
    if SKIP_HP_SEARCH:
        print("\n" + "=" * 70)
        print("Part A: Hyperparameter Search - SKIPPED (using defaults)")
        print("=" * 70)
        print(f"  Default alpha: {DEFAULT_ALPHA}")
        print(f"  Default strategy: {DEFAULT_STRATEGY}")
        print(f"  Default weights: {DEFAULT_WEIGHTS}")

        # Use default hyperparameters
        best_hp = {
            'alpha': DEFAULT_ALPHA,
            'strategy': DEFAULT_STRATEGY,
        }
        experiment_config['hyperparameter_search'] = {
            'enabled': False,
            'default_alpha': DEFAULT_ALPHA,
            'default_strategy': DEFAULT_STRATEGY,
            'default_weights': DEFAULT_WEIGHTS,
        }
    else:
        print("\n" + "=" * 70)
        print("Part A: Hyperparameter Search")
        print("=" * 70)

        # For hyperparameter search, use all variants
        HP_N_VARIANTS = n_unique_variants

        if use_variant_selection:
            # Variant-based: get train_cases (one per variant) and test_cases (all for those variants)
            hp_train_cases_unique, hp_test_cases_all, _ = select_variants_for_experiment(
                variant_df, n_variants=HP_N_VARIANTS, selection_mode=VARIANT_SELECTION_MODE, seed=RANDOM_SEED
            )
            # Apply unique train/test variant settings
            hp_train_cases = hp_train_cases_unique if UNIQUE_TRAIN_VARIANTS else hp_test_cases_all
            hp_test_cases = hp_train_cases_unique if UNIQUE_TEST_VARIANTS else hp_test_cases_all

            print(f"Hyperparameter Search Setup (variant-based selection):")
            print(f"  Training variants: {HP_N_VARIANTS}")
            print(f"  Training traces: {len(hp_train_cases)} ({'unique' if UNIQUE_TRAIN_VARIANTS else 'all'})")
            print(f"  Test traces: {len(hp_test_cases)} ({'unique' if UNIQUE_TEST_VARIANTS else 'all'})")
            print(f"  Variant selection mode: {VARIANT_SELECTION_MODE}")
        else:
            # Standard: use n_train_traces directly, test on all
            hp_train_cases = None
            hp_test_cases = None
            print(f"Hyperparameter Search Setup:")
            print(f"  Training traces (unique variants): {HP_N_VARIANTS}")
            print(f"  Test cases: all")

        print(f"  Alphas: {HP_ALPHAS}")
        print(f"  Strategies: {list(HP_STRATEGIES.keys())}")
        print(f"  Total experiments: {len(HP_ALPHAS) * len(HP_STRATEGIES)}")

        # Run hyperparameter search
        hp_results_dir = results_dir / 'hyperparameter_search'
        hp_results_dir.mkdir(parents=True, exist_ok=True)

        hp_params = [(HP_N_VARIANTS, hp_train_cases, hp_test_cases, a, s, w)
                     for a in HP_ALPHAS
                     for s, w in HP_STRATEGIES.items()]

        print(f"\nRunning {len(hp_params)} hyperparameter experiments...")
        print("=" * 60)

        # Save model only on first experiment (safe since N_PARALLEL_RUNS=1 for HP search)
        # All HP experiments use same training data, so model is identical
        hp_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_single_experiment)(
                n_variants, train_cases, test_cases, alpha, strategy, weights,
                i, len(hp_params), df, softmax_lst, base_config, hp_results_dir,
                "hp_search", DATASET_NAME, MODEL_SOURCE, save_models=(i == 1),
                unique_train=UNIQUE_TRAIN_VARIANTS, unique_test=UNIQUE_TEST_VARIANTS
            )
            for i, (n_variants, train_cases, test_cases, alpha, strategy, weights) in enumerate(hp_params, 1)
        )

        hp_summary_df = pd.DataFrame(hp_results).sort_values('sktr_acc', ascending=False)
        rank_metrics = ['sktr_acc', 'sktr_edit', 'sktr_f1@25']
        ranks = hp_summary_df[rank_metrics].rank(ascending=False, method='average')
        hp_summary_df['avg_rank'] = ranks.mean(axis=1)
        hp_summary_path = hp_results_dir / f"{DATASET_NAME}_{MODEL_SOURCE}_hp_search_summary.csv"
        hp_summary_df.to_csv(hp_summary_path, index=False)
        print(f"\nSaved: {hp_summary_path}")

        # Display hyperparameter search results
        print("\nHyperparameter Search Results (sorted by SKTR accuracy):\n")
        print(hp_summary_df[['alpha', 'strategy', 'avg_rank', 'sktr_acc', 'argmax_acc',
              'sktr_edit', 'argmax_edit', 'sktr_f1@25', 'argmax_f1@25']].to_string())

        # Best hyperparameters (average rank across metrics)
        best_hp = (hp_summary_df
                   .sort_values(['avg_rank', 'sktr_acc'], ascending=[True, False])
                   .iloc[0]
                   .to_dict())
        print(f"\nBest hyperparameters (avg-rank on {', '.join(rank_metrics)}):")
        print(f"  Alpha: {best_hp['alpha']}")
        print(f"  Strategy: {best_hp['strategy']}")
        print(f"  SKTR Accuracy: {best_hp['sktr_acc']:.4f}")
        print(f"  SKTR Edit: {best_hp['sktr_edit']:.4f}")

        experiment_config['hyperparameter_search'] = {
            'enabled': True,
            'alphas': HP_ALPHAS,
            'strategies': HP_STRATEGIES,
            'rank_metrics': rank_metrics,
            'best': {
                'alpha': best_hp['alpha'],
                'strategy': best_hp['strategy'],
                'weights': HP_STRATEGIES[best_hp['strategy']],
                'avg_rank': best_hp['avg_rank'],
                'sktr_acc': best_hp['sktr_acc'],
                'sktr_edit': best_hp['sktr_edit'],
                'sktr_f1@25': best_hp['sktr_f1@25'],
            },
        }

        # Visualize hyperparameter search results
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        plot_cols = ['sktr_acc', 'sktr_edit', 'sktr_f1@25']
        y_min = hp_summary_df[plot_cols].min().min()
        y_max = hp_summary_df[plot_cols].max().max()
        y_lower = math.floor(y_min / 10) * 10
        y_max = max(y_max, 80)
        tick_start = min(y_lower, 50)  # Start from 50 or lower if data goes below
        tick_end = math.ceil(y_max / 10) * 10
        y_limits = (tick_start, tick_end)
        y_ticks = list(range(int(tick_start), int(tick_end) + 1, 10))

        for ax, metric in zip(axes, plot_cols):
            pivot = hp_summary_df.pivot(index='alpha', columns='strategy', values=metric)
            pivot.plot(kind='bar', ax=ax, rot=0)
            ax.set_xlabel('Alpha')
            ax.set_ylabel(metric.replace('sktr_', '').replace('_', ' ').title())
            ax.set_title(f'SKTR {metric.replace("sktr_", "").replace("_", " ").title()}')
            ax.legend(title='Strategy', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            if y_limits is not None:
                ax.set_ylim(*y_limits)
            if y_ticks is not None:
                ax.set_yticks(y_ticks)

        plt.suptitle(f'Hyperparameter Search Results ({DATASET_NAME} - {MODEL_SOURCE})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
        plt.savefig(hp_results_dir / f'{DATASET_NAME}_{MODEL_SOURCE}_hp_search_plots.png', dpi=150, bbox_inches='tight')
        print(f"Saved plot: {hp_results_dir / f'{DATASET_NAME}_{MODEL_SOURCE}_hp_search_plots.png'}")
        plt.close()

    # =========================================================================
    # Part B: Final Experiment (Training Sweep)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Part B: Final Experiment (Training Sweep)")
    print("=" * 70)

    # Use best hyperparameters from search
    FINAL_ALPHA = best_hp['alpha']
    FINAL_STRATEGY = best_hp['strategy']
    FINAL_WEIGHTS = HP_STRATEGIES[best_hp['strategy']]

    # Get sweep range for this dataset
    sweep_variant_counts = TRAIN_VARIANT_SWEEP.get(DATASET_NAME, list(range(1, 11)))

    print(f"Final Experiment Configuration:")
    print(f"  Alpha: {FINAL_ALPHA}")
    print(f"  Strategy: {FINAL_STRATEGY}")
    print(f"  Weights: {FINAL_WEIGHTS}")
    print(f"  Variant-based selection: {use_variant_selection}")
    print(f"  Unique train variants: {UNIQUE_TRAIN_VARIANTS}")
    print(f"  Unique test variants: {UNIQUE_TEST_VARIANTS}")
    print(f"  Training variant sweep: {sweep_variant_counts}")
    print(f"  Total experiments: {len(sweep_variant_counts)}")
    experiment_config['final_experiment'] = {
        'alpha': FINAL_ALPHA,
        'strategy': FINAL_STRATEGY,
        'weights': FINAL_WEIGHTS,
        'variant_based_selection': use_variant_selection,
        'variant_sweep': sweep_variant_counts,
    }

    # Run final experiment sweep
    final_results_dir = results_dir / 'final_experiment'
    final_results_dir.mkdir(parents=True, exist_ok=True)

    # Build sweep parameters with variant-based or standard selection
    sweep_params = []
    for n_variants in sweep_variant_counts:
        if use_variant_selection:
            train_cases_unique, test_cases_all, _ = select_variants_for_experiment(
                variant_df, n_variants=n_variants, selection_mode=VARIANT_SELECTION_MODE, seed=RANDOM_SEED
            )
            # Apply unique train/test variant settings
            train_cases = train_cases_unique if UNIQUE_TRAIN_VARIANTS else test_cases_all
            test_cases = train_cases_unique if UNIQUE_TEST_VARIANTS else test_cases_all
        else:
            train_cases = None
            test_cases = None
        sweep_params.append((n_variants, train_cases, test_cases, FINAL_ALPHA, FINAL_STRATEGY, FINAL_WEIGHTS))

    print(f"\nRunning {len(sweep_params)} sweep experiments...")
    print("=" * 60)

    sweep_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_single_experiment)(
            n_variants, train_cases, test_cases, alpha, strategy, weights,
            i, len(sweep_params), df, softmax_lst, base_config, final_results_dir,
            f"sweep_v{n_variants}", DATASET_NAME, MODEL_SOURCE,
            unique_train=UNIQUE_TRAIN_VARIANTS, unique_test=UNIQUE_TEST_VARIANTS
        )
        for i, (n_variants, train_cases, test_cases, alpha, strategy, weights) in enumerate(sweep_params, 1)
    )

    sweep_summary_df = pd.DataFrame(sweep_results).sort_values('n_train_variants')
    sweep_summary_path = final_results_dir / f"{DATASET_NAME}_{MODEL_SOURCE}_sweep_summary.csv"
    sweep_summary_df.to_csv(sweep_summary_path, index=False)
    print(f"\nSaved: {sweep_summary_path}")
    write_experiment_config(final_results_dir, experiment_config, 'experiment_config.json')

    # Display sweep results
    print("\nTraining Sweep Results:\n")
    display_cols = ['n_train_variants', 'n_train_traces', 'n_test_cases', 'sktr_acc', 'argmax_acc',
                    'sktr_edit', 'argmax_edit', 'sktr_f1@25', 'argmax_f1@25']
    if 'avg_time_per_trace_sec' in sweep_summary_df.columns:
        display_cols.append('avg_time_per_trace_sec')
    # Filter to columns that exist
    display_cols = [c for c in display_cols if c in sweep_summary_df.columns]
    print(sweep_summary_df[display_cols].to_string())

    # Visualization
    plot_sweep_results(sweep_summary_df, final_results_dir, DATASET_NAME, MODEL_SOURCE)

    # Improvement analysis
    analysis = sweep_summary_df.copy()
    analysis['acc_gain'] = analysis['sktr_acc'] - analysis['argmax_acc']
    analysis['edit_gain'] = analysis['sktr_edit'] - analysis['argmax_edit']
    analysis['f1@25_gain'] = analysis['sktr_f1@25'] - analysis['argmax_f1@25']

    print("\nSKTR Improvement over Argmax:")
    print(f"  Accuracy:  mean={analysis['acc_gain'].mean():+.4f}, max={analysis['acc_gain'].max():+.4f}")
    print(f"  Edit:      mean={analysis['edit_gain'].mean():+.4f}, max={analysis['edit_gain'].max():+.4f}")
    print(f"  F1@25:     mean={analysis['f1@25_gain'].mean():+.4f}, max={analysis['f1@25_gain'].max():+.4f}")

    # Timing summary
    if 'avg_time_per_trace_sec' in sweep_summary_df.columns:
        valid_times = sweep_summary_df['avg_time_per_trace_sec'].dropna()
        if len(valid_times) > 0:
            print(f"\nTiming Summary:")
            print(f"  Avg time per trace: {valid_times.mean():.3f}s (min={valid_times.min():.3f}s, max={valid_times.max():.3f}s)")
            total_times = sweep_summary_df['total_time_sec'].dropna()
            if len(total_times) > 0:
                print(f"  Total experiment time: {total_times.sum():.1f}s ({total_times.sum()/60:.1f} min)")

    best_idx = analysis['sktr_acc'].idxmax()
    best_n_variants = analysis.loc[best_idx, 'n_train_variants'] if 'n_train_variants' in analysis.columns else analysis.loc[best_idx, 'n_train_traces']
    print(f"\nBest SKTR accuracy: {analysis.loc[best_idx, 'sktr_acc']:.4f} at n_train_variants={best_n_variants}")

    print("\n" + "=" * 70)
    print("Experiment Complete!")
    print(f"Results saved to: {results_dir}")
    print("=" * 70)
