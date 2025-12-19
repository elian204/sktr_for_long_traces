#!/usr/bin/env python3
"""
Breakfast Dataset - Variant Experiment

This script runs experiments on the Breakfast dataset by controlling which trace variants
are used for training and testing.

Structure:
1. Setup & Data Loading - Load dataset and analyze variants
2. Hyperparameter Search - Fixed train/test split, sweep over alpha and interpolation strategies
3. Final Experiment - Fixed hyperparameters, sweep over number of training cases

Usage:
    python breakfast_variant_experiment.py
"""

import pandas as pd
from typing import List
from pathlib import Path
import random
import logging
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import math
from src.evaluation import compute_sktr_vs_argmax_metrics
from src.incremental_softmax_recovery import incremental_softmax_recovery
from src.utils import (
    prepare_df, linear_prob_combiner,
    get_variant_info, select_variants, get_cases_for_variants, get_variants_for_cases
)
from joblib import Parallel, delayed
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for terminal/tmux


# =============================================================================
# SETUP
# =============================================================================

# Setup workspace path
workspace_root = '/home/dsi/eli-bogdanov/sktr_for_long_traces'
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

print(f"Workspace: {workspace_root}")
print(f"Running in tmux: {'TMUX' in os.environ}")

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Case Selection ---
CASE_IDS = [649, 357, 542, 834, 1006, 385,
            243, 553, 48, 841, 877, 321, 226, 670]

# --- General ---
RANDOM_SEED = 42

# --- Parallelization ---
N_PARALLEL_RUNS = 1  # Sequential hyperparameter experiments
DATASET_PARALLELIZATION = True
N_DATASET_WORKERS = 14  # Workers for dataset parallelization

# --- Output ---
SAVE_PROCESS_MODELS = True

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_variant_info(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze trace variants. Returns DataFrame sorted by frequency (variant_id 0 = most frequent)."""
    trace_variants = {}
    for case_id in df['case:concept:name'].unique():
        trace = tuple(df[df['case:concept:name'] == case_id]
                      ['concept:name'].tolist())
        if trace not in trace_variants:
            trace_variants[trace] = {'case_ids': [], 'length': len(trace)}
        trace_variants[trace]['case_ids'].append(case_id)

    data = [{'variant_id': i, 'trace_signature': sig, 'case_ids': info['case_ids'],
             'frequency': len(info['case_ids']), 'trace_length': info['length']}
            for i, (sig, info) in enumerate(trace_variants.items())]

    variant_df = pd.DataFrame(data).sort_values(
        'frequency', ascending=False).reset_index(drop=True)
    variant_df['variant_id'] = range(len(variant_df))
    return variant_df


def get_cases_for_variants(variant_df: pd.DataFrame, variant_ids: List[int], seed: int = 42) -> List[str]:
    """Get all case IDs for the specified variants."""
    cases = []
    for vid in variant_ids:
        row = variant_df[variant_df['variant_id'] == vid]
        if not row.empty:
            cases.extend(row.iloc[0]['case_ids'])
    return cases


def build_base_config() -> dict:
    """Build base configuration for experiments."""
    return {
        'n_train_traces': None, 'n_test_traces': None,
        'train_cases': None, 'test_cases': None,
        'ensure_train_variant_diversity': False, 'ensure_test_variant_diversity': False,
        'use_same_traces_for_train_test': False, 'allow_train_cases_in_test': True,
        'compute_marking_transition_map': False, 'sequential_sampling': True,
        'n_indices': None, 'n_per_run': 10000, 'independent_sampling': True,
        'prob_threshold': 1e-6, 'chunk_size': 11, 'conformance_switch_penalty_weight': 1.0,
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
        # Conditioning history mode:
        # - "merged" keeps 1 history per state (fastest)
        # - "topm" keeps up to M histories per state (good trade-off)
        # - "exact" keeps all histories (can be too slow)
        'conditioning_state_mode': 'topm',
        'conditioning_top_m': 3,
        # Bound branching factor in conformance search (approximation, but prevents OOM)
        # Uses top-p within top-k on (optionally) conditioned probability vector.
        'candidate_top_p': 0.9,
        'candidate_top_k': 15,
        'candidate_min_k': 1,
        # Use observed (path-independent) candidates for stability/speed.
        'candidate_source': 'observed',
        'candidate_apply_to_sync': True,
    }


def check_existing_result(results_dir, dataset_name, prefix, alpha, strategy):
    """Check if result CSV exists and load metrics from it."""
    csv_path = results_dir / \
        f"{dataset_name}_{prefix}_alpha_{alpha}_weights_{strategy}.csv"
    if not csv_path.exists():
        return None

    try:
        metrics = compute_sktr_vs_argmax_metrics(
            str(csv_path),
            case_col='case:concept:name',
            sktr_pred_col='sktr_activity',
            argmax_pred_col='argmax_activity',
            gt_col='ground_truth',
            background=0
        )
        return metrics
    except Exception as e:
        print(
            f"  Warning: Could not load existing result from {csv_path}: {e}")
        return None


def run_single_experiment(train_cases, test_cases, alpha, strategy, weights,
                          idx, total, df, softmax_lst, base_cfg, results_dir, prefix,
                          skip_existing=True):
    """Run a single experiment and return metrics."""
    print(f"[{idx}/{total}] train={len(train_cases)}, test={len(test_cases)}, alpha={alpha}, strategy={strategy}")

    # Check for existing result to enable resume
    if skip_existing:
        existing_metrics = check_existing_result(
            results_dir, dataset_name, prefix, alpha, strategy)
        if existing_metrics is not None:
            print(f"  -> SKIPPED (already exists)")
            return {
                'n_train_cases': len(train_cases), 'n_test_cases': len(test_cases),
                'alpha': alpha, 'strategy': strategy,
                'sktr_acc': existing_metrics['sktr']['acc_micro'], 'sktr_edit': existing_metrics['sktr']['edit'],
                'sktr_f1@10': existing_metrics['sktr']['f1@10'], 'sktr_f1@25': existing_metrics['sktr']['f1@25'], 'sktr_f1@50': existing_metrics['sktr']['f1@50'],
                'argmax_acc': existing_metrics['argmax']['acc_micro'], 'argmax_edit': existing_metrics['argmax']['edit'],
                'argmax_f1@10': existing_metrics['argmax']['f1@10'], 'argmax_f1@25': existing_metrics['argmax']['f1@25'], 'argmax_f1@50': existing_metrics['argmax']['f1@50'],
            }

    cfg = base_cfg.copy()
    cfg.update({
        'conditioning_alpha': alpha,
        'conditioning_interpolation_weights': weights,
        'train_cases': train_cases,
        'test_cases': test_cases,
        'n_train_traces': len(train_cases),
        'n_test_traces': len(test_cases),
        'save_model': SAVE_PROCESS_MODELS,
        'save_model_path': str(results_dir / f'petri_net_{prefix}'),
    })

    results_df, _, _ = incremental_softmax_recovery(
        df=df, softmax_lst=softmax_lst, **cfg)

    csv_path = results_dir / \
        f"{dataset_name}_{prefix}_alpha_{alpha}_weights_{strategy}.csv"
    results_df.to_csv(csv_path, index=False)

    metrics = compute_sktr_vs_argmax_metrics(
        str(csv_path),
        case_col='case:concept:name',
        sktr_pred_col='sktr_activity',
        argmax_pred_col='argmax_activity',
        gt_col='ground_truth',
        background=0
    )

    return {
        'n_train_cases': len(train_cases), 'n_test_cases': len(test_cases),
        'alpha': alpha, 'strategy': strategy,
        'sktr_acc': metrics['sktr']['acc_micro'], 'sktr_edit': metrics['sktr']['edit'],
        'sktr_f1@10': metrics['sktr']['f1@10'], 'sktr_f1@25': metrics['sktr']['f1@25'], 'sktr_f1@50': metrics['sktr']['f1@50'],
        'argmax_acc': metrics['argmax']['acc_micro'], 'argmax_edit': metrics['argmax']['edit'],
        'argmax_f1@10': metrics['argmax']['f1@10'], 'argmax_f1@25': metrics['argmax']['f1@25'], 'argmax_f1@50': metrics['argmax']['f1@50'],
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # Load Dataset
    # -------------------------------------------------------------------------
    dataset_name = 'breakfast'

    result = prepare_df(dataset_name)
    df, softmax_lst = result[:2]

    print(f"Dataset: {dataset_name}")
    print(f"  Events: {len(df):,}")
    print(f"  Cases: {df['case:concept:name'].nunique()}")
    print(f"  Activities: {df['concept:name'].nunique()}")

    # -------------------------------------------------------------------------
    # Prepare Case Data
    # -------------------------------------------------------------------------
    variant_df = get_variant_info(df)
    print(f"Total variants in dataset: {len(variant_df)}")

    # Validate case IDs and build mappings
    case_ids_str = [str(cid) for cid in CASE_IDS]

    available_cases = df['case:concept:name'].unique().astype(str).tolist()
    missing_cases = [cid for cid in case_ids_str if cid not in available_cases]
    if missing_cases:
        print(
            f"Warning: {len(missing_cases)} case IDs not found: {missing_cases[:10]}...")
        case_ids_str = [cid for cid in case_ids_str if cid in available_cases]
        print(f"Using {len(case_ids_str)} valid case IDs")

    # Build case_id -> variant_id mapping
    case_to_variant = {}
    for _, row in variant_df.iterrows():
        for cid in row['case_ids']:
            case_to_variant[str(cid)] = row['variant_id']

    # Get variants in list order
    experiment_variants = []
    seen_variants = set()
    for cid in case_ids_str:
        vid = case_to_variant.get(cid)
        if vid is not None and vid not in seen_variants:
            experiment_variants.append(vid)
            seen_variants.add(vid)

    # Setup results directory
    results_dir = Path(workspace_root) / 'results' / \
        dataset_name / 'variant_experiment'
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Case IDs: {len(case_ids_str)} cases")
    print(
        f"Unique variants: {len(experiment_variants)} -> {experiment_variants}")
    print(f"\nCase ID -> Variant mapping:")
    for cid in case_ids_str:
        vid = case_to_variant.get(cid, '?')
        print(f"  {cid} -> variant {vid}")

    # Build base config
    base_config = build_base_config()
    print("Base config ready.")

    # =========================================================================
    # Part A: Hyperparameter Search
    # =========================================================================
    print("\n" + "=" * 70)
    print("Part A: Hyperparameter Search")
    print("=" * 70)

    # Hyperparameter search configuration
    HP_TRAIN_CASES = case_ids_str
    HP_TEST_CASES = case_ids_str

    HP_ALPHAS = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
    HP_STRATEGIES = {
        'unigram_super_heavy': [0.75, 0.15, 0.1],
        'balanced': [0.33, 0.33, 0.34],
        'bigram_heavy': [0.2, 0.6, 0.2],
        'trigram_heavy': [0.1, 0.15, 0.75],
    }

    print(f"Hyperparameter Search Setup:")
    print(f"  Train cases: {len(HP_TRAIN_CASES)}")
    print(f"  Test cases: {len(HP_TEST_CASES)}")
    print(f"  Alphas: {HP_ALPHAS}")
    print(f"  Strategies: {list(HP_STRATEGIES.keys())}")
    print(f"  Total experiments: {len(HP_ALPHAS) * len(HP_STRATEGIES)}")

    # Run hyperparameter search
    hp_results_dir = results_dir / 'hyperparameter_search'
    hp_results_dir.mkdir(parents=True, exist_ok=True)

    hp_params = [(HP_TRAIN_CASES, HP_TEST_CASES, a, s, w)
                 for a in HP_ALPHAS
                 for s, w in HP_STRATEGIES.items()]

    n_jobs = N_PARALLEL_RUNS if N_PARALLEL_RUNS is not None else -1

    print(f"Running {len(hp_params)} hyperparameter experiments...")
    print("=" * 60)

    hp_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_single_experiment)(
            train, test, alpha, strategy, weights,
            i, len(
                hp_params), df, softmax_lst, base_config, hp_results_dir, f"hp_search"
        )
        for i, (train, test, alpha, strategy, weights) in enumerate(hp_params, 1)
    )

    hp_summary_df = pd.DataFrame(hp_results).sort_values(
        'sktr_acc', ascending=False)
    hp_summary_path = hp_results_dir / f"{dataset_name}_hp_search_summary.csv"
    hp_summary_df.to_csv(hp_summary_path, index=False)
    print(f"\nSaved: {hp_summary_path}")

    # Display hyperparameter search results
    print("\nHyperparameter Search Results (sorted by SKTR accuracy):\n")
    print(hp_summary_df[['alpha', 'strategy', 'sktr_acc', 'argmax_acc',
          'sktr_edit', 'argmax_edit', 'sktr_f1@25', 'argmax_f1@25']].to_string())

    # Best hyperparameters
    best_hp = hp_summary_df.iloc[0]
    print(f"\nBest hyperparameters:")
    print(f"  Alpha: {best_hp['alpha']}")
    print(f"  Strategy: {best_hp['strategy']}")
    print(f"  SKTR Accuracy: {best_hp['sktr_acc']:.4f}")
    print(f"  SKTR Edit: {best_hp['sktr_edit']:.4f}")

    # Visualize hyperparameter search results
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    plot_cols = ['sktr_acc', 'sktr_edit', 'sktr_f1@25']
    y_min = hp_summary_df[plot_cols].min().min()
    y_max = hp_summary_df[plot_cols].max().max()
    y_lower = math.floor(y_min / 10) * 10
    y_max = max(y_max, 80)
    tick_start = y_lower
    tick_end = math.ceil(y_max / 10) * 10
    y_limits = (tick_start, tick_end)
    y_ticks = list(range(int(tick_start), int(tick_end) + 1, 10))

    for ax, metric in zip(axes, plot_cols):
        pivot = hp_summary_df.pivot(
            index='alpha', columns='strategy', values=metric)
        pivot.plot(kind='bar', ax=ax, rot=0)
        ax.set_xlabel('Alpha')
        ax.set_ylabel(metric.replace('sktr_', '').replace('_', ' ').title())
        ax.set_title(
            f'SKTR {metric.replace("sktr_", "").replace("_", " ").title()}')
        ax.legend(title='Strategy', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        if y_ticks is not None:
            ax.set_yticks(y_ticks)

    plt.suptitle('Hyperparameter Search Results',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(hp_results_dir /
                f'{dataset_name}_hp_search_plots.png', dpi=150)
    print(
        f"Saved plot: {hp_results_dir / f'{dataset_name}_hp_search_plots.png'}")
    plt.close()

    # =========================================================================
    # Part B: Final Experiment (Training Sweep)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Part B: Final Experiment (Training Sweep)")
    print("=" * 70)

    # Final experiment configuration - use best hyperparameters from search
    FINAL_ALPHA = best_hp['alpha']
    FINAL_STRATEGY = best_hp['strategy']
    FINAL_WEIGHTS = HP_STRATEGIES[best_hp['strategy']]

    # Training sweep configuration
    SWEEP_STEP = 1
    SWEEP_START = 1

    # Test set (always all cases)
    FINAL_TEST_CASES = case_ids_str

    # Generate training sweep values
    n_total = len(case_ids_str)
    sweep_values = list(range(SWEEP_START, n_total + 1, SWEEP_STEP))
    if n_total not in sweep_values:
        sweep_values.append(n_total)

    print(f"Final Experiment Configuration:")
    print(f"  Alpha: {FINAL_ALPHA}")
    print(f"  Strategy: {FINAL_STRATEGY}")
    print(f"  Weights: {FINAL_WEIGHTS}")
    print(f"  Test cases: {len(FINAL_TEST_CASES)} (all)")
    print(f"  Training sweep: {sweep_values}")
    print(f"  Total experiments: {len(sweep_values)}")

    # Build training configurations for sweep
    sweep_configs = {}
    for n in sweep_values:
        train_case_ids = case_ids_str[:n]
        train_variant_ids = []
        seen = set()
        for cid in train_case_ids:
            vid = case_to_variant.get(cid)
            if vid is not None and vid not in seen:
                train_variant_ids.append(vid)
                seen.add(vid)
        sweep_configs[n] = {
            'variant_ids': train_variant_ids,
            'case_ids': train_case_ids
        }

    print("\nTraining Sweep Progression:")
    for n, cfg in sweep_configs.items():
        cases_str = str(cfg['case_ids']) if len(
            cfg['case_ids']) <= 5 else f"{cfg['case_ids'][:3]}...{cfg['case_ids'][-1]}"
        print(
            f"  n={n:2d} -> {len(cfg['case_ids']):2d} case(s), {len(cfg['variant_ids']):2d} variant(s)  cases={cases_str}")

    # Run final experiment sweep
    final_results_dir = results_dir / 'final_experiment'
    final_results_dir.mkdir(parents=True, exist_ok=True)

    sweep_params = [(cfg['case_ids'], FINAL_TEST_CASES, FINAL_ALPHA, FINAL_STRATEGY, FINAL_WEIGHTS, n)
                    for n, cfg in sweep_configs.items()]

    print(f"\nRunning {len(sweep_params)} sweep experiments...")
    print("=" * 60)

    sweep_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_single_experiment)(
            train, test, alpha, strategy, weights,
            i, len(
                sweep_params), df, softmax_lst, base_config, final_results_dir, f"sweep_n{n}"
        )
        for i, (train, test, alpha, strategy, weights, n) in enumerate(sweep_params, 1)
    )

    # Add n_train to results and variant info
    for i, (n, cfg) in enumerate(sweep_configs.items()):
        sweep_results[i]['n_train'] = n
        sweep_results[i]['n_variants'] = len(cfg['variant_ids'])

    sweep_summary_df = pd.DataFrame(sweep_results).sort_values('n_train')
    sweep_summary_path = final_results_dir / \
        f"{dataset_name}_sweep_summary.csv"
    sweep_summary_df.to_csv(sweep_summary_path, index=False)
    print(f"\nSaved: {sweep_summary_path}")

    # Display sweep results
    print("\nTraining Sweep Results:\n")
    print(sweep_summary_df[['n_train', 'n_variants', 'sktr_acc', 'argmax_acc',
          'sktr_edit', 'argmax_edit', 'sktr_f1@25', 'argmax_f1@25']].to_string())

    # Visualization of sweep results (match notebook style)
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

    plot_cols = []
    for metric_suffix, _ in metrics_config:
        for method in method_styles:
            col_name = f'{method}_{metric_suffix}'
            if col_name in sweep_summary_df.columns:
                plot_cols.append(col_name)

    y_limits = None
    y_ticks = None
    if plot_cols:
        y_min = sweep_summary_df[plot_cols].min().min()
        y_max = sweep_summary_df[plot_cols].max().max()
        y_lower = math.floor(y_min / 10) * 10
        y_max = max(y_max, 80)
        tick_start = y_lower
        tick_end = math.ceil(y_max / 10) * 10
        y_limits = (tick_start, tick_end)
        y_ticks = list(range(int(tick_start), int(tick_end) + 1, 10))

    for idx, (metric_suffix, title) in enumerate(metrics_config):
        ax = axes[idx]
        for method, style in method_styles.items():
            col_name = f'{method}_{metric_suffix}'
            if col_name in sweep_summary_df.columns:
                sns.lineplot(
                    x=sweep_summary_df['n_train'],
                    y=sweep_summary_df[col_name],
                    ax=ax,
                    linewidth=2.5,
                    markersize=9,
                    **style,
                )
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Cases')
        ax.set_ylabel('Score')
        ax.set_xticks(sweep_summary_df['n_train'].unique())
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        if y_ticks is not None:
            ax.set_yticks(y_ticks)
        ax.legend().remove()

    ax_legend = axes[5]
    ax_legend.axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    ax_legend.legend(
        handles,
        labels,
        loc='center',
        title='Method',
        fontsize=14,
        title_fontsize=16,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    plt.suptitle(
        f'Performance Scaling vs. Training Cases ({dataset_name})', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(final_results_dir / f'{dataset_name}_sweep_plots.png', dpi=150)
    print(
        f"Saved plot: {final_results_dir / f'{dataset_name}_sweep_plots.png'}")
    plt.close()

    # Improvement analysis
    analysis = sweep_summary_df.copy()
    analysis['acc_gain'] = analysis['sktr_acc'] - analysis['argmax_acc']
    analysis['edit_gain'] = analysis['sktr_edit'] - analysis['argmax_edit']
    analysis['f1@25_gain'] = analysis['sktr_f1@25'] - analysis['argmax_f1@25']

    print("\nSKTR Improvement over Argmax:")
    print(
        f"  Accuracy:  mean={analysis['acc_gain'].mean():+.4f}, max={analysis['acc_gain'].max():+.4f}")
    print(
        f"  Edit:      mean={analysis['edit_gain'].mean():+.4f}, max={analysis['edit_gain'].max():+.4f}")
    print(
        f"  F1@25:     mean={analysis['f1@25_gain'].mean():+.4f}, max={analysis['f1@25_gain'].max():+.4f}")

    best_idx = analysis['sktr_acc'].idxmax()
    print(
        f"\nBest SKTR accuracy: {analysis.loc[best_idx, 'sktr_acc']:.4f} at n_train={analysis.loc[best_idx, 'n_train']}")

    print("\n" + "=" * 70)
    print("Experiment Complete!")
    print("=" * 70)
