#!/usr/bin/env python3
"""
Hyperparameter Search for Breakfast Dataset

Evaluates different (alpha, strategy) combinations using a single train/test split
with one trace per variant (~257 unique variants total).

- Training: 100 unique variants
- Testing: ~157 remaining unique variants
- Strategies: trigram_heavy, unigram_super_heavy (top 2 from other datasets)
- Alpha values: 0.1, 0.3, 0.5, 0.7, 0.9, 0.95

Usage:
    python hp_search_breakfast.py -m asformer
    python hp_search_breakfast.py -m mstcn2
    python hp_search_breakfast.py -m asformer --quick  # fewer alphas
"""

from src.evaluation import compute_tas_metrics_asformer
from src.incremental_softmax_recovery import incremental_softmax_recovery
from src.utils import prepare_df_from_model, linear_prob_combiner, get_variant_info
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from itertools import product
import random

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# Setup path
workspace_root = str(Path(__file__).resolve().parent)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)


# Top 2 strategies based on HP search results from 50 Salads and GTEA
HP_STRATEGIES = {
    'trigram_heavy': [0.1, 0.15, 0.75],
    'unigram_super_heavy': [0.75, 0.15, 0.1],
}

# Alpha values to search (including 0.1)
ALPHA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]

# Quick mode uses fewer alphas
ALPHA_VALUES_QUICK = [0.3, 0.7, 0.95]

# Number of training variants
N_TRAIN_VARIANTS = 100


def parse_args():
    parser = argparse.ArgumentParser(
        description='Hyperparameter search for Breakfast dataset (single split, unique variants).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        choices=['asformer', 'mstcn2', 'diffact'],
        default='asformer',
        help='Model source for softmax predictions'
    )
    parser.add_argument(
        '-p', '--parallel',
        type=int,
        default=3,
        help='Number of HP experiments to run in parallel'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=20,
        help='Number of inner workers per experiment (used with --inner-parallel)'
    )
    parser.add_argument(
        '--inner-parallel',
        action='store_true',
        help='Enable dataset-level parallelization inside each HP experiment'
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: fewer alpha values (0.3, 0.7, 0.95)'
    )
    parser.add_argument(
        '--alphas',
        type=str,
        default=None,
        help='Comma-separated alpha values to test (e.g., "0.1,0.5,0.9")'
    )
    parser.add_argument(
        '--n-train',
        type=int,
        default=N_TRAIN_VARIANTS,
        help='Number of training variants'
    )
    parser.add_argument(
        '--n-test',
        type=int,
        default=None,
        help='Number of test variants (default: all remaining after train)'
    )
    # Conformance checking constraints
    parser.add_argument(
        '--restrict-log-moves',
        action='store_true',
        help='Restrict log moves to top-1 + parent last label'
    )
    parser.add_argument(
        '--candidate-top-k',
        type=int,
        default=3,
        help='Max candidate labels per timestamp (top-K for sync moves)'
    )
    parser.add_argument(
        '--top-m',
        type=int,
        default=1,
        help='Number of top states to consider (conditioning_top_m)'
    )
    parser.add_argument(
        '--use-calibration',
        action='store_true',
        help='Enable temperature calibration for test softmax'
    )
    parser.add_argument(
        '--restrict-model-moves-to-tau',
        action='store_true',
        help='Restrict model moves to only tau (silent) transitions'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recompute all experiments, ignoring cached results'
    )
    return parser.parse_args()


def build_base_config(
    alpha: float,
    weights: List[float],
    seed: int,
    max_workers: int,
    dataset_parallelization: bool,
    dataset_parallelization_context: Optional[str],
    restrict_log_moves: bool,
    candidate_top_k: int,
    top_m: int,
    restrict_model_moves_to_tau: bool,
    use_calibration: bool,
) -> dict:
    """Build configuration for a single HP combination."""
    return {
        'n_train_traces': None, 'n_test_traces': None,
        'train_cases': None, 'test_cases': None,
        'ensure_train_variant_diversity': False,
        'ensure_test_variant_diversity': False,
        'use_same_traces_for_train_test': False,
        'allow_train_cases_in_test': False,
        'compute_marking_transition_map': False, 'sequential_sampling': False,
        'n_indices': 10**9, 'n_per_run': None, 'independent_sampling': True,
        'prob_threshold': 1e-6,
        'chunk_size': 11,
        'conformance_switch_penalty_weight': 1.0,
        'merge_mismatched_boundaries': False,
        'conditioning_combine_fn': linear_prob_combiner,
        'conditioning_alpha': alpha,
        'conditioning_interpolation_weights': weights,
        'max_hist_len': 3, 'conditioning_n_prev_labels': 3, 'use_collapsed_runs': True,
        'cost_function': 'linear',
        'model_move_cost': 1.0,
        'log_move_cost': 1.0,
        'tau_move_cost': 1e-6, 'non_sync_penalty': 1.0,
        'use_calibration': use_calibration, 'temp_bounds': (1.0, 10.0), 'temperature': None,
        'verbose': False, 'log_level': 30,  # WARNING level to reduce noise
        'round_precision': 2,
        'random_seed': seed,
        'save_model_path': None,
        'save_model': False,
        'parallel_processing': False,
        'dataset_parallelization': dataset_parallelization,
        'dataset_parallelization_context': dataset_parallelization_context,
        'max_workers': max_workers,
        'conditioning_state_mode': 'topm',
        'conditioning_top_m': top_m,
        'candidate_top_p': 1.0,
        'candidate_top_k': candidate_top_k,
        'candidate_min_k': 1,
        'candidate_source': 'conditioned',
        'candidate_apply_to_sync': True,
        'restrict_log_moves': restrict_log_moves,
        'restrict_model_moves_to_tau': restrict_model_moves_to_tau,
    }


def get_result_filename(alpha: float, strategy: str) -> str:
    """Generate unique filename for a (alpha, strategy) result."""
    return f"result_alpha{alpha}_strategy_{strategy}.json"


def get_error_filename(alpha: float, strategy: str) -> str:
    """Generate unique filename for a (alpha, strategy) error result."""
    return f"error_alpha{alpha}_strategy_{strategy}.json"


def load_existing_result(results_dir: Path, alpha: float, strategy: str) -> Optional[Dict[str, Any]]:
    """Load existing result for a (alpha, strategy) combination if it exists."""
    filename = get_result_filename(alpha, strategy)
    filepath = results_dir / filename
    if filepath.exists():
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def save_result(results_dir: Path, alpha: float, strategy: str, result: Dict[str, Any]) -> None:
    """Save result for a (alpha, strategy) combination."""
    filename = get_result_filename(alpha, strategy)
    filepath = results_dir / filename
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)


def save_error_result(results_dir: Path, alpha: float, strategy: str, result: Dict[str, Any]) -> None:
    """Save error details for a (alpha, strategy) combination."""
    filename = get_error_filename(alpha, strategy)
    filepath = results_dir / filename
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)


def run_hp_experiment(
    df: pd.DataFrame,
    softmax_lst: List[np.ndarray],
    train_case_ids: List[str],
    test_case_ids: List[str],
    alpha: float,
    strategy: str,
    weights: List[float],
    seed: int,
    max_workers: int,
    dataset_parallelization: bool,
    dataset_parallelization_context: Optional[str],
    restrict_log_moves: bool,
    candidate_top_k: int,
    top_m: int,
    restrict_model_moves_to_tau: bool,
    use_calibration: bool,
    results_dir_str: str,
    force: bool = False,
) -> Dict[str, Any]:
    """Run a single HP configuration and return metrics.

    If results already exist for this (alpha, strategy), returns cached result with 'cached': True.
    Use force=True to bypass cache and recompute.
    """
    # Convert string back to Path (for serialization compatibility)
    results_dir = Path(results_dir_str)

    # Check for existing result (unless force recompute)
    if not force:
        existing = load_existing_result(results_dir, alpha, strategy)
        if existing is not None:
            existing['cached'] = True
            return existing

    start_time = time.time()
    try:
        cfg = build_base_config(
            alpha,
            weights,
            seed,
            max_workers,
            dataset_parallelization,
            dataset_parallelization_context,
            restrict_log_moves,
            candidate_top_k,
            top_m,
            restrict_model_moves_to_tau,
            use_calibration,
        )
        cfg.update({
            'train_cases': train_case_ids,
            'n_train_traces': len(train_case_ids),
            'test_cases': test_case_ids,
            'n_test_traces': len(test_case_ids),
        })

        results_df, _, _ = incremental_softmax_recovery(
            df=df, softmax_lst=softmax_lst, **cfg
        )
        elapsed = time.time() - start_time

        # Compute metrics for SKTR
        sktr_metrics = compute_tas_metrics_asformer(
            results_df,
            pred_col='sktr_activity',
            gt_col='ground_truth',
            case_col='case:concept:name',
            background=None,
            dataset_name='breakfast',
        )

        # Compute metrics for argmax
        argmax_metrics = compute_tas_metrics_asformer(
            results_df,
            pred_col='argmax_activity',
            gt_col='ground_truth',
            case_col='case:concept:name',
            background=None,
            dataset_name='breakfast',
        )

        result = {
            'sktr_acc': sktr_metrics['acc'],
            'sktr_edit': sktr_metrics['edit'],
            'sktr_f1@10': sktr_metrics['f1@10'],
            'sktr_f1@25': sktr_metrics['f1@25'],
            'sktr_f1@50': sktr_metrics['f1@50'],
            'argmax_acc': argmax_metrics['acc'],
            'argmax_edit': argmax_metrics['edit'],
            'argmax_f1@25': argmax_metrics['f1@25'],
            'time_sec': round(elapsed, 1),
            'cached': False,
        }

        # Save result for future runs
        save_result(results_dir, alpha, strategy, result)
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        error_result = {
            'error': str(e),
            'error_type': type(e).__name__,
            'traceback': traceback.format_exc(),
            'time_sec': round(elapsed, 1),
            'cached': False,
        }
        print(
            f"  ERROR alpha={alpha}, strategy={strategy}: {e}",
            file=sys.stderr,
            flush=True,
        )
        print(error_result['traceback'], file=sys.stderr, flush=True)
        save_error_result(results_dir, alpha, strategy, error_result)
        return error_result


def get_one_trace_per_variant(variant_df: pd.DataFrame, seed: int = 42) -> List[str]:
    """
    Select one representative trace per variant from all variants.
    Returns list of case IDs (one per variant).
    """
    rng = random.Random(seed)
    selected = []

    for _, row in variant_df.iterrows():
        case_ids = [str(c) for c in row['case_ids']]
        if case_ids:
            selected.append(rng.choice(case_ids))

    return selected


def main():
    args = parse_args()

    print("=" * 70)
    print("HYPERPARAMETER SEARCH - BREAKFAST DATASET")
    print("=" * 70)

    # Determine search space
    if args.alphas:
        alpha_values = [float(a) for a in args.alphas.split(',')]
    elif args.quick:
        alpha_values = ALPHA_VALUES_QUICK
    else:
        alpha_values = ALPHA_VALUES

    strategies = HP_STRATEGIES
    n_train = args.n_train
    dataset_parallelization = args.inner_parallel
    max_workers = args.workers if dataset_parallelization else 1
    dataset_parallelization_context = "spawn" if dataset_parallelization else None

    n_parallel = args.parallel

    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Training variants: {n_train}")
    print(f"  Alpha values: {alpha_values}")
    print(f"  Strategies: {list(strategies.keys())}")
    print(f"  Total experiments: {len(alpha_values) * len(strategies)}")
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
    print(f"  Restrict log moves: {args.restrict_log_moves}")
    print(f"  Restrict model moves to tau: {args.restrict_model_moves_to_tau}")
    print(f"  Candidate top-k: {args.candidate_top_k}")
    print(f"  Top-m states: {args.top_m}")

    # Setup results directory
    results_dir = Path(workspace_root) / 'results' / \
        'breakfast' / 'hp_search' / args.model
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("\nLoading dataset...")
    df, softmax_lst = prepare_df_from_model('breakfast', args.model)
    print(f"  Loaded {len(softmax_lst)} cases")

    print("Computing variant info...")
    variant_df = get_variant_info(df, use_collapsed=True)
    n_variants = len(variant_df)
    print(f"  Found {n_variants} unique variants")

    # Get one trace per variant
    print("\nSelecting one trace per variant...")
    all_variant_traces = get_one_trace_per_variant(variant_df, seed=args.seed)
    print(f"  Selected {len(all_variant_traces)} traces (one per variant)")

    # Shuffle and split into train/test
    rng = random.Random(args.seed)
    shuffled = all_variant_traces.copy()
    rng.shuffle(shuffled)

    train_case_ids = shuffled[:n_train]
    test_case_ids = shuffled[n_train:]

    # Optionally limit test set size
    if args.n_test is not None and args.n_test < len(test_case_ids):
        test_case_ids = test_case_ids[:args.n_test]

    print(f"\nTrain/Test Split:")
    print(f"  Training: {len(train_case_ids)} variants")
    print(f"  Testing: {len(test_case_ids)} variants")

    # Run experiments in parallel
    print("\n" + "-" * 70)
    print("RUNNING EXPERIMENTS")
    print("-" * 70)

    hp_combinations = list(product(alpha_values, strategies.keys()))
    total_exps = len(hp_combinations)

    n_parallel = min(n_parallel, total_exps)
    print(
        f"  Running {total_exps} experiments with {n_parallel} parallel jobs")
    if dataset_parallelization:
        print(
            f"  Each experiment uses dataset-level parallelization ({max_workers} workers)")
    else:
        print(f"  Each experiment processes test traces sequentially")

    # Build job list
    jobs = []
    for alpha, strategy in hp_combinations:
        weights = strategies[strategy]
        jobs.append((alpha, strategy, weights))

    # Check how many are already cached (unless --force)
    if not args.force:
        n_cached = sum(
            1 for alpha, strategy, _ in jobs
            if load_existing_result(results_dir, alpha, strategy) is not None
        )
        n_to_compute = total_exps - n_cached
        print(
            f"\n  Found {n_cached} cached results, {n_to_compute} to compute")
        if n_cached > 0:
            print(f"  (use --force to recompute all)")
    else:
        n_to_compute = total_exps
        print(f"\n  Force mode: recomputing all {total_exps} experiments")

    # Run in parallel using joblib (thread backend when inner parallelization is enabled)
    print(f"\n  Starting {total_exps} experiments...")
    parallel_kwargs = {'n_jobs': n_parallel, 'verbose': 10}
    if dataset_parallelization:
        parallel_kwargs['backend'] = 'threading'

    raw_results = Parallel(**parallel_kwargs)(
        delayed(run_hp_experiment)(
            df, softmax_lst,
            train_case_ids, test_case_ids,
            alpha, strategy, weights,
            args.seed, max_workers, dataset_parallelization, dataset_parallelization_context,
            args.restrict_log_moves, args.candidate_top_k, args.top_m,
            args.restrict_model_moves_to_tau,
            args.use_calibration,
            str(results_dir),  # Convert Path to string for serialization
            args.force
        )
        for alpha, strategy, weights in jobs
    )

    # Process results
    all_results = []
    for (alpha, strategy, _), result in zip(jobs, raw_results):
        if result is None:
            print(
                f"  alpha={alpha}, strategy={strategy} -> ERROR: None result")
            continue
        if 'error' in result:
            print(
                f"  alpha={alpha}, strategy={strategy} -> ERROR: {result['error']}")
            continue

        cached = result.get('cached', False)
        result.update({
            'alpha': alpha,
            'strategy': strategy,
            'n_train': len(train_case_ids),
            'n_test': len(test_case_ids),
        })
        all_results.append(result)

        status = "[CACHED]" if cached else f"({result['time_sec']}s)"
        print(f"  alpha={alpha}, strategy={strategy} -> "
              f"acc={result['sktr_acc']:.1f}, edit={result['sktr_edit']:.1f}, "
              f"f1@25={result['sktr_f1@25']:.1f} {status}")

    # Save results
    if all_results:
        pd.DataFrame(all_results).to_csv(
            results_dir / 'all_results.csv', index=False)

    # Aggregate and rank results
    print("\n" + "-" * 70)
    print("RESULTS SUMMARY")
    print("-" * 70)

    if not all_results:
        print("No results!")
        return

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_dir / 'all_results.csv', index=False)

    # Compute ranks for each metric (higher is better for all TAS metrics)
    metrics = ['sktr_acc', 'sktr_edit',
               'sktr_f1@10', 'sktr_f1@25', 'sktr_f1@50']
    for metric in metrics:
        results_df[f'{metric}_rank'] = results_df[metric].rank(ascending=False)

    # Compute average rank
    rank_cols = [f'{m}_rank' for m in metrics]
    results_df['avg_rank'] = results_df[rank_cols].mean(axis=1)

    # Sort by average rank
    results_df = results_df.sort_values('avg_rank')
    results_df.to_csv(results_dir / 'ranked_results.csv', index=False)

    # Print summary table
    print(f"\n{'Alpha':<8} {'Strategy':<20} {'Acc':<8} {'Edit':<8} {'F1@25':<8} {'AvgRank':<8}")
    print("-" * 60)

    for _, row in results_df.iterrows():
        print(f"{row['alpha']:<8.2f} {row['strategy']:<20} "
              f"{row['sktr_acc']:>6.1f}  {row['sktr_edit']:>6.1f}  "
              f"{row['sktr_f1@25']:>6.1f}  {row['avg_rank']:>6.2f}")

    # Best configuration
    best = results_df.iloc[0]
    print("\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)
    print(f"  Alpha: {best['alpha']}")
    print(f"  Strategy: {best['strategy']}")
    print(f"  Average Rank: {best['avg_rank']:.2f}")
    print(f"\n  Metrics:")
    print(f"    Accuracy: {best['sktr_acc']:.1f}")
    print(f"    Edit:     {best['sktr_edit']:.1f}")
    print(f"    F1@10:    {best['sktr_f1@10']:.1f}")
    print(f"    F1@25:    {best['sktr_f1@25']:.1f}")
    print(f"    F1@50:    {best['sktr_f1@50']:.1f}")

    # Also print argmax baseline
    print(f"\n  Argmax Baseline:")
    print(f"    Accuracy: {best['argmax_acc']:.1f}")
    print(f"    Edit:     {best['argmax_edit']:.1f}")
    print(f"    F1@25:    {best['argmax_f1@25']:.1f}")

    print(f"\nResults saved to: {results_dir}")

    # Save best config as JSON
    best_config = {
        'alpha': float(best['alpha']),
        'strategy': best['strategy'],
        'avg_rank': float(best['avg_rank']),
        'metrics': {
            'acc': float(best['sktr_acc']),
            'edit': float(best['sktr_edit']),
            'f1@10': float(best['sktr_f1@10']),
            'f1@25': float(best['sktr_f1@25']),
            'f1@50': float(best['sktr_f1@50']),
        },
        'argmax_baseline': {
            'acc': float(best['argmax_acc']),
            'edit': float(best['argmax_edit']),
            'f1@25': float(best['argmax_f1@25']),
        },
        'split': {
            'n_train': len(train_case_ids),
            'n_test': len(test_case_ids),
            'seed': args.seed,
        }
    }
    with open(results_dir / 'best_config.json', 'w') as f:
        json.dump(best_config, f, indent=2)


if __name__ == '__main__':
    main()
