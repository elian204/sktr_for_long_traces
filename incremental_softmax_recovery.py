"""
Incremental Softmax Recovery: Main high-level function.

This module provides the main entry point for incremental softmax matrix recovery
using beam search with Petri nets, following the pattern of the existing 
compare_stochastic_vs_argmax_random_indices function.
"""

from typing import Any, Callable, List, Optional, Tuple, Union
import logging
import pandas as pd
import numpy as np
from utils import validate_input_parameters, process_cost_function
from data_processing import prepare_softmax, filter_indices, split_train_test, select_softmax_matrices
from petri_model import discover_petri_net, build_probability_dict

# Configure logger
logger = logging.getLogger(__name__)


def incremental_softmax_recovery(
    df: pd.DataFrame,
    softmax_lst: List[np.ndarray],
    n_train_traces: int = 10,
    n_test_traces: int = 10,
    train_cases: Optional[List[Any]] = None,
    test_cases: Optional[List[Any]] = None,
    ensure_train_variant_diversity: bool = False,
    ensure_test_variant_diversity: bool = False,
    cost_function: Union[str, Callable[[float], float]] = "linear",
    non_sync_penalty: float = 1.0,
    beam_width: int = 10,
    activity_prob_threshold: float = 0.0,
    use_cond_probs: bool = False,
    lambdas: Optional[List[float]] = None,
    alpha: float = 0.5,
    use_ngram_smoothing: bool = True,
    use_calibration: bool = False,
    temp_bounds: Tuple[float, float] = (1.0, 10.0),
    n_indices: Optional[int] = None,
    n_per_run: Optional[int] = None,
    sequential_sampling: bool = False,
    independent_sampling: bool = True,
    round_precision: int = 2,
    random_seed: int = 42,
    return_model: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Any]]:
    """
    Incrementally recover activity sequences using beam search over softmax matrices.

    Discovers a Petri net model from training traces, then processes test softmax
    matrices one step at a time. The beam search maintains top candidates and
    outputs a running accuracy measure.

    Parameters
    ----------
    df
        Event log with 'case:concept:name' and 'concept:name' columns.
    softmax_lst
        Softmax matrices per trace (required).
    n_train_traces
        Number of traces for model discovery (ignored if train_cases is set).
    n_test_traces
        Number of traces to test (ignored if test_cases is set).
    train_cases, test_cases
        Specific case IDs for training/testing (overrides respective counts).
    ensure_train_variant_diversity, ensure_test_variant_diversity
        Enforce distinct trace variants in train/test splits.
    cost_function
        'linear', 'logarithmic', or a callable mapping float→float.
    non_sync_penalty
        Penalty weight for non-synchronous transitions.
    beam_width
        Beam search width (number of candidates retained).
    activity_prob_threshold
        Minimum softmax probability to consider an activity.
    use_cond_probs, lambdas, alpha, use_ngram_smoothing
        Conditional probability settings (weights, blending, smoothing).
    use_calibration, temp_bounds
        Whether to apply temperature scaling and its bounds.
    n_indices : Optional[int]
        Total number of events to sample per trace (for uniform sampling).
        Used when sequential_sampling=False. Mutually exclusive with n_per_run.
    n_per_run : Optional[int]
        Number of events to sample from each activity run (for sequential sampling).
        Used when sequential_sampling=True. Mutually exclusive with n_indices.
    sequential_sampling : bool
        If True, sample from each run of identical activities using n_per_run.
        If False, sample uniformly from entire trace using n_indices.
    independent_sampling : bool
        If True, each trace uses a different random seed derived from base seed.
        If False, all traces use the same random state for sampling.
    round_precision
        Digits to round probabilities to.
    random_seed
        Seed for reproducibility.
    return_model
        If True, also return the discovered Petri net model.

    Returns
    -------
    results_df
        DataFrame with columns:
          - case:concept:name, step, predicted_activity, ground_truth,
            beam_probability, is_correct, cumulative_accuracy
    (results_df, model)
        If return_model is True.
    """
    logger.info("Starting incremental softmax recovery.")

    # Prepare default containers
    lambdas = lambdas or []

    # 1. Validate sampling parameters
    if sequential_sampling and n_per_run is None:
        raise ValueError("n_per_run must be specified when sequential_sampling=True")
    if not sequential_sampling and n_indices is None:
        raise ValueError("n_indices must be specified when sequential_sampling=False")
    if n_indices is not None and n_per_run is not None:
        raise ValueError("n_indices and n_per_run are mutually exclusive")
    if n_indices is None and n_per_run is None:
        raise ValueError("Either n_indices or n_per_run must be specified")

    # 2. Validate other inputs
    sampling_param = n_per_run if sequential_sampling else n_indices
    validate_input_parameters(
        sampling_param, round_precision, non_sync_penalty, alpha, temp_bounds
    )

    # 2. Cost function
    cost_fn = process_cost_function(cost_function, round_precision)

    # 3. Softmax preparation
    softmax_np = prepare_softmax(softmax_lst)

    # 4. Filter data and matrices
    filtered_log, filtered_softmax = filter_indices(
        df, 
        softmax_np, 
        n_indices=n_indices,
        n_per_run=n_per_run,
        sequential_sampling=sequential_sampling,
        independent_sampling=independent_sampling, 
        random_seed=random_seed
    )

    # 5. Train/test split
    train_df, test_df = split_train_test(
        filtered_log,
        n_train_traces,
        n_test_traces,
        train_cases,
        test_cases,
        ensure_train_variant_diversity=ensure_train_variant_diversity,
        ensure_test_variant_diversity=ensure_test_variant_diversity,
        random_seed=random_seed,
    )

    # 6. Model discovery
    logger.info("Discovering Petri net model from training data.")
    model = discover_petri_net(train_df, non_sync_penalty)

    # 7. Conditional probabilities (optional)
    prob_dict = {}
    if use_cond_probs:
        prob_dict = build_probability_dict(train_df, use_cond_probs, lambdas)

    # 8. Prepare test softmax matrices (with optional calibration)
    test_softmax_matrices = select_softmax_matrices(filtered_softmax, test_df)
    if use_calibration:
        train_softmax_matrices = select_softmax_matrices(softmax_np, train_df)
        test_softmax_matrices = calibrate_softmax(
            train_df,
            test_df,
            train_softmax_matrices,
            test_softmax_matrices,
            temp_bounds,
        )

    # 9. Validate alignment between test cases and matrices
    test_cases = test_df["case:concept:name"].drop_duplicates().tolist()
    logger.info(f"Processing {len(test_cases)} cases: {test_cases}")
    if not test_softmax_matrices or len(test_cases) != len(test_softmax_matrices):
        raise ValueError(
            "No softmax matrices provided"
            if not test_softmax_matrices
            else f"Found {len(test_cases)} cases but "
                 f"{len(test_softmax_matrices)} matrices"
        )

    # 10. Incremental beam-search recovery
    recovery_records: List[dict] = []
    for idx, (case, softmax_matrix) in enumerate(
        zip(test_cases, test_softmax_matrices), start=1
    ):
        logger.debug(f"Case {idx}/{len(test_cases)}: {case}")
        records = _process_test_case_incremental(
            trace_case=case,
            test_df=test_df,
            softmax_matrix=softmax_matrix,
            model=model,
            cost_function=cost_fn,
            beam_width=beam_width,
            lambdas=lambdas,
            alpha=alpha,
            use_cond_probs=use_cond_probs,
            prob_dict=prob_dict,
            use_ngram_smoothing=use_ngram_smoothing,
            round_precision=round_precision,
            activity_prob_threshold=activity_prob_threshold,
        )
        recovery_records.extend(records)

    # 11. Build and return results
    results_df = pd.DataFrame(recovery_records)
    logger.info("Incremental recovery completed.")
    return (results_df, model) if return_model else results_df
