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
from utils import validate_input_parameters, make_cost_function
from data_processing import prepare_softmax, filter_indices, split_train_test, select_softmax_matrices, validate_sequential_case_ids
from petri_model import discover_petri_net, build_probability_dict
from calibration import calibrate_softmax
from beam_search import process_test_case_incremental

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
    model_move_cost: Optional[Union[float, str, Callable[[float], float]]] = 1.0,
    log_move_cost:   Optional[Union[float, str, Callable[[float], float]]] = 1.0,
    tau_move_cost:   Optional[Union[float, str, Callable[[float], float]]] = 1e-6,
    non_sync_penalty: float = 1.0,
    beam_width: int = 10,
    activity_prob_threshold: float = 0.0,
    use_cond_probs: bool = False,
    max_hist_len: int = 3,
    lambdas: Optional[List[float]] = None,
    alpha: float = 0.5,
    use_ngram_smoothing: bool = True,
    use_calibration: bool = False,
    temp_bounds: Tuple[float, float] = (1.0, 10.0),
    temperature: Optional[float] = None,
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
    df : pd.DataFrame
        Event log with 'case:concept:name' and 'concept:name' columns.
        **IMPORTANT**: Case IDs must be sequential strings starting from '0'.
        The case IDs must be exactly ['0', '1', '2', ..., 'N-1'] in order
        to maintain alignment with the softmax_lst matrices. This is a 
        restrictive requirement of the current implementation.
    softmax_lst : List[np.ndarray]
        Softmax matrices per trace (required). Must be aligned with case IDs:
        softmax_lst[0] corresponds to case '0', softmax_lst[1] to case '1', etc.
    n_train_traces : int, default=10
        Number of traces for model discovery (ignored if train_cases is set).
    n_test_traces : int, default=10
        Number of traces to test (ignored if test_cases is set).
    train_cases, test_cases : Optional[List[Any]], default=None
        Specific case IDs for training/testing (overrides respective counts).
        If provided, must follow the sequential string format.
    ensure_train_variant_diversity, ensure_test_variant_diversity : bool, default=False
        Enforce distinct trace variants in train/test splits.
    cost_function : str or callable, default='linear'
        'linear', 'logarithmic', or a callable mapping float→float.
    non_sync_penalty : float, default=1.0
        Penalty weight for non-synchronous transitions.
    beam_width : int, default=10
        Beam search width (number of candidates retained).
    activity_prob_threshold : float, default=0.0
        Minimum softmax probability to consider an activity.
    use_cond_probs : bool, default=False
        Whether to use conditional probabilities based on trace history.
    max_hist_len : int, default=3
        Maximum history length for conditional probabilities.
    lambdas : Optional[List[float]], default=None
        Blending weights for conditional probability computation.
    alpha : float, default=0.5
        Blending parameter for conditional probabilities (0=history only, 1=base only).
    use_ngram_smoothing : bool, default=True
        Whether to apply n-gram smoothing for conditional probabilities.
    use_calibration : bool, default=False
        Whether to apply temperature scaling for probability calibration.
    temp_bounds : Tuple[float, float], default=(1.0, 10.0)
        Temperature bounds for calibration optimization.
    temperature : Optional[float], default=None
        Manual temperature value (bypasses optimization if provided).
    n_indices : Optional[int], default=None
        Total number of events to sample per trace (for uniform sampling).
        Used when sequential_sampling=False. Mutually exclusive with n_per_run.
    n_per_run : Optional[int], default=None
        Number of events to sample from each activity run (for sequential sampling).
        Used when sequential_sampling=True. Mutually exclusive with n_indices.
    sequential_sampling : bool, default=False
        If True, sample from each run of identical activities using n_per_run.
        If False, sample uniformly from entire trace using n_indices.
    independent_sampling : bool, default=True
        If True, each trace uses a different random seed derived from base seed.
        If False, all traces use the same random state for sampling.
    round_precision : int, default=2
        Digits to round probabilities to.
    random_seed : int, default=42
        Seed for reproducibility.
    return_model : bool, default=False
        If True, also return the discovered Petri net model.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with columns:
          - case:concept:name, step, predicted_activity, ground_truth,
            beam_probability, is_correct, cumulative_accuracy
    (results_df, model) : tuple
        If return_model is True, returns tuple of (results_df, petri_net_model).
        
    Raises
    ------
    ValueError
        If case IDs don't follow the sequential string format ['0', '1', '2', ...].
        If parameter combinations are invalid (e.g., both n_indices and n_per_run specified).
        If required data is missing or misaligned.
        
    Notes
    -----
    This implementation has a restrictive assumption about case ID format:
    - Case IDs must be sequential strings: ['0', '1', '2', ..., 'N-1']
    - The order in the DataFrame must match the softmax matrix order
    - softmax_lst[i] must correspond to case ID str(i)
    
    For datasets with non-sequential case IDs (e.g., 'CASE_001', 'CASE_042'), 
    you must preprocess to convert to sequential format.
        
    Examples
    --------
    >>> # Correct case ID format
    >>> df = pd.DataFrame({
    ...     'case:concept:name': ['0', '0', '1', '1', '2', '2'],
    ...     'concept:name': ['A', 'B', 'A', 'C', 'B', 'C']
    ... })
    >>> softmax_matrices = [matrix_0, matrix_1, matrix_2]  # Aligned order
    >>> results = incremental_softmax_recovery(df, softmax_matrices, n_indices=2)
    """
    logger.info("Starting incremental softmax recovery.")

    # Prepare default containers
    lambdas = lambdas or []

    # 0. Early validation of sequential case ID assumption
    validate_sequential_case_ids(df, softmax_lst)

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
    assert sampling_param is not None, "sampling_param should not be None after validation"
    validate_input_parameters(
        sampling_param, round_precision, non_sync_penalty, alpha, temp_bounds
    )

    # 3. Cost function
    cost_fn = make_cost_function(
        base=cost_function,
        model_move=model_move_cost,
        log_move=log_move_cost,
        tau_move=tau_move_cost,
        round_precision=round_precision
    )

    # 4. Softmax preparation
    softmax_np = prepare_softmax(softmax_lst)

    # 5. Filter data and matrices
    filtered_log, filtered_softmax = filter_indices(
        df, 
        softmax_np, 
        n_indices=n_indices,
        n_per_run=n_per_run,
        sequential_sampling=sequential_sampling,
        independent_sampling=independent_sampling, 
        random_seed=random_seed
    )

    # 6. Train/test split
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

    # 7. Model discovery
    logger.info("Discovering Petri net model from training data.")
    model = discover_petri_net(train_df, non_sync_penalty)

    # 8. Conditional probabilities (optional)
    prob_dict = {}
    if use_cond_probs:
        prob_dict = build_probability_dict(train_df, max_hist_len)

    # 9. Prepare test softmax matrices (with optional calibration)
    if filtered_softmax is None:
        raise ValueError("Filtered softmax matrices are required but none were generated")
    
    test_softmax_matrices = select_softmax_matrices(filtered_softmax, test_df)
    if use_calibration:
        # Use complete original matrices for calibration training (not filtered)
        train_softmax_matrices = select_softmax_matrices(softmax_np, train_df)
        test_softmax_matrices = calibrate_softmax(
            train_df=train_df,
            test_df=test_df,
            softmax_train=train_softmax_matrices,
            softmax_test=test_softmax_matrices,
            temp_bounds=temp_bounds,
            temperature=temperature,
        )
    
    # 10. Extract test case IDs for processing
    test_case_ids = test_df['case:concept:name'].drop_duplicates().tolist()
    
    # 11. Incremental beam-search recovery
    recovery_records: List[dict] = []
    for idx, (case, softmax_matrix) in enumerate(
        zip(test_case_ids, test_softmax_matrices), start=1
    ):
        logger.debug(f"Case {idx}/{len(test_case_ids)}: {case}")
        
        # Get predicted sequence using beam search
        predicted_sequence = process_test_case_incremental(
            softmax_matrix=softmax_matrix,
            model=model,
            cost_fn=cost_fn,
            beam_width=beam_width,
            lambdas=lambdas,
            alpha=alpha,
            use_cond_probs=use_cond_probs,
            prob_dict=prob_dict,
            use_ngram_smoothing=use_ngram_smoothing,
            activity_prob_threshold=activity_prob_threshold,
        )
        
        # Extract ground truth sequence for accuracy computation
        ground_truth_trace = test_df[test_df['case:concept:name'] == case].copy().reset_index(drop=True)
        ground_truth_sequence = ground_truth_trace['concept:name'].tolist()
        
        # Compute accuracy and create records
        records = _compute_accuracy_records(
            case_id=case,
            predicted_sequence=predicted_sequence,
            ground_truth_sequence=ground_truth_sequence
        )
        recovery_records.extend(records)

    # 12. Build and return results
    results_df = pd.DataFrame(recovery_records)
    logger.info("Incremental recovery completed.")
    return (results_df, model) if return_model else results_df


def _compute_accuracy_records(
    case_id: str,
    predicted_sequence: List[str],
    ground_truth_sequence: List[str]
) -> pd.DataFrame:
    """
    Compute accuracy records by comparing predicted and ground truth sequences.
    
    Parameters
    ----------
    case_id : str
        Case identifier
    predicted_sequence : List[str]
        Predicted activity sequence
    ground_truth_sequence : List[str]
        Ground truth activity sequence
        
    Returns
    -------
    pd.DataFrame
        DataFrame with accuracy metrics for each step
        
    Raises
    ------
    ValueError
        If the sequences have different lengths
    """
    # Check that both sequences have the same length
    if len(predicted_sequence) != len(ground_truth_sequence):
        raise ValueError(
            f"Sequences must have the same length. "
            f"Predicted: {len(predicted_sequence)}, Ground truth: {len(ground_truth_sequence)}"
        )
    
    # Efficiently compute matches using numpy for better performance
    predicted_array = np.array(predicted_sequence)
    ground_truth_array = np.array(ground_truth_sequence)
    matches = predicted_array == ground_truth_array
    
    # Create DataFrame directly
    data = {
        'case:concept:name': [case_id] * len(predicted_sequence),
        'step': list(range(len(predicted_sequence))),
        'predicted_activity': predicted_sequence,
        'ground_truth': ground_truth_sequence,
        'is_correct': matches,
        'cumulative_accuracy': np.cumsum(matches) / np.arange(1, len(matches) + 1)
    }
    
    return pd.DataFrame(data)



