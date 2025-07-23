"""
Incremental Softmax Recovery: Main high-level function.

This module provides the main entry point for incremental softmax matrix recovery
using beam search with Petri nets, following the pattern of the existing 
compare_stochastic_vs_argmax_random_indices function.
"""

from typing import Any, Callable, List, Optional, Tuple, Union, Dict
import logging
import pandas as pd
import numpy as np
from utils import validate_input_parameters, make_cost_function, visualize_petri_net
from data_processing import prepare_softmax, filter_indices, split_train_test, select_softmax_matrices, validate_sequential_case_ids
from petri_model import discover_petri_net, build_probability_dict
from calibration import calibrate_probabilities, calibrate_softmax
from beam_search import process_test_case_incremental

# Configure logger
logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    """
    Set up logging configuration for the incremental softmax recovery module.
    
    Parameters
    ----------
    level : int, default=logging.INFO
        Logging level (logging.DEBUG, logging.INFO, logging.WARNING, etc.)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


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
    beam_width: int = 10,
    activity_prob_threshold: float = 0.0,
    use_cond_probs: bool = False,
    max_hist_len: int = 3,
    lambdas: Optional[List[float]] = None,
    alpha: float = 0.5,
    beam_score_alpha: float = 0.5,
    completion_patience: int = 5,
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
    save_model_path: str = "./discovered_petri_net",
    save_model: bool = True,
    non_sync_penalty: float = 1.0,
    verbose: bool = True,
    log_level: int = logging.INFO,
) -> Tuple[pd.DataFrame, Dict[str, List[float]], Dict[Tuple[str, ...], Dict[str, float]]]:
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
    beam_score_alpha : float, default=0.5
        Beam scoring alpha for blending normalized and total costs.
    completion_patience : int, default=5
        Number of extra iterations to continue beam search after first completion.
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
    save_model_path : str, default="./discovered_petri_net"
        Path (without .pdf extension) to save the Petri net visualization.
    save_model : bool, default=True
        If True, save the Petri net visualization to the specified path.
    verbose : bool, default=True
        If True, set up logging to display progress information.
    log_level : int, default=logging.INFO
        Logging level (logging.DEBUG for more detailed output).
    
    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with columns:
          - case:concept:name, step, predicted_activity, ground_truth,
            beam_probability, is_correct, cumulative_accuracy
    accuracy_dict : Dict[str, List[float]]
        Dictionary with keys 'sktr_accuracy' and 'argmax_accuracy',
        each containing a list of per-trace accuracies (fraction of correct predictions in the trace).
    prob_dict : Dict[Tuple[str, ...], Dict[str, float]]
        Conditional probability dictionary (empty if use_cond_probs=False)
        
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
    >>> results_df, accuracy_dict = incremental_softmax_recovery(df, softmax_matrices, n_indices=2)
    """
    if verbose:
        setup_logging(log_level)
    
    logger.info("Starting incremental softmax recovery.")

    # Prepare default containers
    lambdas = lambdas or []

    # 0. Early validation of sequential case ID assumption
    validate_sequential_case_ids(df, softmax_lst)
    logger.info(f"Validated sequential case IDs (found {len(df['case:concept:name'].unique())} unique cases) and {len(softmax_lst)} softmax matrices.")

    # 1. Validate sampling parameters
    if sequential_sampling and n_per_run is None:
        raise ValueError("n_per_run must be specified when sequential_sampling=True")
    if not sequential_sampling and n_indices is None:
        raise ValueError("n_indices must be specified when sequential_sampling=False")
    if n_indices is not None and n_per_run is not None:
        raise ValueError("n_indices and n_per_run are mutually exclusive")
    if n_indices is None and n_per_run is None:
        raise ValueError("Either n_indices or n_per_run must be specified")
    sampling_method = "sequential runs" if sequential_sampling else "uniform sampling"
    sampling_param_name = "n_per_run" if sequential_sampling else "n_indices"
    sampling_param_value = n_per_run if sequential_sampling else n_indices
    logger.info(f"Validated sampling parameters: {sampling_method} with {sampling_param_name}={sampling_param_value}.")

    # 2. Validate other inputs
    sampling_param = n_per_run if sequential_sampling else n_indices
    assert sampling_param is not None, "sampling_param should not be None after validation"
    validate_input_parameters(
        sampling_param, round_precision, non_sync_penalty, alpha, temp_bounds
    )
    logger.info(f"Validated input parameters: beam_width={beam_width}, alpha={alpha}, round_precision={round_precision}.")

    # 3. Cost function
    cost_fn = make_cost_function(
        base=cost_function,
        model_move=model_move_cost,
        log_move=log_move_cost,
        tau_move=tau_move_cost,
        round_precision=round_precision
    )
    logger.info(f"Prepared cost function: {cost_function} (model={model_move_cost}, log={log_move_cost}, tau={tau_move_cost}).")

    # 4. Softmax preparation
    softmax_np = prepare_softmax(softmax_lst)
    if softmax_np and len(softmax_np) > 0:
        first_matrix_shape = softmax_np[0].shape if len(softmax_np) > 0 else "unknown"
        logger.info(f"Prepared softmax arrays: {len(softmax_np)} traces with individual shape {first_matrix_shape}.")
    else:
        logger.info(f"Prepared softmax arrays: {len(softmax_lst)} traces.")

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
    n_events_before = len(df)
    n_events_after = len(filtered_log)
    logger.info(f"Filtered log and softmax matrices: {n_events_before} -> {n_events_after} events ({n_events_after/n_events_before:.1%} retained).")

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
    n_train_cases = len(train_df['case:concept:name'].unique())
    n_test_cases = len(test_df['case:concept:name'].unique())
    logger.info(f"Performed train/test split: {n_train_cases} train cases, {n_test_cases} test cases.")

    # 7. Model discovery
    logger.info("Discovering Petri net model from training data.")
    model = discover_petri_net(train_df)
    if save_model:
        try:
            visualize_petri_net(model, output_path=save_model_path)
            logger.info(f"Petri net visualization saved to {save_model_path}.pdf")
        except Exception as e:
            logger.warning(f"Failed to save Petri net visualization: {e}")
            logger.warning("Continuing without saving model visualization...")
    n_places = len(model.places)
    n_transitions = len(model.transitions)
    logger.info(f"Discovered Petri net model: {n_places} places, {n_transitions} transitions.")
    
    # Compute marking-to-transition map (reachable markings and their tau-reachable transitions)
    logger.info("Computing marking-to-transition map (tau-reachability) for discovered Petri net...")
    marking_transition_map = model.build_marking_transition_map()
    logger.info(f"Computed marking-to-transition map with {len(marking_transition_map)} reachable markings.")


    # 8. Conditional probabilities (optional)
    prob_dict = {}
    if use_cond_probs:
        prob_dict = build_probability_dict(train_df, max_hist_len)
        if use_cond_probs:
            n_histories = len(prob_dict)
            avg_activities_per_history = np.mean([len(activities) for activities in prob_dict.values()]) if prob_dict else 0
            logger.info(f"Built conditional probability dictionary: {n_histories} histories, avg {avg_activities_per_history:.1f} activities per history.")
    logger.info("Built conditional probability dictionary.")

    # 9. Prepare test softmax matrices (with optional calibration)
    if filtered_softmax is None:
        raise ValueError("Filtered softmax matrices are required but none were generated")
    
    test_softmax_matrices = select_softmax_matrices(filtered_softmax, test_df)
    if use_calibration:
        # Use complete original matrices for calibration training (not filtered)
        train_softmax_matrices = select_softmax_matrices(softmax_np, train_df)
        used_temperature = temperature
        if used_temperature is None:
            used_temperature = calibrate_probabilities(
                softmax_list=train_softmax_matrices,
                df=train_df,
                temp_bounds=temp_bounds,
                only_return_temperature=True
            )
        test_softmax_matrices = calibrate_softmax(
            train_df=train_df,
            test_df=test_df,
            softmax_train=train_softmax_matrices,
            softmax_test=test_softmax_matrices,
            temp_bounds=temp_bounds,
            temperature=used_temperature,
        )
        calibration_info = f" with calibration (temperature={used_temperature:.2f})"
    else:
        calibration_info = ""
    logger.info(f"Prepared {len(test_softmax_matrices)} test softmax matrices{calibration_info}.")
    
    # 10. Extract test case IDs for processing
    test_case_ids = test_df['case:concept:name'].drop_duplicates().tolist()
    logger.info(f"Extracted {len(test_case_ids)} test case IDs for processing.")
    
    # 11. Incremental beam-search recovery
    recovery_records: List[dict] = []
    sktr_accs: List[float] = []
    argmax_accs: List[float] = []
    for idx, (case, softmax_matrix) in enumerate(
        zip(test_case_ids, test_softmax_matrices), start=1
    ):
        logger.debug(f"Case {idx}/{len(test_case_ids)}: {case}")
        
        # Get predicted sequence using beam search
        sktr_preds, sktr_move_costs = process_test_case_incremental(
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
            beam_score_alpha=beam_score_alpha,
            completion_patience=completion_patience,
        )
        
        # Extract ground truth sequence for accuracy computation
        ground_truth_trace = test_df[test_df['case:concept:name'] == case].copy().reset_index(drop=True)
        ground_truth_sequence = ground_truth_trace['concept:name'].tolist()
        
        # Compute argmax predictions
        argmax_indices = np.argmax(softmax_matrix, axis=0)
        argmax_preds = [str(idx) for idx in argmax_indices]
        
        # Compute accuracy and create records
        records_df, sktr_acc, argmax_acc = _compute_accuracy_records(
            case_id=case,
            sktr_preds=sktr_preds,
            argmax_preds=argmax_preds,
            ground_truth_sequence=ground_truth_sequence,
            softmax_matrix=softmax_matrix,
            activity_prob_threshold=activity_prob_threshold,
            sktr_move_costs=sktr_move_costs
        )
        recovery_records.extend(records_df.to_dict('records'))
        sktr_accs.append(sktr_acc)
        argmax_accs.append(argmax_acc)
        logger.info(f"Case {idx}/{len(test_case_ids)} ({case}): SKTR={sktr_acc:.3f}, Argmax={argmax_acc:.3f}, Sequence length={len(sktr_preds)}")

    # 12. Build and return results
    results_df = pd.DataFrame(recovery_records)
    accuracy_dict = {
        'sktr_accuracy': sktr_accs,
        'argmax_accuracy': argmax_accs
    }
    logger.info("Built results DataFrame and accuracy dictionary.")
    logger.info("Incremental recovery completed.")
    return results_df, accuracy_dict, prob_dict


def _compute_accuracy_records(
    case_id: str,
    sktr_preds: List[str],
    argmax_preds: List[str],
    ground_truth_sequence: List[str],
    softmax_matrix: np.ndarray,
    activity_prob_threshold: float = 0.0,
    sktr_move_costs: List[float] = None
) -> Tuple[pd.DataFrame, float, float]:
    """
    Compute accuracy records and trace-level accuracies by comparing SKTR, argmax, and ground truth sequences.
    
    Parameters
    ----------
    case_id : str
        Case identifier
    sktr_preds : List[str]
        SKTR predicted activity sequence (beam search)
    argmax_preds : List[str]
        Argmax predicted sequence
    ground_truth_sequence : List[str]
        Ground truth activity sequence
        
    Returns
    -------
    pd.DataFrame
        DataFrame with per-step accuracy metrics for SKTR predictions
    float
        Trace-level SKTR accuracy (fraction correct)
    float
        Trace-level argmax accuracy (fraction correct)
        
    Raises
    ------
    ValueError
        If the sequences have different lengths
    """
    # Check that all sequences have the same length
    seq_len = len(ground_truth_sequence)
    if len(sktr_preds) != seq_len or len(argmax_preds) != seq_len:
        raise ValueError(
            f"All sequences must have the same length. "
            f"SKTR: {len(sktr_preds)}, Argmax: {len(argmax_preds)}, Ground truth: {seq_len}"
        )
    
    # Efficiently compute matches using numpy for better performance
    sktr_array = np.array(sktr_preds)
    argmax_array = np.array(argmax_preds)
    ground_truth_array = np.array(ground_truth_sequence)
    sktr_matches = sktr_array == ground_truth_array
    argmax_matches = argmax_array == ground_truth_array
    
    # Compute trace accuracies
    sktr_acc = np.mean(sktr_matches)
    argmax_acc = np.mean(argmax_matches)
    
    # Replace fixed all_activities
    all_probs_filtered = []
    all_activities_filtered = []
    n_classes = softmax_matrix.shape[0]
    all_possible_activities = [str(i) for i in range(n_classes)]
    for t in range(seq_len):
        probs = softmax_matrix[:, t]
        filtered_indices = np.where(probs >= activity_prob_threshold)[0]
        filtered_probs = [round(p, 2) for p in probs[filtered_indices]]
        filtered_acts = [all_possible_activities[i] for i in filtered_indices]
        all_probs_filtered.append(filtered_probs)
        all_activities_filtered.append(filtered_acts)
    
    # Add to data
    data = {
        'case:concept:name': [case_id] * seq_len,
        'step': list(range(seq_len)),
        'sktr_activity': sktr_preds,
        'argmax_activity': argmax_preds,
        'ground_truth': ground_truth_sequence,
        'all_probs': all_probs_filtered,
        'all_activities': all_activities_filtered,
        'is_correct': sktr_matches,
        'cumulative_accuracy': np.cumsum(sktr_matches) / np.arange(1, seq_len + 1),
        'sktr_move_cost': [round(cost, 2) for cost in sktr_move_costs]
    }
    
    return pd.DataFrame(data), sktr_acc, argmax_acc



