"""
Softmax Trace Recovery: Main high-level function.

This module provides the main entry point for recovering activity sequences from 
softmax probability matrices using Petri net models. Supports both beam search 
and conformance checking approaches for flexible trace recovery.
"""

from typing import Any, Callable, List, Optional, Tuple, Union, Dict, Set
import logging
import pandas as pd
import numpy as np
import concurrent.futures
import multiprocessing
from utils import validate_input_parameters, make_cost_function, visualize_petri_net
from data_processing import prepare_softmax, filter_indices, split_train_test, select_softmax_matrices, validate_sequential_case_ids, _extract_cases
from petri_model import discover_petri_net, build_probability_dict
from calibration import calibrate_probabilities, calibrate_softmax
from conformance_checking import process_trace_chunked

# Configure logger
logger = logging.getLogger(__name__)
import sys


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


def _process_single_test_case(
    case_id: str,
    softmax_matrix: np.ndarray,
    test_df: pd.DataFrame,
    model: Any,
    cost_fn: Callable,
    prob_threshold: float,
    prob_dict: Dict,
    chunk_size: int,
    conformance_switch_penalty_weight: float,
    use_state_caching: bool,
    merge_mismatched_boundaries: bool,
) -> Tuple[str, List[str], List[float], float, float, pd.DataFrame]:
    """
    Process a single test case using conformance checking. Used for parallel processing.

    Returns:
        Tuple of (case_id, predictions, move_costs, sktr_accuracy, argmax_accuracy, records_df)
    """
    # Extract ground truth sequence for accuracy computation
    ground_truth_trace = test_df[test_df['case:concept:name'] == case_id].copy().reset_index(drop=True)
    ground_truth_sequence = ground_truth_trace['concept:name'].tolist()

    # Compute argmax predictions
    argmax_indices = np.argmax(softmax_matrix, axis=0)
    argmax_preds = [str(idx) for idx in argmax_indices]

    # Conformance prediction
    sktr_preds, sktr_move_costs = process_trace_chunked(
        softmax_matrix=softmax_matrix,
        model=model,
        cost_fn=cost_fn,
        chunk_size=chunk_size,
        eps=prob_threshold,
        inline_progress=False,  # Disable progress for parallel processing
        prob_dict=prob_dict,
        switch_penalty_weight=conformance_switch_penalty_weight,
        use_state_caching=use_state_caching,
        merge_mismatched_boundaries=merge_mismatched_boundaries,
    )

    # Compute accuracy
    records_df, sktr_acc, argmax_acc = _compute_accuracy_records(
        case_id=case_id,
        sktr_preds=sktr_preds,
        argmax_preds=argmax_preds,
        ground_truth_sequence=ground_truth_sequence,
        softmax_matrix=softmax_matrix,
        activity_prob_threshold=prob_threshold,
        sktr_move_costs=sktr_move_costs,
        chunk_size=chunk_size
    )

    return case_id, sktr_preds, sktr_move_costs, sktr_acc, argmax_acc, records_df


def incremental_softmax_recovery(
    df: pd.DataFrame,
    softmax_lst: List[np.ndarray],
    n_train_traces: int = 10,
    n_test_traces: int = 10,
    train_cases: Optional[List[Any]] = None,
    test_cases: Optional[List[Any]] = None,
    ensure_train_variant_diversity: bool = False,
    ensure_test_variant_diversity: bool = False,
    allow_train_cases_in_test: bool = False,
    cost_function: Union[str, Callable[[float], float]] = "linear",
    model_move_cost: Optional[Union[float, str, Callable[[float], float]]] = 1.0,
    log_move_cost:   Optional[Union[float, str, Callable[[float], float]]] = 1.0,
    tau_move_cost:   Optional[Union[float, str, Callable[[float], float]]] = 1e-6,
    prob_threshold: float = 1e-12,
    max_hist_len: int = 3,
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
    chunk_size: int = 10,
    # Conformance-specific: switch penalty weight on label change (uses bigram prob_dict)
    conformance_switch_penalty_weight: float = 0.0,
    # Performance optimization parameters
    adaptive_chunk_sizing: bool = True,
    max_chunk_size: int = 50,
    use_state_caching: bool = True,
    parallel_processing: bool = False,
    max_workers: Optional[int] = None,
    merge_mismatched_boundaries: bool = True,
    # removed: restrict_to_observed_moves
    verbose: bool = True,
    log_level: int = logging.INFO,
) -> Tuple[pd.DataFrame, Dict[str, List[float]], Dict[Tuple[str, ...], Dict[str, float]]]:
    """
    Recover activity sequences from softmax matrices using Petri net models (conformance only).

    Discovers a Petri net model from training traces, then processes test softmax
    matrices using chunked conformance checking.
    """
    if verbose:
        setup_logging(log_level)
    
    logger.info("Starting incremental softmax recovery (conformance-only).")

    # Prepare default containers
    
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
    # alpha removed from validation, default a safe value for API compatibility
    validate_input_parameters(
        sampling_param, round_precision, non_sync_penalty, temp_bounds
    )
    logger.info(f"Validated input parameters: round_precision={round_precision}, prob_threshold={prob_threshold}.")

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
        allow_train_cases_in_test=allow_train_cases_in_test,
        random_seed=random_seed,
    )
    n_train_cases = len(train_df['case:concept:name'].unique())
    n_test_cases = len(test_df['case:concept:name'].unique())
    logger.info(f"Performed train/test split: {n_train_cases} train cases, {n_test_cases} test cases.")
    overlap_cases = set(train_df['case:concept:name'].unique()).intersection(test_df['case:concept:name'].unique())
    if overlap_cases:
        logger.info(f"Train/test overlap enabled for {len(overlap_cases)} case(s).")
    print(test_df['case:concept:name'].unique())

    # 7. Model discovery
    logger.info("Discovering Petri net model from training data.")
    model = discover_petri_net(train_df)
    model.enable_caching(True)
    
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

    # 8. Conditional probabilities (build when needed for switch penalty)
    prob_dict: Dict[Tuple[str, ...], Dict[str, float]] = {}
    if conformance_switch_penalty_weight > 0.0:
        prob_dict = build_probability_dict(train_df, max_hist_len)
        n_histories = len(prob_dict)
        avg_activities_per_history = np.mean([len(activities) for activities in prob_dict.values()]) if prob_dict else 0
        logger.info(f"Built conditional probability dictionary: {n_histories} histories, avg {avg_activities_per_history:.1f} activities per history.")
    else:
        logger.info("Skipping probability dictionary build (not requested).")

    # 9. Prepare test softmax matrices (with optional calibration)
    if filtered_softmax is None:
        raise ValueError("Filtered softmax matrices are required but none were generated")
    
    test_softmax_matrices = select_softmax_matrices(filtered_softmax, test_df)
    
    if use_calibration:
        # Learn temperature on FULL, unfiltered training traces to ensure alignment
        train_case_ids = train_df['case:concept:name'].drop_duplicates().tolist()
        train_df_full = _extract_cases(df, train_case_ids)
        train_softmax_matrices = select_softmax_matrices(softmax_np, train_df_full)
        used_temperature = temperature
        if used_temperature is None:
            used_temperature = calibrate_probabilities(
                softmax_list=train_softmax_matrices,
                df=train_df_full,
                temp_bounds=temp_bounds,
                only_return_temperature=True
            )
        test_softmax_matrices = calibrate_softmax(
            train_df=train_df_full,
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
    
    # 11. Incremental recovery with optional parallel processing
    recovery_records: List[dict] = []
    sktr_accs: List[float] = []
    argmax_accs: List[float] = []

    # Adaptive chunk sizing based on model complexity (for sequential processing)
    effective_chunk_size = chunk_size
    if adaptive_chunk_sizing and not parallel_processing:
        # Scale chunk size based on model complexity (places + transitions)
        model_complexity = len(model.places) + len(model.transitions)
        # Conservative heuristic: slightly smaller chunks for very complex models
        # to avoid exponential search space growth
        if model_complexity > 100:
            # Reduce chunk size for very complex models to control search space
            complexity_factor = max(0.7, 100 / model_complexity)  # Min 0.7x
            effective_chunk_size = max(5, int(chunk_size * complexity_factor))
        elif model_complexity > 50:
            # Keep original size for moderately complex models
            effective_chunk_size = chunk_size
        else:
            # Slightly increase for simple models (less overhead)
            complexity_factor = min(model_complexity / 50, 1.5)  # Max 1.5x
            effective_chunk_size = min(int(chunk_size * complexity_factor), max_chunk_size)

        if effective_chunk_size != chunk_size:
            logger.debug(f"Using adaptive chunk size: {effective_chunk_size} (base: {chunk_size}, complexity: {model_complexity})")

    if parallel_processing and len(test_case_ids) > 1:
        # Parallel processing
        logger.info(f"Processing {len(test_case_ids)} test cases in parallel")

        # Determine number of workers
        n_workers = max_workers or min(len(test_case_ids), multiprocessing.cpu_count())
        logger.info(f"Using {n_workers} parallel workers")

        # Prepare arguments for parallel processing
        parallel_args = []
        for case, softmax_matrix in zip(test_case_ids, test_softmax_matrices):
            args = (
                case, softmax_matrix, test_df, model, cost_fn,
                prob_threshold, prob_dict, effective_chunk_size,
                conformance_switch_penalty_weight, use_state_caching,
                merge_mismatched_boundaries,
            )
            parallel_args.append(args)

        # Process in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_process_single_test_case, *args) for args in parallel_args]

            for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                try:
                    case_id, sktr_preds, sktr_move_costs, sktr_acc, argmax_acc, records_df = future.result()
                    recovery_records.extend(records_df.to_dict('records'))
                    sktr_accs.append(sktr_acc)
                    argmax_accs.append(argmax_acc)
                    logger.debug(f"Case {idx}/{len(test_case_ids)} ({case_id}) [conformance]: SKTR={sktr_acc:.3f}, Argmax={argmax_acc:.3f}, Sequence length={len(sktr_preds)}")
                except Exception as exc:
                    logger.error(f"Parallel processing failed for case {idx}: {exc}")

    else:
        # Sequential processing
        for idx, (case, softmax_matrix) in enumerate(
            zip(test_case_ids, test_softmax_matrices), start=1
        ):
            # In-place progress update for traces (single updating line)
            try:
                sys.stdout.write(f"\rcase {idx}/{len(test_case_ids)} — conformance")
                sys.stdout.flush()
            except Exception:
                pass
            # Avoid extra per-trace INFO logs when using inline progress
            logger.debug(
                f"Processing test case {idx}/{len(test_case_ids)} ({case}) using 'conformance'"
            )

            sktr_preds, sktr_move_costs = process_trace_chunked(
                softmax_matrix=softmax_matrix,
                model=model,
                cost_fn=cost_fn,
                chunk_size=effective_chunk_size,
                eps=prob_threshold,
                inline_progress=True,
                progress_prefix=f"case {idx}/{len(test_case_ids)}",
                prob_dict=prob_dict,
                switch_penalty_weight=conformance_switch_penalty_weight,
                use_state_caching=use_state_caching,
                merge_mismatched_boundaries=merge_mismatched_boundaries,
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
                activity_prob_threshold=prob_threshold,
                sktr_move_costs=sktr_move_costs,
                chunk_size=effective_chunk_size
            )
            recovery_records.extend(records_df.to_dict('records'))
            sktr_accs.append(sktr_acc)
            argmax_accs.append(argmax_acc)
            logger.debug(f"Case {idx}/{len(test_case_ids)} ({case}) [conformance]: SKTR={sktr_acc:.3f}, Argmax={argmax_acc:.3f}, Sequence length={len(sktr_preds)}")

    # 12. Build and return results
    results_df = pd.DataFrame(recovery_records)
    accuracy_dict = {
        'sktr_accuracy': sktr_accs,
        'argmax_accuracy': argmax_accs
    }
    # End the in-place progress line
    try:
        sys.stdout.write("\n")
        sys.stdout.flush()
    except Exception:
        pass
    logger.info("Built results DataFrame and accuracy dictionary.")
    logger.info("Softmax trace recovery completed using conformance method.")
    return results_df, accuracy_dict, prob_dict


def _compute_accuracy_records(
    case_id: str,
    sktr_preds: List[str],
    argmax_preds: List[str],
    ground_truth_sequence: List[str],
    softmax_matrix: np.ndarray,
    activity_prob_threshold: float = 0.0,
    sktr_move_costs: List[float] = None,
    chunk_size: int = None
) -> Tuple[pd.DataFrame, float, float]:
    """
    Compute accuracy records and trace-level accuracies by comparing SKTR, argmax, and ground truth sequences.
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
    
    # Create window-relative steps
    if chunk_size is not None and chunk_size > 0:
        # Create steps within each window: 0, 1, 2, ..., chunk_size-1, 0, 1, 2, ...
        window_steps = []
        for i in range(seq_len):
            window_steps.append(i % chunk_size)
    else:
        # Fall back to global steps if no chunk_size provided
        window_steps = list(range(seq_len))

    # Add to data
    data = {
        'case:concept:name': [case_id] * seq_len,
        'step': window_steps,
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


def _build_prefix_to_next_run_duration_map_from_labels(
   labels: List[Any],
   prefix_len: int,
   key_by_activity: bool = False,
) -> Dict[Union[Tuple[Any, ...], Tuple[Tuple[Any, ...], Any]], Set[int]]:
   """
   Internal helper: compute the prefix-to-next-run-duration map for a single
   sequence of labels (already provided as a list).
   
   Parameters
   ----------
   labels : List[Any]
       Sequence of activity labels.
   prefix_len : int
       Exact prefix length to extract.
   key_by_activity : bool, default=False
       If True, use (prefix, next_activity) as the key; otherwise use prefix only.
       This allows separating durations by the next activity to avoid false conflicts.
       
   Returns
   -------
   Dict[Union[Tuple[Any, ...], Tuple[Tuple[Any, ...], Any]], Set[int]]
       Mapping from key to the set of observed next run durations. The key is either the
       prefix tuple (when key_by_activity=False) or a 2-tuple of (prefix tuple, next activity)
       when key_by_activity=True.
   """
   if prefix_len <= 0 or not labels:
       return {}

   collapsed_labels: List[Any] = []
   run_lengths: List[int] = []
   for label in labels:
       if not collapsed_labels or label != collapsed_labels[-1]:
           collapsed_labels.append(label)
           run_lengths.append(1)
       else:
           run_lengths[-1] += 1

   num_runs = len(collapsed_labels)
   if num_runs <= prefix_len:
       return {}

   result: Dict[Union[Tuple[Any, ...], Tuple[Tuple[Any, ...], Any]], Set[int]] = {}
   for start_idx in range(0, num_runs - prefix_len):
       prefix = tuple(collapsed_labels[start_idx:start_idx + prefix_len])
       next_activity = collapsed_labels[start_idx + prefix_len]
       key = (prefix, next_activity) if key_by_activity else prefix
       next_run_length = run_lengths[start_idx + prefix_len]
       
       if key in result:
           result[key].add(next_run_length)
       else:
           result[key] = {next_run_length}

   return result


def build_prefix_to_next_run_duration_map(
  df: pd.DataFrame,
  prefix_len: int,
  case_col: str = 'case:concept:name',
  activity_col: str = 'concept:name',
  key_by_activity: bool = False,
) -> Union[Dict[Tuple[Any, ...], int], Dict[Tuple[Any, ...], Dict[Any, int]]]:
  """
  Build a dictionary mapping run-collapsed prefixes of EXACT length `prefix_len` to
  the next activity's run duration by aggregating across all cases in a DataFrame.

  Parameters
  ----------
  df : pd.DataFrame
      Event log containing at least case and activity columns.
  prefix_len : int
      Exact prefix length to include per case/run-collapsed sequence.
  case_col : str, default='case:concept:name'
      Column name identifying the case identifier.
  activity_col : str, default='concept:name'
      Column name identifying the activity label.
  key_by_activity : bool, default=False
      If True, keys are (prefix tuple, next activity) and values are (duration, activity).
      If False, keys are prefix tuples and values are durations.

  Returns
  -------
  Union[Dict[Tuple[Any, ...], int], Dict[Tuple[Any, ...], Dict[Any, int]]]
      When key_by_activity=False: Dict[prefix_tuple, duration]
      When key_by_activity=True: Dict[prefix_tuple, Dict[next_activity, duration]]
      Aggregated across all cases. Keys with conflicting durations are dropped.

  Raises
  ------
  ValueError
      If required columns are missing from the DataFrame.

  Notes
  -----
  - Grouping preserves the row order within each case as it appears in `df`.
  - If you require a different ordering, pre-sort `df` before calling this function.
  - When conflicts occur, the key is dropped.

  Examples
  --------
  >>> df = pd.DataFrame({
  ...     'case:concept:name': ['case1', 'case1', 'case1', 'case1', 'case1'],
  ...     'concept:name': ['A', 'A', 'B', 'B', 'C']
  ... })
  >>> build_prefix_to_next_run_duration_map(df, prefix_len=2)
  {('A', 'B'): 1}
  >>> build_prefix_to_next_run_duration_map(df, prefix_len=1, key_by_activity=True)
  {('A',): {'B': 2}, ('B',): {'C': 1}}
  """
  if prefix_len <= 0:
      return {}

  if case_col not in df.columns or activity_col not in df.columns:
      raise ValueError(
          f"DataFrame must contain columns '{case_col}' and '{activity_col}'."
      )

  aggregated_sets: Dict[Union[Tuple[Any, ...], Tuple[Tuple[Any, ...], Any]], Set[int]] = {}
  for _, case_df in df.groupby(case_col, sort=False):
      labels = case_df[activity_col].tolist()
      local_map = _build_prefix_to_next_run_duration_map_from_labels(labels, prefix_len, key_by_activity=key_by_activity)
      for key, duration_set in local_map.items():
          if key in aggregated_sets:
              aggregated_sets[key].update(duration_set)
          else:
              aggregated_sets[key] = set(duration_set)

  # Keep only keys with a single unique duration; drop conflicting ones
  if key_by_activity:
      nested: Dict[Tuple[Any, ...], Dict[Any, int]] = {}
      for key, duration_set in aggregated_sets.items():
          if len(duration_set) == 1:
              duration = next(iter(duration_set))
              prefix, next_activity = key  # key is (prefix, next_activity)
              if prefix not in nested:
                  nested[prefix] = {}
              nested[prefix][next_activity] = duration
          else:
              logger.warning(
                  f"Removing key {key} due to conflicting durations: {sorted(duration_set)}"
              )
      return nested
  else:
      flat: Dict[Tuple[Any, ...], int] = {}
      for key, duration_set in aggregated_sets.items():
          if len(duration_set) == 1:
              flat[key] = next(iter(duration_set))  # key is prefix
          else:
              logger.warning(
                  f"Removing key {key} due to conflicting durations: {sorted(duration_set)}"
              )
      return flat