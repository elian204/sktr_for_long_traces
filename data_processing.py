"""
Data Processing Utilities for Process Mining
============================================

Provides utilities for event log manipulation, trace sampling, and neural network
output processing in process mining applications.

Key Features:
- Softmax matrix processing and PyTorch tensor conversion
- Trace sampling (uniform or sequential strategies) 
- Train/test splitting with variant diversity support
- Reproducible random sampling with configurable independence

Main Functions:
- prepare_softmax: Convert tensors to NumPy arrays
- filter_indices: Sample events from traces  
- split_train_test: Create train/test splits with diversity constraints

Example:
    >>> filtered_df, _ = filter_indices(df, softmax_list, n_indices=5)
    >>> train_df, test_df = split_train_test(df, 100, 50, random_seed=42)
"""

from __future__ import annotations
from typing import Any, Optional, List, Sequence, Union, Tuple
import numpy as np
import pandas as pd
import torch
import random
import hashlib


def prepare_softmax(
    softmax_list: List
) -> List[np.ndarray]:
    """
    Convert a list of softmax arrays (or tensors) into a list of NumPy arrays.
    """
    if softmax_list is None:
        raise ValueError("softmax_list cannot be None - softmax matrices are required")
    if not softmax_list:
        raise ValueError("softmax_list cannot be empty - softmax matrices are required")
    try:
        return convert_tensors_to_numpy(softmax_list)
    except Exception as e:
        raise ValueError(f"Failed to convert softmax list to numpy arrays: {e}") from e


def convert_tensors_to_numpy(
    softmax_list: List[Union[torch.Tensor, np.ndarray]]
) -> List[np.ndarray]:
    """
    Convert a list of PyTorch tensors or NumPy arrays into NumPy arrays,
    squeezing out the leading singleton dimension.

    Parameters
    ----------
    softmax_list : List[Union[torch.Tensor, np.ndarray]]
        A sequence where each element is either:
        - a PyTorch Tensor of shape (1, …), possibly on GPU, or
        - a NumPy array (any shape).

    Returns
    -------
    List[np.ndarray]
        A list of NumPy arrays. For tensor inputs, the returned array is
        `tensor.cpu().numpy().squeeze(0)`. For array inputs, `np.asarray`
        is used to ensure a NumPy array.
    """
    return [
        # GPU → CPU → NumPy → squeeze leading dim
        item.cpu().numpy().squeeze(0)  # type: ignore[attr-defined]
        if isinstance(item, torch.Tensor) else np.asarray(item)
        for item in softmax_list
    ]


def filter_indices(
    df: pd.DataFrame,
    softmax_list: Optional[List[np.ndarray]],
    n_indices: Optional[int] = None,
    n_per_run: Optional[int] = None,
    sequential_sampling: bool = False,
    random_seed: int = 42,
    independent_sampling: bool = True
) -> Tuple[pd.DataFrame, Optional[List[np.ndarray]]]:
    """
    Sample events from each trace in the dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Event log with 'case:concept:name' and 'concept:name' columns.
    softmax_list : Optional[List[np.ndarray]]
        Softmax matrices aligned with unique cases in df.
        Each matrix shape: (n_classes, n_events_in_trace).
    n_indices : Optional[int]
        Total number of events to sample per trace (for uniform sampling).
        Used when sequential_sampling=False. Mutually exclusive with n_per_run.
    n_per_run : Optional[int]  
        Number of events to sample from each activity run (for sequential sampling).
        Used when sequential_sampling=True. Mutually exclusive with n_indices.
    sequential_sampling : bool, default=False
        If True, sample from each run of identical activities using n_per_run.
        If False, sample uniformly from entire trace using n_indices.
    random_seed : int, default=42
        Base random seed for reproducible sampling.
    independent_sampling : bool, default=True
        If True, each trace uses a different random seed derived from base seed.
        If False, all traces use the same random state.
        
    Returns
    -------
    tuple
        (filtered_df, filtered_softmax_list)
        - filtered_df: DataFrame with sampled events
        - filtered_softmax_list: None or list of filtered softmax matrices
        
    Raises
    ------
    ValueError
        If parameter combination is invalid or alignment issues exist.
        
    Examples
    --------
    # Uniform sampling: 5 total events per trace
    >>> filter_indices(df, softmax, n_indices=5, sequential_sampling=False)
    
    # Sequential sampling: 2 events from each activity run  
    >>> filter_indices(df, softmax, n_per_run=2, sequential_sampling=True)
    
    Notes
    -----
    When independent_sampling=True, each trace gets its own random seed:
    trace_seed = base_seed + trace_index. This ensures reproducible but
    diverse sampling patterns across traces.
    """
    # Validate parameter combinations
    if sequential_sampling and n_per_run is None:
        raise ValueError("n_per_run must be specified when sequential_sampling=True")
    if not sequential_sampling and n_indices is None:
        raise ValueError("n_indices must be specified when sequential_sampling=False")
    if n_indices is not None and n_per_run is not None:
        raise ValueError("n_indices and n_per_run are mutually exclusive")
    
    # Get unique cases in appearance order
    case_ids = df['case:concept:name'].drop_duplicates().tolist()
    
    # Validate softmax list if provided
    if softmax_list is not None:
        _validate_softmax_alignment(df, softmax_list, case_ids)
    
    # Setup random number generation strategy
    if independent_sampling:
        def get_rng_for_trace(trace_idx: int) -> random.Random:
            return random.Random(random_seed + trace_idx)
    else:
        shared_rng = random.Random(random_seed)
        def get_rng_for_trace(trace_idx: int) -> random.Random:
            return shared_rng
    
    # Process each trace independently
    filtered_dfs = []
    filtered_softmax = [] if softmax_list is not None else None
    
    for i, case_id in enumerate(case_ids):
        # Validate that case_id matches the index
        if str(i) != str(case_id):
            raise ValueError(f"Case ID '{case_id}' at index {i} doesn't match expected value '{i}'")
        
        # Get trace data
        trace_df = df[df['case:concept:name'] == case_id].reset_index(drop=True)
        trace_length = len(trace_df)
        
        # Get RNG for this specific trace
        trace_rng = get_rng_for_trace(i)
        
        # Sample indices based on strategy
        if sequential_sampling:
            assert n_per_run is not None  # Type narrowing for mypy
            sampled_indices = _sample_sequential_indices(
                trace_df, n_per_run, trace_rng
            )
        else:
            assert n_indices is not None  # Type narrowing for mypy
            sampled_indices = _sample_uniform_indices(
                trace_length, n_indices, trace_rng
            )
        
        # Filter dataframe and softmax
        filtered_trace = trace_df.iloc[sampled_indices].reset_index(drop=True)
        filtered_dfs.append(filtered_trace)
        
        if softmax_list is not None:
            filtered_matrix = softmax_list[i][:, sampled_indices]
            filtered_softmax.append(filtered_matrix)
        
    # Combine results
    result_df = pd.concat(filtered_dfs, ignore_index=True)
    
    return result_df, filtered_softmax


def _sample_uniform_indices(
    trace_length: int,
    n_indices: int,
    rng: random.Random,
) -> List[int]:
    """
    Uniformly sample up to `n_indices` distinct positions from [0, trace_length).
    Indices are returned sorted in ascending order (chronological).

    Complexity: O(k log k) where k = min(n_indices, trace_length).
    """
    if trace_length <= 0 or n_indices <= 0:
        return []
    k = min(n_indices, trace_length)
    # random.sample supports range directly without materializing a list
    return sorted(rng.sample(range(trace_length), k))


def _sample_sequential_indices(
    trace_df: pd.DataFrame,
    n_per_run: int,
    rng: random.Random,
) -> List[int]:
    """
    Sample events sequentially from each run of identical activities.

    This is an optimized version that uses vectorized operations where possible.
    """
    """
    Sample up to `n_per_run` positions from each maximal run of identical
    'concept:name' values within the trace. Indices are returned sorted.

    - Consecutive NaNs are treated as the *same* activity (no boundary).
    - Other transitions (including None↔non-None) start a new run.

    Complexity: O(n + r log n_per_run), where n is trace length and r is number of runs.
    """
    if n_per_run <= 0:
        return []

    names = trace_df["concept:name"].to_numpy()
    n = len(names)
    if n == 0:
        return []

    def _equal(a, b) -> bool:
        # Treat NaN == NaN as equal; otherwise normal equality
        if pd.isna(a) and pd.isna(b):
            return True
        return a == b

    sampled: List[int] = []
    run_start = 0

    # scan runs without sentinels
    for i in range(1, n + 1):
        at_boundary = (i == n) or (not _equal(names[i], names[i - 1]))
        if at_boundary:
            # [run_start, i) is a maximal run
            length = i - run_start
            k = min(n_per_run, length)
            if k:
                sampled.extend(rng.sample(range(run_start, i), k))
            run_start = i

    return sorted(sampled)


def _validate_softmax_alignment(
    df: pd.DataFrame, 
    softmax_list: List[np.ndarray], 
    case_ids: List[str]
) -> None:
    """Validate that softmax matrices align with dataframe traces."""
    if len(case_ids) != len(softmax_list):
        raise ValueError(
            f"Number of cases ({len(case_ids)}) doesn't match "
            f"number of softmax matrices ({len(softmax_list)})"
        )
    
    trace_lengths = df['case:concept:name'].value_counts()
    for i, case_id in enumerate(case_ids):
        expected_length = trace_lengths[case_id]
        actual_length = softmax_list[i].shape[1]
        
        if expected_length != actual_length:
            raise ValueError(
                f"Case '{case_id}': trace has {expected_length} events "
                f"but softmax matrix has {actual_length} columns"
            )


def validate_sequential_case_ids(df: pd.DataFrame, softmax_lst: List[np.ndarray]) -> None:
    """
    Validate that case IDs are sequential strings starting from '0'.
    
    This function ensures the restrictive assumption required by the current
    implementation: case IDs must be ['0', '1', '2', ...] in order.
    
    Parameters
    ----------
    df : pd.DataFrame
        Event log with 'case:concept:name' column
    softmax_lst : List[np.ndarray]
        List of softmax matrices, one per case
        
    Raises
    ------
    ValueError
        If case IDs don't match the sequential string pattern
    """
    # Get unique case IDs in order of appearance
    unique_cases = df['case:concept:name'].drop_duplicates().tolist()
    
    # Check count alignment
    if len(unique_cases) != len(softmax_lst):
        raise ValueError(
            f"Number of unique cases ({len(unique_cases)}) doesn't match "
            f"number of softmax matrices ({len(softmax_lst)})"
        )
    
    # Check sequential pattern
    expected_cases = [str(i) for i in range(len(unique_cases))]
    
    if unique_cases != expected_cases:
        raise ValueError(
            f"Case IDs must be sequential strings starting from '0'.\n"
            f"Expected: {expected_cases}\n"
            f"Found: {unique_cases}\n"
            f"This implementation requires case IDs to be exactly ['0', '1', '2', ...] "
            f"to maintain alignment between traces and softmax matrices."
        )


def split_train_test(
    df: pd.DataFrame,
    n_train_traces: int,
    n_test_traces: int,
    train_cases: Optional[List[Any]] = None,
    test_cases: Optional[List[Any]] = None,
    ensure_train_variant_diversity: bool = False,
    ensure_test_variant_diversity: bool = False,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split event log into train and test sets based on traces.
    
    Parameters
    ----------
    df : pd.DataFrame
        Event log with 'case:concept:name' column.
    n_train_traces : int
        Number of training traces to select (only used when train_cases is None).
    n_test_traces : int
        Number of test traces to select (only used when test_cases is None).
    train_cases : Optional[List[Any]]
        Specific case IDs for training. If provided, ALL these cases are used
        and n_train_traces is ignored.
    test_cases : Optional[List[Any]]
        Specific case IDs for testing. If provided, ALL these cases are used
        and n_test_traces is ignored.
    ensure_train_variant_diversity : bool, default=False
        If True, select train cases from different trace variants.
        Only used when train_cases is None.
    ensure_test_variant_diversity : bool, default=False
        If True, select test cases from different trace variants.
        Only used when test_cases is None.
    random_seed : int, default=42
        Random seed for reproducible selection.
        
    Returns
    -------
    tuple
        (train_df, test_df) - DataFrames containing train and test events.
        
    Examples
    --------
    >>> train_df, test_df = split_train_test(
    ...     df, n_train_traces=10, n_test_traces=5,
    ...     ensure_test_variant_diversity=True, random_seed=42
    ... )
    """
    # Get all unique cases
    all_cases = df['case:concept:name'].drop_duplicates().tolist()
    
    # Generate derived seeds for train and test to ensure independence
    train_seed = _get_derived_seed(random_seed, "train")
    test_seed = _get_derived_seed(random_seed, "test")
    
    # Determine train cases
    if train_cases is not None:
        final_train_cases = train_cases
    elif ensure_train_variant_diversity:
        final_train_cases = _select_diverse_cases(df, n_train_traces, train_seed)
    else:
        final_train_cases = _select_random_cases(all_cases, n_train_traces, train_seed)
    
    # Determine test cases (excluding train cases to prevent data leakage)
    remaining_cases = [c for c in all_cases if c not in final_train_cases]
    
    if test_cases is not None:
        final_test_cases = test_cases
    elif ensure_test_variant_diversity:
        remaining_df = df.loc[df['case:concept:name'].isin(remaining_cases)]
        final_test_cases = _select_diverse_cases(remaining_df, n_test_traces, test_seed)
    else:
        final_test_cases = _select_random_cases(remaining_cases, n_test_traces, test_seed)
    
    # Create train and test DataFrames
    train_df = _extract_cases(df, final_train_cases)
    test_df = _extract_cases(df, final_test_cases)
    
    return train_df, test_df


def _select_random_cases(
    all_cases: List[str], 
    n_cases: int, 
    seed: int
) -> List[str]:
    """Randomly select n_cases from all available cases."""
    rng = random.Random(seed)
    n_to_select = min(n_cases, len(all_cases))
    return rng.sample(all_cases, n_to_select)


def _select_diverse_cases(
    df: pd.DataFrame, 
    n_cases: int, 
    seed: int
) -> List[str]:
    """
    Select cases from different trace variants to ensure diversity.
    Each variant represents a unique sequence of activities.
    """
    rng = random.Random(seed)
    
    # Group cases by their trace variant (sequence of activities)
    trace_variants = {}
    for case_id in df['case:concept:name'].unique():
        case_trace = df[df['case:concept:name'] == case_id]['concept:name'].tolist()
        trace_signature = tuple(case_trace)
        
        if trace_signature not in trace_variants:
            trace_variants[trace_signature] = []
        trace_variants[trace_signature].append(case_id)
    
    selected_cases = []
    available_variants = list(trace_variants.keys())
    
    # Shuffle the variants to randomize selection order
    rng.shuffle(available_variants)
    
    # First, select one case from each different variant
    for i in range(min(n_cases, len(available_variants))):
        variant = available_variants[i]
        selected_case = rng.choice(trace_variants[variant])
        selected_cases.append(selected_case)
    
    # If we still need more cases, select randomly from remaining cases
    if len(selected_cases) < n_cases:
        all_cases = df['case:concept:name'].unique().tolist()
        remaining_cases = [c for c in all_cases if c not in selected_cases]
        
        n_additional = n_cases - len(selected_cases)
        if len(remaining_cases) >= n_additional:
            additional_cases = rng.sample(remaining_cases, n_additional)
            selected_cases.extend(additional_cases)
        else:
            # If not enough remaining cases, select all remaining
            selected_cases.extend(remaining_cases)
    
    return selected_cases[:n_cases]


def _extract_cases(df: pd.DataFrame, case_ids: List[str]) -> pd.DataFrame:
    """Extract events for specified case IDs, maintaining order."""
    if not case_ids:
        return df.iloc[0:0].copy()
    
    # Filter and preserve original row order
    filtered = df[df['case:concept:name'].isin(case_ids)].reset_index()
    
    # Sort by case_ids order, then by original position
    case_order = {case_id: i for i, case_id in enumerate(case_ids)}
    filtered['_order'] = filtered['case:concept:name'].map(lambda x: case_order[x])
    
    return (filtered.sort_values(['_order', 'index'])
                   .drop(columns=['_order', 'index'])
                   .reset_index(drop=True))


def _get_derived_seed(base_seed: int, context: str) -> int:
    """Generate a derived seed for different contexts."""
    seed_str = f"{base_seed}_{context}"
    return int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)


def select_softmax_matrices(
    softmax_matrices: Sequence[np.ndarray],
    df: pd.DataFrame
) -> List[np.ndarray]:
    """
    Select softmax matrices corresponding to case IDs in DataFrame.
    
    Maps case IDs to softmax matrix indices. Supports both integer and string case IDs.
    Uses standard event log columns: 'case:concept:name' and 'concept:name'.
    
    Parameters
    ----------
    softmax_matrices : Sequence[np.ndarray]
        Sequence of softmax matrices, indexed by case ID
    df : pd.DataFrame
        DataFrame with columns 'case:concept:name' and 'concept:name'
        
    Returns
    -------
    List[np.ndarray]
        Softmax matrices corresponding to unique case IDs in df
        
    Raises
    ------
    ValueError
        If required columns missing, case IDs invalid, or indices out of bounds
        
    Examples
    --------
    >>> matrices = [np.array([[0.8, 0.2], [0.3, 0.7]]), 
    ...            np.array([[0.9, 0.1], [0.4, 0.6]])]
    >>> df = pd.DataFrame({'case:concept:name': ['0', '1', '0'],
    ...                   'concept:name': ['A', 'B', 'C']})
    >>> result = select_softmax_matrices(matrices, df)
    >>> len(result)  # Should be 2 (unique cases 0 and 1)
    2
    """
    case_column = 'case:concept:name'
    activity_column = 'concept:name'
    
    # Validate required columns
    if case_column not in df.columns:
        raise ValueError(f"DataFrame missing required column: '{case_column}'")
    if activity_column not in df.columns:
        raise ValueError(f"DataFrame missing required column: '{activity_column}'")
    
    # Get unique case IDs in appearance order
    unique_case_ids = df[case_column].drop_duplicates().tolist()
    
    if not unique_case_ids:
        raise ValueError("No case IDs found in DataFrame")
    
    # Convert case IDs to matrix indices
    try:
        indices = _convert_case_ids_to_indices(unique_case_ids, len(softmax_matrices))
    except ValueError as e:
        raise ValueError(f"Case ID conversion failed: {e}") from e
    
    # Select and return matrices
    selected_matrices = [softmax_matrices[idx] for idx in indices]
    

    return selected_matrices


def _convert_case_ids_to_indices(
    case_ids: List[Union[str, int]],
    max_matrices: int
) -> List[int]:
    """
    Convert case IDs to valid matrix indices.

    Supports both integer and string case IDs.
    """
    indices = []

    for case_id in case_ids:
        try:
            # Convert to integer index
            idx = int(case_id)

            # Validate bounds
            if idx < 0:
                raise ValueError(f"Negative case ID not allowed: {case_id}")
            if idx >= max_matrices:
                raise ValueError(f"Case ID {case_id} out of bounds (max: {max_matrices - 1})")

            indices.append(idx)

        except (ValueError, TypeError) as e:
            if "invalid literal" in str(e):
                raise ValueError(f"Case ID '{case_id}' is not convertible to integer")
            raise ValueError(f"Invalid case ID '{case_id}': {e}")

    return indices


def write_collapsed_traces_to_file(
    df: pd.DataFrame,
    output_file_path: str,
    case_column: str = 'case:concept:name',
    activity_column: str = 'concept:name',
    separator: str = ' '
) -> None:
    """
    Compatibility wrapper. Use trace_export.write_collapsed_traces instead.
    """
    # Absolute import so it works in notebooks/scripts without a package context
    from trace_export import write_collapsed_traces  # type: ignore

    write_collapsed_traces(
        df=df,
        output_file_path=output_file_path,
        case_column=case_column,
        activity_column=activity_column,
        separator=separator,
        line_prefix='* ',
        line_suffix=' #',
    )