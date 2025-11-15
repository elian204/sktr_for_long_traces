"""
Standalone Trace Sampler Module
================================

This module provides functionality to sample training and test traces using the same
logic as incremental_softmax_recovery, ensuring identical case selection given the
same parameters.

You can copy this module to other projects to maintain consistent trace sampling.

Key Features:
- Identical train/test case selection logic as incremental_softmax_recovery
- Support for random sampling or variant diversity sampling
- Reproducible results with random seeds
- Export collapsed traces (self-loops removed) to text files
- Optional test set selection with full control over overlap

Example Usage:
    >>> from trace_sampler import sample_and_write_collapsed_traces

    >>> # Training only
    >>> train_ids, _ = sample_and_write_collapsed_traces(
    ...     df, n=10, train_output_file_path='train.txt', random_seed=42
    ... )

    >>> # Training and test
    >>> train_ids, test_ids = sample_and_write_collapsed_traces(
    ...     df, n=10, train_output_file_path='train.txt',
    ...     n_test=5, test_output_file_path='test.txt',
    ...     ensure_train_variant_diversity=True,
    ...     ensure_test_variant_diversity=True,
    ...     random_seed=42
    ... )

Dependencies:
    - pandas
    - hashlib (standard library)
    - random (standard library)
"""

from __future__ import annotations
from typing import Any, Optional, List, Iterable, Sequence, Union, Tuple
import pandas as pd
import random
import hashlib


def collapse_consecutive_runs(activities: Iterable[Any]) -> List[Any]:
    """
    Collapse consecutive repeated items in an iterable, preserving order.

    NaN values are considered equal to NaN for the purpose of runs.
    Returns a list.

    Parameters
    ----------
    activities : Iterable[Any]
        Sequence of activity names

    Returns
    -------
    List[Any]
        Activities with consecutive duplicates removed

    Examples
    --------
    >>> collapse_consecutive_runs(['A', 'A', 'B', 'B', 'B', 'C', 'A'])
    ['A', 'B', 'C', 'A']
    """
    activities_list = list(activities)
    if not activities_list:
        return []

    def _equal(a, b) -> bool:
        if pd.isna(a) and pd.isna(b):
            return True
        return a == b

    collapsed: List[Any] = []
    current = activities_list[0]
    for item in activities_list[1:]:
        if not _equal(item, current):
            collapsed.append(current)
            current = item
    collapsed.append(current)
    return collapsed


def write_collapsed_traces(
    df: pd.DataFrame,
    output_file_path: str,
    case_column: str = 'case:concept:name',
    activity_column: str = 'concept:name',
    separator: str = ' ',
    line_prefix: str = '* ',
    line_suffix: str = ' #',
) -> None:
    """
    Write one collapsed trace per line to a text file.

    Consecutive repeated activities are collapsed into one (self-loops removed).

    Parameters
    ----------
    df : pd.DataFrame
        Event log DataFrame
    output_file_path : str
        Path where to write the output file
    case_column : str, default 'case:concept:name'
        Column name for case IDs
    activity_column : str, default 'concept:name'
        Column name for activities
    separator : str, default ' '
        Separator between activities in output
    line_prefix : str, default '* '
        Prefix for each line
    line_suffix : str, default ' #'
        Suffix for each line

    Examples
    --------
    >>> write_collapsed_traces(df, 'traces.txt')
    # Creates file with format:
    # * activity1 activity2 activity3 #
    # * activity1 activity2 #
    """
    if case_column not in df.columns:
        raise ValueError(f"DataFrame missing required column: '{case_column}'")
    if activity_column not in df.columns:
        raise ValueError(f"DataFrame missing required column: '{activity_column}'")

    unique_case_ids = df[case_column].drop_duplicates().tolist()

    with open(output_file_path, 'w', encoding='utf-8') as f:
        for case_id in unique_case_ids:
            case_activities = df[df[case_column] == case_id][activity_column].tolist()
            if not case_activities:
                f.write(f"{line_prefix}{line_suffix}\n")
                continue

            collapsed = collapse_consecutive_runs(case_activities)
            line = separator.join(str(x) for x in collapsed)
            f.write(f"{line_prefix}{line}{line_suffix}\n")


def _get_derived_seed(base_seed: int, context: str) -> int:
    """
    Generate a derived seed for different contexts.

    This ensures that train and test splits use different random sequences
    even when starting from the same base seed.
    """
    seed_str = f"{base_seed}_{context}"
    return int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)


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

    Uses hardcoded column names 'case:concept:name' and 'concept:name'
    to match incremental_softmax_recovery behavior.
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


def _extract_cases(df: pd.DataFrame, case_ids: List[Any]) -> pd.DataFrame:
    """Extract all events for the requested case IDs preserving selection order."""
    if not case_ids:
        return df.iloc[0:0].copy()

    filtered = df[df['case:concept:name'].isin(case_ids)].reset_index()
    case_order = {case_id: idx for idx, case_id in enumerate(case_ids)}
    filtered['_case_order'] = filtered['case:concept:name'].map(
        lambda x: case_order.get(x, len(case_order))
    )

    return (
        filtered.sort_values(['_case_order', 'index'])
        .drop(columns=['_case_order', 'index'])
        .reset_index(drop=True)
    )


def _convert_case_ids_to_indices(
    case_ids: List[Union[str, int]],
    max_matrices: int
) -> List[int]:
    """Convert case identifiers into positional indices for softmax matrices."""
    indices: List[int] = []

    for case_id in case_ids:
        try:
            idx = int(case_id)
        except (ValueError, TypeError) as err:
            if isinstance(err, ValueError) and "invalid literal" in str(err):
                raise ValueError(f"Case ID '{case_id}' is not convertible to integer")
            raise ValueError(f"Invalid case ID '{case_id}': {err}")

        if idx < 0:
            raise ValueError(f"Negative case ID not allowed: {case_id}")
        if idx >= max_matrices:
            raise ValueError(
                f"Case ID {case_id} out of bounds (max: {max_matrices - 1})"
            )

        indices.append(idx)

    return indices


def _select_softmax_matrices(
    softmax_matrices: Sequence[Any],
    df: pd.DataFrame
) -> List[Any]:
    """Select softmax matrices that align with the case order inside df."""
    case_column = 'case:concept:name'
    if case_column not in df.columns:
        raise ValueError(f"DataFrame missing required column: '{case_column}'")

    unique_case_ids = df[case_column].drop_duplicates().tolist()
    if not unique_case_ids:
        return []

    indices = _convert_case_ids_to_indices(unique_case_ids, len(softmax_matrices))
    return [softmax_matrices[idx] for idx in indices]


def sample_and_write_collapsed_traces(
    df: pd.DataFrame,
    n: int = 10,
    train_output_file_path: str = 'sampled_traces.txt',
    case_column: str = 'case:concept:name',
    activity_column: str = 'concept:name',
    random_seed: int = 42,
    ensure_train_variant_diversity: bool = False,
    train_cases: Optional[List[Any]] = None,
    n_test: Optional[int] = None,
    test_output_file_path: Optional[str] = None,
    ensure_test_variant_diversity: bool = False,
    test_cases: Optional[List[Any]] = None,
    allow_train_cases_in_test: bool = False,
    softmax_matrices: Optional[Sequence[Any]] = None,
    verbose: bool = True
) -> Union[
    Tuple[List[str], Optional[List[str]]],
    Tuple[List[str], Optional[List[str]], List[Any], Optional[List[Any]]]
]:
    """
    Sample training (and optionally test) cases using the same logic as
    incremental_softmax_recovery and write collapsed traces to file(s).

    This function selects cases using the exact same logic as the train/test split in
    incremental_softmax_recovery. Given the same parameters, it will select the same cases.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe containing the event log data with columns
        'case:concept:name' and 'concept:name'
    n : int, default 10
        Number of training cases to select (equivalent to n_train_traces
        in incremental_softmax_recovery)
    train_output_file_path : str, default 'sampled_traces.txt'
        Path where to write the training collapsed traces
    case_column : str, default 'case:concept:name'
        Name of the case identifier column in df.
        Note: For selection logic, hardcoded 'case:concept:name' is used.
        This parameter only affects how traces are written to disk.
    activity_column : str, default 'concept:name'
        Name of the activity column in df.
        Note: For selection logic, hardcoded 'concept:name' is used.
        This parameter only affects how traces are written to disk.
    random_seed : int, default 42
        Random seed for reproducible sampling. Internally derives train/test seeds using
        _get_derived_seed(random_seed, "train"/"test") to match incremental_softmax_recovery.
    ensure_train_variant_diversity : bool, default False
        If True, select training cases from different trace variants (ensures diversity).
        Equivalent to ensure_train_variant_diversity in incremental_softmax_recovery.
    train_cases : List[Any], optional
        Specific case IDs to use for training. If provided, these cases are used directly
        and n, ensure_train_variant_diversity are ignored.
    n_test : int, optional
        Number of test cases to select. If None, no test cases are selected.
        If provided, supply test_output_file_path unless you pass softmax_matrices
        to only retrieve the selected matrices without writing.
        Equivalent to n_test_traces in incremental_softmax_recovery.
    test_output_file_path : str, optional
        Path where to write the test collapsed traces. Required when writing tests
        to disk; optional if you only need the sampled IDs/matrices.
    ensure_test_variant_diversity : bool, default False
        If True, select test cases from different trace variants (ensures diversity).
        Only used when test_cases is None.
    test_cases : List[Any], optional
        Specific case IDs to use for testing. If provided, these cases are used directly
        and n_test, ensure_test_variant_diversity are ignored.
    allow_train_cases_in_test : bool, default False
        If True, allow train cases to remain eligible for test selection.
        Only applies when test_cases is None.
    softmax_matrices : Sequence[Any], optional
        Sequence of softmax matrices indexed by case ID. When provided, the function
        also returns the matrices corresponding to the sampled train and (if selected)
        test cases, mimicking incremental_softmax_recovery.
    verbose : bool, default True
        Whether to print the sampled trace IDs

    Returns
    -------
    tuple
        Returns (train_case_ids, test_case_ids) when softmax_matrices is None.
        If softmax_matrices is provided, returns
        (train_case_ids, test_case_ids, train_softmax_matrices, test_softmax_matrices),
        where the final two elements are lists aligned with the sampled case IDs
        (test_softmax_matrices is None when no test selection occurs).

    Examples
    --------
    # Training only
    >>> train_ids, _ = sample_and_write_collapsed_traces(df, n=10, random_seed=42)
    Sampled 10 train trace IDs: ['5', '12', '18', ...]

    # Training and test with variant diversity
    >>> train_ids, test_ids = sample_and_write_collapsed_traces(
    ...     df,
    ...     n=20,
    ...     train_output_file_path='train_traces.txt',
    ...     random_seed=42,
    ...     ensure_train_variant_diversity=True,
    ...     n_test=10,
    ...     test_output_file_path='test_traces.txt',
    ...     ensure_test_variant_diversity=True
    ... )

    # Using all remaining cases for test
    >>> train_ids, test_ids = sample_and_write_collapsed_traces(
    ...     df,
    ...     n=10,
    ...     train_output_file_path='train.txt',
    ...     n_test=None,  # Use None to select all remaining cases
    ...     test_output_file_path='test.txt'
    ... )

    # Returning aligned softmax matrices for downstream use
    >>> train_ids, test_ids, train_mats, test_mats = sample_and_write_collapsed_traces(
    ...     df,
    ...     n=10,
    ...     train_output_file_path='train.txt',
    ...     n_test=5,
    ...     test_output_file_path='test.txt',
    ...     softmax_matrices=softmax_np
    ... )

    Notes
    -----
    - Self-loops (consecutive duplicate activities) are automatically removed
    - Output format: "* activity1 activity2 activity3 #" (one trace per line)
    - Case selection uses hardcoded 'case:concept:name' and 'concept:name' columns
      to match incremental_softmax_recovery behavior
    """
    # Validate required columns for selection logic
    if 'case:concept:name' not in df.columns:
        raise ValueError("DataFrame missing required column: 'case:concept:name'")
    if 'concept:name' not in df.columns:
        raise ValueError("DataFrame missing required column: 'concept:name'")

    # Determine whether test selection/output is requested
    wants_test_selection = (
        n_test is not None or test_cases is not None or test_output_file_path is not None
    )

    # Validate test parameters
    if (
        (n_test is not None or test_cases is not None)
        and test_output_file_path is None
        and softmax_matrices is None
    ):
        raise ValueError(
            "Provide test_output_file_path or softmax_matrices when selecting test cases"
        )

    # Get all unique cases (using hardcoded column for selection)
    all_cases = df['case:concept:name'].drop_duplicates().tolist()

    # Generate derived seeds for train and test (matches incremental_softmax_recovery)
    train_seed = _get_derived_seed(random_seed, "train")
    test_seed = _get_derived_seed(random_seed, "test")

    # === Train case selection ===
    # Determine selected cases using the same logic as split_train_test
    if train_cases is not None:
        train_case_ids = train_cases
    elif ensure_train_variant_diversity:
        train_case_ids = _select_diverse_cases(df, n, train_seed)
    else:
        train_case_ids = _select_random_cases(all_cases, n, train_seed)

    # Print train trace IDs if verbose is True
    if verbose:
        print(f"Sampled {len(train_case_ids)} train trace IDs: {train_case_ids}")

    # Filter (preserving selection order) and write train traces
    train_filtered_df = _extract_cases(df, train_case_ids)
    write_collapsed_traces(
        df=train_filtered_df,
        output_file_path=train_output_file_path,
        case_column=case_column,
        activity_column=activity_column
    )

    train_softmax_matrices: Optional[List[Any]] = None
    if softmax_matrices is not None:
        train_softmax_matrices = _select_softmax_matrices(softmax_matrices, train_filtered_df)

    # === Test case selection (optional) ===
    test_case_ids = None
    test_softmax_matrices: Optional[List[Any]] = None
    if wants_test_selection:
        # Determine remaining cases (optionally allowing overlap with training cases)
        if allow_train_cases_in_test:
            remaining_cases = all_cases
        else:
            remaining_cases = [c for c in all_cases if c not in train_case_ids]

        # Select test cases using the same logic as split_train_test
        if test_cases is not None:
            test_case_ids = test_cases
        elif n_test is None:
            # Use all available cases for testing
            test_case_ids = remaining_cases
        elif ensure_test_variant_diversity:
            candidate_df = df if allow_train_cases_in_test else df.loc[df['case:concept:name'].isin(remaining_cases)]
            test_case_ids = _select_diverse_cases(candidate_df, n_test, test_seed)
        else:
            test_case_ids = _select_random_cases(remaining_cases, n_test, test_seed)

        # Print test trace IDs if verbose is True
        if verbose:
            print(f"Sampled {len(test_case_ids)} test trace IDs: {test_case_ids}")

        # Filter test traces while preserving selection order
        test_filtered_df = _extract_cases(df, test_case_ids)

        if test_output_file_path is not None:
            write_collapsed_traces(
                df=test_filtered_df,
                output_file_path=test_output_file_path,
                case_column=case_column,
                activity_column=activity_column
            )

        if softmax_matrices is not None:
            test_softmax_matrices = _select_softmax_matrices(
                softmax_matrices,
                test_filtered_df
            )

    if softmax_matrices is not None:
        return (
            train_case_ids,
            test_case_ids,
            train_softmax_matrices or [],
            test_softmax_matrices,
        )

    return train_case_ids, test_case_ids


if __name__ == '__main__':
    # Example usage
    print("Trace Sampler Module")
    print("=" * 50)
    print("\nThis module provides standalone trace sampling functionality.")
    print("Import and use sample_and_write_collapsed_traces() in your code.")
    print("\nExamples:")
    print("\n  # Training only")
    print("  from trace_sampler import sample_and_write_collapsed_traces")
    print("  train_ids, _ = sample_and_write_collapsed_traces(df, n=10, random_seed=42)")
    print("\n  # Training and test")
    print("  train_ids, test_ids = sample_and_write_collapsed_traces(")
    print("      df, n=10, train_output_file_path='train.txt',")
    print("      n_test=5, test_output_file_path='test.txt', random_seed=42)")
    print("\n  # With variant diversity")
    print("  train_ids, test_ids = sample_and_write_collapsed_traces(")
    print("      df, n=20, ensure_train_variant_diversity=True,")
    print("      n_test=10, ensure_test_variant_diversity=True,")
    print("      train_output_file_path='train.txt', test_output_file_path='test.txt')")
