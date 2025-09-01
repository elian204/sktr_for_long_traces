"""
Incremental Softmax Recovery: Main high-level function.

This module provides the main entry point for incremental softmax matrix recovery
using beam search with Petri nets, following the pattern of the existing 
compare_stochastic_vs_argmax_random_indices function.
"""

import pickle
from typing import Tuple, Union, Callable, Optional, Dict, List
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import io
from graphviz import Digraph
import os
from contextlib import redirect_stderr
import logging

MoveType = str

logger = logging.getLogger(__name__)
        
def validate_input_parameters(
    n_indices: int,
    round_precision: int,
    non_sync_penalty: float,
    alpha: float,
    temp_bounds: Tuple[Union[int, float], Union[int, float]],
) -> None:
    """
    Validate input parameters for the incremental softmax recovery function.
    
    Parameters
    ----------
    n_indices : int
        Number of indices/positions per trace. Must be positive.
    round_precision : int
        Digits to round probabilities to. Must be non-negative.
    non_sync_penalty : float
        Penalty weight for non-synchronous transitions. Must be non-negative.
    alpha : float
        Blending parameter for conditional probabilities. Must be between 0 and 1.
    temp_bounds : Tuple[float, float]
        Temperature bounds for calibration. Must be a valid range with min < max.
        The values must be positive.
    Raises
    ------
    ValueError
        If any parameter is invalid.
    TypeError
        If parameters have incorrect types.
    """
    # Validate n_indices
    if not isinstance(n_indices, int):
        raise TypeError(f"n_indices must be an integer, got {type(n_indices)}")
    if n_indices <= 0:
        raise ValueError(f"n_indices must be positive, got {n_indices}")
    
    # Validate round_precision
    if not isinstance(round_precision, int):
        raise TypeError(f"round_precision must be an integer, got {type(round_precision)}")
    if round_precision < 0:
        raise ValueError(f"round_precision must be non-negative, got {round_precision}")
    
    # Validate non_sync_penalty
    if not isinstance(non_sync_penalty, (int, float)):
        raise TypeError(f"non_sync_penalty must be a number, got {type(non_sync_penalty)}")
    if non_sync_penalty < 0:
        raise ValueError(f"non_sync_penalty must be non-negative, got {non_sync_penalty}")
    
    # Validate alpha
    if not isinstance(alpha, (int, float)):
        raise TypeError(f"alpha must be a number, got {type(alpha)}")
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    
    # Validate temp_bounds
    if not isinstance(temp_bounds, tuple):
        raise TypeError(f"temp_bounds must be a tuple, got {type(temp_bounds)}")
    if len(temp_bounds) != 2:
        raise ValueError(f"temp_bounds must be a tuple of length 2, got length {len(temp_bounds)}")
    
    min_temp, max_temp = temp_bounds
    if not isinstance(min_temp, (int, float)) or not isinstance(max_temp, (int, float)):
        raise TypeError(f"temp_bounds values must be numbers, got ({type(min_temp)}, {type(max_temp)})")
    if min_temp >= max_temp:
        raise ValueError(f"temp_bounds min must be less than max, got {temp_bounds}")
    if min_temp <= 0:
        raise ValueError(f"temp_bounds values must be positive, got {temp_bounds}")


def make_cost_function(
    base: Union[str, Callable[[float], float]] = "linear",
    model_move: Optional[Union[float, str, Callable[[float], float]]] = 1.0,
    log_move:   Optional[Union[float, str, Callable[[float], float]]] = 1.0,
    tau_move:   Optional[Union[float, str, Callable[[float], float]]] = 1e-6,
    round_precision: int = 2
) -> Callable[[float, MoveType], float]:
    """
    Build f(p: float, move_type: MoveType) → cost.

    Defaults:
      - sync: `base` (linear/logarithmic/callable)
      - model: constant 1.0
      - log:   constant 1.0
      - tau:   constant 1e-6
    """
    def normalize(cf):
        if isinstance(cf, (float, int)):
            return lambda _: float(cf)
        if isinstance(cf, str):
            if cf == "linear":
                return lambda p: 1 - p
            if cf == "logarithmic":
                min_p = 10 ** (-round_precision)
                scale = -np.log(min_p)
                return lambda p: -np.log(max(min(p, 1.0), min_p)) / scale
            raise ValueError(f"Unknown cost '{cf}'")
        if callable(cf):
            return cf
        raise TypeError(f"Cost spec must be float, str or callable, got {type(cf)}")

    base_fn = normalize(base)
    overrides = {
        "model": normalize(model_move),
        "log":   normalize(log_move),
        "tau":   normalize(tau_move),
    }

    def cost_fn(prob: float, move_type: MoveType = "sync") -> float:
        return overrides.get(move_type, base_fn)(prob)

    return cost_fn


def inverse_softmax(
    softmax_probs: np.ndarray,
    epsilon: float = 1e-9
) -> np.ndarray:
    """
    Convert softmax probabilities to logits (inverse of softmax).

    Parameters
    ----------
    softmax_probs : np.ndarray
        Array of softmax probabilities, shape (n_classes, n_events) or (n_classes,).
        Each probability should be in (0, 1) and columns should sum to 1.
    epsilon : float, default=1e-9
        Small value to avoid log(0) or log(1) which would result in -inf/inf.

    Returns
    -------
    np.ndarray
        Logits with same shape as input.

    Notes
    -----
    - This computes log(p), not the logit function log(p/(1-p)).
    - Softmax logit recovery can only retrieve logits *up to a constant shift*
      since softmax is invariant to adding constants: softmax(x) = softmax(x + c).
    - Used primarily for temperature scaling in probability calibration.

    Examples
    --------
    >>> probs = np.array([[0.7, 0.2], [0.3, 0.8]])
    >>> logits = inverse_softmax(probs)
    >>> # logits ≈ [[-0.357, -1.609], [-1.204, -0.223]]
    """
    # Clip probabilities to avoid log(0) and log(1)
    probs = np.clip(softmax_probs, epsilon, 1.0 - epsilon)
    return np.log(probs)


def compute_conditional_probability(
    path_prefix: Tuple[str, ...],
    activity_name: str,
    base_probability: float,
    prob_dict: dict,
    lambdas: List[float],
    alpha: float,
    use_ngram_smoothing: bool,
) -> float:
    """
    Compute conditional probability based on path history.
    
    Supports both n-gram smoothing and prefix search approaches for
    computing conditional probabilities based on path history.
    
    Parameters
    ----------
    path_prefix : tuple of str
        Sequence of activities leading to current transition
    activity_name : str
        Name of the current activity
    base_probability : float
        Base probability from softmax
    prob_dict : dict
        Probability dictionary for conditional weights
    lambdas : list of float
        Weights for n-gram lengths (required for n-gram smoothing)
    alpha : float
        Blending factor between base and conditional probabilities
    use_ngram_smoothing : bool
        Whether to use n-gram smoothing approach
        
    Returns
    -------
    float
        Conditional probability
    """
    if not prob_dict:
        return base_probability
    
    if use_ngram_smoothing:
        # Validate lambdas when using n-gram smoothing
        if not lambdas:
            raise ValueError("lambdas must be provided when use_ngram_smoothing is True")
        
        conditional_prob = compute_ngram_probability(
            path_prefix, activity_name, prob_dict, lambdas
        )
    else:
        conditional_prob = compute_prefix_search_probability(
            path_prefix, activity_name, prob_dict
        )
    
    # Blend conditional probability with base probability
    return (1 - alpha) * conditional_prob + alpha * base_probability


def simple_bigram_blend(
    path_prefix: Tuple[str, ...],
    activity_name: str,
    base_probability: float,
    prob_dict: Dict[Tuple[str, ...], Dict[str, float]],
    alpha: float,
) -> float:
    """
    Simple function to blend base probability with bigram conditional probability for testing.
    
    Uses bigram P(activity|previous) if previous exists, unigram P(activity) otherwise.
    
    Parameters same as compute_conditional_probability, but ignores lambdas and use_ngram_smoothing.
    """
    if not path_prefix:
        conditional_prob = prob_dict.get((), {}).get(activity_name, 0.0)
    else:
        previous = path_prefix[-1]
        conditional_prob = prob_dict.get((previous,), {}).get(activity_name, 0.0)
    
    return (1 - alpha) * conditional_prob + alpha * base_probability


def compute_ngram_probability(
    path_prefix_tuple: Tuple[str, ...],
    activity_name: str,
    prob_dict: Dict[Tuple[str, ...], Dict[str, float]],
    lambdas: List[float]
) -> float:
    """
    Compute probability using n-gram smoothing.
    
    Parameters
    ----------
    path_prefix_tuple : tuple of str
        Sequence of activities in path prefix
    activity_name : str
        Name of current activity
    prob_dict : dict
        N-gram probability dictionary
    lambdas : list of float
        Weights for different n-gram lengths, ordered from unigram to higher n-grams.
        lambdas[0] = unigram weight, lambdas[1] = bigram weight, 
        lambdas[2] = trigram weight, etc.
        
    Returns
    -------
    float
        Computed probability
    """
    logger = logging.getLogger("ngram_prob")
    if not lambdas:
        raise ValueError("lambdas cannot be empty for n-gram probability computation")
    
    if not path_prefix_tuple:
        base_prob = prob_dict.get((), {}).get(activity_name, 0.0)
        logger.debug(
            f"[n-gram] Empty prefix: returning unigram prob for '{activity_name}': {base_prob}"
        )
        return base_prob
    
    total_weighted_prob = 0.0
    total_lambda_weight = 0.0
    max_n = min(len(path_prefix_tuple), len(lambdas))
    ngram_details = []
    
    for n in range(1, max_n + 1):
        prefix_n_gram = path_prefix_tuple[-n:]
        prob = prob_dict.get(prefix_n_gram, {}).get(activity_name, 0.0)
        lambda_weight = lambdas[n - 1]
        contribution = lambda_weight * prob
        total_weighted_prob += contribution
        total_lambda_weight += lambda_weight
        ngram_details.append(
            f"n={n} prefix={prefix_n_gram} lambda={lambda_weight:.3f} prob={prob:.5f} contrib={contribution:.5f}"
        )
    
    if total_lambda_weight == 0:
        logger.debug(
            f"[n-gram] All lambda weights zero for prefix {path_prefix_tuple} and activity '{activity_name}'. Returning 0.0."
        )
        return 0.0
    
    final_prob = total_weighted_prob / total_lambda_weight
    logger.debug(
        f"[n-gram] path_prefix={path_prefix_tuple}, activity='{activity_name}'\n"
        f"  Details: " + "; ".join(ngram_details) + f"\n"
        f"  total_weighted_prob={total_weighted_prob:.5f}, total_lambda_weight={total_lambda_weight:.5f}, final_prob={final_prob:.5f}"
    )
    return final_prob


def compute_prefix_search_probability(
    path_prefix_tuple: Tuple[str, ...],
    activity_name: str,
    prob_dict: Dict[Tuple[str, ...], Dict[str, float]]
) -> float:
    """
    Compute probability using prefix search.
    
    Parameters
    ----------
    path_prefix_tuple : tuple of str
        Sequence of activities in path prefix
    activity_name : str
        Name of current activity
    prob_dict : dict
        Prefix probability dictionary
        
    Returns
    -------
    float
        Computed probability
    """
    if not path_prefix_tuple:
        return prob_dict.get((), {}).get(activity_name, 0.0)
    
    # Check exact prefix match
    if path_prefix_tuple in prob_dict:
        return prob_dict[path_prefix_tuple].get(activity_name, 0.0)
    
    # Find longest matching prefix
    longest_prefix = find_longest_prefix(path_prefix_tuple, prob_dict)
    if longest_prefix:
        return prob_dict[longest_prefix].get(activity_name, 0.0)
    
    return 0.0


def find_longest_prefix(
    path_prefix_tuple: Tuple[str, ...],
    prob_dict: Dict[Tuple[str, ...], Dict[str, float]]
) -> Optional[Tuple[str, ...]]:
    """
    Find longest prefix that exists in dictionary.
    
    Parameters
    ----------
    path_prefix_tuple : tuple of str
        Complete path to search in
    prob_dict : dict
        Dictionary to search for prefixes
        
    Returns
    -------
    tuple of str or None
        Longest matching prefix, or None if no match found
    """
    for i in range(len(path_prefix_tuple), 0, -1):
        sub_prefix = path_prefix_tuple[-i:]
        if sub_prefix in prob_dict:
            return sub_prefix
    return None


def prepare_df(
    dataset_name: str,
    path: Optional[Union[str, Path]] = None,
    return_mapping: bool = False,
) -> Union[
    Tuple[pd.DataFrame, List[np.ndarray]],
    Tuple[pd.DataFrame, List[np.ndarray], Dict[str, str]]
]:
    """
    Prepare a DataFrame from a specified video dataset.

    Loads target labels and softmax predictions, builds a combined DataFrame,
    auto-fixes matrix orientation, and optionally returns activity mapping.

    Parameters
    ----------
    dataset_name : str
        One of: '50salads', 'gtea', 'breakfast'.
    path : str or Path, optional
        Base directory containing the dataset pickles. This can point either
        directly to the directory with the files or to a parent directory that
        contains a 'video' subfolder. If None, the function will try, in order:
          1) Environment variables 'SKTR_DATASETS_DIR' and 'DATASETS_DIR' (both
             the value itself and '<value>/video')
          2) The user's home directory under 'Datasets/video' and 'Datasets'
          3) Common project-relative locations: 'Datasets/video' and 'Datasets'
             one, two, and three levels above this file
        Required files:
          - {dataset_name}_softmax_lst.pickle
          - {dataset_name}_target_lst.pickle
    return_mapping : bool, optional
        If True, returns the mapping dict as the third element.

    Returns
    -------
    (df, softmax_list) or (df, softmax_list, mapping_dict)
    """
    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            else: return super().find_class(module, name)

    # 1) validate dataset
    valid = {'50salads', 'gtea', 'breakfast'}
    if dataset_name not in valid:
        raise ValueError(f"dataset_name must be one of {valid}")

    # 2) resolve path
    dirs_to_try = []

    # If explicit path provided, prefer a 'video' child first, then the path itself
    if path:
        base = Path(path)
        dirs_to_try += [base / 'video', base]
    else:
        # Environment variables
        for env_var in ('SKTR_DATASETS_DIR', 'DATASETS_DIR'):
            env_val = os.environ.get(env_var)
            if env_val:
                env_base = Path(env_val)
                # Try the 'video' subfolder first
                dirs_to_try += [env_base / 'video', env_base]

        # Home directory conventions: ~/Datasets/video or ~/Datasets (common on servers)
        home = Path.home()
        dirs_to_try += [
            home / 'Datasets' / 'video',
            home / 'Datasets',
            home / 'datasets' / 'video',
            home / 'datasets',
        ]

        # Project-relative fallbacks
        here = Path(__file__).resolve().parent
        dirs_to_try += [
            here / 'Datasets' / 'video',
            here / 'Datasets',
            here.parent / 'Datasets' / 'video',
            here.parent / 'Datasets',
            here.parents[1] / 'Datasets' / 'video',
            here.parents[1] / 'Datasets'
        ]

    for d in dirs_to_try:
        if d.is_dir():
            data_dir = d
            break
    else:
        raise FileNotFoundError(
            "Could not locate dataset directory. Tried (in order): "
            + ", ".join(str(p) for p in dirs_to_try)
        )

    sf_path = data_dir / f"{dataset_name}_softmax_lst.pickle"
    tg_path = data_dir / f"{dataset_name}_target_lst.pickle"
    if not sf_path.exists() or not tg_path.exists():
        raise FileNotFoundError(f"Missing files in {data_dir}")

    # 3) load pickles
    with open(sf_path, "rb") as f:
        raw_softmax = CPU_Unpickler(f).load()
    with open(tg_path, "rb") as f:
        target_list = CPU_Unpickler(f).load()

    # 4) fix orientation
    softmax_list: List[np.ndarray] = []
    for idx, entry in enumerate(raw_softmax):
        arr = entry.cpu().numpy() if isinstance(entry, torch.Tensor) else np.asarray(entry)
        arr = np.squeeze(arr)
        L = len(target_list[idx])
        if arr.ndim == 1:
            # reshape 1D → (n_classes, L)
            c = arr.size // L
            if arr.size % L:
                raise ValueError(f"Cannot reshape array of size {arr.size} to match length {L}")
            arr = arr.reshape(c, L)
        else:
            c, e = arr.shape
            if e != L and c == L:
                arr = arr.T
                c, e = arr.shape
            if e != L:
                raise ValueError(f"Case {idx}: expected {L} columns but got {e}")

        softmax_list.append(arr)

    # 5) build df
    recs = []
    for idx, trace in enumerate(target_list):
        cid = str(idx)
        recs += [{"case:concept:name": cid, "concept:name": str(int(x))} for x in trace.tolist()]
    df = pd.DataFrame.from_records(recs)

    # 6) mapping
    df, mapping_dict = map_to_string_numbers(df)

    return (df, softmax_list, mapping_dict) if return_mapping else (df, softmax_list)
   
    
def map_to_string_numbers(
    df: pd.DataFrame,
    map_strings_to_integer_strings: bool = False
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Ensure `df['concept:name']` contains stringified integers.

    - If `map_strings_to_integer_strings=False`, values are cast to int then str.
    - If `map_strings_to_integer_strings=True`, every unique label is assigned
      a new integer string starting from 0.

    Returns
    -------
    df
        Modified DataFrame (copy-on-write semantics).
    mapping_dict
        The mapping used (newly generated or empty if direct casting).
    """
    df = df.copy()  # avoid in-place changes

    if map_strings_to_integer_strings:
        mapping_dict = {}
        next_id = 0

        def _map_new(label: str) -> str:
            nonlocal next_id
            if label not in mapping_dict:
                mapping_dict[label] = str(next_id)
                next_id += 1
            return mapping_dict[label]

        df["concept:name"] = df["concept:name"].astype(str).map(_map_new)
    else:
        # Cast to int then str
        df["concept:name"] = (
            df["concept:name"]
            .astype(int)
            .astype(str)
        )
        mapping_dict = {}

    return df, mapping_dict


def group_cases_by_trace(df: pd.DataFrame) -> pd.DataFrame:
    # Group by 'case:concept:name' and aggregate the 'concept:name' into a tuple
    grouped = df.groupby('case:concept:name')['concept:name'].apply(tuple).reset_index()
    
    # Group by the trace (sequence of activities) and aggregate the case IDs
    trace_groups = grouped.groupby('concept:name')['case:concept:name'].apply(list).reset_index()
    
    # Add a column for the length of each trace
    trace_groups['trace_length'] = trace_groups['concept:name'].apply(len)
    
    # Sort case_list numerically
    trace_groups['case:concept:name'] = trace_groups['case:concept:name'].apply(lambda x: sorted(x, key=int))
    
    # Create result DataFrame with explicit column names
    result = pd.DataFrame({
        'case_list': trace_groups['case:concept:name'],
        'trace_length': trace_groups['trace_length']
    })
    
    return result


def compute_activity_run_counts(
    dataset_name: str,
    path: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Compute the number of sequential runs for each activity in each trace.

    A "run" is a maximal contiguous block of the same activity within a trace.
    For example, the sequence [A, A, B, A, A, A] has:
      - A: 2 runs ("A, A" and "A, A, A")
      - B: 1 run

    Parameters
    ----------
    dataset_name : str
        One of: '50salads', 'gtea', 'breakfast'.
    path : str or Path, optional
        Base directory containing dataset pickles (passed to prepare_df).

    Returns
    -------
    Dict[str, Dict[str, int]]
        Nested dictionary mapping:
          activity_label -> { case_id -> number_of_runs }
        Only traces where an activity has at least one run are included in that
        activity's inner dictionary.
    """
    # Load the event DataFrame for the dataset
    df, _ = prepare_df(dataset_name, path)

    # Ensure expected columns exist
    required_cols = {"case:concept:name", "concept:name"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {missing}")

    # Result: activity -> { case_id -> run_count }
    activity_to_case_runs: Dict[str, Dict[str, int]] = {}

    # Iterate per trace preserving order
    for case_id, group in df.groupby("case:concept:name", sort=False):
        labels = group["concept:name"].tolist()
        if not labels:
            continue

        previous_label: Optional[str] = None
        # Temporary per-trace run counts by activity
        per_trace_runs: Dict[str, int] = {}

        for label in labels:
            # A new run starts whenever the label changes
            if label != previous_label:
                per_trace_runs[label] = per_trace_runs.get(label, 0) + 1
                previous_label = label

        # Merge per-trace counts into the global structure
        for activity_label, run_count in per_trace_runs.items():
            if activity_label not in activity_to_case_runs:
                activity_to_case_runs[activity_label] = {}
            activity_to_case_runs[activity_label][str(case_id)] = int(run_count)

    return activity_to_case_runs


def compute_activity_run_lengths(
    dataset_name: str,
    path: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, List[int]]]:
    """
    Compute the lengths of each sequential run for every activity in each trace.

    A "run" is a maximal contiguous block of the same activity within a trace.
    For example, the sequence [A, A, B, A, A, A] has A-run lengths [2, 3] and B-run lengths [1].

    Parameters
    ----------
    dataset_name : str
        One of: '50salads', 'gtea', 'breakfast'.
    path : str or Path, optional
        Base directory containing dataset pickles (passed to prepare_df).

    Returns
    -------
    Dict[str, Dict[str, List[int]]]
        Nested dictionary mapping:
          activity_label -> { case_id -> [run_length_1, run_length_2, ...] }
        Only traces where an activity appears are included in that activity's inner dictionary.
    """
    df, _ = prepare_df(dataset_name, path)

    required_cols = {"case:concept:name", "concept:name"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {missing}")

    activity_to_case_run_lengths: Dict[str, Dict[str, List[int]]] = {}

    for case_id, group in df.groupby("case:concept:name", sort=False):
        labels = group["concept:name"].tolist()
        if not labels:
            continue

        # Build per-trace run lengths by activity
        per_trace_lengths: Dict[str, List[int]] = {}

        current_label: Optional[str] = None
        current_run_length: int = 0

        for label in labels:
            if label == current_label:
                current_run_length += 1
            else:
                if current_label is not None:
                    per_trace_lengths.setdefault(current_label, []).append(current_run_length)
                current_label = label
                current_run_length = 1

        # Flush last run
        if current_label is not None:
            per_trace_lengths.setdefault(current_label, []).append(current_run_length)

        # Merge into global structure
        for activity_label, lengths in per_trace_lengths.items():
            activity_to_case_run_lengths.setdefault(activity_label, {})[str(case_id)] = [int(x) for x in lengths]

    return activity_to_case_run_lengths


def compute_unique_activity_run_lengths(
    dataset_name: str,
    path: Optional[Union[str, Path]] = None,
) -> Dict[str, List[int]]:
    """
    Return, for each activity, the list of unique contiguous run lengths across all traces.

    Parameters
    ----------
    dataset_name : str
        One of: '50salads', 'gtea', 'breakfast'.
    path : str or Path, optional
        Base directory containing dataset pickles (passed to prepare_df).

    Returns
    -------
    Dict[str, List[int]]
        Mapping: activity_label -> sorted list of unique run lengths.
    """
    activity_to_case_lengths = compute_activity_run_lengths(dataset_name, path)

    activity_to_unique_lengths: Dict[str, List[int]] = {}
    for activity_label, case_to_lengths in activity_to_case_lengths.items():
        unique_lengths = set()
        for lengths in case_to_lengths.values():
            unique_lengths.update(int(x) for x in lengths)
        activity_to_unique_lengths[activity_label] = sorted(unique_lengths)

    return activity_to_unique_lengths

def visualize_petri_net(net, marking=None, output_path="./model"):
    """
    Generates a visual representation of a Petri Net model using Graphviz.
    Transitions are displayed as rectangles, and places as circles.
    Tokens are represented as filled black circles within places if a marking is provided.

    Args:
        net: The self-defined Petri Net model defining the structure of the net (places, transitions, arcs).
        marking (tuple, optional): A tuple representing the current marking of the net.
            Each entry corresponds to a place in the order defined by the place_mapping.
            If None, the net will be visualized without tokens.
        output_path (str, optional): Path (without extension) to save the visualization. Defaults to "./model".
    """
    if not hasattr(net, 'place_mapping') or not hasattr(net, 'reverse_place_mapping'):
        raise AttributeError("The provided Petri net does not have the required place_mapping and reverse_place_mapping attributes")

    viz = Digraph(engine='dot')
    viz.attr(rankdir='TB')  # Set rank direction to top-to-bottom

    # Convert tuple marking to dictionary if provided
    marking_dict = {}
    if marking is not None:
        if len(marking) != len(net.place_mapping):
            raise ValueError(f"The length of the marking tuple ({len(marking)}) does not match the number of places in the net ({len(net.place_mapping)})")
        for idx, tokens in enumerate(marking):
            if tokens > 0:
                place = net.reverse_place_mapping[idx]
                marking_dict[place.name] = tokens  # Store by place name

    # Add Places (Circles)
    for place in net.places:
        label = place.name
        if marking is not None and place.name in marking_dict:  # Check by place name
            tokens = marking_dict[place.name]
            if tokens > 0:
                # Add a large token to the label with larger font size
                label += f"\n<FONT POINT-SIZE='30'>●</FONT>"
        
        # Ensure that the label is treated as an HTML-like label
        viz.node(str(place), label=f"<{label}>", shape='circle', style='filled', fillcolor='white', fixedsize='true', width='0.75', height='0.75')

    # Add Transitions (Rectangles)
    for transition in net.transitions:
        label = transition.label if transition.label else str(transition)
        viz.node(str(transition), label=label, shape='box')

    # Add Arcs
    for arc in net.arcs:
        viz.edge(str(arc.source), str(arc.target))

    # Explicitly set the rank of the source and sink nodes
    with viz.subgraph() as s:
        s.attr(rank='source')
        s.node(str(net.places[0]))  # Assuming the first place is the source

    with viz.subgraph() as s:
        s.attr(rank='sink')
        s.node(str(net.places[-1]))  # Assuming the last place is the sink

    # Redirect stderr to null to suppress warnings
    with open(os.devnull, 'w') as f:
        saved_paths = []
        
        # Try to save in PNG format
        try:
            png_path = viz.render(output_path, format='png', cleanup=True)
            saved_paths.append(png_path)
            print(f"PNG visualization saved to: {png_path}")
        except Exception as e:
            print(f"Failed to generate PNG visualization: {e}")
        
        # Try to save in PDF format
        try:
            pdf_path = viz.render(output_path, format='pdf', cleanup=True)
            saved_paths.append(pdf_path)
            print(f"PDF visualization saved to: {pdf_path}")
        except Exception as e:
            print(f"Failed to generate PDF visualization: {e}")
        
        # If neither format worked, raise an exception
        if not saved_paths:
            raise RuntimeError("Failed to generate visualization in both PNG and PDF formats")
