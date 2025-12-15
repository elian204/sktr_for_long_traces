"""
Incremental Softmax Recovery: Main high-level function.

This module provides the main entry point for incremental softmax matrix recovery
using beam search with Petri nets, following the pattern of the existing 
compare_stochastic_vs_argmax_random_indices function.
"""

import os
import io
import pickle
import logging
import random
from pathlib import Path

from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
)

import numpy as np
import pandas as pd
import torch
from graphviz import Digraph


MoveType = str

logger = logging.getLogger(__name__)
        
def validate_input_parameters(
    n_indices: int,
    round_precision: int,
    non_sync_penalty: float,
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
    Build f(p: float, move_type: MoveType) ג†’ cost.

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
    >>> # logits ג‰ˆ [[-0.357, -1.609], [-1.204, -0.223]]
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


def linear_prob_combiner(
    observed: float,
    conditioned: float,
    alpha: float
) -> float:
    """
    Linearly combine observed and conditioned probabilities.

    Returns (1 - alpha) * conditioned + alpha * observed, mirroring the
    blending convention used elsewhere in this module.
    """
    return (1.0 - alpha) * float(conditioned) + float(alpha) * float(observed)


def get_run_context_labels(
    predicted_sequence: Sequence[str]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Return the (current_run_label, last_different_label) from a predicted sequence.

    - current_run_label: the label of the final contiguous run (i.e., last element).
    - last_different_label: the most recent label in the sequence that differs from
      the current run label (i.e., label of the previous run). If none exists,
      returns None.
    """
    if not predicted_sequence:
        return None, None

    current_label = predicted_sequence[-1]

    # Walk backwards to find the last label that differs from the current run
    i = len(predicted_sequence) - 1
    while i >= 0 and predicted_sequence[i] == current_label:
        i -= 1

    last_different = predicted_sequence[i] if i >= 0 else None
    return current_label, last_different


def get_run_context_labels_extended(
    predicted_sequence: Sequence[str],
    n_prev_labels: int = 1
) -> Tuple[Optional[str], List[str]]:
    """
    Return the (current_run_label, list_of_previous_different_labels) from a predicted sequence.

    - current_run_label: the label of the final contiguous run (i.e., last element).
    - list_of_previous_different_labels: list of up to n_prev_labels most recent labels
      that differ from the current run label AND from each other, ordered from most
      recent to oldest. If fewer than n_prev_labels are available, returns what's available.

    Example:
        sequence = ['A', 'B', 'B', 'C', 'C', 'C', 'D', 'D']
        n_prev_labels = 3
        Returns: ('D', ['C', 'B', 'A'])

    Parameters
    ----------
    predicted_sequence : Sequence[str]
        Sequence of predicted labels.
    n_prev_labels : int, default 1
        Number of previous different labels to extract.

    Returns
    -------
    Tuple[Optional[str], List[str]]
        (current_run_label, list_of_previous_different_labels)
    """
    if not predicted_sequence:
        return None, []

    current_label = predicted_sequence[-1]
    previous_different_labels: List[str] = []

    # Walk backwards through the sequence, collecting labels different from current
    # and different from each other
    i = len(predicted_sequence) - 1
    last_collected: Optional[str] = current_label

    while i >= 0 and len(previous_different_labels) < n_prev_labels:
        label = predicted_sequence[i]
        if label != last_collected:
            previous_different_labels.append(label)
            last_collected = label
        i -= 1

    return current_label, previous_different_labels


def interpolate_conditional_probs(
    target_label: str,
    context_history: List[str],
    prob_dict: Dict[Tuple[str, ...], Dict[str, float]],
    interpolation_weights: Optional[List[float]] = None,
) -> float:
    """
    Compute interpolated conditional probability P(target_label | context_history)
    using n-gram probabilities from prob_dict.

    Uses linear interpolation across all available n-gram levels. If a specific
    n-gram is not in prob_dict, it contributes 0 to the weighted sum (automatic
    smoothing effect).

    Example:
        context_history = ['D', 'C', 'B']
        target_label = 'E'
        interpolation_weights = [0.5, 0.3, 0.2]

        Computes:
        P_interpolated = 0.5 * P(E|D) + 0.3 * P(E|D,C) + 0.2 * P(E|D,C,B)

        If P(E|D,C,B) is not in prob_dict, that term contributes 0.

    Parameters
    ----------
    target_label : str
        The label to predict.
    context_history : List[str]
        Context labels ordered from most recent to oldest (e.g., ['D', 'C', 'B']).
    prob_dict : Dict[Tuple[str, ...], Dict[str, float]]
        N-gram probability dictionary mapping history tuples to next-label distributions.
    interpolation_weights : Optional[List[float]]
        Weights for each n-gram level [λ₁, λ₂, λ₃, ...].
        λ₁ is for unigram, λ₂ for bigram, etc.
        If None, uses equal weights.
        Should sum to 1.0 for proper probability interpretation.

    Returns
    -------
    float
        Interpolated conditional probability. Returns 0.0 if no n-grams are available.
    """
    if not context_history:
        return 0.0

    n_levels = len(context_history)

    # Default to equal weights if not specified
    if interpolation_weights is None:
        interpolation_weights = [1.0 / n_levels] * n_levels

    # If we have more context levels than weights, pad weights with 0
    # If we have fewer context levels than weights, truncate weights
    if len(interpolation_weights) < n_levels:
        # Pad with zeros
        weights = list(interpolation_weights) + [0.0] * (n_levels - len(interpolation_weights))
    else:
        # Truncate to match context length
        weights = interpolation_weights[:n_levels]

    interpolated_prob = 0.0

    # Iterate through n-gram levels: unigram, bigram, trigram, etc.
    for i in range(n_levels):
        # Build the history tuple for this n-gram level
        # i=0: unigram (context_history[0],)
        # i=1: bigram (context_history[0], context_history[1])
        # i=2: trigram (context_history[0], context_history[1], context_history[2])
        history_tuple = tuple(context_history[:i+1])

        # Lookup the conditional probability
        next_map = prob_dict.get(history_tuple, {})
        cond_prob = next_map.get(target_label, 0.0)

        # Add weighted contribution
        interpolated_prob += weights[i] * cond_prob

    return interpolated_prob


def adjust_probs_with_conditioning_vector(
    observed_probs: np.ndarray,
    class_labels: Sequence[str],
    *,
    current_run_label: Optional[str],
    last_different_label: Optional[str],
    cond_prob_bigram: Dict[str, Dict[str, float]],
    alpha: float = 0.5,
    combine_fn: Optional[Callable[[float, float, float], float]] = None,
) -> np.ndarray:
    """
    Adjust a single timestep's probability vector using conditioned probabilities
    based on the run context, then normalize to sum to 1.

    Rules (using example current run label 'b' and last different label 'a'):
      - For any activity c != 'b': blend observed P(c) with conditioned P(c|'b').
      - For activity 'b' itself: blend observed P('b') with conditioned P('b'|'a').
        If there is no last different label (first run), leave as observed.

    Parameters
    ----------
    observed_probs : np.ndarray
        1D array of size C with observed probabilities (e.g., softmax at time t).
    class_labels : Sequence[str]
        Labels corresponding to indices of observed_probs.
    current_run_label : Optional[str]
        Label of the current run (last predicted label).
    last_different_label : Optional[str]
        Label of the previous run (most recent label that differs from current).
    cond_prob_bigram : Dict[str, Dict[str, float]]
        Bigram conditional probabilities mapping prev -> {next -> P(next|prev)}.
    alpha : float, default 0.5
        Blend weight: result = (1 - alpha)*conditioned + alpha*observed.
    combine_fn : Optional[Callable]
        Custom combiner (observed, conditioned, alpha) -> float. Defaults to linear.

    Returns
    -------
    np.ndarray
        Normalized adjusted probabilities (1D array of size C).
    """
    if combine_fn is None:
        combine_fn = linear_prob_combiner

    probs = np.asarray(observed_probs, dtype=float).copy()
    if probs.ndim != 1 or len(probs) != len(class_labels):
        raise ValueError("observed_probs must be 1D and match class_labels length")

    # If we don't have a current run label or no conditioning dict, return observed (normalized)
    if current_run_label is None or not cond_prob_bigram:
        s = probs.sum()
        return probs / s if s > 0 else probs

    adjusted = np.zeros_like(probs)

    # Pre-fetch conditional maps with graceful fallbacks
    cond_from_current = cond_prob_bigram.get(current_run_label, {})
    cond_from_previous = cond_prob_bigram.get(last_different_label, {}) if last_different_label is not None else {}

    for idx, label in enumerate(class_labels):
        observed_p = probs[idx]
        if label == current_run_label:
            # For the current run label 'b', use P(b|'a') if available; if not, keep observed
            if last_different_label is None:
                conditioned_p = observed_p
            else:
                conditioned_p = cond_from_previous.get(label, observed_p)
        else:
            # For any other label c, use P(c|'b')
            conditioned_p = cond_from_current.get(label, observed_p)

        adjusted[idx] = combine_fn(observed_p, conditioned_p, alpha)

    total = float(adjusted.sum())
    if total <= 0:
        # Degenerate case: fallback to observed
        s = probs.sum()
        return probs / s if s > 0 else probs

    return adjusted / total


def adjust_probs_with_conditioning_vector_extended(
    observed_probs: np.ndarray,
    class_labels: Sequence[str],
    *,
    current_run_label: Optional[str],
    previous_different_labels: List[str],
    prob_dict_uncollapsed: Dict[Tuple[str, ...], Dict[str, float]],
    prob_dict_collapsed: Optional[Dict[Tuple[str, ...], Dict[str, float]]] = None,
    alpha: float = 0.5,
    combine_fn: Optional[Callable[[float, float, float], float]] = None,
    interpolation_weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Adjust a single timestep's probability vector using interpolated conditioned
    probabilities based on multiple previous different labels, then normalize to sum to 1.

    This is the extended version that supports N previous different labels with
    interpolation smoothing and uses TWO probability dictionaries:
    - Uncollapsed dict for CONTINUATION probability (current run label)
    - Collapsed dict for TRANSITION probability (other labels)

    Rules (using example current='D', previous_different=['C','B','A']):
      - For activity 'D' (current run): blend observed P('D') with interpolated
        P('D'|'C','B','A') using UNCOLLAPSED dictionary (captures within-run transitions).
      - For any activity x != 'D': blend observed P(x) with interpolated P(x|'D','C','B')
        using COLLAPSED dictionary (captures run-to-run transitions).

    Interpolation example for P(x|'D','C','B') with weights [0.5, 0.3, 0.2]:
        P_interp = 0.5 * P(x|'D') + 0.3 * P(x|'D','C') + 0.2 * P(x|'D','C','B')

    Parameters
    ----------
    observed_probs : np.ndarray
        1D array of size C with observed probabilities (e.g., softmax at time t).
    class_labels : Sequence[str]
        Labels corresponding to indices of observed_probs.
    current_run_label : Optional[str]
        Label of the current run (last predicted label).
    previous_different_labels : List[str]
        List of previous different labels, ordered from most recent to oldest.
    prob_dict_uncollapsed : Dict[Tuple[str, ...], Dict[str, float]]
        N-gram probability dictionary built from UNCOLLAPSED traces (for continuation).
    prob_dict_collapsed : Optional[Dict[Tuple[str, ...], Dict[str, float]]]
        N-gram probability dictionary built from COLLAPSED traces (for transitions).
        If None, uses prob_dict_uncollapsed for both (backward compatibility).
    alpha : float, default 0.5
        Blend weight: result = (1 - alpha)*conditioned + alpha*observed.
    combine_fn : Optional[Callable]
        Custom combiner (observed, conditioned, alpha) -> float. Defaults to linear.
    interpolation_weights : Optional[List[float]]
        Weights for interpolation [λ₁, λ₂, λ₃, ...]. If None, uses equal weights.

    Returns
    -------
    np.ndarray
        Normalized adjusted probabilities (1D array of size C).
    """
    if combine_fn is None:
        combine_fn = linear_prob_combiner

    # Backward compatibility: if no collapsed dict provided, use uncollapsed for both
    if prob_dict_collapsed is None:
        prob_dict_collapsed = prob_dict_uncollapsed

    probs = np.asarray(observed_probs, dtype=float).copy()
    if probs.ndim != 1 or len(probs) != len(class_labels):
        raise ValueError("observed_probs must be 1D and match class_labels length")

    # If we don't have a current run label or no conditioning dict, return observed (normalized)
    if current_run_label is None or not prob_dict_uncollapsed:
        s = probs.sum()
        return probs / s if s > 0 else probs

    adjusted = np.zeros_like(probs)

    for idx, label in enumerate(class_labels):
        observed_p = probs[idx]

        if label == current_run_label:
            # CONTINUATION: For the current run label, use P(current|previous_different_labels)
            # Use UNCOLLAPSED dictionary to capture within-run transition patterns
            if not previous_different_labels:
                conditioned_p = observed_p
            else:
                # Build context from previous different labels
                context_history = previous_different_labels
                conditioned_p = interpolate_conditional_probs(
                    target_label=label,
                    context_history=context_history,
                    prob_dict=prob_dict_uncollapsed,  # UNCOLLAPSED for continuation
                    interpolation_weights=interpolation_weights,
                )
                # If interpolation returns 0 (no n-grams available), use observed
                if conditioned_p == 0.0:
                    conditioned_p = observed_p
        else:
            # TRANSITION: For any other label, use P(label|current, previous_different_labels)
            # Use COLLAPSED dictionary to capture run-to-run transition patterns
            # Build context: current + previous different labels
            context_history = [current_run_label] + previous_different_labels
            conditioned_p = interpolate_conditional_probs(
                target_label=label,
                context_history=context_history,
                prob_dict=prob_dict_collapsed,  # COLLAPSED for transitions
                interpolation_weights=interpolation_weights,
            )
            # If interpolation returns 0 (no n-grams available), use observed
            if conditioned_p == 0.0:
                conditioned_p = observed_p

        adjusted[idx] = combine_fn(observed_p, conditioned_p, alpha)

    total = float(adjusted.sum())
    if total <= 0:
        # Degenerate case: fallback to observed
        s = probs.sum()
        return probs / s if s > 0 else probs

    return adjusted / total


def adjust_probs_with_sequence_context(
    observed_probs: np.ndarray,
    class_labels: Sequence[str],
    predicted_sequence: Sequence[str],
    *,
    cond_prob_bigram: Optional[Dict[str, Dict[str, float]]] = None,
    prob_dict_uncollapsed: Optional[Dict[Tuple[str, ...], Dict[str, float]]] = None,
    prob_dict_collapsed: Optional[Dict[Tuple[str, ...], Dict[str, float]]] = None,
    alpha: float = 0.5,
    combine_fn: Optional[Callable[[float, float, float], float]] = None,
    n_prev_labels: int = 1,
    interpolation_weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Convenience wrapper: derives run context from the predicted sequence and
    applies probability adjustment with conditioning.

    Supports both legacy (single previous label) and extended (multiple previous labels)
    conditioning modes:
    - If n_prev_labels=1 and cond_prob_bigram is provided: uses legacy mode
    - If n_prev_labels>1 or prob_dict_uncollapsed is provided: uses extended mode with interpolation

    Parameters
    ----------
    observed_probs : np.ndarray
        1D array of observed probabilities at current timestep.
    class_labels : Sequence[str]
        Labels corresponding to indices of observed_probs.
    predicted_sequence : Sequence[str]
        Sequence of predicted labels so far.
    cond_prob_bigram : Optional[Dict[str, Dict[str, float]]]
        Legacy bigram map (for backward compatibility). Used when n_prev_labels=1.
    prob_dict_uncollapsed : Optional[Dict[Tuple[str, ...], Dict[str, float]]]
        N-gram probability dictionary built from UNCOLLAPSED traces (for continuation).
    prob_dict_collapsed : Optional[Dict[Tuple[str, ...], Dict[str, float]]]
        N-gram probability dictionary built from COLLAPSED traces (for transitions).
    alpha : float, default 0.5
        Blend weight: result = (1 - alpha)*conditioned + alpha*observed.
    combine_fn : Optional[Callable]
        Custom combiner function. Defaults to linear_prob_combiner.
    n_prev_labels : int, default 1
        Number of previous different labels to use for conditioning.
        If 1, uses legacy single-label mode. If >1, uses extended interpolation mode.
    interpolation_weights : Optional[List[float]]
        Weights for interpolation [λ₁, λ₂, λ₃, ...]. If None, uses equal weights.
        Only used in extended mode.

    Returns
    -------
    np.ndarray
        Normalized adjusted probabilities.
    """
    # Legacy mode: n_prev_labels=1 with cond_prob_bigram
    if n_prev_labels == 1 and cond_prob_bigram is not None and prob_dict_uncollapsed is None:
        current_label, last_diff = get_run_context_labels(predicted_sequence)
        return adjust_probs_with_conditioning_vector(
            observed_probs,
            class_labels,
            current_run_label=current_label,
            last_different_label=last_diff,
            cond_prob_bigram=cond_prob_bigram,
            alpha=alpha,
            combine_fn=combine_fn,
        )

    # Extended mode: n_prev_labels>=1 with prob_dict_uncollapsed
    if prob_dict_uncollapsed is not None:
        current_label, previous_labels = get_run_context_labels_extended(
            predicted_sequence, n_prev_labels=n_prev_labels
        )
        return adjust_probs_with_conditioning_vector_extended(
            observed_probs,
            class_labels,
            current_run_label=current_label,
            previous_different_labels=previous_labels,
            prob_dict_uncollapsed=prob_dict_uncollapsed,
            prob_dict_collapsed=prob_dict_collapsed,
            alpha=alpha,
            combine_fn=combine_fn,
            interpolation_weights=interpolation_weights,
        )

    # Fallback: no conditioning, return normalized observed
    s = observed_probs.sum()
    return observed_probs / s if s > 0 else observed_probs


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
            # reshape 1D ג†’ (n_classes, L)
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

    # 5) build df (vectorized)
    lengths = [len(trace) for trace in target_list]
    total_events = sum(lengths)
    if total_events == 0:
        # Preserve original behavior on empty input by creating an empty DataFrame with required columns
        df = pd.DataFrame({
            'case:concept:name': pd.Series([], dtype=str),
            'concept:name': pd.Series([], dtype='int64'),
        })
    else:
        case_ids = np.repeat(np.arange(len(target_list)).astype(str), lengths)
        concept_values = np.concatenate([
            np.asarray(trace, dtype=int)
            for trace in target_list
            if len(trace) > 0
        ])
        df = pd.DataFrame({
            "case:concept:name": case_ids,
            "concept:name": concept_values,
        })

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


def get_ground_truth_sequences(df: pd.DataFrame, case_ids: List[str]) -> List[str]:
    """
    Get ground truth activity sequences for specified case IDs as a single sequential list.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'case:concept:name' and 'concept:name' columns,
        typically from prepare_df().
    case_ids : List[str]
        List of case IDs (as strings) to get sequences for.

    Returns
    -------
    List[str]
        Single sequential list containing all ground truth activities from the
        specified cases in order. Activities from each case are appended in sequence.

    Example
    -------
    >>> result = prepare_df('50salads')
    >>> df, softmax_lst = result
    >>> sequence = get_ground_truth_sequences(df, ['20', '12'])
    >>> # sequence contains activities for case '20' followed by activities for case '12'
    """
    all_activities = []
    for case_id in case_ids:
        # Filter to this case and get the concept:name sequence
        case_df = df[df['case:concept:name'] == case_id]
        sequence = case_df['concept:name'].tolist()
        all_activities.extend(sequence)
    return all_activities


def get_ground_truth_sequences_by_case(df: pd.DataFrame, case_ids: List[str]) -> List[List[str]]:
    """
    Get ground truth activity sequences for specified case IDs as a list of lists.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'case:concept:name' and 'concept:name' columns,
        typically from prepare_df().
    case_ids : List[str]
        List of case IDs (as strings) to get sequences for.

    Returns
    -------
    List[List[str]]
        List of lists, where each inner list contains the ground truth activities
        for the corresponding case in case_ids (in the same order).

    Example
    -------
    >>> result = prepare_df('50salads')
    >>> df, softmax_lst = result
    >>> sequences = get_ground_truth_sequences_by_case(df, ['20', '12'])
    >>> # sequences[0] contains activities for case '20'
    >>> # sequences[1] contains activities for case '12'
    """
    all_sequences = []
    for case_id in case_ids:
        # Filter to this case and get the concept:name sequence
        case_df = df[df['case:concept:name'] == case_id]
        sequence = case_df['concept:name'].tolist()
        all_sequences.append(sequence)
    return all_sequences


def get_sequences_by_case(df: pd.DataFrame, case_ids: List[str], label_col: str = 'concept:name') -> List[List[str]]:
    """
    Get sequences for specified case IDs from any label column as a list of lists.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'case:concept:name' and the specified label column.
    case_ids : List[str]
        List of case IDs (as strings) to get sequences for.
    label_col : str, default 'concept:name'
        Column name containing the labels/activities to extract.

    Returns
    -------
    List[List[str]]
        List of lists, where each inner list contains the labels
        for the corresponding case in case_ids (in the same order).

    Example
    -------
    >>> result = prepare_df('50salads')
    >>> df, softmax_lst = result
    >>> gt_sequences = get_sequences_by_case(df, ['20', '12'], 'concept:name')
    >>> pred_sequences = get_sequences_by_case(df, ['20', '12'], 'predictions')
    """
    all_sequences = []
    for case_id in case_ids:
        # Filter to this case and get the label sequence
        case_df = df[df['case:concept:name'] == case_id]
        sequence = case_df[label_col].tolist()
        all_sequences.append(sequence)
    return all_sequences


def normalize_sequences_for_evaluation(gt_sequences: List[List[str]], pred_sequences: List[np.ndarray]) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Normalize ground truth and prediction sequences to the same type for evaluation.

    This function ensures both sequences are lists of lists of strings, which is
    the expected format for evaluation functions like tas_metrics.

    Parameters
    ----------
    gt_sequences : List[List[str]]
        Ground truth sequences as lists of strings.
    pred_sequences : List[np.ndarray]
        Prediction sequences as numpy arrays (typically of int64).

    Returns
    -------
    Tuple[List[List[str]], List[List[str]]]
        Normalized (gt_sequences, pred_sequences) where both are lists of lists of strings.

    Raises
    ------
    ValueError
        If the sequences have different lengths or if individual sequences don't match in length.

    Example
    -------
    >>> gt_seqs = [['17', '11', '13'], ['17', '7', '8']]
    >>> pred_seqs = [np.array([17, 11, 13]), np.array([17, 7, 8])]
    >>> gt_norm, pred_norm = normalize_sequences_for_evaluation(gt_seqs, pred_seqs)
    >>> # Both are now lists of lists of strings
    """
    if len(gt_sequences) != len(pred_sequences):
        raise ValueError(f"Number of sequences don't match: GT has {len(gt_sequences)}, Pred has {len(pred_sequences)}")

    normalized_pred_sequences = []

    for i, (gt_seq, pred_seq) in enumerate(zip(gt_sequences, pred_sequences)):
        # Convert numpy array to list of strings
        pred_seq_list = pred_seq.tolist() if hasattr(pred_seq, 'tolist') else list(pred_seq)
        pred_seq_str = [str(x) for x in pred_seq_list]

        # Check lengths match
        if len(gt_seq) != len(pred_seq_str):
            raise ValueError(f"Sequence {i} length mismatch: GT has {len(gt_seq)}, Pred has {len(pred_seq_str)}")

        normalized_pred_sequences.append(pred_seq_str)

    return gt_sequences, normalized_pred_sequences


def get_variant_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze trace variants. Returns DataFrame sorted by frequency (variant_id 0 = most frequent).
    
    Parameters
    ----------
    df : pd.DataFrame
        Event log DataFrame with 'case:concept:name' and 'concept:name' columns.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - variant_id: Integer ID (0 = most frequent variant)
        - trace_signature: Tuple representing the activity sequence
        - case_ids: List of case IDs belonging to this variant
        - frequency: Number of cases with this variant
        - trace_length: Length of the trace sequence
    """
    trace_variants = {}
    for case_id in df['case:concept:name'].unique():
        trace = tuple(df[df['case:concept:name'] == case_id]['concept:name'].tolist())
        if trace not in trace_variants:
            trace_variants[trace] = {'case_ids': [], 'length': len(trace)}
        trace_variants[trace]['case_ids'].append(case_id)
    
    data = [{'variant_id': i, 'trace_signature': sig, 'case_ids': info['case_ids'],
             'frequency': len(info['case_ids']), 'trace_length': info['length']}
            for i, (sig, info) in enumerate(trace_variants.items())]
    
    variant_df = pd.DataFrame(data).sort_values('frequency', ascending=False).reset_index(drop=True)
    variant_df['variant_id'] = range(len(variant_df))
    return variant_df


def select_variants(variant_df: pd.DataFrame, n: int, method: str = 'frequency', seed: int = 42) -> List[int]:
    """
    Select n variants by frequency or randomly.
    
    Parameters
    ----------
    variant_df : pd.DataFrame
        DataFrame from get_variant_info().
    n : int
        Number of variants to select.
    method : str, default 'frequency'
        Selection method: 'frequency' (top N by frequency) or 'random'.
    seed : int, default 42
        Random seed for reproducible selection.
    
    Returns
    -------
    List[int]
        List of variant IDs.
    """
    n = min(n, len(variant_df))
    if method == 'frequency':
        return variant_df['variant_id'].head(n).tolist()
    else:  # random
        return random.Random(seed).sample(variant_df['variant_id'].tolist(), n)


def get_cases_for_variants(variant_df: pd.DataFrame, variant_ids: List[int], seed: int = 42) -> List[str]:
    """
    Get all case IDs for the specified variants.
    
    Parameters
    ----------
    variant_df : pd.DataFrame
        DataFrame from get_variant_info().
    variant_ids : List[int]
        List of variant IDs to get cases for.
    seed : int, default 42
        Random seed (currently unused, kept for API consistency).
    
    Returns
    -------
    List[str]
        List of all case IDs belonging to the specified variants.
    """
    cases = []
    for vid in variant_ids:
        row = variant_df[variant_df['variant_id'] == vid]
        if not row.empty:
            cases.extend(row.iloc[0]['case_ids'])
    return cases


def get_variants_for_cases(variant_df: pd.DataFrame, case_ids: List[Union[str, int]]) -> List[int]:
    """
    Get unique variant IDs for the specified case IDs.
    
    Parameters
    ----------
    variant_df : pd.DataFrame
        DataFrame from get_variant_info().
    case_ids : List[Union[str, int]]
        List of case IDs to find variants for.
    
    Returns
    -------
    List[int]
        Sorted list of unique variant IDs that contain the specified cases.
    """
    # Convert case_ids to strings for comparison (case IDs in variant_df are stored as strings)
    case_ids_str = [str(cid) for cid in case_ids]
    variant_ids = set()
    
    for _, row in variant_df.iterrows():
        # Check if any of the requested case IDs are in this variant's case_ids
        variant_case_ids = [str(cid) for cid in row['case_ids']]
        if any(cid in variant_case_ids for cid in case_ids_str):
            variant_ids.add(row['variant_id'])
    
    return sorted(list(variant_ids))


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
                label += f"\n<FONT POINT-SIZE='30'>ג—</FONT>"
        
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
            logger.info(f"PNG visualization saved to: {png_path}")
        except Exception as e:
            logger.warning(f"Failed to generate PNG visualization: {e}")
        
        # Try to save in PDF format
        try:
            pdf_path = viz.render(output_path, format='pdf', cleanup=True)
            saved_paths.append(pdf_path)
            logger.info(f"PDF visualization saved to: {pdf_path}")
        except Exception as e:
            logger.warning(f"Failed to generate PDF visualization: {e}")
        
        # If neither format worked, raise an exception
        if not saved_paths:
            raise RuntimeError("Failed to generate visualization in both PNG and PDF formats")


def sample_sequence_preserving_runs(sequence: Union[str, List[str]], sampling_ratio: float, min_run_length: int = 1) -> Union[str, List[str]]:
    """
    Sample a sequence while preserving the frequency and structure of activity runs.

    This function identifies contiguous runs of the same activity and samples from each run
    proportionally to the desired sampling ratio, ensuring each run is represented by at least
    min_run_length activities. The goal is to maintain the approximate frequency distribution
    of activities while reducing sequence length.

    Parameters
    ----------
    sequence : str or List[str]
        The input sequence to sample from. Can be a string or list of activity labels.
    sampling_ratio : float
        The fraction of the original sequence to retain (between 0 and 1).
        For example, 0.2 means approximately 20% of activities will be kept.
    min_run_length : int, default=1
        Minimum number of activities to preserve from each run. Each run will have
        at least this many activities in the sampled sequence.

    Returns
    -------
    str or List[str]
        The sampled sequence with the same type as input (string or list).

    Examples
    --------
    >>> sample_sequence_preserving_runs("aaaaaaaaaabbbbbaaaaacccc", 0.2)
    'aabac'

    >>> sample_sequence_preserving_runs(['a', 'a', 'a', 'b', 'b', 'c'], 0.5)
    ['a', 'a', 'b', 'c']

    Notes
    -----
    - Each run in the original sequence is guaranteed to have at least min_run_length
      activities in the sampled sequence.
    - The sampling ratio is approximate and may vary slightly due to the minimum run
      length constraint.
    - Run structure is preserved: activities from the same run remain contiguous in
      the sampled sequence.
    """
    if not sequence:
        return sequence

    # Convert string to list for processing if needed
    is_string = isinstance(sequence, str)
    if is_string:
        seq_list = list(sequence)
    else:
        seq_list = sequence.copy()

    if sampling_ratio <= 0:
        return [] if not is_string else ""
    if sampling_ratio >= 1:
        return sequence

    # Identify runs in the sequence
    runs = []
    current_run = [seq_list[0]]

    for activity in seq_list[1:]:
        if activity == current_run[-1]:
            current_run.append(activity)
        else:
            runs.append(current_run)
            current_run = [activity]
    runs.append(current_run)  # Don't forget the last run

    # Allocate samples per run using proportional ceil while
    # enforcing a minimum per-run length and not exceeding run size.
    sampled_runs = []
    for run in runs:
        run_length = len(run)
        # Proportional target from this run (ceil to avoid under-sampling long runs)
        proportional = int(np.ceil(run_length * sampling_ratio))
        samples_from_run = min(run_length, max(min_run_length, proportional))
        sampled_runs.append(run[:samples_from_run])

    # Flatten the sampled runs back into a sequence
    sampled_sequence = []
    for run in sampled_runs:
        sampled_sequence.extend(run)

    # Convert back to original type
    if is_string:
        return ''.join(sampled_sequence)
    else:
        return sampled_sequence


def get_activity_run_lengths_by_case(
    df: pd.DataFrame,
    activity_label: str,
    min_runs: int = 0,
    include_preceding_sequence: bool = False,
    exclude_empty_cases: bool = True
) -> Union[Dict[str, List[int]], Dict[str, Tuple[List[int], List[Tuple[str, ...]]]]]:
    """
    Compute the lengths of each sequential run of a specific activity label in each case.

    A "run" is a maximal contiguous block of the same activity within a trace.
    For example, if a trace has activities [A, B, B, B, A, C, C] and we look for runs of B,
    the runs would be [3] (one run of length 3).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'case:concept:name' and 'concept:name' columns.
    activity_label : str
        The activity label to find runs for.
    min_runs : int, default=0
        Minimum number of runs required for a case to be included in the result.
        Cases with fewer runs than this threshold will be excluded from the dictionary.
    include_preceding_sequence : bool, default=False
        If True, also track the sequence of activities that preceded each run.
        For each run, includes a string showing one activity from each run that
        appeared before the current run in the trace (excluding the current run).
    exclude_empty_cases : bool, default=True
        If True, cases with zero runs of the target activity are not returned.
        If False, such cases are included with an empty list.

    Returns
    -------
    Dict[str, List[int]] or Dict[str, Tuple[List[int], List[Tuple[str, ...]]]]
        - If include_preceding_sequence=False:
          case_id -> [run_length, ...]
        - If include_preceding_sequence=True:
          case_id -> ([run_length, ...], [preceding_sequence, ...]) where each preceding_sequence
          is a tuple of one activity label per prior run (excluding the current run). The preceding
          sequence updates only after the activity switches.
        Only cases with at least min_runs runs of the activity are included (and cases with zero
        runs are excluded by default unless exclude_empty_cases=False).

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'case:concept:name': ['case1', 'case1', 'case1', 'case2', 'case2', 'case3', 'case3'],
    ...     'concept:name': ['A', 'B', 'B', 'A', 'B', 'A', 'A']
    ... })
    >>> get_activity_run_lengths_by_case(df, 'B')
    {'case1': [2], 'case2': [1]}
    >>> get_activity_run_lengths_by_case(df, 'B', min_runs=2)
    {'case1': [2]}
    >>> get_activity_run_lengths_by_case(df, 'B', include_preceding_sequence=True)
    {'case1': ([2], [('A',)]), 'case2': ([1], [('A',)])}
    >>> # For sequence 'aaaabbbccccaaa', tracking 'a' gives:
    >>> # {'case1': ([4, 3], [(), ('a', 'b', 'c')])}
    """
    # Ensure required columns exist
    required_cols = {"case:concept:name", "concept:name"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {missing}")

    if min_runs < 0:
        raise ValueError(f"min_runs must be non-negative, got {min_runs}")

    if include_preceding_sequence:
        result: Dict[str, Tuple[List[int], List[Tuple[str, ...]]]] = {}
    else:
        result: Dict[str, List[int]] = {}

    # Group by case and process each trace
    for case_id, group in df.groupby("case:concept:name", sort=False):
        labels = group["concept:name"].tolist()
        if not labels:
            continue

        if include_preceding_sequence:
            # Find all runs in the sequence and their preceding context
            run_lengths: List[int] = []
            preceding_list: List[Tuple[str, ...]] = []
            current_run_length = 0
            prior_runs: List[str] = []  # One activity from each prior run

            i = 0
            while i < len(labels):
                if labels[i] == activity_label:
                    # Start of a target activity run
                    while i < len(labels) and labels[i] == activity_label:
                        current_run_length += 1
                        i += 1

                    # Build sequence of runs seen so far (excluding current)
                    preceding_seq = tuple(prior_runs)
                    run_lengths.append(current_run_length)
                    preceding_list.append(preceding_seq)
                    # Add current run to prior_runs for future runs
                    prior_runs.append(activity_label)
                    current_run_length = 0
                else:
                    # Non-target activity - find the run it belongs to
                    run_activity = labels[i]
                    while i < len(labels) and labels[i] == run_activity:
                        i += 1
                    # Add this activity to prior runs (one from each run)
                    prior_runs.append(run_activity)

            # Only include cases that meet the minimum runs threshold
            if len(run_lengths) >= min_runs and (len(run_lengths) > 0 or not exclude_empty_cases):
                result[str(case_id)] = (run_lengths, preceding_list)
        else:
            # Original logic for just run lengths
            run_lengths: List[int] = []
            current_run_length = 0

            for label in labels:
                if label == activity_label:
                    current_run_length += 1
                else:
                    # End of a run
                    if current_run_length > 0:
                        run_lengths.append(current_run_length)
                        current_run_length = 0

            # Don't forget the last run if it ends with the activity
            if current_run_length > 0:
                run_lengths.append(current_run_length)

            # Only include cases that meet the minimum runs threshold
            if len(run_lengths) >= min_runs and (len(run_lengths) > 0 or not exclude_empty_cases):
                result[str(case_id)] = run_lengths

    return result


def compute_argmax_accuracy_from_softmax(
    df: pd.DataFrame,
    softmax_lst: List[np.ndarray],
    case_ids: List[str]
) -> float:
    """
    Compute argmax accuracy for specified cases from softmax matrices.

    For each case, computes the accuracy as the proportion of timesteps where
    the argmax (highest probability) prediction matches the ground truth.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'case:concept:name' and 'concept:name' columns.
    softmax_lst : List[np.ndarray]
        List of softmax probability matrices, where each matrix has shape
        (n_classes, n_timesteps) and corresponds to a case.
    case_ids : List[str]
        List of case IDs to compute accuracy for.

    Returns
    -------
    float
        Mean argmax accuracy across all specified cases.
    """
    total_correct = 0
    total_predictions = 0

    for case_id in case_ids:
        # Get ground truth sequence for this case
        case_df = df[df['case:concept:name'] == case_id]
        gt_sequence = case_df['concept:name'].tolist()

        if not gt_sequence:
            continue

        # Get softmax matrix for this case (case_id is the index)
        case_idx = int(case_id)
        if case_idx >= len(softmax_lst):
            continue

        softmax_matrix = softmax_lst[case_idx]

        # Compute argmax predictions
        argmax_predictions = np.argmax(softmax_matrix, axis=0)

        # Convert ground truth to indices (assuming they are string labels)
        # For now, we'll assume ground truth is already in the correct format
        # This might need adjustment based on your specific label encoding
        gt_indices = [int(label) for label in gt_sequence]

        # Count correct predictions
        correct = sum(1 for pred, gt in zip(argmax_predictions, gt_indices) if pred == gt)
        total_correct += correct
        total_predictions += len(gt_sequence)

    if total_predictions == 0:
        return 0.0

    return total_correct / total_predictions


def compute_group_statistics(
    df: pd.DataFrame,
    softmax_lst: List[np.ndarray],
    group1_cases: List[str],
    group2_cases: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compute argmax accuracy statistics for two groups of cases.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'case:concept:name' and 'concept:name' columns.
    softmax_lst : List[np.ndarray]
        List of softmax probability matrices.
    group1_cases : List[str]
        List of case IDs for the first group.
    group2_cases : List[str]
        List of case IDs for the second group.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary with group statistics:
        {
            'group1': {
                'mean_accuracy': float,
                'individual_accuracies': List[float],
                'num_cases': int
            },
            'group2': {
                'mean_accuracy': float,
                'individual_accuracies': List[float],
                'num_cases': int
            }
        }
    """
    results = {}

    for group_name, cases in [('group1', group1_cases), ('group2', group2_cases)]:
        accuracies = []

        for case_id in cases:
            case_df = df[df['case:concept:name'] == case_id]
            gt_sequence = case_df['concept:name'].tolist()

            if not gt_sequence:
                continue

            case_idx = int(case_id)
            if case_idx >= len(softmax_lst):
                continue

            softmax_matrix = softmax_lst[case_idx]
            argmax_predictions = np.argmax(softmax_matrix, axis=0)

            # Convert ground truth to indices
            gt_indices = [int(label) for label in gt_sequence]

            # Compute accuracy for this case
            if len(argmax_predictions) == len(gt_indices):
                correct = sum(1 for pred, gt in zip(argmax_predictions, gt_indices) if pred == gt)
                accuracy = correct / len(gt_sequence)
                accuracies.append(accuracy)

        if accuracies:
            mean_accuracy = np.mean(accuracies)
            results[group_name] = {
                'mean_accuracy': mean_accuracy,
                'individual_accuracies': accuracies,
                'num_cases': len(accuracies)
            }
        else:
            results[group_name] = {
                'mean_accuracy': 0.0,
                'individual_accuracies': [],
                'num_cases': 0
            }

    return results


def compute_entropy(probs: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute entropy of probability distributions.

    Parameters
    ----------
    probs : np.ndarray
        Probability matrix of shape (n_classes, n_timesteps)
    axis : int, default=0
        Axis along which to compute entropy (0 for per timestep, 1 for per class)

    Returns
    -------
    np.ndarray
        Entropy values
    """
    # Clip probabilities to avoid log(0)
    probs = np.clip(probs, 1e-10, 1.0)
    # Compute entropy: -sum(p * log(p))
    entropy = -np.sum(probs * np.log(probs), axis=axis)
    return entropy


def compute_comprehensive_group_statistics(
    df: pd.DataFrame,
    softmax_lst: List[np.ndarray],
    group1_cases: List[str],
    group2_cases: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Compute comprehensive statistics for two groups of cases including:
    - Argmax accuracy
    - Sequence length statistics
    - Prediction confidence statistics
    - Entropy statistics (prediction uncertainty)
    - Class distribution statistics

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'case:concept:name' and 'concept:name' columns.
    softmax_lst : List[np.ndarray]
        List of softmax probability matrices.
    group1_cases : List[str]
        List of case IDs for the first group.
    group2_cases : List[str]
        List of case IDs for the second group.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary with comprehensive statistics for each group.
    """
    results = {}

    for group_name, cases in [('group1', group1_cases), ('group2', group2_cases)]:
        accuracies = []
        sequence_lengths = []
        max_probs = []
        entropies = []
        gt_class_counts = {}
        pred_class_counts = {}

        for case_id in cases:
            case_df = df[df['case:concept:name'] == case_id]
            gt_sequence = case_df['concept:name'].tolist()

            if not gt_sequence:
                continue

            case_idx = int(case_id)
            if case_idx >= len(softmax_lst):
                continue

            softmax_matrix = softmax_lst[case_idx]
            argmax_predictions = np.argmax(softmax_matrix, axis=0)
            max_prob_values = np.max(softmax_matrix, axis=0)

            # Compute entropy per timestep
            timestep_entropies = compute_entropy(softmax_matrix, axis=0)
            entropies.extend(timestep_entropies)

            # Convert ground truth to indices
            gt_indices = [int(label) for label in gt_sequence]

            # Only process if lengths match
            if len(argmax_predictions) == len(gt_indices):
                # Accuracy
                correct = sum(1 for pred, gt in zip(argmax_predictions, gt_indices) if pred == gt)
                accuracy = correct / len(gt_sequence)
                accuracies.append(accuracy)

                # Sequence length
                sequence_lengths.append(len(gt_sequence))

                # Prediction confidence (max probabilities)
                max_probs.extend(max_prob_values)

                # Class distributions
                for gt_class in gt_indices:
                    gt_class_counts[gt_class] = gt_class_counts.get(gt_class, 0) + 1

                for pred_class in argmax_predictions:
                    pred_class_counts[pred_class] = pred_class_counts.get(pred_class, 0) + 1

        if accuracies:
            results[group_name] = {
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'individual_accuracies': accuracies,
                'num_cases': len(accuracies),
                'mean_sequence_length': np.mean(sequence_lengths),
                'std_sequence_length': np.std(sequence_lengths),
                'sequence_lengths': sequence_lengths,
                'mean_max_prob': np.mean(max_probs),
                'std_max_prob': np.std(max_probs),
                'max_probs': max_probs,
                'mean_entropy': np.mean(entropies),
                'std_entropy': np.std(entropies),
                'entropies': entropies,
                'gt_class_distribution': gt_class_counts,
                'pred_class_distribution': pred_class_counts
            }
        else:
            results[group_name] = {
                'mean_accuracy': 0.0,
                'std_accuracy': 0.0,
                'individual_accuracies': [],
                'num_cases': 0,
                'mean_sequence_length': 0.0,
                'std_sequence_length': 0.0,
                'sequence_lengths': [],
                'mean_max_prob': 0.0,
                'std_max_prob': 0.0,
                'max_probs': [],
                'mean_entropy': 0.0,
                'std_entropy': 0.0,
                'entropies': [],
                'gt_class_distribution': {},
                'pred_class_distribution': {}
            }

    return results


def compute_accuracies_by_case(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SKTR and Argmax accuracy for each case in the results DataFrame.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing the results for recovered traces (must include
        'case:concept:name', 'sktr_activity', 'argmax_activity', and 'ground_truth' columns).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['case:concept:name', 'sktr_accuracy', 'argmax_accuracy'].
        The last row contains the mean accuracy for both columns (case:concept:name = 'MEAN').
    """
    def _case_acc(group):
        return pd.Series({
            'sktr_accuracy': (group['sktr_activity'] == group['ground_truth']).mean(),
            'argmax_accuracy': (group['argmax_activity'] == group['ground_truth']).mean()
        })
    acc_df = results_df.groupby('case:concept:name').apply(_case_acc, include_groups=False).reset_index()
    # Compute mean accuracies
    mean_sktr = acc_df['sktr_accuracy'].mean()
    mean_argmax = acc_df['argmax_accuracy'].mean()
    # Append mean row
    mean_row = pd.DataFrame({
        'case:concept:name': ['MEAN'],
        'sktr_accuracy': [mean_sktr],
        'argmax_accuracy': [mean_argmax]
    })
    acc_df = pd.concat([acc_df, mean_row], ignore_index=True)
    return acc_df


def compute_group_accuracies_from_results(results_df: pd.DataFrame, group1_cases: List[str], group2_cases: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compute SKTR and argmax accuracies for two groups of cases from results DataFrame.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with 'case:concept:name', 'sktr_activity', 'argmax_activity', 'ground_truth' columns.
    group1_cases : List[str]
        List of case IDs for the first group.
    group2_cases : List[str]
        List of case IDs for the second group.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary with group statistics:
        {
            'group1': {
                'sktr_accuracy': float,
                'argmax_accuracy': float,
                'sktr_individual': List[float],
                'argmax_individual': List[float],
                'num_cases': int
            },
            'group2': {
                'sktr_accuracy': float,
                'argmax_accuracy': float,
                'sktr_individual': List[float],
                'argmax_individual': List[float],
                'num_cases': int
            }
        }
    """
    def _compute_case_accuracies(group):
        sktr_correct = (group['sktr_activity'] == group['ground_truth']).sum()
        argmax_correct = (group['argmax_activity'] == group['ground_truth']).sum()
        total = len(group)
        return pd.Series({
            'sktr_accuracy': sktr_correct / total if total > 0 else 0,
            'argmax_accuracy': argmax_correct / total if total > 0 else 0
        })

    results = {}

    for group_name, cases in [('group1', group1_cases), ('group2', group2_cases)]:
        # Filter to cases in this group
        group_df = results_df[results_df['case:concept:name'].isin(cases)]

        if len(group_df) > 0:
            # Compute per-case accuracies
            case_accuracies = group_df.groupby('case:concept:name').apply(_compute_case_accuracies, include_groups=False).reset_index()

            # Overall group statistics
            sktr_individual = case_accuracies['sktr_accuracy'].tolist()
            argmax_individual = case_accuracies['argmax_accuracy'].tolist()

            results[group_name] = {
                'sktr_accuracy': np.mean(sktr_individual),
                'argmax_accuracy': np.mean(argmax_individual),
                'sktr_individual': sktr_individual,
                'argmax_individual': argmax_individual,
                'num_cases': len(sktr_individual)
            }
        else:
            results[group_name] = {
                'sktr_accuracy': 0.0,
                'argmax_accuracy': 0.0,
                'sktr_individual': [],
                'argmax_individual': [],
                'num_cases': 0
            }

    return results


def compute_evaluation_metrics(
    results_df: pd.DataFrame,
    *,
    background: Optional[Any] = None,
    label_names: Optional[Sequence[Any]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute comprehensive TAS evaluation metrics for SKTR and argmax predictions.

    This function takes a results DataFrame (typically from recovery experiments)
    and computes the standard Temporal Action Segmentation (TAS) metrics for both
    SKTR and argmax predictions. It uses the evaluation functions from evaluation.py
    following the standard TAS protocol.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing recovery results with columns:
        - 'case:concept:name': case identifier
        - 'sktr_activity': SKTR predicted activities
        - 'argmax_activity': argmax predicted activities
        - 'ground_truth': ground truth activities

    Returns
    -------
    Dict[str, Dict[str, float]]
        Nested dictionary with metrics for SKTR and argmax:
        {
            'sktr': {
                'acc_micro': micro frame accuracy,
                'edit': macro mean edit score,
                'f1@10': macro mean F1@10,
                'f1@25': macro mean F1@25,
                'f1@50': macro mean F1@50
            },
            'argmax': {
                same metrics as above
            }
        }

    Examples
    --------
    >>> import pandas as pd
    >>> from utils import compute_evaluation_metrics
    >>> # Load your recovery results CSV
    >>> df = pd.read_csv('recovery_results_50salads_complete_15.csv')
    >>> metrics = compute_evaluation_metrics(df)
    >>> print("SKTR F1@10:", metrics['sktr']['f1@10'])
    >>> print("Argmax F1@10:", metrics['argmax']['f1@10'])

    Notes
    -----
    - Uses compute_tas_metrics_macro from evaluation.py
    - Follows standard TAS evaluation protocol with macro averaging for Edit/F1
    - Micro accuracy is computed globally over all frames
    - Activities are automatically converted to string format if needed
    - Returns separate metrics for SKTR and argmax approaches
    """
    try:
        from .evaluation import compute_tas_metrics_macro
    except ImportError:
        # If running as standalone script, try direct import
        try:
            from evaluation import compute_tas_metrics_macro
        except ImportError:
            raise ImportError("Cannot import compute_tas_metrics_macro from evaluation.py")

    # Validate required columns
    required_cols = {'case:concept:name', 'sktr_activity', 'argmax_activity', 'ground_truth'}
    missing = required_cols.difference(results_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Make a copy to avoid modifying original data
    df = results_df.copy()

    # Convert activity columns to strings if they're numeric
    for col in ['sktr_activity', 'argmax_activity', 'ground_truth']:
        if df[col].dtype != 'object':
            df[col] = df[col].astype(str)

    if label_names is None:
        label_names = pd.concat(
            [df['ground_truth'], df['sktr_activity'], df['argmax_activity']]
        ).unique().tolist()

    print(f"Computing evaluation metrics for {df['case:concept:name'].nunique()} cases...")

    # Compute metrics for SKTR
    print("Computing SKTR metrics...")
    sktr_metrics = compute_tas_metrics_macro(
        df=df,
        pred_col='sktr_activity',
        gt_col='ground_truth',
        case_col='case:concept:name',
        background=background,
        label_names=label_names,
    )

    # Compute metrics for argmax
    print("Computing argmax metrics...")
    argmax_metrics = compute_tas_metrics_macro(
        df=df,
        pred_col='argmax_activity',
        gt_col='ground_truth',
        case_col='case:concept:name',
        background=background,
        label_names=label_names,
    )

    # Organize results
    results = {
        'sktr': sktr_metrics,
        'argmax': argmax_metrics
    }

    print("Evaluation metrics computed successfully!")
    return results


def compute_comprehensive_tas_comparison(
    results_df: pd.DataFrame,
    kari_pkl_path: str,
    dataset_name: str = "50salads"
) -> Dict[str, Dict[str, float]]:
    """
    Compute comprehensive TAS statistics for all three approaches: argmax, SKTR, and KARI.

    This function takes recovery results (for SKTR/argmax) and KARI results, then computes
    TAS metrics for all three approaches using the appropriate evaluation functions.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing recovery results with columns:
        - 'case:concept:name': case identifier
        - 'sktr_activity': SKTR predicted activities
        - 'argmax_activity': argmax predicted activities
        - 'ground_truth': ground truth activities
    kari_pkl_path : str
        Path to the pickle file containing KARI results
    dataset_name : str, default "50salads"
        Dataset name for loading ground truth data

    Returns
    -------
    Dict[str, Dict[str, float]]
        Nested dictionary with metrics for all three approaches:
        {
            'sktr': {
                'acc_micro': micro frame accuracy,
                'edit': macro mean edit score,
                'f1@10': macro mean F1@10,
                'f1@25': macro mean F1@25,
                'f1@50': macro mean F1@50
            },
            'argmax': {
                same metrics as above
            },
            'kari': {
                same metrics as above
            }
        }

    Raises
    ------
    FileNotFoundError
        If KARI pickle file doesn't exist.
    ValueError
        If required columns are missing or data validation fails.

    Examples
    --------
    >>> import pandas as pd
    >>> from utils import compute_comprehensive_tas_comparison
    >>> # Load recovery results
    >>> results = pd.read_csv('recovery_results_50salads_complete_15.csv')
    >>> # Compute comprehensive comparison
    >>> all_metrics = compute_comprehensive_tas_comparison(results, 'kari_results_50salads_complete.pkl')
    >>> print("SKTR F1@10:", all_metrics['sktr']['f1@10'])
    >>> print("Argmax F1@10:", all_metrics['argmax']['f1@10'])
    >>> print("KARI F1@10:", all_metrics['kari']['f1@10'])
    """
    from pathlib import Path

    # Validate inputs
    required_cols = {'case:concept:name', 'sktr_activity', 'argmax_activity', 'ground_truth'}
    missing = required_cols.difference(results_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in results_df: {sorted(missing)}")

    # Check KARI file exists
    pkl_path = Path(kari_pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"KARI results file not found: {pkl_path}")

    print(f"Computing comprehensive TAS comparison for {results_df['case:concept:name'].nunique()} cases...")

    # Compute SKTR and argmax metrics using existing function
    print("Computing SKTR and argmax metrics...")
    sktr_argmax_metrics = compute_evaluation_metrics(results_df)

    # Compute KARI metrics using existing function
    print("Computing KARI metrics...")
    case_order = results_df['case:concept:name'].astype(str).drop_duplicates().tolist()
    # Adapt results_df to the schema expected by compute_kari_metrics/get_sequences_by_case:
    # it needs a DataFrame with columns 'case:concept:name' and 'concept:name' for GT.
    gt_df_for_kari = (
        results_df[["case:concept:name", "ground_truth"]]
        .rename(columns={"ground_truth": "concept:name"})
        .copy()
    )
    # Ensure both columns are strings so filtering by case_id matches
    gt_df_for_kari["concept:name"] = gt_df_for_kari["concept:name"].astype(str)
    gt_df_for_kari["case:concept:name"] = gt_df_for_kari["case:concept:name"].astype(str)
    kari_metrics = compute_kari_metrics(pkl_path, gt_df_for_kari, case_order, method_name="kari")

    # Combine all metrics into a single comparison dictionary
    comprehensive_comparison = {
        'sktr': sktr_argmax_metrics['sktr'],
        'argmax': sktr_argmax_metrics['argmax'],
        'kari': kari_metrics['kari']
    }

    print("✅ Comprehensive TAS comparison completed!")
    print(f"SKTR F1@10: {comprehensive_comparison['sktr']['f1@10']:.3f}")
    print(f"Argmax F1@10: {comprehensive_comparison['argmax']['f1@10']:.3f}")
    print(f"KARI F1@10: {comprehensive_comparison['kari']['f1@10']:.3f}")

    return comprehensive_comparison


def print_tas_comparison(
    metrics: Dict[str, Dict[str, float]],
    sort_by: Optional[str] = "f1@10",
    ascending: bool = False,
    highlight_best: bool = False,
    return_df: bool = False,
    precision: int = 2,
) -> Optional[pd.DataFrame]:
    """
    Print a compact comparison table of TAS metrics.

    Rows correspond to approaches (argmax, sktr, kari) and columns to metrics.
    By default no special highlighting is applied; set ``highlight_best=True``
    to show a short summary of the best performer per metric.
    """
    import pandas as pd

    metric_names = ["acc_micro", "edit", "f1@10", "f1@25", "f1@50"]

    # Build table with approaches as rows
    desired_order = ["argmax", "sktr", "kari"]
    df = pd.DataFrame.from_dict(metrics, orient="index")
    existing_rows = [row for row in desired_order if row in df.index]
    # Append any unexpected rows at the end to avoid dropping information
    existing_rows += [row for row in df.index if row not in existing_rows]
    df = df.loc[existing_rows]
    # Keep only known metric columns (in order) that exist
    existing = [m for m in metric_names if m in df.columns]
    df = df.reindex(columns=existing)

    df_sorted = df.copy()
    if sort_by is not None:
        if sort_by not in df_sorted.columns:
            # Fall back gracefully to first metric if available
            sort_by = existing[0] if existing else None
        if sort_by is not None:
            df_sorted = df_sorted.sort_values(by=sort_by, ascending=ascending)

    # Optional highlighting of best values per column
    df_display = df_sorted.round(precision)

    if sort_by is None:
        print("\nTAS comparison (original order)")
    else:
        print("\nTAS comparison (sorted by '{}' {})".format(sort_by, "asc" if ascending else "desc"))
    print(df_display.to_string())

    if highlight_best:
        print("\nBest per metric:")
        for col in df_sorted.columns:
            best_row = df_sorted[col].idxmax()
            best_val = df_sorted.loc[best_row, col]
            print(f"  {col}: {best_row} ({best_val:.{precision}f})")

    if return_df:
        return df_sorted
    return None


def compute_kari_metrics(
    pkl_file_path: Union[str, Path],
    df: pd.DataFrame,
    case_id_order: List[str],
    method_name: str = "kari",
    background: Optional[Any] = '0',
) -> Dict[str, Dict[str, float]]:
    """
    Compute TAS evaluation metrics for KARI approach predictions.

    This function loads KARI results from a pickle file, extracts frame-level
    predictions from the 'labels' field, and computes comprehensive TAS metrics
    exactly like compute_evaluation_metrics.

    Parameters
    ----------
    pkl_file_path : str or Path
        Path to the pickle file containing KARI results with 'labels' field
        containing frame-level predictions.
    df : pd.DataFrame
        DataFrame containing ground truth sequences with 'case:concept:name'
        and 'concept:name' columns.
    case_id_order : List[str]
        List of case IDs in the order they appear in the pickle file results.
        Each element should be a string matching the case identifiers in df.
    method_name : str, default "kari"
        Name/key to use for the results dictionary.
    background : Any, optional
        Background label for edit/F1 computation (default '0').

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary with method name as key and metrics as values:
        {
            method_name: {
                'acc_micro': global frame accuracy,
                'edit': macro mean edit score,
                'f1@10': macro mean F1@10,
                'f1@25': macro mean F1@25,
                'f1@50': macro mean F1@50
            }
        }

    Raises
    ------
    FileNotFoundError
        If pickle file doesn't exist.
    ValueError
        If sequence lengths don't match or other validation errors.
    """
    import pickle
    try:
        # Try relative import (for when used as part of the package)
        from .evaluation import compute_tas_metrics_from_sequences
    except ImportError:
        # Fall back to absolute import (for standalone usage)
        from evaluation import compute_tas_metrics_from_sequences

    # Load KARI results
    pkl_path = Path(pkl_file_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"KARI results file not found: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        kari_results = pickle.load(f)

    # Validate we have the expected number of results
    if len(kari_results) != len(case_id_order):
        raise ValueError(
            f"Number of KARI results ({len(kari_results)}) doesn't match "
            f"number of case IDs ({len(case_id_order)})"
        )

    # Get ground truth sequences
    gt_sequences = get_sequences_by_case(df, case_id_order)

    # Extract predictions from labels field
    pred_sequences = []
    for i, result in enumerate(kari_results):
        case_id = case_id_order[i]

        # Validate that 'labels' field exists
        if 'labels' not in result:
            raise ValueError(f"Result for case {case_id} missing 'labels' field")

        # Convert numpy array to list of strings to match ground truth format
        labels_array = result['labels']
        pred_seq = [str(x) for x in labels_array]
        pred_sequences.append(pred_seq)

    # Validate that all sequences have matching lengths
    length_mismatches = []
    for i, (gt_seq, pred_seq) in enumerate(zip(gt_sequences, pred_sequences)):
        if len(gt_seq) != len(pred_seq):
            length_mismatches.append(
                f"Case {case_id_order[i]}: GT={len(gt_seq)}, Pred={len(pred_seq)}"
            )

    if length_mismatches:
        raise ValueError(
            f"Sequence length mismatches found:\n" + "\n".join(length_mismatches)
        )

    # Compute metrics using the standard function
    metrics = compute_tas_metrics_from_sequences(
        gt_sequences=gt_sequences,
        pred_sequences=pred_sequences,
        background=background
    )

    return {method_name: metrics}


def filter_dataframe_by_case_ids(
    df: pd.DataFrame,
    case_ids: List[str]
) -> pd.DataFrame:
    """
    Filter a DataFrame to include only specified case IDs in the exact order provided.

    This function selects specific cases from the DataFrame while preserving the exact
    order of activities within each case as they appear in the original DataFrame.
    The output DataFrame will contain only the specified case IDs in the order given,
    with all activities from each case preserved in their original sequence.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'case:concept:name' and 'concept:name' columns,
        typically from prepare_df().
    case_ids : List[str]
        List of case IDs (as strings) to include in the filtered DataFrame.
        Cases will appear in the output DataFrame in the same order as in this list.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only the specified case IDs in the order provided,
        with the exact activity sequences preserved within each case.

    Example
    -------
    >>> result = prepare_df('50salads')
    >>> df, softmax_lst = result
    >>> test_cases = ['20', '11', '5', '36', '14']
    >>> filtered_df = filter_dataframe_by_case_ids(df, test_cases)
    >>> # filtered_df now contains only cases '20', '11', '5', '36', '14'
    >>> # in that exact order, with activities in original sequence per case
    """
    # Validate required columns exist
    required_cols = {"case:concept:name", "concept:name"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {missing}")

    # Create a categorical with the specified order to preserve case order
    df = df.copy()  # Avoid modifying the original DataFrame
    df['case:concept:name'] = pd.Categorical(
        df['case:concept:name'],
        categories=case_ids,
        ordered=True
    )

    # Filter to only the specified case IDs and sort by the categorical order
    filtered_df = df[df['case:concept:name'].isin(case_ids)].sort_values('case:concept:name')

    # Reset the categorical to regular strings to avoid issues downstream
    filtered_df['case:concept:name'] = filtered_df['case:concept:name'].astype(str)

    return filtered_df


def compute_kari_metrics_from_pkl(
    pkl_file_path: Union[str, Path],
    dataset_name: str = "50salads",
    case_id_order: Optional[List[str]] = None,
    method_name: str = "kari",
    background: Optional[Any] = '0',
    path: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Convenience function to compute KARI metrics with automatic DataFrame loading.

    This function automatically loads the DataFrame using prepare_df and
    provides a default case_id_order for 50salads dataset.

    Parameters
    ----------
    pkl_file_path : str or Path
        Path to the pickle file containing KARI results.
    dataset_name : str, default "50salads"
        Dataset name for prepare_df.
    case_id_order : List[str], optional
        List of case IDs in correct order. If None, uses default order for 50salads.
    method_name : str, default "kari"
        Name/key for results dictionary.
    background : Any, optional
        Background label (default '0').
    path : str or Path, optional
        Path parameter for prepare_df.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Same format as compute_kari_metrics.
    """
    # Load DataFrame
    # Only pass path if it's not None or empty string
    if path:
        result = prepare_df(dataset_name, path=path)
    else:
        result = prepare_df(dataset_name)
    if len(result) == 2:
        df, _ = result
    else:
        df, _, _ = result

    # Default case order for 50salads
    if case_id_order is None:
        case_id_order = ['30', '17', '9', '8', '20', '7', '23', '5', '28', '2', '1', '0',
                        '13', '36', '33', '3', '14', '10', '31', '22', '34', '38', '37', '6',
                        '24', '27', '21', '15', '11', '19', '16', '12', '32', '25', '35', '39',
                        '26', '29', '4', '18']

    return compute_kari_metrics(
        pkl_file_path=pkl_file_path,
        df=df,
        case_id_order=case_id_order,
        method_name=method_name,
        background=background
    )


def attach_column_by_case_and_position(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    source_value_col: str,
    *,
    case_col: str = "case:concept:name",
    new_col_name: Optional[str] = None,
    validate_lengths: bool = True,
) -> pd.DataFrame:
    """
    Attach a column from ``source_df`` to ``target_df`` by aligning rows
    using (case_id, within-case order) without changing the row order of ``target_df``.

    - The case ordering in ``target_df`` is preserved exactly.
    - The within-case ordering in both dataframes is preserved using per-case positions.
    - ``source_df`` can have cases in any order; only within-case order matters.

    Parameters
    ----------
    target_df : pd.DataFrame
        Left dataframe whose row order must be preserved.
    source_df : pd.DataFrame
        Right dataframe providing the values to attach.
    source_value_col : str
        Column name in ``source_df`` containing the values to attach.
    case_col : str, default 'case:concept:name'
        Column name for the case identifier present in both dataframes.
    new_col_name : str, optional
        Name of the new column in the returned dataframe. Defaults to ``source_value_col``.
    validate_lengths : bool, default True
        If True, verifies that for every case_id the number of rows in ``source_df``
        matches the number of rows in ``target_df``. Raises ``ValueError`` on mismatch.

    Returns
    -------
    pd.DataFrame
        A new dataframe equal to ``target_df`` with the additional column attached.

    Raises
    ------
    ValueError
        If required columns are missing or, when ``validate_lengths=True``, any per-case
        length mismatches are found between ``target_df`` and ``source_df``.
    """
    if new_col_name is None:
        new_col_name = source_value_col

    # Basic column checks
    for col_name, df_name, df in (
        (case_col, "target_df", target_df),
        (case_col, "source_df", source_df),
        (source_value_col, "source_df", source_df),
    ):
        if col_name not in df.columns:
            raise ValueError(f"Missing required column '{col_name}' in {df_name}")

    # Optimize: extract only needed columns from source_df before copying
    # This reduces memory usage significantly for wide dataframes
    right = source_df[[case_col, source_value_col]].copy()
    right[case_col] = right[case_col].astype(str)
    
    # Optimize: convert case column without copying entire target_df
    left_case_str = target_df[case_col].astype(str)

    if validate_lengths:
        left_counts = left_case_str.value_counts(sort=False)
        right_counts = right[case_col].value_counts(sort=False)

        mismatches = []
        for case_id, left_n in left_counts.items():
            right_n = int(right_counts.get(case_id, 0))
            if right_n != int(left_n):
                mismatches.append((case_id, int(left_n), right_n))

        if mismatches:
            sample = "\n".join(
                f"  {cid}: target={tn}, source={sn}" for cid, tn, sn in mismatches[:10]
            )
            raise ValueError(
                "Per-case length mismatch between target_df and source_df for "
                f"{len(mismatches)} cases. First mismatches:\n{sample}"
            )

    # Compute per-case positions efficiently
    left_pos = left_case_str.groupby(left_case_str, sort=False).cumcount()
    right["__pos__"] = right.groupby(case_col, sort=False).cumcount()

    # Create minimal lookup dataframe
    right_lookup = right.rename(columns={case_col: "__case__", source_value_col: new_col_name})
    
    # Build merge keys without copying target_df
    merge_keys = pd.DataFrame({
        "__case__": left_case_str.values,
        "__pos__": left_pos.values
    })
    
    # Merge to get the values
    merged_col = merge_keys.merge(right_lookup, on=["__case__", "__pos__"], how="left")[new_col_name]
    
    # Only copy target_df once at the end and add the new column
    result = target_df.copy()
    result[new_col_name] = merged_col.values
    # Preserve original behavior: ensure case_col is string in the output
    result[case_col] = result[case_col].astype(str)
    return result


def add_kari_column_to_results(
    target_csv_path: Union[str, Path],
    kari_csv_path: Union[str, Path],
    *,
    case_col: str = "case:concept:name",
    kari_col: str = "kari_activity",
    output_path: Optional[Union[str, Path]] = None,
    validate_lengths: bool = True,
) -> pd.DataFrame:
    """
    Convenience wrapper to add ``kari_activity`` from a KARI results CSV to a
    recovery results CSV by aligning on (case_id, within-case order).

    Parameters
    ----------
    target_csv_path : str or Path
        Path to the recovery results CSV (left dataframe, order preserved).
    kari_csv_path : str or Path
        Path to the KARI results CSV containing the column specified by ``kari_col``.
    case_col : str, default 'case:concept:name'
        Case identifier column present in both CSVs.
    kari_col : str, default 'kari_activity'
        Column in the KARI CSV to attach.
    output_path : str or Path, optional
        If provided, saves the merged dataframe to this path.
    validate_lengths : bool, default True
        If True, checks equal per-case lengths and raises on mismatch.

    Returns
    -------
    pd.DataFrame
        The merged dataframe. If ``output_path`` is provided, the same content is also saved.
    """
    import pandas as pd

    left = pd.read_csv(target_csv_path)
    # Optimize: only load the columns we need from kari_csv
    right = pd.read_csv(kari_csv_path, usecols=[case_col, kari_col])

    merged = attach_column_by_case_and_position(
        left,
        right,
        source_value_col=kari_col,
        case_col=case_col,
        new_col_name=kari_col,
        validate_lengths=validate_lengths,
    )

    if output_path is not None:
        merged.to_csv(output_path, index=False)

    return merged
