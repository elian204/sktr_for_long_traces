"""
Incremental Softmax Recovery: Main high-level function.

This module provides the main entry point for incremental softmax matrix recovery
using beam search with Petri nets, following the pattern of the existing 
compare_stochastic_vs_argmax_random_indices function.
"""

from typing import Tuple, Union, Callable, Optional, Dict, List
import numpy as np

MoveType = str 

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
    if not lambdas:
        raise ValueError("lambdas cannot be empty for n-gram probability computation")
    
    if not path_prefix_tuple:
        return prob_dict.get((), {}).get(activity_name, 0.0)
    
    total_weighted_prob = 0.0
    total_lambda_weight = 0.0
    max_n = min(len(path_prefix_tuple), len(lambdas))
    
    for n in range(1, max_n + 1):
        prefix_n_gram = path_prefix_tuple[-n:]
        prob = prob_dict.get(prefix_n_gram, {}).get(activity_name, 0.0)
        lambda_weight = lambdas[n - 1]
        
        total_weighted_prob += lambda_weight * prob
        total_lambda_weight += lambda_weight
    
    if total_lambda_weight == 0:
        return 0.0
    
    return total_weighted_prob / total_lambda_weight


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


