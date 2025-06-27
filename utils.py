"""
Incremental Softmax Recovery: Main high-level function.

This module provides the main entry point for incremental softmax matrix recovery
using beam search with Petri nets, following the pattern of the existing 
compare_stochastic_vs_argmax_random_indices function.
"""

from typing import Tuple, Union, Callable
import numpy as np


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


def process_cost_function(
    cost_function: Union[str, Callable[[float], float]], 
    round_precision: int = 2
) -> Callable[[float], float]:
    """
    Process and validate the cost function parameter.
    
    Parameters
    ----------
    cost_function : str or callable
        Either 'linear', 'logarithmic', or a custom callable that maps float → float.
    round_precision : int, default=2
        Number of decimal places for probability rounding. Used to determine
        the minimum probability (10^-round_precision) for logarithmic scaling.
        
    Returns
    -------
    callable
        A cost function that takes a float and returns a float.
        For logarithmic functions, costs are normalized to [0,1] range.
        
    Raises
    ------
    ValueError
        If cost_function is a string but not 'linear' or 'logarithmic'.
    TypeError
        If cost_function is neither a string nor callable.
        
    Notes
    -----
    - Linear cost: cost = 1 - probability
    - Logarithmic cost: cost = -ln(max(prob, min_prob)) / (-ln(min_prob))
      where min_prob = 10^(-round_precision)
    """
    if isinstance(cost_function, str):
        if cost_function == "linear":
            return lambda x: 1.0 - x
        elif cost_function == "logarithmic":
            # Calculate minimum probability and normalization factor
            min_prob = 10 ** (-round_precision)
            scale_factor = -np.log(min_prob)  # This ensures max cost = 1.0
            
            def logarithmic_cost(x: float) -> float:
                # Clamp probability to valid range [min_prob, 1.0]
                prob = max(min(x, 1.0), min_prob)
                return -np.log(prob) / scale_factor
            
            return logarithmic_cost
        else:
            raise ValueError(f"Unknown cost function string: '{cost_function}'. "
                           f"Supported values are 'linear' and 'logarithmic'.")
    elif callable(cost_function):
        return cost_function
    else:
        raise TypeError(f"cost_function must be a string or callable, got {type(cost_function)}")


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


