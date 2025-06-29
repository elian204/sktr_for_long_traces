"""
Incremental Softmax Recovery: Main high-level function.

This module provides the main entry point for incremental softmax matrix recovery
using beam search with Petri nets, following the pattern of the existing 
compare_stochastic_vs_argmax_random_indices function.
"""

from typing import Tuple, Union, Callable, Optional
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


