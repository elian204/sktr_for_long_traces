"""
Temperature scaling and calibration for softmax probabilities.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from utils import inverse_softmax


def softmax_numpy(logits: np.ndarray, axis: int = 0) -> np.ndarray:
    """Stable softmax implementation using numpy."""
    # Subtract max for numerical stability
    shifted_logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)


def cross_entropy_loss_numpy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Compute cross-entropy loss using numpy."""
    # Apply softmax to get probabilities
    probs = softmax_numpy(logits, axis=1)
    
    # Clip probabilities to avoid log(0)
    probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
    
    # Compute cross-entropy loss
    log_probs = np.log(probs)
    n_samples = logits.shape[0]
    
    # For each sample, select the log probability of the true class
    loss = -np.sum(log_probs[np.arange(n_samples), labels]) / n_samples
    return loss


def calibrate_probabilities(
    softmax_list: List[np.ndarray],
    df: pd.DataFrame,
    temp_bounds: Tuple[float, float],
    only_return_temperature: bool = False,
    global_temperature: Optional[float] = None
) -> Any:
    """
    CPU-only temperature scaling using numpy and scipy optimization.
    
    Parameters
    ----------
    softmax_list : List[np.ndarray]
        List of softmax probability matrices, one per case
    df : pd.DataFrame
        DataFrame with case and label information
    temp_bounds : Tuple[float, float]
        Lower and upper bounds for temperature parameter
    only_return_temperature : bool, default=False
        If True, only return the optimal temperature value
    global_temperature : float, optional
        Pre-computed temperature to use instead of learning
        
    Returns
    -------
    Union[float, List[np.ndarray]]
        Either the optimal temperature or list of calibrated matrices
    """
    all_logits, all_labels = [], []
    num_classes = softmax_list[0].shape[0]
    unique_cases = df['case:concept:name'].unique()

    # Create case to index mapping for safer access
    case_to_idx = {case: idx for idx, case in enumerate(unique_cases)}
    
    if len(softmax_list) != len(unique_cases):
        raise ValueError(f"Mismatch: {len(softmax_list)} softmax matrices but {len(unique_cases)} cases")

    # Collect logits and labels from all cases
    for case in unique_cases:
        case_idx = case_to_idx[case]
        probs = softmax_list[case_idx]
        case_data = df[df['case:concept:name'] == case]['concept:name']
        
        # Convert string integers to integers
        try:
            labels = case_data.astype(int).values
        except (ValueError, TypeError) as e:
            raise ValueError(f"concept:name contains non-integer values: {e}")
        
        logits = inverse_softmax(probs)
        L = min(logits.shape[1], len(labels))
        logits, labels = logits[:, :L], labels[:L]
        
        if logits.shape[0] != num_classes:
            raise ValueError(f"Logits shape mismatch: expected {num_classes} classes, got {logits.shape[0]}")
        
        all_logits.append(logits)
        all_labels.extend(labels)

    # Combine all data (transpose to get samples x classes format)
    combined_logits = np.hstack(all_logits).T  # Shape: (n_samples, n_classes)
    combined_labels = np.array(all_labels, dtype=int)

    # Learn temperature if not provided
    if global_temperature is None:
        def temperature_loss(temp: float) -> float:
            """Objective function for temperature optimization."""
            scaled_logits = combined_logits / temp
            return cross_entropy_loss_numpy(scaled_logits, combined_labels)
        
        # Use scipy for optimization (much faster than PyTorch for this simple case)
        result = minimize_scalar(
            temperature_loss, 
            bounds=temp_bounds, 
            method='bounded'
        )
        
        if not result.success:
            print(f"Warning: Temperature optimization failed: {result.message}")
            global_temperature = 1.0  # Fallback to no scaling
        else:
            global_temperature = result.x

    if only_return_temperature:
        return global_temperature

    # Apply temperature scaling to all cases
    calibrated = []
    for logits in all_logits:
        scaled_logits = logits / global_temperature
        calibrated_probs = softmax_numpy(scaled_logits, axis=0)
        calibrated.append(calibrated_probs)
    
    return calibrated


def calibrate_softmax(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    softmax_train: List[np.ndarray],
    softmax_test: List[np.ndarray],
    temp_bounds: Tuple[float, float],
    temperature: Optional[float] = None
) -> List[np.ndarray]:
    """
    Wrapper to first find the best temperature on training data, 
    then apply it to test softmax matrices.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame with case and label columns
    test_df : pd.DataFrame
        Test DataFrame with case and label columns
    softmax_train : List[np.ndarray]
        Training softmax matrices for temperature learning
    softmax_test : List[np.ndarray]
        Test softmax matrices to be calibrated
    temp_bounds : Tuple[float, float]
        Temperature bounds for calibration
    temperature : float, optional
        If provided, use this temperature instead of learning from training data
        
    Returns
    -------
    List[np.ndarray]
        Calibrated test softmax matrices

    Notes
    -----
    Assumes:
    - softmax_list and df cases are in the same order
    - df['concept:name'] values are string integers (e.g., '0', '1', '2')
    """
    # Use provided temperature or learn optimal temperature on training data
    if temperature is not None:
        temp = temperature
    else:
        temp = calibrate_probabilities(
            softmax_list=softmax_train,
            df=train_df,
            temp_bounds=temp_bounds,
            only_return_temperature=True
        )
    
    # Apply temperature to test data
    return calibrate_probabilities(
        softmax_list=softmax_test,
        df=test_df,
        temp_bounds=temp_bounds,
        global_temperature=temp
    )