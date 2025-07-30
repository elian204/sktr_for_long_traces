"""
Beam Search for Incremental Softmax Recovery.

This module implements beam search algorithms for incrementally recovering 
activity sequences from softmax probability matrices using Petri net models.
The beam search maintains multiple candidate paths and selects the most 
promising ones at each step based on probability scores and model constraints.

Main Functions:
    process_test_case_incremental: Main entry point for processing a single test case
    
Helper Classes and Functions:
    Various probability computation and ranking helpers
"""

from typing import Callable, Dict, List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


def process_test_case_incremental(
    softmax_matrix: np.ndarray,
    cost_fn: Callable[[float, str], float],
    short_term_window: int,
    long_term_window: int,
    alpha: float,
    beta: float, # Added beta parameter
    prob_dict: Dict[Tuple[str, ...], Dict[str, float]],
    zero_penalty: float = 0.5
) -> Tuple[List[str], List[float]]:
    """
    Process a single test case using greedy lookahead prediction.

    Parameters
    ----------
    softmax_matrix : np.ndarray
        Softmax probability matrix for this trace, shape (n_activities, n_timestamps).
    cost_fn : Callable[[float, str], float]
        Cost function for probability-to-cost conversion.
    short_term_window : int
        Number of steps to look ahead for short-term averaging.
    long_term_window : int
        Number of steps to look ahead for long-term averaging.
    alpha : float
        Weight for past conditional probabilities.
    beta : float
        Weight for current probabilities.
    prob_dict : Dict[Tuple[str, ...], Dict[str, float]]
        Dictionary of conditional probabilities.

    Returns
    -------
    Tuple[List[str], List[float]]
        Predicted sequence of activity labels and their associated costs.
    """
    n_timestamps = softmax_matrix.shape[1]
    predicted_sequence = []
    predicted_costs = []
    prev_label = None

    for step in range(n_timestamps):
        current_probs = softmax_matrix[:, step]

        if prev_label is None:
            max_idx = np.argmax(current_probs)
            chosen_label = str(max_idx)
            p = current_probs[max_idx]
        else:
            max_idx = np.argmax(current_probs)
            max_label = str(max_idx)

            prev_idx = int(prev_label)
            prev_prob = current_probs[prev_idx]

            if max_label == prev_label:
                chosen_label = max_label
                p = current_probs[max_idx]
            else:
                higher_labels = [str(i) for i in range(len(current_probs)) if current_probs[i] > prev_prob]
                candidates = set(higher_labels) | {prev_label}

                short_avgs = {}
                for cand_label in candidates:
                    cand_idx = int(cand_label)
                    avail_future = n_timestamps - step - 1
                    short_avg = 0.0
                    if avail_future > 0:
                        short_end = step + 1 + min(short_term_window, avail_future)
                        if short_end > step + 1:
                            short_avg = np.mean(softmax_matrix[cand_idx, step + 1 : short_end])

                    current_prob = current_probs[int(cand_label)]  # Get current prob for this candidate
                    conditional = prob_dict.get((prev_label,), {}).get(cand_label, 0.0)
                    blended = alpha * conditional + beta * current_prob + (1 - alpha - beta) * short_avg
                    if conditional == 0:
                        blended *= zero_penalty
                    short_avgs[cand_label] = blended

                max_short_label = max(short_avgs, key=short_avgs.get) if short_avgs else prev_label

                if max_short_label == prev_label:
                    chosen_label = prev_label
                    p = prev_prob
                else:
                    long_avgs = {}
                    for cand_label in candidates:
                        cand_idx = int(cand_label)
                        avail_future = n_timestamps - step - 1
                        long_avg = 0.0
                        if avail_future > 0:
                            long_end = step + 1 + min(long_term_window, avail_future)
                            if long_end > step + 1:
                                long_avg = np.mean(softmax_matrix[cand_idx, step + 1 : long_end])

                            current_prob = current_probs[int(cand_label)]
                            conditional = prob_dict.get((prev_label,), {}).get(cand_label, 0.0)
                            blended = alpha * conditional + beta * current_prob + (1 - alpha - beta) * long_avg
                            if conditional == 0:
                                blended *= zero_penalty
                            long_avgs[cand_label] = blended

                    chosen_label = max(long_avgs, key=long_avgs.get) if long_avgs else prev_label
                    chosen_idx = int(chosen_label)
                    p = current_probs[chosen_idx]

        predicted_sequence.append(chosen_label)
        assumed_cost = cost_fn(p, "sync")
        predicted_costs.append(assumed_cost)
        prev_label = chosen_label

    return predicted_sequence, predicted_costs
