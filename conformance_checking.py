"""
Conformance checking functions for trace recovery.

This module provides conformance-based alternatives to beam search
for recovering activity sequences from softmax probability matrices.
"""

from typing import Callable, List, Tuple, Dict, Any, Mapping, Optional
import numpy as np
from classes import PetriNet


def process_trace_chunked(
    softmax_matrix: np.ndarray,
    model: PetriNet,
    cost_fn: Callable[[float, str], float],
    chunk_size: int = 10,
    eps: float = 1e-12,
    inline_progress: bool = False,
    progress_prefix: str = "",
    prob_dict: Optional[Mapping[Tuple[str, ...], Mapping[str, float]]] = None,
    switch_penalty_weight: float = 0.0,
    use_state_caching: bool = True,
    merge_mismatched_boundaries: bool = True,
) -> Tuple[List[str], List[float]]:
    """
    Conformance-based recovery for a single trace processed in chunks.
    
    Args:
        softmax_matrix: Softmax probability matrix (n_activities, n_timestamps)
        model: PetriNet model to use for conformance checking
        cost_fn: Cost function for moves
        chunk_size: Size of chunks to process iteratively
        eps: Minimum probability threshold - activities below this are filtered out
        inline_progress: Whether to print inline progress during processing
        progress_prefix: Prefix string for progress display
        prob_dict: Optional conditional probability dictionary for switch penalties
        switch_penalty_weight: Weight for penalizing label switches across chunks
        use_state_caching: Enable caching of intermediate states for speed
        merge_mismatched_boundaries: If True, merge adjacent chunks when boundary labels disagree
    
    Returns:
        Tuple[List[str], List[float]]: (predicted_sequence, move_costs)
    """
    return model.process_trace_conformance(
        softmax_matrix=softmax_matrix,
        cost_fn=cost_fn,
        chunk_size=chunk_size,
        eps=eps,
        inline_progress=inline_progress,
        progress_prefix=progress_prefix,
        prob_dict=prob_dict,
        switch_penalty_weight=switch_penalty_weight,
        use_state_caching=use_state_caching,
        merge_mismatched_boundaries=merge_mismatched_boundaries,
    )