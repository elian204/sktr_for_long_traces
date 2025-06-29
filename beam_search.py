"""
Beam Search for Incremental Softmax Recovery.

This module implements beam search algorithms for incrementally recovering 
activity sequences from softmax probability matrices using Petri net models.
The beam search maintains multiple candidate paths and selects the most 
promising ones at each step based on probability scores and model constraints.

Main Functions:
    process_test_case_incremental: Main entry point for processing a single test case
    
Helper Classes and Functions:
    BeamCandidate: Represents a candidate path in the beam
    BeamState: Manages the beam search state
    Various probability computation and ranking helpers
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from classes import PetriNet, Marking


class BeamCandidate:
    """
    Represents a candidate path in the beam search.
    
    Attributes
    ----------
    path : List[str]
        Sequence of predicted activities so far
    cumulative_cost : float
        Total cost accumulated along this path
    marking : Any
        Current Petri net marking state
    timestamp : int
        Current timestamp position in the softmax matrix (how many timestamps recovered)
    """
    
    def __init__(self, path: List[str], cumulative_cost: float, marking: Marking, timestamp: int):
        self.path = path.copy()
        self.cumulative_cost = cumulative_cost
        self.marking = marking
        self.timestamp = timestamp
    
    def __repr__(self) -> str:
        return f"BeamCandidate(path={self.path}, cost={self.cumulative_cost:.4f}, timestamp={self.timestamp})"
    
    def copy(self) -> 'BeamCandidate':
        """Create a copy of this candidate."""
        return BeamCandidate(self.path, self.cumulative_cost, self.marking, self.timestamp)


class BeamState:
    """
    Manages the beam search state and operations.
    
    Attributes
    ----------
    candidates : List[BeamCandidate]
        Current beam candidates
    beam_width : int
        Maximum number of candidates to maintain
    """
    
    def __init__(self, beam_width: int):
        self.candidates: List[BeamCandidate] = []
        self.beam_width = beam_width
    
    def add_candidate(self, candidate: BeamCandidate) -> None:
        """Add a candidate to the beam."""
        self.candidates.append(candidate)
    
    def prune_beam(self) -> None:
        """Keep only the top beam_width candidates (lowest cost)."""
        if len(self.candidates) > self.beam_width:
            # Sort by cost (ascending) and keep top candidates
            self.candidates.sort(key=lambda c: c.cumulative_cost)
            self.candidates = self.candidates[:self.beam_width]
    
    def get_best_candidate(self) -> Optional[BeamCandidate]:
        """Get the candidate with lowest cumulative cost."""
        if not self.candidates:
            return None
        return min(self.candidates, key=lambda c: c.cumulative_cost)
    
    def is_empty(self) -> bool:
        """Check if beam is empty."""
        return len(self.candidates) == 0


def process_test_case_incremental(
    softmax_matrix: np.ndarray,
    model: PetriNet,
    cost_fn: Callable[[float, str], float],
    beam_width: int,
    lambdas: List[float],
    alpha: float,
    use_cond_probs: bool,
    prob_dict: dict,
    use_ngram_smoothing: bool,
    activity_prob_threshold: float,
) -> List[str]:
    """
    Process a single test case using incremental beam search.
    
    Performs step-by-step recovery of activity sequences from softmax matrices
    using beam search with Petri net constraints and optional conditional probabilities.
    
    Parameters
    ----------
    softmax_matrix : np.ndarray
        Softmax probability matrix for this trace, shape (n_activities, n_timestamps)
    model : PetriNet
        Discovered Petri net model
    cost_function : Callable[[float], float]
        Cost function for probability-to-cost conversion
    beam_width : int
        Maximum number of candidates to maintain in beam
    lambdas : List[float]
        Blending weights for conditional probabilities
    alpha : float
        Blending parameter between base and conditional probabilities
    use_cond_probs : bool
        Whether to use conditional probabilities
    prob_dict : dict
        Dictionary of conditional probabilities
    use_ngram_smoothing : bool
        Whether to apply n-gram smoothing
    activity_prob_threshold : float
        Minimum probability threshold for considering activities
        
    Returns
    -------
    List[str]
        Predicted sequence of activities
    """    
    # Input validation
    if softmax_matrix.ndim != 2:
        raise ValueError(f"Softmax matrix must be 2-dimensional, got {softmax_matrix.ndim} dimensions")
    
    if model.init_mark is None:
        raise ValueError("Model must have a valid initial marking (init_mark)")
    
    # Initialize beam search
    beam_state = BeamState(beam_width)
    
    # Initialize with empty path (zero initial cost)
    initial_candidate = BeamCandidate(
        path=[],
        cumulative_cost=0.0,
        marking=model.init_mark,
        timestamp=0
    )
    beam_state.add_candidate(initial_candidate)
    
    while True:
        # Expand beam candidates
        new_candidates = _expand_beam_candidates(
            beam_state=beam_state,
            softmax_matrix=softmax_matrix,
            model=model,
            cost_fn=cost_fn,
            use_cond_probs=use_cond_probs,
            prob_dict=prob_dict,
            lambdas=lambdas,
            alpha=alpha,
            use_ngram_smoothing=use_ngram_smoothing,
            activity_prob_threshold=activity_prob_threshold,
        )
        
        # Update beam with new candidates
        beam_state.candidates = new_candidates
        beam_state.prune_beam()
        
        # If beam becomes empty, use fallback
        if beam_state.is_empty():
            fallback_activity = _get_fallback_prediction(step_probabilities, model)
            # Create fallback candidate for remaining steps
            fallback_candidate = BeamCandidate(
                path=[fallback_activity],
                cumulative_cost=float('inf'),
                marking=model.init_mark,  # Reset to initial marking
                timestamp=step + 1
            )
            beam_state.add_candidate(fallback_candidate)
    
    # Get the best complete path
    best_candidate = beam_state.get_best_candidate()
    if best_candidate:
        predicted_sequence = best_candidate.path
    else:
        # Complete fallback - predict most likely activity for each step
        predicted_sequence = []
        for step in range(n_steps):
            step_probabilities = softmax_matrix[:, step] if softmax_matrix.ndim == 2 else softmax_matrix[step]
            predicted_sequence.append(_get_fallback_prediction(step_probabilities, model))
    
    return predicted_sequence


def _expand_beam_candidates(
    beam_state: BeamState,
    softmax_matrix: np.ndarray,
    model: PetriNet,
    cost_fn: Callable[[float, str], float],
    use_cond_probs: bool,
    prob_dict: Dict[Tuple[str, ...], Dict[str, float]],
    lambdas: List[float],
    alpha: float,
    use_ngram_smoothing: bool,
    activity_prob_threshold: float,
) -> List[BeamCandidate]:
    """
    Expand each beam candidate by all valid next transitions:
      - sync moves: match log & model (advance timestamp & path)
      - log moves: log insertion (advance path only)
      - tau moves: silent model moves (advance timestamp only)
      - model moves: model insertion (advance marking only)
    Compute adjusted probabilities, dispatch cost_fn by move_type.
    """
    new_candidates: List[BeamCandidate] = []

    for cand in beam_state.candidates:
        # Partition available transitions
        available = model._find_available_transitions(cand.marking.places)
        sync_trans = {t.label: t for t in available if t.label is not None}
        tau_trans  = {t.name:  t for t in available if t.label is None}

        # Raw softmax probs at this timestamp
        probs = softmax_matrix[:, cand.timestamp]

        # 1) handle sync/log/tau via softmax indices
        for idx, raw_p in enumerate(probs):
            if raw_p < activity_prob_threshold:
                continue

            act = str(idx)
            p = (
                _compute_conditional_probability(
                    cand.path, act, raw_p,
                    prob_dict, lambdas, alpha, use_ngram_smoothing
                )
                if use_cond_probs and prob_dict else raw_p
            )

            if act in sync_trans:
                # synchronous move
                move_type = "sync"
                next_path = cand.path + [act]
                next_timestamp = cand.timestamp + 1
                next_marking = model._fire_transition(cand.marking, sync_trans[act])

            elif act in tau_trans:
                # silent τ-move triggered by an activity name
                move_type = "tau"
                next_path = cand.path
                next_timestamp = cand.timestamp
                next_marking = model._fire_transition(cand.marking, tau_trans[act])

            else:
                # log move: insert log event without model firing
                move_type = "log"
                next_path = cand.path + [act]
                next_timestamp = cand.timestamp
                next_marking = cand.marking

            step_cost = cost_fn(p, move_type)
            new_candidates.append(
                BeamCandidate(
                    path=next_path,
                    cumulative_cost=cand.cumulative_cost + step_cost,
                    marking=next_marking,
                    timestamp=next_timestamp
                )
            )

        # 2) handle pure model moves (model insertion)
        for trans in sync_trans.values():
            next_marking = model._fire_transition(cand.marking, trans)
            model_cost = cost_fn(0.0, "model")
            new_candidates.append(
                BeamCandidate(
                    path=cand.path,
                    cumulative_cost=cand.cumulative_cost + model_cost,
                    marking=next_marking,
                    timestamp=cand.timestamp
                )
            )

    return new_candidates





def _get_activity_names_from_model(model: Any) -> List[str]:
    """Extract activity names from the Petri net model."""
    # TODO: Implement based on model structure
    # This is a placeholder - need to understand model structure
    return [t.label for t in model.transitions if t.label is not None]


def _is_activity_valid(marking: Any, activity_name: str, model: Any) -> bool:
    """Check if an activity can be executed from the current marking."""
    # TODO: Implement Petri net firing rule validation
    # This is a placeholder - need to implement actual validation
    return True


def _fire_transition(marking: Any, activity_name: str, model: Any) -> Any:
    """Fire a transition and return the new marking."""
    # TODO: Implement transition firing logic
    # This is a placeholder - need to implement actual firing
    return marking


def _compute_conditional_probability(
    path_prefix: List[str],
    activity_name: str,
    base_probability: float,
    prob_dict: dict,
    lambdas: List[float],
    alpha: float,
    use_ngram_smoothing: bool,
) -> float:
    """Compute conditional probability based on path history."""
    # TODO: Implement conditional probability computation
    # This is a placeholder - need to implement the logic from classes.py
    return base_probability


def _get_fallback_prediction(step_probabilities: np.ndarray, model: Any) -> str:
    """Get fallback prediction when beam search fails."""
    activity_names = _get_activity_names_from_model(model)
    best_idx = np.argmax(step_probabilities)
    return activity_names[best_idx] if best_idx < len(activity_names) else f"activity_{best_idx}" 