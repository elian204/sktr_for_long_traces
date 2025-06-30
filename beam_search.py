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
from utils import compute_conditional_probability


class BeamCandidate:
    """
    Represents a candidate path in the beam search.

    Attributes
    ----------
    path : Tuple[str, ...]
        Sequence of predicted activities so far
    cumulative_cost : float
        Total cost accumulated along this path
    marking : Any
        Current Petri net marking state
    timestamp : int
        Current timestamp position in the softmax matrix
    """
    def __init__(self,
                 path: Tuple[str, ...],
                 cumulative_cost: float,
                 marking: Any,
                 timestamp: int):
        self.path = path
        self.cumulative_cost = cumulative_cost
        self.marking = marking
        self.timestamp = timestamp

    def __repr__(self) -> str:
        return (f"BeamCandidate(path={self.path}, "
                f"cost={self.cumulative_cost:.4f}, "
                f"timestamp={self.timestamp})")

    def copy(self) -> 'BeamCandidate':
        """Create a copy of this candidate."""
        return BeamCandidate(
            path=self.path,
            cumulative_cost=self.cumulative_cost,
            marking=self.marking,
            timestamp=self.timestamp
        )


class BeamState:
    """
    Manages the beam search state and operations.

    Attributes
    ----------
    candidates : List[BeamCandidate]
        Current beam candidates
    beam_width : int
        Maximum number of candidates to maintain
    epsilon : float
        Smoothing factor for average-cost normalization
    alpha : float
        Interpolation weight between normalized and total cost
    """
    def __init__(self,
                 beam_width: int,
                 epsilon: float = 1e-2,
                 alpha: float = 0.5):
        """
        :param beam_width: max number of beams to keep
        :param epsilon: smoothing for average-cost per move
        :param alpha: weight on normalized cost vs total cost
        """
        self.candidates: List[BeamCandidate] = []
        self.beam_width = beam_width
        self.epsilon = epsilon
        self.alpha = alpha

    def add_candidate(self, candidate: BeamCandidate) -> None:
        """Add a candidate to the beam."""
        self.candidates.append(candidate)

    def _beam_score(self, cand: BeamCandidate) -> float:
        """
        Compute a blended score:
          score = alpha * (avg_cost per move) + (1 - alpha) * total_cost
        Depth is taken as len(path).
        Lower is better.
        """
        depth = max(len(cand.path), 1)
        avg_cost = cand.cumulative_cost / (depth + self.epsilon)
        total_cost = cand.cumulative_cost
        return self.alpha * avg_cost + (1 - self.alpha) * total_cost

    def prune_beam(self) -> None:
        """Keep only the top beam_width candidates by blended score."""
        if len(self.candidates) > self.beam_width:
            self.candidates.sort(key=self._beam_score)
            self.candidates = self.candidates[:self.beam_width]

    def get_best_candidate(self) -> Optional[BeamCandidate]:
        """Get the candidate with lowest cumulative cost."""
        if not self.candidates:
            return None
        return min(self.candidates, key=lambda c: c.cumulative_cost)


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
    n_timestamps = softmax_matrix.shape[1]
    
    # Initialize with empty path (zero initial cost)
    initial_candidate = BeamCandidate(
        path=(),
        cumulative_cost=0.0,
        marking=model.init_mark,
        timestamp=0
    )
    beam_state.add_candidate(initial_candidate)
    
    # Main beam search loop - continue until all candidates complete the sequence
    max_iterations = n_timestamps * 10  # Safety limit to prevent infinite loops
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Expand beam candidates (generate ALL successors from current beam)
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
        
        # Replace ALL old candidates with new ones (proper beam search)
        beam_state.candidates = new_candidates
        beam_state.prune_beam()
        
        # Check for completed candidates AFTER expansion and pruning
        completed_candidates = [c for c in beam_state.candidates if c.timestamp >= n_timestamps]
        if completed_candidates:
            # Return the best completed candidate
            best_completed = min(completed_candidates, key=lambda c: beam_state._beam_score(c))
            return list(best_completed.path)
             
    # Fallback if we exit the loop without completing any candidates
    return _generate_fallback_sequence(beam_state, softmax_matrix, n_timestamps)


def _generate_fallback_sequence(
    beam_state: BeamState,
    softmax_matrix: np.ndarray,
    n_timestamps: int
) -> List[str]:
    """
    Generate fallback sequence when beam search doesn't complete successfully.
    
    If there's a best candidate, extend it with highest probability predictions.
    Otherwise, predict highest probability activity for each timestamp.
    
    Parameters
    ----------
    beam_state : BeamState
        Current beam search state
    softmax_matrix : np.ndarray
        Softmax probability matrix
    n_timestamps : int
        Total number of timestamps to predict
        
    Returns
    -------
    List[str]
        Predicted sequence of activities
    """
    best_candidate = beam_state.get_best_candidate()
    if best_candidate:
        predicted_sequence = list(best_candidate.path)
        # If the best candidate didn't complete the full sequence, extend with highest probabilities
        for step in range(best_candidate.timestamp, n_timestamps):
            best_idx = np.argmax(softmax_matrix[:, step])
            predicted_sequence.append(str(best_idx))
    else:
        # Complete fallback - predict highest probability activity for each step
        predicted_sequence = []
        for step in range(n_timestamps):
            best_idx = np.argmax(softmax_matrix[:, step])
            predicted_sequence.append(str(best_idx))
    
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
      - log moves: log insertion (advance timestamp & path)
      - tau moves: silent model moves (advance marking only)
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

        # 1) handle sync/log moves via softmax indices
        for idx, raw_p in enumerate(probs):
            if raw_p < activity_prob_threshold:
                continue

            act = str(idx)
            p = (
                compute_conditional_probability(
                    cand.path, act, raw_p,
                    prob_dict, lambdas, alpha, use_ngram_smoothing
                )
                if use_cond_probs and prob_dict else raw_p
            )

            if act in sync_trans:
                # synchronous move
                move_type = "sync"
                next_path = cand.path + (act,)
                next_timestamp = cand.timestamp + 1
                next_marking = model._fire_transition(cand.marking, sync_trans[act])
            else:
                # log move: insert log event without model firing
                move_type = "log"
                next_path = cand.path + (act,)
                next_timestamp = cand.timestamp + 1
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

        # 2) handle tau transitions separately (silent model moves)
        for trans in tau_trans.values():
            next_marking = model._fire_transition(cand.marking, trans)
            tau_cost = cost_fn(0.0, "tau")
            new_candidates.append(
                BeamCandidate(
                    path=cand.path,
                    cumulative_cost=cand.cumulative_cost + tau_cost,
                    marking=next_marking,
                    timestamp=cand.timestamp
                )
            )

        # 3) handle pure model moves (model insertion)
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
