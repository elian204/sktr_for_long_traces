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
import numpy as np
from classes import PetriNet
from utils import compute_conditional_probability, simple_bigram_blend
import logging

logger = logging.getLogger(__name__)


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
                 timestamp: int,
                 move_costs: List[float] = None):
        self.path = path
        self.cumulative_cost = cumulative_cost
        self.marking = marking
        self.timestamp = timestamp
        self.move_costs = [] if move_costs is None else move_costs

    def __repr__(self) -> str:
        marking_repr = self.marking.places if hasattr(self.marking, 'places') else self.marking
        return (f"BeamCandidate(path={self.path}, "
                f"cost={self.cumulative_cost:.4f}, "
                f"marking={marking_repr}, "
                f"timestamp={self.timestamp}, "
                f"move_costs={self.move_costs})")

    def copy(self) -> 'BeamCandidate':
        """Create a copy of this candidate."""
        return BeamCandidate(
            path=self.path,
            cumulative_cost=self.cumulative_cost,
            marking=self.marking,
            timestamp=self.timestamp,
            move_costs=self.move_costs[:]
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
                 epsilon: float = 1e-3,
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
        return self.alpha * total_cost + (1 - self.alpha) * avg_cost

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
    beam_score_alpha: float = 0.5,
    completion_patience: int = 5,
    lookahead_window: int = 5,
    beta: float = 0.0
) -> Tuple[List[str], List[float]]:
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
    beam_score_alpha : float, default=0.5
        Interpolation weight for beam scoring (normalized vs total cost)
    completion_patience : int, default=5
        Number of extra iterations to continue after first completion to find potentially better paths.
        
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
    
    if model.final_mark is None:
        raise ValueError("Model must have a valid final marking (final_mark)")
    
    # Initialize beam search
    beam_state = BeamState(beam_width, alpha=beam_score_alpha)
    n_timestamps = softmax_matrix.shape[1]
    
    # Initialize with empty path (zero initial cost)
    initial_candidate = BeamCandidate(
        path=(),
        cumulative_cost=0.0,
        marking=model.init_mark,
        timestamp=0,
        move_costs=[]
    )
    beam_state.add_candidate(initial_candidate)
    
    # Main beam search loop - continue until all candidates complete the sequence
    max_iterations = n_timestamps * 10  # Safety limit to prevent infinite loops
    iteration = 0
    all_completed: List[BeamCandidate] = []
    patience_counter = 0
    first_completion = False
    
    prev_beam_state_keys = None

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
            lookahead_window=lookahead_window,
            beta=beta
        )
        
        # --- Deduplication Step ---
        # Group candidates by their state (marking + timestamp) and keep only the best one.
        best_candidates_per_state: Dict[Tuple, BeamCandidate] = {}
        for cand in new_candidates:
            state_key = (cand.marking.places, cand.timestamp)
            if state_key not in best_candidates_per_state or \
               cand.cumulative_cost < best_candidates_per_state[state_key].cumulative_cost:
                best_candidates_per_state[state_key] = cand
        
        beam_state.candidates = list(best_candidates_per_state.values())
        beam_state.prune_beam()
        

        # --- Stall Detection & Completion Checks ---
        
        # A candidate is complete if it has processed the entire trace.
        completed_this_step = [c for c in beam_state.candidates if c.timestamp >= n_timestamps]
        if completed_this_step:
            if not first_completion:
                first_completion = True
            all_completed.extend(completed_this_step)

        # The active beam for the next iteration contains only unfinished candidates.
        active_beam = [c for c in beam_state.candidates if c.timestamp < n_timestamps]
        
        # Check for stalls only on the active part of the beam.
        current_beam_state_keys = {(c.marking.places, c.timestamp) for c in active_beam}
        if prev_beam_state_keys is not None and current_beam_state_keys == prev_beam_state_keys:
            logger.warning(f"Beam search stalled on active candidates at iteration {iteration}. Terminating.")
            break
        prev_beam_state_keys = current_beam_state_keys
        
        beam_state.candidates = active_beam
        
        # Patience check: if we've found a completed candidate, wait a few steps for better ones.
        if first_completion:
            patience_counter += 1
            if patience_counter >= completion_patience:
                break
        
        # If the active beam is empty, there's nothing more to expand.
        if not beam_state.candidates:
            logger.info("Active beam is empty. Stopping search.")
            break
    
    # --- Final Selection ---
    if all_completed:
        logger.info(f"Beam search finished. Found {len(all_completed)} candidates that completed the trace.")
        # Score all completed candidates and return the path of the best one.
        best_completed = min(all_completed, key=lambda c: beam_state._beam_score(c))
        return list(best_completed.path), list(best_completed.move_costs)
    
    # Fallback if we exit the loop without any candidate ever completing the trace.
    logger.warning(
        f"Beam search failed to find any path that completes the trace after {iteration} iterations. "
        "Falling back to greedy prediction."
    )
    return _generate_fallback_sequence(beam_state, softmax_matrix, n_timestamps, cost_fn)


def _generate_fallback_sequence(
    beam_state: BeamState,
    softmax_matrix: np.ndarray,
    n_timestamps: int,
    cost_fn: Callable[[float, str], float]
) -> Tuple[List[str], List[float]]:
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
        predicted_costs = list(best_candidate.move_costs)
        # If the best candidate didn't complete the full sequence, extend with highest probabilities
        for step in range(best_candidate.timestamp, n_timestamps):
            max_p = np.max(softmax_matrix[:, step])
            assumed_cost = cost_fn(max_p, "log")  # Assume log move cost for fallback
            predicted_costs.append(assumed_cost)
            best_idx = np.argmax(softmax_matrix[:, step])
            predicted_sequence.append(str(best_idx))
    else:
        # Complete fallback - predict highest probability activity for each step
        predicted_sequence = []
        predicted_costs = []
        for step in range(n_timestamps):
            max_p = np.max(softmax_matrix[:, step])
            assumed_cost = cost_fn(max_p, "log")
            predicted_costs.append(assumed_cost)
            best_idx = np.argmax(softmax_matrix[:, step])
            predicted_sequence.append(str(best_idx))
    
    return predicted_sequence, predicted_costs


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
    lookahead_window: int,
    beta: float
) -> List[BeamCandidate]:
    """
    Expand each beam candidate by all valid next transitions:
      - sync moves: match log & model (advance timestamp & path)
      - log moves: log insertion (advance timestamp & path)
      - model moves: model insertion (advance marking only)
    Compute adjusted probabilities, dispatch cost_fn by move_type.
    """
    new_candidates: List[BeamCandidate] = []

    for cand in beam_state.candidates:
        # Get all available transitions from the current marking.
        available = model._find_available_transitions(cand.marking.places)
        sync_trans = {t.label: t for t in available if t.label is not None}

        predicted_acts = set()
        if cand.timestamp < softmax_matrix.shape[1]:
            # Raw softmax probs at this timestamp
            probs = softmax_matrix[:, cand.timestamp]

            # 1) handle sync/log moves via softmax indices
            for idx, raw_p in enumerate(probs):
                if raw_p < activity_prob_threshold:
                    continue

                act = str(idx)
                predicted_acts.add(act)
                available_future = softmax_matrix.shape[1] - cand.timestamp - 1
                future_avg = 0.0
                if available_future > 0 and lookahead_window > 0 and beta > 0:
                    end_idx = cand.timestamp + 1 + min(lookahead_window, available_future)
                    future_avg = np.mean(softmax_matrix[idx, cand.timestamp + 1 : end_idx])

                p = (
                    simple_bigram_blend(
                        cand.path, act, raw_p,
                        prob_dict, alpha,
                        future_avg=future_avg,
                        beta=beta
                    )
                    if use_cond_probs and prob_dict else raw_p
                )

                if not (use_cond_probs and prob_dict) and beta > 0 and future_avg > 0:
                    p = (1 - beta) * p + beta * future_avg

                # ALWAYS consider a log move for any activity with sufficient probability
                log_move_cost = cost_fn(p, "log")
                new_candidates.append(
                    BeamCandidate(
                        path=cand.path + (act,),
                        cumulative_cost=cand.cumulative_cost + log_move_cost,
                        marking=cand.marking,
                        timestamp=cand.timestamp + 1,
                        move_costs=cand.move_costs[:] + [log_move_cost]
                    )
                )

                # IF the activity is also a valid synchronous move, consider that path too
                if act in sync_trans:
                    if prob_dict and len(cand.path) > 0:
                        prev_act = cand.path[-1]
                        bigram = (prev_act,)
                        if bigram in prob_dict and act in prob_dict[bigram]:
                            sync_move_cost = cost_fn(p, "sync")
                        else:
                            sync_move_cost = cost_fn(0.0, "model")
                    else:
                        sync_move_cost = cost_fn(p, "sync")
                    next_marking_sync = model._fire_macro_transition(cand.marking, sync_trans[act])
                    new_candidates.append(
                        BeamCandidate(
                            path=cand.path + (act,),
                            cumulative_cost=cand.cumulative_cost + sync_move_cost,
                            marking=next_marking_sync,
                            timestamp=cand.timestamp + 1,
                            move_costs=cand.move_costs[:] + [sync_move_cost]
                        )
                    )

            # 2) handle pure model moves (model insertion)
            for trans in sync_trans.values():
                # Use fire_macro_transition to correctly handle any required tau-path
                next_marking = model._fire_macro_transition(cand.marking, trans)
                model_cost = cost_fn(0.0, "model")
                new_candidates.append(
                    BeamCandidate(
                        path=cand.path,
                        cumulative_cost=cand.cumulative_cost + model_cost,
                        marking=next_marking,
                        timestamp=cand.timestamp,
                        move_costs=cand.move_costs[:]
                    )
                )

    return new_candidates
