"""
Beam Search for Incremental Softmax Recovery.

This module implements beam search algorithms for incrementally recovering 
activity sequences from softmax probability matrices using Petri net models.
The beam search maintains multiple candidate paths and selects the most 
promising ones at each step based on probability scores and model constraints.

Main Functions:
    process_test_case_beam_search: Main entry point for processing a single test case using beam search
    
Helper Classes and Functions:
    BeamCandidate: Represents a candidate path in the beam
    BeamState: Manages the beam search state
    Various probability computation and ranking helpers
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Mapping, Sequence
import numpy as np
from classes import PetriNet
from utils import simple_bigram_blend
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


def process_test_case_beam_search(
    softmax_matrix: np.ndarray,
    model: "PetriNet",
    cost_fn: Callable[[float, str], float],
    beam_width: int,
    alpha: float,
    use_cond_probs: bool,
    prob_dict: Mapping[Any, Any],
    activity_prob_threshold: float,
    beam_score_alpha: float = 0.5,
    completion_patience: int = 5,
) -> Tuple[List[str], List[float]]:
    """
    Recover an activity sequence for a single trace via incremental beam search.

    The search expands candidates under Petri net constraints while optionally
    blending conditional probabilities (with n-gram smoothing) into the scoring.

    Parameters
    ----------
    softmax_matrix : np.ndarray
        Probability matrix shaped (n_activities, n_timestamps).
    model : PetriNet
        Petri net with valid initial and final markings (init_mark, final_mark).
    cost_fn : Callable[[float, str], float]
        Converts (probability, move_type/label) to a nonnegative cost.
    beam_width : int
        Maximum number of candidates kept in the beam after pruning.
    alpha : float
        Base/conditional blending parameter in [0, 1].
    use_cond_probs : bool
        Whether to incorporate conditional probabilities.
    prob_dict : Mapping
        Dictionary supplying conditional/transition probabilities.
    activity_prob_threshold : float
        Minimum per-step probability cutoff to consider an activity.
    beam_score_alpha : float, default=0.5
        Interpolation weight for beam scoring (normalized vs. total cost).
    completion_patience : int, default=5
        After the first complete candidate is found, continue this many
        iterations to allow potentially better completed paths to appear.

    Returns
    -------
    Tuple[List[str], List[float]]
        The best recovered activity sequence and its per-move costs.
    """
    # ---- Validation ---------------------------------------------------------
    if softmax_matrix.ndim != 2:
        raise ValueError(
            f"softmax_matrix must be 2D (n_activities, n_timestamps); "
            f"got ndim={softmax_matrix.ndim}"
        )
    if getattr(model, "init_mark", None) is None:
        raise ValueError("Model must define a valid initial marking: model.init_mark")
    if getattr(model, "final_mark", None) is None:
        raise ValueError("Model must define a valid final marking: model.final_mark")

    n_timestamps = int(softmax_matrix.shape[1])
    if n_timestamps == 0:
        # Degenerate trace: nothing to decode.
        return [], []

    # ---- Initialize beam ----------------------------------------------------
    beam_state = BeamState(beam_width, alpha=beam_score_alpha)
    beam_state.add_candidate(
        BeamCandidate(
            path=(),
            cumulative_cost=0.0,
            marking=model.init_mark,
            timestamp=0,
            move_costs=[],
        )
    )

    # Safety cap to avoid infinite loops if expansion stalls pathologically.
    max_iterations = n_timestamps * 10

    iteration = 0
    first_completion_found = False
    patience = 0
    completed: List["BeamCandidate"] = []
    prev_active_keys: "set[Tuple[Tuple[int, ...], int]]" = set()

    # ---- Main loop ----------------------------------------------------------
    while iteration < max_iterations:
        iteration += 1

        # Expand all current candidates in the beam.
        new_candidates = _expand_beam_candidates(
            beam_state=beam_state,
            softmax_matrix=softmax_matrix,
            model=model,
            cost_fn=cost_fn,
            use_cond_probs=use_cond_probs,
            prob_dict=prob_dict,
            alpha=alpha,
            activity_prob_threshold=activity_prob_threshold,
        )

        # Deduplicate by state (marking, timestamp): keep the minimum-cost candidate per state.
        best_by_state: Dict[Tuple[Tuple[int, ...], int], "BeamCandidate"] = {}
        for cand in new_candidates:
            key = (cand.marking.places, cand.timestamp)
            if key not in best_by_state or cand.cumulative_cost < best_by_state[key].cumulative_cost:
                best_by_state[key] = cand

        beam_state.candidates = list(best_by_state.values())
        beam_state.prune_beam()

        # Split completed vs active for the next round.
        done_now = [c for c in beam_state.candidates if c.timestamp >= n_timestamps]
        if done_now:
            completed.extend(done_now)
            if not first_completion_found:
                first_completion_found = True

        active = [c for c in beam_state.candidates if c.timestamp < n_timestamps]
        active_keys = {(c.marking.places, c.timestamp) for c in active}

        # Stall detection on active beam: identical frontier as previous iteration.
        if active and active_keys == prev_active_keys:
            logger.warning("Beam search stalled on active candidates. Terminating.")
            break
        prev_active_keys = active_keys

        beam_state.candidates = active

        # If we have at least one completed path, allow a few more rounds.
        if first_completion_found:
            patience += 1
            if patience >= completion_patience:
                break

        if not beam_state.candidates:
            logger.info("Active beam is empty. Stopping search.")
            break

    # ---- Final selection / fallback ----------------------------------------
    if completed:
        logger.info("Beam search finished with %d completed candidates.", len(completed))
        best = min(completed, key=lambda c: beam_state._beam_score(c))
        return list(best.path), list(best.move_costs)

    logger.warning(
        "Beam search failed to complete after %d iterations; falling back to greedy.",
        iteration,
    )
    return _generate_fallback_sequence(beam_state, softmax_matrix, n_timestamps, cost_fn)


def process_test_case_hybrid(
    softmax_matrix: np.ndarray,
    model: "PetriNet",
    cost_fn: Callable[[float, str], float],
    beam_width: int,
    alpha: float,
    use_cond_probs: bool,
    prob_dict: Mapping[Any, Any],
    activity_prob_threshold: float,
    beam_score_alpha: float = 0.5,
    completion_patience: int = 5,
    conformance_chunk_size: int = 10,
    conformance_eps: float = 1e-12,
    decider: Optional[Callable[[int, np.ndarray, BeamCandidate, "PetriNet"], Optional[int]]] = None,
    # Agreement-based conformance settings
    min_disagree_run: int = 1,
) -> Tuple[List[str], List[float]]:
    """
    Hybrid recovery: run beam search, but when the provided `decider` indicates,
    switch to conformance for a chunk and then resume beam search from the updated state.

    Parameters mirror `process_test_case_beam_search`, with extra:
    - conformance_chunk_size: default fallback chunk size if decider returns None/invalid
    - conformance_eps: probability threshold for conformance filtering
    - decider: function(timestamp, remaining_softmax, best_candidate, model) -> Optional[int]
        Return a positive integer chunk length to trigger conformance from the current timestamp.
        Return None/0 to continue beam search.
    """
    # ---- Validation ---------------------------------------------------------
    if softmax_matrix.ndim != 2:
        raise ValueError(
            f"softmax_matrix must be 2D (n_activities, n_timestamps); got ndim={softmax_matrix.ndim}"
        )
    if getattr(model, "init_mark", None) is None:
        raise ValueError("Model must define a valid initial marking: model.init_mark")
    if getattr(model, "final_mark", None) is None:
        raise ValueError("Model must define a valid final marking: model.final_mark")

    n_timestamps = int(softmax_matrix.shape[1])
    if n_timestamps == 0:
        return [], []

    # ---- Initialize beam ----------------------------------------------------
    beam_state = BeamState(beam_width, alpha=beam_score_alpha)
    beam_state.add_candidate(
        BeamCandidate(
            path=(),
            cumulative_cost=0.0,
            marking=model.init_mark,
            timestamp=0,
            move_costs=[],
        )
    )

    max_iterations = n_timestamps * 10
    iteration = 0
    first_completion_found = False
    patience = 0
    completed: List["BeamCandidate"] = []
    prev_active_keys: "set[Tuple[Tuple[int, ...], int]]" = set()

    # Pre-compute argmax labels for agreement tracking
    argmax_labels = [str(int(i)) for i in np.argmax(softmax_matrix, axis=0)]
    # Track contiguous runs of disagreement between best path and argmax
    disagree_start: Optional[int] = None
    candidate_at_disagree_start: Optional[BeamCandidate] = None

    # Helper: apply conformance on the best candidate for a given chunk length
    def _apply_conformance(best: BeamCandidate, chunk_len: int) -> BeamCandidate:
        nonlocal completed
        if chunk_len <= 0:
            chunk_len = conformance_chunk_size
        start_ts = best.timestamp
        end_ts = min(start_ts + chunk_len, n_timestamps)
        if start_ts >= end_ts:
            return best

        chunk = softmax_matrix[:, start_ts:end_ts]
        # Run partial conformance from current marking
        result = model.partial_trace_conformance(
            softmax_matrix=chunk,
            initial_marking=best.marking,
            cost_fn=cost_fn,
            eps=conformance_eps,
        )

        # Update candidate with alignment results
        new_path: List[str] = list(best.path)
        new_move_costs: List[float] = list(best.move_costs)
        new_cost = best.cumulative_cost
        new_timestamp = best.timestamp
        for move_type, move_label, move_cost in result["alignment"]:
            new_cost += float(move_cost)
            # Only sync/log advance the trace and contribute to predicted sequence
            if move_type in ("sync", "log"):
                new_path.append(str(move_label))
                new_move_costs.append(float(move_cost))
                new_timestamp += 1

        updated = BeamCandidate(
            path=tuple(new_path),
            cumulative_cost=new_cost,
            marking=result["final_marking"],
            timestamp=new_timestamp,
            move_costs=new_move_costs,
        )
        # If completed via conformance, remember it
        if updated.timestamp >= n_timestamps:
            completed.append(updated)
        return updated

    # ---- Main loop ----------------------------------------------------------
    while iteration < max_iterations:
        iteration += 1

        # Snapshot current best before any expansion to detect where a run starts
        best_before = beam_state.get_best_candidate()
        if best_before is None:
            logger.info("Active beam is empty. Stopping hybrid search.")
            break

        # Optionally switch to conformance for a chunk based on external decider
        if decider is not None:
            try:
                remaining = softmax_matrix[:, best_before.timestamp:]
                chunk_len = decider(best_before.timestamp, remaining, best_before, model)
            except Exception as exc:
                logger.warning(f"Hybrid decider errored ({exc}); continuing with beam search.")
                chunk_len = None
            if isinstance(chunk_len, int) and chunk_len > 0:
                # Apply conformance to best candidate and continue
                updated_best = _apply_conformance(best_before, chunk_len)
                beam_state.candidates = [updated_best]
                # Reset any disagreement tracking since timeline changed
                disagree_start = None
                candidate_at_disagree_start = None
                if updated_best.timestamp >= n_timestamps:
                    first_completion_found = True
                if first_completion_found:
                    patience += 1
                    if patience >= completion_patience:
                        break
                continue

        # Otherwise run a normal beam expansion step
        new_candidates = _expand_beam_candidates(
            beam_state=beam_state,
            softmax_matrix=softmax_matrix,
            model=model,
            cost_fn=cost_fn,
            use_cond_probs=use_cond_probs,
            prob_dict=prob_dict,
            alpha=alpha,
            activity_prob_threshold=activity_prob_threshold,
        )

        best_by_state: Dict[Tuple[Tuple[int, ...], int], "BeamCandidate"] = {}
        for cand in new_candidates:
            key = (cand.marking.places, cand.timestamp)
            if key not in best_by_state or cand.cumulative_cost < best_by_state[key].cumulative_cost:
                best_by_state[key] = cand

        beam_state.candidates = list(best_by_state.values())
        beam_state.prune_beam()

        done_now = [c for c in beam_state.candidates if c.timestamp >= n_timestamps]
        if done_now:
            completed.extend(done_now)
            if not first_completion_found:
                first_completion_found = True

        active = [c for c in beam_state.candidates if c.timestamp < n_timestamps]
        active_keys = {(c.marking.places, c.timestamp) for c in active}
        if active and active_keys == prev_active_keys:
            logger.warning("Hybrid beam search stalled on active candidates. Terminating.")
            break
        prev_active_keys = active_keys
        beam_state.candidates = active

        # Agreement-based conformance: detect disagreement runs and align them via conformance
        best_after = beam_state.get_best_candidate()
        if best_after is not None:
            # Check if we advanced in timestamps this round (i.e., produced a new label)
            if best_after.timestamp > best_before.timestamp:
                step_idx = best_after.timestamp - 1  # last produced timestamp
                if 0 <= step_idx < n_timestamps:
                    predicted_label = best_after.path[-1] if best_after.path else None
                    argmax_label = argmax_labels[step_idx]
                    if predicted_label is not None and predicted_label != argmax_label:
                        # Start of a disagreement run
                        if disagree_start is None:
                            disagree_start = step_idx
                            candidate_at_disagree_start = best_before.copy()
                    else:
                        # We are in agreement at this step; if a run was active, close and align it
                        if disagree_start is not None:
                            run_len = step_idx - disagree_start
                            if run_len >= min_disagree_run:
                                updated_best = _apply_conformance(candidate_at_disagree_start, run_len)
                                beam_state.candidates = [updated_best]
                                # If updated completes, mark completion
                                if updated_best.timestamp >= n_timestamps:
                                    completed.append(updated_best)
                                    first_completion_found = True
                                # Reset tracking
                                disagree_start = None
                                candidate_at_disagree_start = None
                            else:
                                # Run too short; discard tracking
                                disagree_start = None
                                candidate_at_disagree_start = None

            # If we reached the end while in a disagreement run, align the tail
            if best_after.timestamp >= n_timestamps and disagree_start is not None:
                run_len_tail = n_timestamps - disagree_start
                if run_len_tail >= min_disagree_run:
                    updated_best = _apply_conformance(candidate_at_disagree_start, run_len_tail)
                    beam_state.candidates = [updated_best]
                    if updated_best.timestamp >= n_timestamps:
                        completed.append(updated_best)
                        first_completion_found = True
                # Reset tracking regardless
                disagree_start = None
                candidate_at_disagree_start = None

        if first_completion_found:
            patience += 1
            if patience >= completion_patience:
                break

        if not beam_state.candidates:
            logger.info("Active beam is empty. Stopping hybrid search.")
            break

    # ---- Final selection / fallback ----------------------------------------
    if completed:
        logger.info("Hybrid search finished with %d completed candidates.", len(completed))
        best_final = min(completed, key=lambda c: beam_state._beam_score(c))
        return list(best_final.path), list(best_final.move_costs)

    logger.warning(
        "Hybrid search failed to complete after %d iterations; falling back to greedy.",
        iteration,
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
    beam_state: "BeamState",
    softmax_matrix: np.ndarray,
    model: "PetriNet",
    cost_fn: Callable[[float, str], float],
    use_cond_probs: bool,
    prob_dict: Mapping[Tuple[str, ...], Mapping[str, float]],
    alpha: float,
    activity_prob_threshold: float,
) -> List["BeamCandidate"]:
    """
    Expand each beam candidate by all valid next transitions at its current timestamp.

    Moves
    -----
    - log  : advance timestamp & path; marking unchanged
    - sync : advance timestamp & path; marking advanced via `_fire_macro_transition`
    - model: advance marking only; timestamp & path unchanged

    Notes
    -----
    - Activity labels are assumed to be `str(index)` for the softmax row index.
      Ensure your Petri net transition labels follow this convention or adapt upstream.
    - By design, model moves update `cumulative_cost` but do not append to `move_costs`
      (to keep per-step cost history aligned to timestamps). Change if you want per-move accounting.
    

    Returns
    -------
    List[BeamCandidate]
        Unpruned list of new candidates (may contain multiple candidates per state).
    """
    new_candidates: List["BeamCandidate"] = []
    n_ts = int(softmax_matrix.shape[1])

    for cand in beam_state.candidates:
        t = cand.timestamp
        if t >= n_ts:
            # No expansion once we've consumed all timestamps.
            continue

        # Map available labeled transitions at the current marking.
        available = model._find_available_transitions(cand.marking.places)
        sync_by_label = {tr.label: tr for tr in available if tr.label is not None}

        # Raw probabilities at the current timestamp.
        probs = softmax_matrix[:, t]

        # Consider only activities above threshold for log/sync moves.
        eligible_idxs = np.flatnonzero(probs >= activity_prob_threshold)
        for idx in eligible_idxs:
            raw_p = float(probs[int(idx)])
            act = str(int(idx))

            if use_cond_probs and prob_dict:
                p = simple_bigram_blend(cand.path, act, raw_p, prob_dict, alpha)
            else:
                p = raw_p

            # Log move (always allowed)
            log_cost = cost_fn(p, "log")
            new_candidates.append(
                BeamCandidate(
                    path=cand.path + (act,),
                    cumulative_cost=cand.cumulative_cost + log_cost,
                    marking=cand.marking,
                    timestamp=t + 1,
                    move_costs=cand.move_costs + [log_cost],
                )
            )

            # Sync move (allowed only if a matching labeled transition exists)
            tr = sync_by_label.get(act)
            if tr is not None:
                sync_cost = cost_fn(p, "sync")
                next_marking = model._fire_macro_transition(cand.marking, tr)
                new_candidates.append(
                    BeamCandidate(
                        path=cand.path + (act,),
                        cumulative_cost=cand.cumulative_cost + sync_cost,
                        marking=next_marking,
                        timestamp=t + 1,
                        move_costs=cand.move_costs + [sync_cost],
                    )
                )

        # Model moves: explore model insertions (timestamp unchanged)
        for tr in sync_by_label.values():
            next_marking = model._fire_macro_transition(cand.marking, tr)
            model_cost = cost_fn(0.0, "model")
            new_candidates.append(
                BeamCandidate(
                    path=cand.path,
                    cumulative_cost=cand.cumulative_cost + model_cost,
                    marking=next_marking,
                    timestamp=t,
                    move_costs=cand.move_costs,  # intentionally unchanged
                )
            )

    return new_candidates