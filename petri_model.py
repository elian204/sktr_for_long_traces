"""
Petri net discovery and conversion utilities.
"""

from collections import Counter, defaultdict
import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pm4py

from classes import Arc, Marking, PetriNet, Place, Transition


def discover_petri_net(
    train_df: pd.DataFrame,
    non_sync_penalty: float = 1.0
) -> PetriNet:
    """
    Discover an inductive Petri net from training traces.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame with columns 'case:concept:name' and 'concept:name'
    non_sync_penalty : float
        Penalty weight for non-synchronous transitions
        
    Returns
    -------
    PetriNet
        Discovered Petri net converted to internal format
    """
    prep_df = prepare_df_for_discovery(train_df)
    
    net, init_marking, final_marking = pm4py.discover_petri_net_inductive(prep_df)
    
    model = convert_pm4py_to_petrinet(
        discovered_model=net,
        non_sync_penalty=non_sync_penalty,
        pm4py_init_marking=init_marking,
        pm4py_final_marking=final_marking
    )
    
    return model


def prepare_df_for_discovery(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare DataFrame columns for pm4py discovery.
    
    Adds required columns: 'order' and 'time:timestamp' if missing.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with event log data
        
    Returns
    -------
    pd.DataFrame
        DataFrame prepared for pm4py discovery
    """
    df_copy = df.copy()
    
    # Add order column for event ordering within cases
    df_copy['order'] = df_copy.groupby('case:concept:name').cumcount()
    
    # Ensure timestamp column exists
    if 'time:timestamp' in df_copy.columns:
        df_copy['time:timestamp'] = pd.to_datetime(df_copy['time:timestamp'])
    else:
        # Use order as proxy timestamp if no timestamp exists
        df_copy['time:timestamp'] = pd.to_datetime(df_copy['order'])
    
    return df_copy


def convert_pm4py_to_petrinet(
    discovered_model: Any,
    non_sync_penalty: float = 1.0,
    name: str = 'discovered_model',
    cost_function: Optional[Any] = None,
    conditioned_prob_compute: bool = False,
    quiet_moves_weight: float = 1e-8,
    pm4py_init_marking: Optional[Any] = None,
    pm4py_final_marking: Optional[Any] = None
) -> PetriNet:
    """
    Convert a pm4py discovered model to internal PetriNet format.
    
    Parameters
    ----------
    discovered_model : pm4py.PetriNet
        The pm4py discovered Petri net
    non_sync_penalty : float, default=1.0
        Weight penalty for non-synchronous moves
    name : str, default='discovered_net'
        Name for the new PetriNet
    cost_function : callable, optional
        Custom cost function for the PetriNet
    conditioned_prob_compute : bool, default=False
        Whether to enable conditional probability computation
    quiet_moves_weight : float, default=1e-8
        Weight for silent transitions (tau moves)
    pm4py_init_marking : pm4py.Marking, optional
        Initial marking from pm4py
    pm4py_final_marking : pm4py.Marking, optional
        Final marking from pm4py
        
    Returns
    -------
    PetriNet
        Converted PetriNet in internal format
    """
    # Convert places with proper ordering
    places = _convert_places(discovered_model.places)
    place_mapping = _create_place_mapping(discovered_model.places, places)
    
    # Convert transitions
    transitions = _convert_transitions(
        discovered_model.transitions,
        non_sync_penalty,
        quiet_moves_weight
    )
    
    # Convert arcs and establish connections
    arcs = _convert_arcs(discovered_model.transitions, places, transitions)
    
    # Create and configure PetriNet
    petri_net = _create_configured_petrinet(
        name=name,
        places=places,
        transitions=transitions,
        arcs=arcs,
        cost_function=cost_function,
        conditioned_prob_compute=conditioned_prob_compute,
        discovered_model=discovered_model,
        pm4py_init_marking=pm4py_init_marking,
        pm4py_final_marking=pm4py_final_marking,
        place_mapping=place_mapping
    )
    
    return petri_net


def _convert_places(pm4py_places: List[Any]) -> List[Place]:
    """Convert pm4py places to internal Place objects with proper ordering."""
    sorted_places = _sort_places_by_type(pm4py_places)
    return [Place(p.name) for p in sorted_places]


def _sort_places_by_type(places: List[Any]) -> List[Any]:
    """
    Sort places: source first, then inner places, then sink.
    
    This ensures proper marking structure for the Petri net.
    """
    source_places = [p for p in places if p.name == 'source']
    sink_places = [p for p in places if p.name == 'sink']
    inner_places = [p for p in places if p.name not in {'source', 'sink'}]
    
    # Sort inner places numerically if they follow pattern like 'p_1', 'p_2', etc.
    try:
        inner_places_sorted = sorted(
            inner_places, 
            key=lambda x: float(x.name.split('_')[-1]) if '_' in x.name else 0
        )
    except (ValueError, IndexError):
        # Fallback to alphabetical sorting if numeric parsing fails
        inner_places_sorted = sorted(inner_places, key=lambda x: x.name)
    
    return source_places + inner_places_sorted + sink_places


def _create_place_mapping(pm4py_places: List[Any], internal_places: List[Place]) -> Dict[Any, int]:
    """Create mapping from pm4py places to internal place indices."""
    sorted_pm4py_places = _sort_places_by_type(pm4py_places)
    return {old_place: idx for idx, old_place in enumerate(sorted_pm4py_places)}


def _convert_transitions(
    pm4py_transitions: List[Any],
    non_sync_penalty: float,
    quiet_moves_weight: float
) -> List[Transition]:
    """Convert pm4py transitions to internal Transition objects."""
    transitions = []
    
    for pm4py_trans in pm4py_transitions:
        # Determine weight based on whether transition is silent (tau)
        weight = quiet_moves_weight if pm4py_trans.label is None else non_sync_penalty
        
        transition = Transition(
            name=pm4py_trans.name,
            label=pm4py_trans.label,
            in_arcs=set(),
            out_arcs=set(),
            move_type='model',
            weight=weight
        )
        
        # Use label as name if available (for better readability)
        if transition.label is not None:
            transition.name = transition.label
            
        transitions.append(transition)
    
    return transitions


def _convert_arcs(
    pm4py_transitions: List[Any],
    places: List[Place],
    transitions: List[Transition]
) -> List[Arc]:
    """Convert pm4py arcs and establish connections between places and transitions."""
    # Create lookup dictionaries for efficient access
    place_dict = {p.name: p for p in places}
    trans_dict = {t.name: t for t in transitions}
    
    arcs = []
    
    for pm4py_trans, internal_trans in zip(pm4py_transitions, transitions):
        # Convert input arcs (place -> transition)
        for pm4py_arc in pm4py_trans.in_arcs:
            source_place = place_dict[pm4py_arc.source.name]
            arc = Arc(source_place, internal_trans)
            arcs.append(arc)
            internal_trans.in_arcs.add(arc)
        
        # Convert output arcs (transition -> place)
        for pm4py_arc in pm4py_trans.out_arcs:
            target_place = place_dict[pm4py_arc.target.name]
            arc = Arc(internal_trans, target_place)
            arcs.append(arc)
            internal_trans.out_arcs.add(arc)
    
    return arcs


def _create_configured_petrinet(
    name: str,
    places: List[Place],
    transitions: List[Transition],
    arcs: List[Arc],
    cost_function: Optional[Any],
    conditioned_prob_compute: bool,
    discovered_model: Any,
    pm4py_init_marking: Optional[Any],
    pm4py_final_marking: Optional[Any],
    place_mapping: Dict[Any, int]
) -> PetriNet:
    """Create and configure the final PetriNet object."""
    # Create PetriNet and add components
    petri_net = PetriNet(name)
    petri_net.add_places(places)
    petri_net.add_transitions(transitions)
    petri_net.arcs = arcs
    
    # Set initial and final markings
    # Default: token in first place initially, token in last place finally
    num_places = len(places)
    petri_net.init_mark = Marking((1,) + (0,) * (num_places - 1))
    petri_net.final_mark = Marking((0,) * (num_places - 1) + (1,))
    
    # Set configuration
    petri_net.cost_function = cost_function
    petri_net.conditioned_prob_compute = conditioned_prob_compute
    
    # Store pm4py references for potential future use
    petri_net.pm4py_net = discovered_model
    petri_net.pm4py_initial_marking = pm4py_init_marking
    petri_net.pm4py_final_marking = pm4py_final_marking
    petri_net.place_mapping = place_mapping
    petri_net.reverse_place_mapping = {idx: place for place, idx in place_mapping.items()}
    
    # CRITICAL: Activate all optimizations
    petri_net.finalize()
    
    return petri_net


def build_probability_dict(
    train_df: pd.DataFrame,
    max_hist_len: int = 3,
    precision: int = 2
) -> Dict[Tuple[str, ...], Dict[str, float]]:
    """
    Build a conditional probability dictionary from training traces.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame with 'case:concept:name' and 'concept:name' columns
    max_hist_len : int, default=3
        Maximum history length for n-grams
    precision : int, default=2
        Decimal precision for probabilities
        
    Returns
    -------
    Dict[Tuple[str, ...], Dict[str, float]]
        Dictionary mapping history tuples to activity probabilities
    """
    return _build_conditioned_prob_dict(
        train_df, 
        max_hist_len=max_hist_len, 
        precision=precision
    )


def _build_conditioned_prob_dict(
    df_train: pd.DataFrame,
    max_hist_len: int = 2,
    precision: int = 2,
) -> Dict[Tuple[str, ...], Dict[str, float]]:
    """
    Build conditional probabilities P(activity | history) for all histories of
    length ≤ max_hist_len, using relative frequencies (no smoothing).

    Parameters
    ----------
    df_train : pd.DataFrame
        Log with 'case:concept:name' and 'concept:name'.
    max_hist_len : int, default=2
        Maximum history length for n-grams.
    precision : int, default=2
        Decimal places for probabilities.

    Returns
    -------
    Dict[history, Dict[next_activity, probability]]
    """
    # 1) Extract sequences per case
    activity_sequences = [
        group['concept:name'].tolist()
        for _, group in df_train.groupby('case:concept:name', sort=False)
    ]

    # 2) Count (history → next_activity) exactly as original
    history_counts: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for seq in activity_sequences:
        pairs = _get_histories_up_to_length_k(seq, max_hist_len)
        for history, activity in pairs:
            history_counts[history][activity] += 1

    # 3) Build frequency-based probabilities
    prob_dict: Dict[Tuple[str, ...], Dict[str, float]] = {}
    for history, counts in history_counts.items():
        total = sum(counts.values())
        if total <= 0:
            continue
        # normalize and round
        probs = {a: round(c / total, precision) for a, c in counts.items()}
        prob_dict[history] = probs

    return prob_dict


def _get_histories_up_to_length_k(
    activities_seq_list: List[str], 
    k: int
) -> List[Tuple[Tuple[str, ...], str]]:
    """
    Generate all possible (history, activity) pairs up to length k.
    
    Parameters
    ----------
    activities_seq_list : List[str]
        Sequence of activity names
    k : int
        Maximum history length
        
    Returns
    -------
    List[Tuple[Tuple[str, ...], str]]
        List of (history_tuple, next_activity) pairs
    """
    if not activities_seq_list:
        return []
    
    histories = []
    
    # Include the first activity with empty history
    histories.append(((), activities_seq_list[0]))
    
    # Generate histories for subsequent activities
    for i in range(1, len(activities_seq_list)):
        current_activity = activities_seq_list[i]
        
        # Try different history lengths up to k
        for j in range(1, min(i + 1, k + 1)):
            history = tuple(activities_seq_list[i-j:i])
            histories.append((history, current_activity))
    
    return histories