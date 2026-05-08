"""
Core classes for Petri net modeling and process conformance checking.

This module provides the fundamental data structures and algorithms for:
- Petri net representation (places, transitions, arcs, markings)
- Reachability graph construction and exploration
- Synchronous product construction for alignment-based conformance checking
- Chunked conformance checking for softmax trace recovery
- Support for partial conformance checking and trace recovery

Main Classes:
    Place, Transition, Arc: Basic Petri net components
    Marking: Token distribution representation
    PetriNet: Main Petri net class with reachability analysis and conformance checking
    Graph, Node, Edge: Graph representation structures
    SearchNode: Search state for alignment algorithms

The module supports probabilistic conformance checking with conditional
probability-based cost functions and n-gram smoothing for improved alignment quality.
"""

import numpy as np
import copy
import logging
import time
import sys
from collections import deque, Counter, defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Any, Union
import heapq
from .utils import adjust_probs_with_sequence_context

# Configure logger
logger = logging.getLogger(__name__)


def _extend_path_prefix_bounded(
    current_prefix: Tuple[str, ...],
    new_label: str,
    max_distinct_labels: int
) -> Tuple[str, ...]:
    """
    Extend path_prefix with a new label, keeping only enough history for conditioning.

    This stores a "collapsed" representation - consecutive runs of the same label
    are represented by a single entry. We keep at most max_distinct_labels entries
    to bound memory usage while preserving correctness for conditioning lookups.

    The get_run_context_labels_extended function needs to:
    1. Identify the current run label (last element)
    2. Find up to n_prev_labels previous DIFFERENT labels

    By storing collapsed runs, we preserve this information with bounded memory.

    Example with max_distinct_labels=4:
        ('A', 'B', 'C') + 'C' -> ('A', 'B', 'C')  # same label, no change
        ('A', 'B', 'C') + 'D' -> ('A', 'B', 'C', 'D')  # new label, append
        ('A', 'B', 'C', 'D') + 'E' -> ('B', 'C', 'D', 'E')  # new label, drop oldest
    """
    # Intern the label to ensure identical strings share memory across all nodes
    new_label = sys.intern(new_label)

    if not current_prefix:
        return (new_label,)

    # If new label matches the last label in prefix, no change needed (still same run)
    if current_prefix[-1] == new_label:
        return current_prefix

    # New distinct label - append it
    new_prefix = current_prefix + (new_label,)

    # If we exceed the limit, drop the oldest entry
    if len(new_prefix) > max_distinct_labels:
        new_prefix = new_prefix[1:]  # Drop first (oldest) element

    return new_prefix


class Place:
    def __init__(self, name, in_arcs=None, out_arcs=None, properties={}):
        self.name = name
        self.in_arcs = set() if in_arcs is None else in_arcs
        self.out_arcs = set() if out_arcs is None else out_arcs
        self.properties = properties
        
    def __repr__(self):
        return self.name
    
    
class Transition:
    def __init__(self, name, label, in_arcs=None, out_arcs=None, move_type=None, 
                 prob=None, weight=None, properties=None, cost_function=None):
        self.name = name
        self.label = label
        self.in_arcs = set() if in_arcs is None else in_arcs 
        self.out_arcs = set() if out_arcs is None else out_arcs
        self.move_type = move_type
        self.prob = prob
        self.cost_function = cost_function
        self.weight = self.__initialize_weight(weight)
        self.properties = properties or {}

    def prepare_fire(self, places_indices: dict) -> None:
        """Prepare for optimized operations."""
        # Precompute input arcs
        self.in_idx_weights = tuple(
            (places_indices[arc.source.name], arc.weight) 
            for arc in self.in_arcs
        )
        
        # Precompute output arcs
        self.out_idx_weights = tuple(
            (places_indices[arc.target.name], arc.weight) 
            for arc in self.out_arcs
        )
        
        # Check for weighted inputs
        self.has_weighted_inputs = any(w > 1 for _, w in self.in_idx_weights)
        
        # Store just indices for unweighted case (common, faster)
        if not self.has_weighted_inputs:
            self.in_indices = tuple(idx for idx, _ in self.in_idx_weights)

        # Precompute sparse net token deltas for firing.
        delta: Dict[int, int] = defaultdict(int)
        for idx, weight in self.in_idx_weights:
            delta[idx] -= weight
        for idx, weight in self.out_idx_weights:
            delta[idx] += weight
        self.delta_idx_weights = tuple(
            (idx, weight_delta)
            for idx, weight_delta in delta.items()
            if weight_delta != 0
        )
    
    def is_enabled_optimized(self, mark_tuple: Tuple[int, ...]) -> bool:
        """Ultra-fast enabled check."""
        if not self.in_idx_weights:
            return True  # No inputs = always enabled
        
        if self.has_weighted_inputs:
            # General case with weights
            for idx, weight in self.in_idx_weights:
                if mark_tuple[idx] < weight:
                    return False
        else:
            # Fast path for weight=1 (common case)
            for idx in self.in_indices:
                if mark_tuple[idx] < 1:
                    return False
        return True

    def __repr__(self):
        return self.name
    
    def __initialize_weight(self, weight):
        """Calculate the weight for this transition based on probability and cost function."""
        if self.prob == 0:
            raise ValueError("Probability cannot be zero.")
        
        # If weight is explicitly provided, use it
        if weight is not None:
            return weight
        
        # If no cost function, use default cost based on move_type
        if self.cost_function is None:
            return 1e-6 if self.move_type == 'sync' else 1
        
        # Try to call cost_function with both parameters, fallback to prob only
        try:
            return self.cost_function(self.prob, self.move_type)
        except TypeError:
            # Fallback for older cost functions that only accept probability
            return self.cost_function(self.prob)
        
    
class Arc:
    def __init__(self, source, target, weight=1, properties={}):
        self.source = source
        self.target = target
        self.weight = weight
        self.properties = properties
        
    def __repr__(self):
        return self.source.name + ' -> ' + self.target.name 
    
    
class Marking:
    __slots__ = ('places',)
    def __init__(self, places=None):
        # Convert `places` to a tuple if it's not None, otherwise initialize an empty tuple
        if places is None:
            self.places = tuple()
        else:
            # Ensure `places` is a tuple. If it's not, convert it to a tuple.
            self.places = tuple(places) if not isinstance(places, tuple) else places
    
    def __repr__(self):
        return str(self.places)
    
    
class Node:
    def __init__(self, marking):
        self.marking = marking
        self.neighbors = set()
    
    def __repr__(self):
        return str(self.marking)
    
    def add_neighbor(self, node, transition):
        self.neighbors.add((node, transition)) 
        
        
class Edge:
    def __init__(self, name, source_marking, target_marking, move_type):
        self.name = name
        self.source_marking = source_marking
        self.target_marking = target_marking
        self.move_type = move_type
        
        
    def __repr__(self):
        return f'{self.source_marking} -> {self.name} -> {self.target_marking}'
    
    
class Graph:
    def __init__(self, nodes = None, edges = None, starting_node = None, ending_node = None):
        self.nodes = list() if nodes is None else nodes
        self.edges = list() if edges is None else edges
        self.starting_node = starting_node
        self.ending_node = ending_node
        self.nodes_indices = {}
        
    def __repr__(self):
        return f'Nodes:{self.nodes}, \n edges:{self.edges}'
    
    def __get_markings(self):
        return set([node.marking for node in self.nodes])
    
    def add_node(self, node):
        self.nodes.append(node)
        self.nodes_indices[node.marking] = len(self.nodes) - 1
        
    def add_edge(self, edge): 
        self.edges.append(edge)


class SearchNode:
    __slots__ = (
        'marking',
        'cost',
        'ancestor',
        'move_type',
        'move_label',
        'move_cost',
        'timestamp',
        'last_label',
        'path_prefix'
    )

    def __init__(
        self,
        marking: 'Marking',
        cost: float = float('inf'),
        ancestor: Optional['SearchNode'] = None,
        move_type: Optional[str] = None,
        move_label: Optional[str] = None,
        move_cost: float = 0.0,
        timestamp: int = 0,
        # Track last emitted label and full path prefix for conditional p_stay
        last_label: Optional[str] = None,
        path_prefix: Tuple[str, ...] = tuple(),
    ):
        self.cost = cost
        self.ancestor = ancestor
        self.move_type = move_type
        self.move_label = move_label
        self.move_cost = move_cost
        self.marking = marking
        self.timestamp = timestamp
        self.last_label = last_label
        self.path_prefix: Tuple[str, ...] = path_prefix

    def __lt__(self, other: 'SearchNode'):
        return self.cost < other.cost

    def reconstruct_path(self) -> List[Tuple[str, str, float]]:
        """Reconstruct the alignment as a list of (move_type, move_label, move_cost)."""
        seq: List[Tuple[str, str, float]] = []
        node = self
        while node.ancestor is not None:
            seq.append((node.move_type, node.move_label, node.move_cost))
            node = node.ancestor
        seq.reverse()
        return seq

    
class PetriNet:
    def __init__(self, name='net', places=None, transitions=None, arcs=None, properties=None, conditioned_prob_compute=False):
        """
        Initialize a Petri Net.
        
        Args:
            name: Name of the Petri net
            places: List of places (default: empty list)
            transitions: List of transitions (default: empty list) 
            arcs: List of arcs (default: empty list)
            properties: Dictionary of properties (default: empty dict)
            conditioned_prob_compute: Whether to compute conditioned probabilities
        """
        self.name = name
        self.transitions = list() if transitions is None else transitions
        self.places = list() if places is None else places
        self.arcs = list() if arcs is None else arcs
        self.properties = properties or {}
        
        # Initialize core attributes
        self.init_mark = None
        self.final_mark = None
        self.reachability_graph = None
        self.cost_function = None
        self.conditioned_prob_compute = conditioned_prob_compute
        self.alive_transitions_map = None

        # Build indices safely
        self._build_indices()
        self._finalized = False
        self._enabled_cache = {}
        self._cache_max_size = 10000
        self._use_cache = False  # Initialize caching as disabled by default

        # Memory optimization: lazy loading for marking transition map
        self._marking_transition_map = None
        self._marking_transition_map_max_tau = None
        self._allow_lazy_map_build = True  # Control whether lazy loading is allowed
        # Bound the size of the on-demand marking→tau-reachable cache.
        # When exceeded, we evict older entries and recompute as needed.
        self._marking_transition_map_cache_max_size = 20000

    @property
    def marking_transition_map(self):
        """Lazy-loaded marking transition map."""
        return self._marking_transition_map

    @marking_transition_map.setter
    def marking_transition_map(self, value):
        """Set the marking transition map."""
        self._marking_transition_map = value

    def finalize(self) -> None:
        """Prepare all transitions for optimized operations."""
        self._build_indices()  # Ensure indices are current
        
        for transition in self.transitions:
            transition.prepare_fire(self.places_indices)
        
        self._finalized = True
        self._enabled_cache.clear()

    def _build_indices(self):
        """Build lookup indices for places and transitions."""
        self.places_indices = {
            place.name: i for i, place in enumerate(self.places)
        }
        self.transitions_indices = {
            transition.name: i for i, transition in enumerate(self.transitions)
        }
    
    def construct_reachability_graph(self):   
        """Construct the reachability graph for this Petri net."""
        if self.init_mark is None:
            raise ValueError("Initial marking must be set before constructing reachability graph")
            
        # Initialize graph and starting node
        self.reachability_graph = self._initialize_graph()
        
        # Explore all reachable markings
        self._explore_reachable_markings()
    
    def _initialize_graph(self):
        """Initialize the reachability graph with starting node."""
        starting_node = Node(self.init_mark.places)
        graph = Graph()
        graph.add_node(starting_node)
        graph.starting_node = starting_node
        
        if self.final_mark is not None:
            graph.ending_node = Node(self.final_mark.places)
            
        return graph
    
    def _explore_reachable_markings(self):
        """Explore all reachable markings using breadth-first search."""
        exploration_queue = deque()
        visited_markings = set()
        
        # Initialize with transitions from initial marking
        starting_node = self.reachability_graph.starting_node
        initial_transitions = self._find_available_transitions(self.init_mark.places)
        
        for transition in initial_transitions:
            exploration_queue.append((self.init_mark, transition, starting_node))
            
        visited_markings.add(self.init_mark.places)

        # Process all nodes in the queue
        while exploration_queue:
            self._process_exploration_step(exploration_queue, visited_markings)
    
    def _process_exploration_step(self, exploration_queue, visited_markings):
        """Process a single step in the reachability graph exploration."""
        prev_marking, transition, prev_node = exploration_queue.popleft()
        
        # Verify transition can fire (assertion from original)
        assert self.__check_transition_prerequesits(transition, prev_marking.places) == True
        
        # Fire transition to get new marking
        new_marking = self._fire_transition(prev_marking, transition)
        
        # Get or create node for new marking
        current_node = self._get_or_create_node(new_marking, visited_markings)
        
        # Add connection between nodes
        self._connect_nodes(prev_node, current_node, transition, prev_marking, new_marking)
        
        # If marking is new, explore its transitions
        if new_marking.places not in visited_markings:
            self._add_new_marking_for_exploration(
                new_marking, current_node, exploration_queue, visited_markings
            )
    
    def _get_or_create_node(self, marking, visited_markings):
        """Get existing node or create new one for the given marking."""
        if marking.places in visited_markings:
            node_index = self.reachability_graph.nodes_indices[marking.places]
            return self.reachability_graph.nodes[node_index]
        else:
            return Node(marking.places)
    
    def _connect_nodes(self, prev_node, current_node, transition, prev_marking, new_marking):
        """Connect two nodes with transition and add edge to graph."""
        prev_node.add_neighbor(current_node, transition)
        edge = Edge(transition.name, prev_marking, new_marking, transition.move_type)
        self.reachability_graph.add_edge(edge)
    
    def _add_new_marking_for_exploration(self, marking, node, exploration_queue, visited_markings):
        """Add new marking to exploration queue and mark as visited."""
        # Add available transitions to exploration queue
        available_transitions = self._find_available_transitions(marking.places)
        for transition in available_transitions:
            exploration_queue.append((marking, transition, node))
        
        # Mark as visited and add node to graph
        visited_markings.add(marking.places)
        self.reachability_graph.add_node(node)
    
    def construct_synchronous_product(self, trace_model, cost_function):
        """
        Construct a synchronous product between model and trace.
        
        Assigns move types:
        - Model transitions: move_type=model
        - Trace transitions: move_type=trace  
        - Sync transitions: move_type=sync
        
        Args:
            trace_model: The trace model to synchronize with
            cost_function: Cost function for synchronous transitions
            
        Returns:
            PetriNet: The synchronous product Petri net
        """
        # Assign move types to transitions
        self._setup_move_types(trace_model)
        
        # Create synchronized components
        sync_components = self._create_sync_components(trace_model)
        
        # Build the synchronous product
        sync_product = self._build_sync_product(sync_components, trace_model, cost_function)
        
        return sync_product
    
    def _setup_move_types(self, trace_model):
        """Assign appropriate move types to model and trace transitions."""
        self.assign_model_transitions_move_type()   
        trace_model.assign_trace_transitions_move_type()
    
    def _create_sync_components(self, trace_model):
        """Create synchronized places, transitions, and arcs."""
        return {
            'places': copy.deepcopy(self.places + trace_model.places),
            'transitions': copy.deepcopy(self.transitions + trace_model.transitions),
            'arcs': copy.deepcopy(self.arcs + trace_model.arcs)
        }
    
    def _build_sync_product(self, components, trace_model, cost_function):
        """Build the final synchronous product Petri net."""
        # Generate synchronous transitions
        sync_transitions = self._generate_all_sync_transitions(trace_model, cost_function)
        
        # Create the synchronous product net
        sync_product = PetriNet('sync_prod', 
                               components['places'], 
                               components['transitions'], 
                               components['arcs'])
        
        # Add synchronous transitions
        sync_product.add_transitions_with_arcs(sync_transitions)
        
        # Set initial and final markings
        sync_product.init_mark = Marking(self.init_mark.places + trace_model.init_mark.places)
        sync_product.final_mark = Marking(self.final_mark.places + trace_model.final_mark.places)
        
        # Update transition names
        self.update_sync_product_trans_names(sync_product)
        
        return sync_product
        
        
    def add_places(self, places):
        if isinstance(places, list):
            self.places += places
        
        else:
            self.places.append(places)
        
        self.__update_indices_p_dict(places)
     
    
    def add_transitions(self, transitions):
        if isinstance(transitions, list):
            self.transitions += transitions
        
        else:
            self.transitions.append(transitions)
        
        self.__update_indices_t_dict(transitions)
       
    
    def add_transitions_with_arcs(self, transitions):
        if isinstance(transitions, list):
            self.transitions += transitions
            for transition in transitions:
                self.arcs += list(transition.in_arcs.union(transition.out_arcs))

        else:
            self.transitions.append(transitions) 
            self.arcs += list(transition.in_arcs.union(transition.out_arcs))

        self.__update_indices_t_dict(transitions)
  

    def add_arc_from_to(self, source, target, weight=None):
            if weight is None:
                arc = Arc(source, target)
            else:
                arc = Arc(source, target, weight)
            source.out_arcs.add(arc)
            target.in_arcs.add(arc)
            self.arcs.append(arc)

    
    def _generate_all_sync_transitions(self, trace_model, cost_function):
        sync_transitions = []
        counter = 1

        for trans in self.transitions:
            # trans.label is guaranteed to be unique in the discovered model (from docs)
            if trans.label is not None:
                # Find in the trace model all the transitions with the same label
                same_label_transitions = self.__find_simillar_label_transitions(trace_model, trans.label)

                for trace_trans in same_label_transitions:
                    new_sync_trans = self.__generate_new_trans(trans, trace_trans, counter, cost_function)
                    sync_transitions.append(new_sync_trans)
                    counter += 1
     
        return sync_transitions
    
    
    def __find_simillar_label_transitions(self, trace_model, activity_label):
        '''Returns all the transitions in the trace with a specified activity label'''
        same_label_trans = [transition for transition in trace_model.transitions if transition.label == activity_label]
                                                                                                   
        return same_label_trans
        
           
    def __generate_new_trans(self, trans, trace_trans, counter, cost_function):
        name = f'sync_{trace_trans.name}'
        new_sync_transition = Transition(name=name, label=trans.label, move_type='sync', prob=trace_trans.prob, cost_function=cost_function)
        
        input_arcs = trans.in_arcs.union(trace_trans.in_arcs)
        new_input_arcs = []
        for arc in input_arcs:
            new_arc = Arc(arc.source, new_sync_transition, arc.weight)
            new_input_arcs.append(new_arc)
            
        output_arcs = trans.out_arcs.union(trace_trans.out_arcs)
        new_output_arcs = []
        for arc in output_arcs:
            new_arc = Arc(new_sync_transition, arc.target, arc.weight)
            new_output_arcs.append(new_arc)
       
        new_sync_transition.in_arcs = new_sync_transition.in_arcs.union(new_input_arcs)
        new_sync_transition.out_arcs = new_sync_transition.out_arcs.union(new_output_arcs)
       
        return new_sync_transition        

    
    def __update_indices_p_dict(self, places):
        curr_idx = len(self.places_indices)
        if isinstance(places, list):
            for p in places:
                self.places_indices[p.name] = curr_idx
                curr_idx += 1
        else:
            self.places_indices[places.name] = curr_idx
     
    
    def __update_indices_t_dict(self, transitions):
        curr_idx = len(self.transitions_indices)
        if isinstance(transitions, list):
            for t in transitions:
                self.transitions_indices[t.name] = curr_idx
                curr_idx += 1
        else:
            self.transitions_indices[transitions.name] = curr_idx            
     
    
    def _find_directly_enabled_transitions(self, mark_tuple: Tuple[int, ...]) -> List[Transition]:
        """
        Finds all transitions that are directly enabled from a given marking,
        without considering tau-reachability.
        """
        available_transitions = []
        for transition in self.transitions:
            if self.__check_transition_prerequesits(transition, mark_tuple):
                available_transitions.append(transition)
        return available_transitions


    def _find_available_transitions(self, mark_tuple: Tuple[int, ...], max_tau_depth: int = 100) -> List[Transition]:
        # Use cache if enabled
        if self._use_cache:
            cached_enabled = self._enabled_cache.get(mark_tuple)
            if cached_enabled is not None:
                return cached_enabled
        
        # Use lazy caching for marking transition map only if allowed. Unlike the previous
        # "lazy" behavior (which built the *entire* map), this caches per-marking entries
        # on demand to avoid OOM on complex models.
        transition_map = (
            self.get_or_build_marking_transition_map(max_tau_depth)
            if self._allow_lazy_map_build
            else self._marking_transition_map
        )

        if transition_map is not None:
            entry = transition_map.get(mark_tuple)
            if entry and "available_transitions" in entry and self._allow_lazy_map_build:
                # Refresh entry to behave like an LRU cache under dict insertion ordering.
                transition_map.pop(mark_tuple, None)
                transition_map[mark_tuple] = entry

            if not entry or "available_transitions" not in entry:
                # Compute tau-reachable visible transitions for this marking and cache.
                try:
                    tau_reachable = self._compute_reachable_transitions_via_tau(mark_tuple, max_tau_depth)
                except Exception:
                    tau_reachable = {}
                entry = {"available_transitions": tau_reachable}
                if self._allow_lazy_map_build:
                    transition_map[mark_tuple] = entry
                    if len(transition_map) > self._marking_transition_map_cache_max_size:
                        pinned = self.init_mark.places if self.init_mark is not None else None
                        while len(transition_map) > self._marking_transition_map_cache_max_size:
                            oldest_key = next(iter(transition_map))
                            if pinned is not None and oldest_key == pinned:
                                # Keep the initial marking hot.
                                transition_map[pinned] = transition_map.pop(pinned)
                                continue
                            transition_map.pop(oldest_key, None)

            # Return the list of non-silent transitions reachable via tau-moves
            result = list(entry["available_transitions"].keys())
        else:
            # No transition map available: fall back to direct enabled transitions.
            result = self._find_directly_enabled_transitions(mark_tuple)
        
        # Cache result if enabled
        if self._use_cache:
            self._enabled_cache[mark_tuple] = result
            # Simple cache eviction
            if len(self._enabled_cache) > self._cache_max_size:
                # Remove first half of entries
                keys_to_remove = list(self._enabled_cache.keys())[:self._cache_max_size // 2]
                for key in keys_to_remove:
                    del self._enabled_cache[key]
        
        return result

    
    def __check_transition_prerequesits(self, transition: Transition, mark_tuple: Tuple[int, ...]) -> bool:
        """
        Check if the given transition is enabled under the current marking.

        Args:
            transition (Transition): The transition to check.
            mark_tuple (Tuple[int, ...]): The current marking as a tuple of token counts.

        Returns:
            bool: True if the transition is enabled (all input places have enough tokens), False otherwise.
        """
        if self._finalized:
            return transition.is_enabled_optimized(mark_tuple)
        else:
            # Fallback to original implementation
            for arc in transition.in_arcs:
                arc_weight = arc.weight
                source_idx = self.places_indices[arc.source.name]
                if mark_tuple[source_idx] < arc_weight:
                    return False
            return True

    def enable_caching(self, enable: bool = True, max_size: int = 10000) -> None:
        self._use_cache = enable
        self._cache_max_size = max_size
        if not enable:
            self._enabled_cache.clear() 


    def __assign_trace_transitions_move_type(self):
        for trans in self.transitions:
            trans.move_type = 'trace'
            
    
    def assign_trace_transitions_move_type(self):
        return self.__assign_trace_transitions_move_type()   
    
    
    def assign_model_transitions_move_type(self):
        return self.__assign_model_transitions_move_type()
    
    
    def __assign_model_transitions_move_type(self):
        for trans in self.transitions:
                trans.move_type = 'model'
                
        


    def _fire_transition_original(
        self,
        mark: Union[Marking, Tuple[int, ...]],
        transition: "Transition"
    ) -> "Marking":
        """
        Fire a transition on a given marking (original implementation).

        Parameters
        ----------
        mark
            Either a Marking object or a raw tuple of token counts.
        transition
            The transition to fire.

        Returns
        -------
        Marking
            The new marking after firing.

        Raises
        ------
        TypeError
            If `mark` is neither a tuple nor a Marking.
        ValueError
            If firing would produce negative tokens.
        """
        # 1) Normalize and type‐check
        if isinstance(mark, tuple):
            places = mark
        elif hasattr(mark, 'places') and isinstance(mark.places, tuple):
            # Handle Marking objects (more robust than isinstance check)
            places = mark.places
        else:
            raise TypeError(f"mark must be a tuple or Marking object with .places attribute, got {type(mark)}")

        # 2) Build net token‐change per place
        delta = Counter()
        for arc in transition.in_arcs:
            idx = self.places_indices[arc.source.name]
            delta[idx] -= arc.weight
        for arc in transition.out_arcs:
            idx = self.places_indices[arc.target.name]
            delta[idx] += arc.weight

        # 3) Apply delta and check for negatives
        new_places = []
        for i, old in enumerate(places):
            new = old + delta[i]
            if new < 0:
                raise ValueError(
                    f"Firing '{transition.name}' yields negative tokens at place {i}: "
                    f"{old} + ({delta[i]}) = {new}"
                )
            new_places.append(new)

        # 4) Wrap in Marking
        return Marking(tuple(new_places))

    def _fire_transition(self, mark: Union['Marking', Tuple[int, ...]], 
                        transition: 'Transition') -> 'Marking':
        # Choose implementation based on finalization status
        if self._finalized and hasattr(transition, 'in_idx_weights') and transition.in_idx_weights is not None:
            if not hasattr(transition, 'delta_idx_weights'):
                transition.prepare_fire(self.places_indices)

            # Normalize marking
            places = mark if isinstance(mark, tuple) else mark.places
            
            # FIX: Proper check for empty transitions
            if not transition.in_idx_weights and not transition.out_idx_weights:
                return Marking(places)
            
            # Validate inputs before applying the sparse net delta.
            for idx, weight in transition.in_idx_weights:
                if places[idx] < weight:
                    place_name = self.places[idx].name
                    raise ValueError(
                        f"Firing '{transition.name}' yields negative tokens at {place_name}: "
                        f"{places[idx]} - {weight} = {places[idx] - weight}"
                    )

            if not transition.delta_idx_weights:
                return Marking(places)

            new_places = list(places)
            for idx, weight_delta in transition.delta_idx_weights:
                new_places[idx] += weight_delta
            
            return Marking(tuple(new_places))
        else:
            # Use original implementation
            return self._fire_transition_original(mark, transition)

    def _fire_transition_sequence(self, marking, transitions):
        """
        Fires a sequence of transitions starting from a given marking.

        Args:
            marking (Marking or tuple): The starting marking.
            transitions (list or tuple of Transition): The sequence of transitions to fire.

        Returns:
            Marking: The marking after firing the entire sequence.
        """
        current_marking = marking
        for transition in transitions:
            current_marking = self._fire_transition(current_marking, transition)
        return current_marking

    def _fire_macro_transition(self, marking, target_transition):
        """
        Fires a τ-path and then the target transition.
        
        Parameters
        ----------
        marking : Any
            A Marking-like object (with a `.places` tuple) or a raw tuple of ints.
        target_transition : Transition
            The visible transition to fire after the τ-path.
        
        Returns
        -------
        Marking
            The final marking after firing the τ-path and the target transition.
        
        Raises
        ------
        TypeError
            If `marking` is neither Marking-like nor a tuple of ints.
        ValueError
            If the transition-map isn’t computed, or target is unreachable.
        """
        # 1) Normalize input: accept any object with .places, or a tuple
        if hasattr(marking, "places"):
            marking_tuple = marking.places
        elif isinstance(marking, tuple):
            marking_tuple = marking
        else:
            raise TypeError(
                f"_fire_macro_transition expected a Marking-like or tuple, "
                f"got {type(marking)}"
            )

        # 2) Ensure the τ-reachability map exists
        if not getattr(self, "marking_transition_map", None):
            raise ValueError("marking_transition_map is not available or not computed.")

        # 3) Lookup the τ-path for this marking → transition
        entry = self.marking_transition_map.get(marking_tuple)
        if not entry or "available_transitions" not in entry:
            raise ValueError(f"Marking {marking_tuple} not in transition map.")

        tau_path = entry["available_transitions"].get(target_transition)
        if tau_path is None:
            raise ValueError(
                f"Transition {target_transition.name} not reachable from marking {marking_tuple}."
            )

        # 4) Fire the τ-sequence, then the target transition
        marking_after_tau = self._fire_transition_sequence(marking_tuple, tau_path)
        final_marking     = self._fire_transition(marking_after_tau, target_transition)
        return final_marking


    def convert_marking_to_pm4py(self, marking: Any) -> Dict[Any, int]:
        return {self.reverse_place_mapping[idx]: tokens 
                for idx, tokens in enumerate(marking.places) 
                if tokens > 0}
    
      
    def _compute_reachable_transitions_via_tau(
        self,
        marking_places: Tuple[int, ...],
        max_tau_depth: int = 100
    ) -> Dict[Transition, Tuple[Transition, ...]]:
        """
        Compute all non-silent transitions reachable from marking_places via τ-moves.

        Returns a mapping from each reachable non-silent transition to the shortest
        τ-path that enables it.
        """
        if max_tau_depth <= 0:
            raise ValueError("max_tau_depth must be positive")

        reachable_transitions: Dict[Transition, Tuple[Transition, ...]] = {}
        queue = deque([(marking_places, tuple())])
        visited_markings: Dict[Tuple[int, ...], int] = {marking_places: 0}

        # Early exit if no silent transitions exist
        has_tau_transitions = any(t.label is None for t in self.transitions)
        if not has_tau_transitions:
            # No tau transitions, so just return directly enabled non-silent transitions
            try:
                enabled = self._find_directly_enabled_transitions(marking_places)
                for transition in enabled:
                    if transition.label is not None:
                        reachable_transitions[transition] = tuple()
                return reachable_transitions
            except Exception:
                return {}

        while queue:
            current_marking, tau_path = queue.popleft()

            # Skip if we've seen this marking with a shorter path
            if visited_markings.get(current_marking, float('inf')) < len(tau_path):
                continue

            try:
                enabled_transitions = self._find_directly_enabled_transitions(current_marking)
            except Exception as exc:
                logger.warning(f"Could not get transitions for marking {current_marking}: {exc}")
                continue

            # Process tau transitions first (more common)
            tau_transitions = [t for t in enabled_transitions if t.label is None]
            visible_transitions = [t for t in enabled_transitions if t.label is not None]

            # Handle visible transitions (can be reached immediately)
            for transition in visible_transitions:
                if transition not in reachable_transitions or len(tau_path) < len(reachable_transitions[transition]):
                    reachable_transitions[transition] = tau_path

            # Handle tau transitions
            for transition in tau_transitions:
                if len(tau_path) >= max_tau_depth:
                    continue

                try:
                    successor = self._fire_transition(
                        Marking(current_marking),
                        transition
                    )
                    new_tau_path = tau_path + (transition,)

                    if len(new_tau_path) <= visited_markings.get(successor.places, float('inf')):
                        visited_markings[successor.places] = len(new_tau_path)
                        queue.append((successor.places, new_tau_path))

                except Exception as exc:
                    logger.warning(f"Could not fire τ transition {transition.name}: {exc}")

        return reachable_transitions


    def get_or_build_marking_transition_map(self, max_tau_depth: int = 100) -> Dict[Tuple[int, ...], Dict]:
        """
        Get an existing marking-transition cache or initialize an empty on-demand cache.

        Note: This intentionally does NOT build the full reachability map. Full-map
        precomputation can be extremely memory-heavy for complex discovered models.
        Use `build_marking_transition_map()` explicitly if you truly need the complete map.
        """
        if self._marking_transition_map is None:
            self._marking_transition_map = {}
            self._marking_transition_map_max_tau = max_tau_depth
            return self._marking_transition_map

        # If we have an existing map but no recorded depth, assume it's valid for the
        # requested depth (this is the case for eagerly precomputed maps).
        if self._marking_transition_map_max_tau is None:
            self._marking_transition_map_max_tau = max_tau_depth
            return self._marking_transition_map

        if self._marking_transition_map_max_tau != max_tau_depth:
            # Depth changed; invalidate any cached entries to preserve correctness.
            self._marking_transition_map = {}
            self._marking_transition_map_max_tau = max_tau_depth
        return self._marking_transition_map

    def build_marking_transition_map(self, max_tau_depth: int = 100) -> Dict[Tuple[int, ...], Dict]:
        """Build complete marking-to-transition map and store on self with optimizations."""
        if self.init_mark is None:
            raise ValueError("Initial marking must be set before building transition map")
        if max_tau_depth <= 0:
            raise ValueError("max_tau_depth must be positive")

        # Ensure optimization before heavy computation
        if not self._finalized:
            self.finalize()

        result: Dict[Tuple[int, ...], Dict] = {}
        visited = set()
        queue = deque([self.init_mark.places])

        # Pre-compute directly enabled transitions for all markings we encounter
        direct_enabled_cache: Dict[Tuple[int, ...], List[Transition]] = {}

        while queue:
            current_marking = queue.popleft()

            if current_marking in visited:
                continue
            visited.add(current_marking)

            # Compute τ-reachable transitions
            try:
                tau_reachable = self._compute_reachable_transitions_via_tau(
                    current_marking, max_tau_depth
                )
                result[current_marking] = {"available_transitions": tau_reachable}

                # Cache directly enabled transitions
                if current_marking not in direct_enabled_cache:
                    try:
                        direct_enabled_cache[current_marking] = self._find_directly_enabled_transitions(current_marking)
                    except Exception:
                        direct_enabled_cache[current_marking] = []

                # Add successors to queue using cached transitions
                enabled_transitions = direct_enabled_cache[current_marking]
                for transition in enabled_transitions:
                    try:
                        successor = self._fire_transition(
                            Marking(current_marking),
                            transition
                        )
                        if successor.places not in visited and successor.places not in direct_enabled_cache:
                            # Pre-compute enabled transitions for successor to avoid recomputation
                            direct_enabled_cache[successor.places] = self._find_directly_enabled_transitions(successor.places)
                            queue.append(successor.places)
                    except Exception as exc:
                        logger.warning(f"Could not fire {transition.name} from {current_marking}: {exc}")
            except Exception as exc:
                logger.warning(f"Could not compute τ-reachability for {current_marking}: {exc}")
                result[current_marking] = {"available_transitions": {}}

        self.marking_transition_map = result
        self._marking_transition_map_max_tau = max_tau_depth
        logger.info(f"Built marking transition map with {len(result)} markings")
        return result

    def get_tau_reachable_transitions(self, marking=None, max_tau_depth=100):
        """Get all tau-reachable transitions for a given marking."""
        # Normalize marking to tuple
        if marking is None:
            if self.init_mark is None:
                raise ValueError("No marking provided and initial marking not set")
            marking_tuple = self.init_mark.places
        elif isinstance(marking, Marking):
            marking_tuple = marking.places
        elif isinstance(marking, tuple):
            marking_tuple = marking
        else:
            raise TypeError("Marking must be a Marking object, tuple, or None")
        
        # Check cache first
        if getattr(self, "_marking_transition_map", None) and self._marking_transition_map_max_tau == max_tau_depth:
            entry = self._marking_transition_map.get(marking_tuple)
            if entry and "available_transitions" in entry:
                return entry["available_transitions"]

        # Compute and optionally cache
        reachable = self._compute_reachable_transitions_via_tau(marking_tuple, max_tau_depth)
        if self._allow_lazy_map_build:
            transition_map = self.get_or_build_marking_transition_map(max_tau_depth)
            transition_map[marking_tuple] = {"available_transitions": reachable}
            if len(transition_map) > self._marking_transition_map_cache_max_size:
                pinned = self.init_mark.places if self.init_mark is not None else None
                while len(transition_map) > self._marking_transition_map_cache_max_size:
                    oldest_key = next(iter(transition_map))
                    if pinned is not None and oldest_key == pinned:
                        transition_map[pinned] = transition_map.pop(pinned)
                        continue
                    transition_map.pop(oldest_key, None)
        return reachable
    
    def get_tau_reachable_transitions_initial(self, max_tau_depth=100):
        """Get tau-reachable transitions for the initial marking."""
        return self.get_tau_reachable_transitions(self.init_mark, max_tau_depth)
    
    def get_tau_reachable_transitions_final(self, max_tau_depth=100):
        """Get tau-reachable transitions for the final marking."""
        if self.final_mark is None:
            raise ValueError("Final marking not set")
        return self.get_tau_reachable_transitions(self.final_mark, max_tau_depth)
    

    # [removed] Dijkstra-based conformance checking helpers (dijkstra_no_rg_construct, etc.)
    # These were broken and unused. Use partial_trace_conformance / conformance_chunked instead.

    def partial_trace_conformance(
        self,
        softmax_matrix: np.ndarray,
        initial_marking: 'Marking',
        cost_fn: Callable[[float, str], float],
        eps: float = 1e-12,
        prob_dict_uncollapsed: Optional[Dict[Tuple[str, ...], Dict[str, float]]] = None,
        prob_dict_collapsed: Optional[Dict[Tuple[str, ...], Dict[str, float]]] = None,
        switch_penalty_weight: float = 0.0,
        initial_last_label: Optional[str] = None,
        state_cache: Optional[Dict] = None,
        conditioning_alpha: Optional[float] = None,
        conditioning_combine_fn: Optional[Callable[[float, float, float], float]] = None,
        conditioning_n_prev_labels: int = 1,
        conditioning_interpolation_weights: Optional[List[float]] = None,
        # Conditioning history handling:
        # - "exact": include path_prefix in the state key (can be slow/large)
        # - "topm": like exact, but keep only top-M histories per (places, ts, last_label)
        # - "merged": equivalent to topm with M=1
        conditioning_state_mode: str = "exact",
        conditioning_top_m: int = 3,
        # Candidate pruning (approximation): bound branching factor for log/sync moves.
        candidate_top_p: Optional[float] = None,
        candidate_top_k: Optional[int] = None,
        candidate_min_k: int = 1,
        candidate_source: str = "conditioned",  # Deprecated: candidates always use conditioned probs
        candidate_apply_to_sync: bool = True,
        # Restricted log moves: only allow top-1 probability + parent's last_label
        restrict_log_moves: bool = False,
        # Restricted model moves: only allow tau (silent) transitions
        restrict_model_moves_to_tau: bool = False,
        # Optional cap on consecutive direct tau/model-quiet moves
        max_consecutive_tau_moves: Optional[int] = None,
        profile_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compute a partial trace conformance alignment using Dijkstra/A*-style search.

        Memory-optimized version:
        - Uses dist-on-push pattern (update distance when relaxing, skip stale on pop)
        - Uses came_from dict for path reconstruction instead of ancestor chain
        - Periodically compacts heap to remove stale entries

        Args:
            softmax_matrix: Probability matrix (n_activities, n_timestamps)
            initial_marking: Starting marking
            cost_fn: Cost function for moves
            eps: Minimum probability threshold - activities below this are filtered out

        Returns a dict with:
        - 'alignment': List[(move_type, move_label, move_cost)]
        - 'total_cost': float
        - 'final_marking': Marking
        """
        n_acts, n_ts = softmax_matrix.shape
        # Use sys.intern for label strings to ensure identical labels share memory
        label2idx = {sys.intern(str(i)): i for i in range(n_acts)}
        idx2label = {i: sys.intern(str(i)) for i in range(n_acts)}
        class_labels = [idx2label[i] for i in range(n_acts)]
        raw_top1_by_ts = np.argmax(softmax_matrix, axis=0) if n_ts > 0 else np.array([], dtype=int)
        tau_cost_const = cost_fn(0.0, 'tau')
        model_cost_const = cost_fn(0.0, 'model')
        INF = float('inf')
        transition_is_tau: Dict[Transition, bool] = {}
        transition_label_idx: Dict[Transition, int] = {}
        for transition in self.transitions:
            is_tau = transition.label is None
            transition_is_tau[transition] = is_tau
            transition_label_idx[transition] = -1 if is_tau else label2idx.get(transition.label, -1)
        n_transitions_total = len(self.transitions)
        use_tau_cap = max_consecutive_tau_moves is not None
        tau_cap = int(max_consecutive_tau_moves) if use_tau_cap else None
        if tau_cap is not None and tau_cap < 1:
            raise ValueError("max_consecutive_tau_moves must be positive when set")

        # Memory optimization: bound path_prefix to only store what's needed for conditioning.
        # We need n_prev_labels previous DIFFERENT labels plus the current label.
        # Add +1 buffer for safety in edge cases.
        max_prefix_distinct_labels = conditioning_n_prev_labels + 2

        # Extract bigram map prev -> {next -> P(next|prev)} if provided (legacy mode)
        # For extended mode (n_prev_labels > 1), we'll pass prob_dict_uncollapsed and prob_dict_collapsed directly
        bigram_map: Dict[str, Dict[str, float]] = {}
        if prob_dict_uncollapsed is not None and conditioning_n_prev_labels == 1:
            for prefix, next_map in prob_dict_uncollapsed.items():
                if isinstance(prefix, tuple) and len(prefix) == 1:
                    bigram_map[prefix[0]] = dict(next_map)

        # If switch penalties are active, the future cost depends on last_label.
        # Extend the state with last_label to preserve optimality under pruning.
        use_last_label_in_state: bool = switch_penalty_weight > 0.0

        mode = (conditioning_state_mode or "exact").lower()
        if mode not in ("exact", "topm", "merged"):
            raise ValueError("conditioning_state_mode must be one of: exact, topm, merged")

        # If conditioning is active, future costs depend on path_prefix (history).
        # Include it in the key for correctness, unless conditioning is disabled entirely.
        use_conditioning_context: bool = conditioning_alpha is not None and conditioning_n_prev_labels >= 1 and mode in ("exact", "topm", "merged")

        # Top-M cap on number of histories per merged base state.
        # Applies only when conditioning context is part of the key.
        top_m: Optional[int] = None
        if use_conditioning_context:
            if mode == "merged":
                top_m = 1
            elif mode == "topm":
                if conditioning_top_m < 1:
                    raise ValueError("conditioning_top_m must be >= 1")
                top_m = int(conditioning_top_m)

        use_specialized_key_layout = (
            use_conditioning_context
            and use_last_label_in_state
            and use_tau_cap
        )
        use_compact_state_keys = (
            use_specialized_key_layout
            and (state_cache is None or len(state_cache) == 0)
        )

        label_none_id = -1
        marking_id_by_places: Dict[Tuple[int, ...], int] = {initial_marking.places: 0}
        next_marking_id = 1
        prefix_id_by_tuple: Dict[Tuple[str, ...], int] = {tuple(): 0}
        prefix_by_id: List[Tuple[str, ...]] = [tuple()]
        prefix_extend_cache: Dict[Tuple[int, int], int] = {}

        def _label_to_key_id(label: Optional[str]) -> int:
            return label_none_id if label is None else label2idx.get(label, label_none_id)

        def _get_marking_id(marking: 'Marking') -> int:
            nonlocal next_marking_id
            places_key = marking.places
            marking_id = marking_id_by_places.get(places_key)
            if marking_id is None:
                marking_id = next_marking_id
                next_marking_id += 1
                marking_id_by_places[places_key] = marking_id
            return marking_id

        def _get_marking_id_from_places(places: Tuple[int, ...]) -> int:
            nonlocal next_marking_id
            marking_id = marking_id_by_places.get(places)
            if marking_id is None:
                marking_id = next_marking_id
                next_marking_id += 1
                marking_id_by_places[places] = marking_id
            return marking_id

        def _get_prefix_id(prefix: Tuple[str, ...]) -> int:
            prefix_id = prefix_id_by_tuple.get(prefix)
            if prefix_id is None:
                prefix_id = len(prefix_by_id)
                prefix_id_by_tuple[prefix] = prefix_id
                prefix_by_id.append(prefix)
            return prefix_id

        def _extend_prefix_id(prefix_id: int, label_idx: int) -> int:
            cache_key = (prefix_id, label_idx)
            cached_prefix_id = prefix_extend_cache.get(cache_key)
            if cached_prefix_id is not None:
                return cached_prefix_id

            prefix = prefix_by_id[prefix_id]
            label = idx2label[label_idx]
            if prefix and prefix[-1] == label:
                new_prefix_id = prefix_id
            elif len(prefix) >= max_prefix_distinct_labels:
                new_prefix_id = _get_prefix_id(prefix[1:] + (label,))
            else:
                new_prefix_id = _get_prefix_id(prefix + (label,))
            prefix_extend_cache[cache_key] = new_prefix_id
            return new_prefix_id

        def make_key(
            places: Tuple[int, ...],
            ts: int,
            last_label: Optional[str],
            context: Tuple[str, ...],
            tau_run_len: int = 0,
        ):
            if use_compact_state_keys:
                return (
                    _get_marking_id_from_places(places),
                    ts,
                    _label_to_key_id(last_label),
                    _get_prefix_id(context),
                    tau_run_len,
                )
            if use_specialized_key_layout:
                return (places, ts, last_label, context, tau_run_len)
            if use_conditioning_context:
                # Include full context for conditioning correctness
                base = (places, ts, last_label, context) if use_last_label_in_state else (places, ts, context)
            else:
                base = (places, ts, last_label) if use_last_label_in_state else (places, ts)
            return (*base, int(tau_run_len)) if use_tau_cap else base

        def make_base_key(
            places: Tuple[int, ...],
            ts: int,
            last_label: Optional[str],
            tau_run_len: int = 0,
        ):
            if use_compact_state_keys:
                return (
                    _get_marking_id_from_places(places),
                    ts,
                    _label_to_key_id(last_label),
                    tau_run_len,
                )
            if use_specialized_key_layout:
                return (places, ts, last_label, tau_run_len)
            base = (places, ts, last_label) if use_last_label_in_state else (places, ts)
            return (*base, int(tau_run_len)) if use_tau_cap else base

        def make_unlabeled_key(places: Tuple[int, ...], ts: int, tau_run_len: int = 0):
            if use_compact_state_keys:
                return (_get_marking_id_from_places(places), ts, tau_run_len)
            if use_specialized_key_layout:
                return (places, ts, tau_run_len)
            base = (places, ts)
            return (*base, int(tau_run_len)) if use_tau_cap else base

        # Start state key
        start_key = make_key(initial_marking.places, 0, initial_last_label, tuple(), 0)

        # dist: best-known cost per state key - updated on PUSH (relaxation)
        # This is the key optimization: update dist when we find a better path,
        # not when we pop. This prevents pushing many duplicates for the same state.
        dist: Dict[Any, float] = state_cache if state_cache is not None else {}
        dist[start_key] = 0.0

        # came_from: for path reconstruction - stores (parent_key, move_type, move_label, move_cost)
        # This eliminates the ancestor chain that prevented garbage collection.
        came_from: Dict[Any, Tuple[Any, str, str, float]] = {}

        # Store marking for each key (needed for expansion). When conditioning is enabled,
        # path_prefix is part of the key, so we avoid duplicating it here.
        marking_by_key: Dict[Any, 'Marking'] = {start_key: initial_marking}

        # For top-M / merged modes: keep only the best M history variants per merged base state.
        # We store the full keys (which include context) and their current best costs.
        kept_single_by_base: Dict[Any, Tuple[Any, float]] = {}
        kept_by_base: Dict[Any, Dict[Any, float]] = (
            {} if top_m == 1 else defaultdict(dict)
        )
        if top_m == 1:
            base_start = make_base_key(initial_marking.places, 0, initial_last_label, 0)
            kept_single_by_base[base_start] = (start_key, 0.0)
        elif top_m is not None:
            base_start = make_base_key(initial_marking.places, 0, initial_last_label, 0)
            kept_by_base[base_start][start_key] = 0.0

        # Additional dominance pruning across last_label for the same (places, timestamp):
        # Any two nodes that only differ in last_label can differ at most by one switch penalty
        # before the next timestamp advance. If a node is already worse than the current best
        # for (places, ts) by more than `switch_penalty_weight`, it can never catch up.
        best_unlabeled: Dict[Any, float] = defaultdict(lambda: INF)
        best_unlabeled[make_unlabeled_key(initial_marking.places, 0, 0)] = 0.0

        # Min-heap of (cost, counter, key) - counter for tie-breaking to avoid comparing keys
        counter = 0
        open_set: List[Tuple[float, int, Any]] = []
        heapq.heappush(open_set, (0.0, counter, start_key))
        counter += 1

        # For periodic heap compaction
        COMPACT_THRESHOLD = 100000
        COMPACT_MIN_REMOVED = 10000
        iterations_since_compact = 0
        removed_since_compact = 0

        profile_completed = False
        profile_no_path = False
        heap_pops = 0
        heap_pushes = 1
        stale_pops = 0
        missing_marking_pops = 0
        dominance_prunes = 0
        states_expanded = 0
        enabled_calls = 0
        enabled_total = 0
        enabled_scanned_transitions = 0
        conditioning_calls = 0
        candidate_selection_calls = 0
        candidate_total = 0
        model_edges_considered = 0
        model_moves_skipped_restrict = 0
        tau_moves_skipped_cap = 0
        macro_tau_paths_skipped_cap = 0
        max_tau_run_seen = 0
        model_relaxations = 0
        model_pushes = 0
        log_edges_considered = 0
        log_edges_below_eps = 0
        log_relaxations = 0
        log_pushes = 0
        sync_edges_considered = 0
        sync_edges_skipped_candidate = 0
        sync_edges_below_eps = 0
        sync_relaxations = 0
        sync_pushes = 0
        fire_transition_calls = 0
        macro_fire_calls = 0
        topm_rejects = 0
        topm_evictions = 0
        heap_compactions = 0
        max_heap_size = len(open_set)
        max_dist_size = len(dist)
        max_marking_by_key_size = len(marking_by_key)
        last_timestamp_seen = 0
        prob_vec_cache: Dict[Any, np.ndarray] = {}
        candidate_cache: Dict[Any, Tuple[Optional[np.ndarray], Optional[Set[int]]]] = {}
        cost_vec_cache: Dict[Any, Tuple[np.ndarray, np.ndarray]] = {}
        enabled_info_cache: Dict[
            Any,
            Tuple[List[Transition], List[Transition], List[Tuple[Transition, int, str]]],
        ] = {}
        direct_fire_cache: Dict[Tuple[Any, Transition], Any] = {}
        macro_fire_cache: Dict[Tuple[Any, Transition], Any] = {}
        prob_vec_cache_hits = 0
        prob_vec_cache_misses = 0
        candidate_cache_hits = 0
        candidate_cache_misses = 0
        cost_vec_cache_hits = 0
        cost_vec_cache_misses = 0
        fire_cache_hits = 0
        fire_cache_misses = 0
        switch_penalty_active = switch_penalty_weight > 0.0 and prob_dict_uncollapsed is not None
        switch_penalty_by_label: Dict[str, float] = {}
        switch_penalty_by_label_idx = [0.0] * n_acts
        if switch_penalty_active:
            for label_idx, label in enumerate(class_labels):
                bigram_prefix = (label,)
                p_stay = float(prob_dict_uncollapsed.get(bigram_prefix, {}).get(label, 0.0))
                p_stay = max(min(p_stay, 1.0), 0.0)
                penalty = switch_penalty_weight * p_stay
                switch_penalty_by_label[label] = penalty
                switch_penalty_by_label_idx[label_idx] = penalty

        def _build_enabled_transition_info(
            cache_key: Any,
            places_key: Tuple[int, ...],
        ) -> Tuple[List[Transition], List[Transition], List[Tuple[Transition, int, str]]]:
            enabled_transitions = self._find_available_transitions(places_key)
            tau_transitions: List[Transition] = []
            visible_transitions: List[Tuple[Transition, int, str]] = []
            for transition in enabled_transitions:
                label_idx = transition_label_idx[transition]
                if label_idx >= 0:
                    visible_transitions.append((transition, label_idx, idx2label[label_idx]))
                elif transition_is_tau[transition]:
                    tau_transitions.append(transition)

            cached = (enabled_transitions, tau_transitions, visible_transitions)
            enabled_info_cache[cache_key] = cached
            return cached

        def _select_candidate_indices(probabilities: np.ndarray) -> Optional[np.ndarray]:
            """
            Return sorted indices of candidate labels to expand (top-p within top-k),
            or None to indicate "no candidate pruning".
            """
            if candidate_top_p is None and candidate_top_k is None:
                return None
            if candidate_min_k < 1:
                raise ValueError("candidate_min_k must be >= 1")

            p = np.asarray(probabilities, dtype=float)
            n = int(p.shape[0])
            if n == 0:
                return None

            k_cap = n if candidate_top_k is None else int(candidate_top_k)
            if k_cap <= 0:
                raise ValueError("candidate_top_k must be positive when set")
            k_cap = min(max(k_cap, candidate_min_k), n)

            # Select top-k indices efficiently
            topk_idx = np.argpartition(-p, k_cap - 1)[:k_cap]
            topk_idx = topk_idx[np.argsort(-p[topk_idx])]

            if candidate_top_p is None or float(candidate_top_p) >= 1.0:
                return topk_idx

            top_p = float(candidate_top_p)
            if top_p <= 0.0 or top_p > 1.0:
                raise ValueError("candidate_top_p must be in (0, 1]")

            cum = np.cumsum(p[topk_idx])
            m = int(np.searchsorted(cum, top_p, side="left") + 1)
            m = max(m, candidate_min_k)
            m = min(m, topk_idx.size)
            return topk_idx[:m]

        def _apply_top_m(base_key: Any, state_key: Any, new_cost: float) -> bool:
            """
            Return True if this (base_key, state_key) variant is allowed under the top-M cap.
            May evict a worse variant from the same base_key to make room.
            """
            nonlocal topm_rejects, topm_evictions, removed_since_compact
            if top_m is None:
                return True

            if top_m == 1:
                existing = kept_single_by_base.get(base_key)
                if existing is None:
                    kept_single_by_base[base_key] = (state_key, new_cost)
                    return True

                existing_key, existing_cost = existing
                if existing_key == state_key:
                    if new_cost < existing_cost - 1e-12:
                        kept_single_by_base[base_key] = (state_key, new_cost)
                    return True

                if new_cost < existing_cost - 1e-12:
                    topm_evictions += 1
                    removed_since_compact += 1
                    # Remove heavy per-state data to free memory; leave came_from for reconstruction.
                    dist.pop(existing_key, None)
                    marking_by_key.pop(existing_key, None)
                    kept_single_by_base[base_key] = (state_key, new_cost)
                    return True

                topm_rejects += 1
                return False

            variants = kept_by_base[base_key]
            existing = variants.get(state_key)
            if existing is not None:
                if new_cost < existing - 1e-12:
                    variants[state_key] = new_cost
                return True

            if len(variants) < top_m:
                variants[state_key] = new_cost
                return True

            # Evict the current worst variant for this base state if this one is better.
            worst_key, worst_cost = max(variants.items(), key=lambda kv: kv[1])
            if new_cost < worst_cost - 1e-12:
                del variants[worst_key]
                topm_evictions += 1
                removed_since_compact += 1
                # Remove heavy per-state data to free memory; leave came_from for reconstruction
                dist.pop(worst_key, None)
                marking_by_key.pop(worst_key, None)
                variants[state_key] = new_cost
                return True

            topm_rejects += 1
            return False

        try:
            while open_set:
                heap_pops += 1
                cost, _, key = heapq.heappop(open_set)

                # Skip stale entries: if we've already found a better path to this state
                if cost > dist.get(key, INF):
                    stale_pops += 1
                    removed_since_compact += 1
                    continue

                marking = marking_by_key.get(key)
                if marking is None:
                    missing_marking_pops += 1
                    removed_since_compact += 1
                    continue

                # Extract key components (key structure depends on flags)
                if use_compact_state_keys:
                    marking_id, timestamp, last_label_id, prefix_id, tau_run_len = key
                    places = marking.places
                    last_label = idx2label[last_label_id] if last_label_id >= 0 else None
                    path_prefix = None
                elif use_specialized_key_layout:
                    places, timestamp, last_label, path_prefix, tau_run_len = key
                    marking_id = None
                    last_label_id = _label_to_key_id(last_label)
                    prefix_id = None
                else:
                    if use_tau_cap:
                        key_parts = key[:-1]
                        tau_run_len = int(key[-1])
                    else:
                        key_parts = key
                        tau_run_len = 0
                    if use_conditioning_context:
                        if use_last_label_in_state:
                            places, timestamp, last_label, path_prefix = key_parts
                        else:
                            places, timestamp, path_prefix = key_parts
                            last_label = None
                    else:
                        if use_last_label_in_state:
                            places, timestamp, last_label = key_parts
                        else:
                            places, timestamp = key_parts
                            last_label = None
                        path_prefix = tuple()
                    marking_id = None
                    last_label_id = _label_to_key_id(last_label)
                    prefix_id = None
                if timestamp > last_timestamp_seen:
                    last_timestamp_seen = int(timestamp)
                if tau_run_len > max_tau_run_seen:
                    max_tau_run_seen = tau_run_len

                unlabeled_key = (
                    (marking_id, timestamp, tau_run_len)
                    if use_compact_state_keys
                    else (places, timestamp, tau_run_len)
                    if use_specialized_key_layout
                    else make_unlabeled_key(places, timestamp, tau_run_len)
                )

                # Dominance prune across different last_label variants
                if switch_penalty_weight > 0.0:
                    current_best_unlabeled = best_unlabeled[unlabeled_key]
                    max_advantage = switch_penalty_weight
                    if cost > current_best_unlabeled + max_advantage + 1e-12:
                        dominance_prunes += 1
                        continue

                # Update unlabeled best after acceptance
                if cost < best_unlabeled[unlabeled_key]:
                    best_unlabeled[unlabeled_key] = cost
                states_expanded += 1

                # Goal reached: consumed all timestamps
                if timestamp == n_ts:
                    # Reconstruct path from came_from
                    alignment = []
                    current_key = key
                    while current_key in came_from:
                        parent_key, move_type, move_label, move_cost = came_from[current_key]
                        alignment.append((move_type, move_label, move_cost))
                        current_key = parent_key
                    alignment.reverse()

                    result = {
                        'alignment': alignment,
                        'total_cost': cost,
                        'final_marking': marking
                    }
                    profile_completed = True
                    return result

                enabled_calls += 1
                enabled_scanned_transitions += n_transitions_total
                enabled_cache_key = marking_id if use_compact_state_keys else places
                enabled_info = enabled_info_cache.get(enabled_cache_key)
                if enabled_info is None:
                    enabled_info = _build_enabled_transition_info(enabled_cache_key, places)
                enabled, tau_enabled, sync_enabled = enabled_info
                enabled_len = len(enabled)
                enabled_total += enabled_len

                # Prepare per-timestamp probability vector (optionally conditioned)
                raw_vec = softmax_matrix[:, timestamp]
                if conditioning_alpha is not None and (bigram_map or prob_dict_uncollapsed):
                    conditioning_calls += 1
                    prob_cache_key = (
                        (timestamp, prefix_id)
                        if use_compact_state_keys
                        else (timestamp, path_prefix)
                    )
                    cached_prob_vec = prob_vec_cache.get(prob_cache_key)
                    if cached_prob_vec is not None:
                        prob_vec_cache_hits += 1
                        prob_vec = cached_prob_vec
                    else:
                        prob_vec_cache_misses += 1
                        conditioning_prefix = (
                            prefix_by_id[prefix_id]
                            if use_compact_state_keys
                            else path_prefix
                        )
                        # Determine which mode to use based on conditioning_n_prev_labels
                        if conditioning_n_prev_labels == 1 and bigram_map:
                            # Legacy mode: single previous label with bigram_map
                            prob_vec = adjust_probs_with_sequence_context(
                                observed_probs=raw_vec,
                                class_labels=class_labels,
                                predicted_sequence=list(conditioning_prefix),
                                cond_prob_bigram=bigram_map,
                                alpha=conditioning_alpha,
                                combine_fn=conditioning_combine_fn,
                                n_prev_labels=1,
                            )
                        elif conditioning_n_prev_labels > 1 and prob_dict_uncollapsed is not None:
                            # Extended mode: multiple previous labels with interpolation
                            # Uses TWO dictionaries: uncollapsed for continuation, collapsed for transitions
                            prob_vec = adjust_probs_with_sequence_context(
                                observed_probs=raw_vec,
                                class_labels=class_labels,
                                predicted_sequence=list(conditioning_prefix),
                                prob_dict_uncollapsed=prob_dict_uncollapsed,
                                prob_dict_collapsed=prob_dict_collapsed,
                                alpha=conditioning_alpha,
                                combine_fn=conditioning_combine_fn,
                                n_prev_labels=conditioning_n_prev_labels,
                                interpolation_weights=conditioning_interpolation_weights,
                            )
                        else:
                            prob_vec = raw_vec
                        prob_vec_cache[prob_cache_key] = prob_vec
                else:
                    prob_vec = raw_vec

                # Candidate pruning: always use conditioned probabilities (prob_vec).
                # When conditioning is disabled, prob_vec == raw_vec.
                candidate_selection_calls += 1
                candidate_cache_key = (
                    (timestamp, prefix_id)
                    if use_compact_state_keys
                    else (timestamp, path_prefix)
                )
                cached_candidates = candidate_cache.get(candidate_cache_key)
                if cached_candidates is not None:
                    candidate_cache_hits += 1
                    cand_idx, cand_idx_set = cached_candidates
                else:
                    candidate_cache_misses += 1
                    cand_idx = _select_candidate_indices(prob_vec)
                    cand_idx_set = set(map(int, cand_idx)) if cand_idx is not None and candidate_apply_to_sync else None
                    candidate_cache[candidate_cache_key] = (cand_idx, cand_idx_set)
                if cand_idx is not None:
                    candidate_total += int(len(cand_idx))

                cached_cost_vecs = cost_vec_cache.get(candidate_cache_key)
                if cached_cost_vecs is not None:
                    cost_vec_cache_hits += 1
                    sync_cost_vec, log_cost_vec = cached_cost_vecs
                else:
                    cost_vec_cache_misses += 1
                    sync_cost_vec = np.empty(n_acts, dtype=float)
                    log_cost_vec = np.empty(n_acts, dtype=float)
                    for cost_idx in range(n_acts):
                        p_cost = max(float(prob_vec[cost_idx]), 1e-12)
                        sync_cost_vec[cost_idx] = cost_fn(p_cost, 'sync')
                        log_cost_vec[cost_idx] = cost_fn(p_cost, 'log')
                    cost_vec_cache[candidate_cache_key] = (sync_cost_vec, log_cost_vec)

                marking_transition_map = self.marking_transition_map
                marking_transition_entry = (
                    marking_transition_map.get(places)
                    if marking_transition_map is not None
                    else None
                )
                available_transition_map = (
                    marking_transition_entry.get("available_transitions")
                    if marking_transition_entry is not None
                    else None
                )
                fire_cache_marking_key = marking_id if use_compact_state_keys else places
                next_timestamp = timestamp + 1

                # 1) Model moves (silent τ or labeled model moves; timestamp unchanged)
                model_edges_considered += enabled_len
                if restrict_model_moves_to_tau:
                    model_moves_skipped_restrict += enabled_len - len(tau_enabled)
                    model_iter = tau_enabled
                else:
                    model_iter = enabled

                if restrict_model_moves_to_tau:
                    for t in model_iter:
                        tau_cost_total = 0.0
                        # Use macro transition if this transition comes from marking_transition_map
                        if available_transition_map is not None and t in available_transition_map:
                            # This transition requires τ-path firing; include τ costs
                            tau_path = available_transition_map[t]
                            tau_path_len = len(tau_path)
                            if tau_cap is not None and tau_path_len > tau_cap:
                                macro_tau_paths_skipped_cap += 1
                                continue
                            tau_cost_total = tau_path_len * tau_cost_const
                            new_tau_run_len = 0
                            fire_key = (fire_cache_marking_key, t)
                            cached_fire = macro_fire_cache.get(fire_key)
                            if cached_fire is None:
                                fire_cache_misses += 1
                                new_mark = self._fire_macro_transition(marking, t)
                                if use_compact_state_keys:
                                    new_marking_id = _get_marking_id(new_mark)
                                    macro_fire_cache[fire_key] = (new_mark, new_marking_id)
                                else:
                                    new_marking_id = None
                                    macro_fire_cache[fire_key] = new_mark
                                macro_fire_calls += 1
                            else:
                                fire_cache_hits += 1
                                if use_compact_state_keys:
                                    new_mark, new_marking_id = cached_fire
                                else:
                                    new_mark = cached_fire
                                    new_marking_id = None
                        else:
                            # This is a directly enabled tau transition
                            if tau_cap is not None and tau_run_len >= tau_cap:
                                tau_moves_skipped_cap += 1
                                continue
                            new_tau_run_len = tau_run_len + 1 if use_tau_cap else 0
                            fire_key = (fire_cache_marking_key, t)
                            cached_fire = direct_fire_cache.get(fire_key)
                            if cached_fire is None:
                                fire_cache_misses += 1
                                new_mark = self._fire_transition(marking, t)
                                if use_compact_state_keys:
                                    new_marking_id = _get_marking_id(new_mark)
                                    direct_fire_cache[fire_key] = (new_mark, new_marking_id)
                                else:
                                    new_marking_id = None
                                    direct_fire_cache[fire_key] = new_mark
                                fire_transition_calls += 1
                            else:
                                fire_cache_hits += 1
                                if use_compact_state_keys:
                                    new_mark, new_marking_id = cached_fire
                                else:
                                    new_mark = cached_fire
                                    new_marking_id = None

                        new_cost = cost + tau_cost_total + tau_cost_const
                        if new_tau_run_len > max_tau_run_seen:
                            max_tau_run_seen = new_tau_run_len
                        if use_compact_state_keys:
                            new_key = (new_marking_id, timestamp, last_label_id, prefix_id, new_tau_run_len)
                        elif use_specialized_key_layout:
                            new_key = (new_mark.places, timestamp, last_label, path_prefix, new_tau_run_len)
                        else:
                            new_key = make_key(new_mark.places, timestamp, last_label, path_prefix, new_tau_run_len)

                        # Dist-on-push: only push if this is a better path
                        model_relaxations += 1
                        if new_cost < dist.get(new_key, INF):
                            if use_compact_state_keys:
                                new_base_key = (new_marking_id, timestamp, last_label_id, new_tau_run_len)
                            elif use_specialized_key_layout:
                                new_base_key = (new_mark.places, timestamp, last_label, new_tau_run_len)
                            else:
                                new_base_key = make_base_key(new_mark.places, timestamp, last_label, new_tau_run_len)
                            if not _apply_top_m(new_base_key, new_key, new_cost):
                                continue
                            dist[new_key] = new_cost
                            came_from[new_key] = (key, 'tau', 'τ', tau_cost_const + tau_cost_total)
                            marking_by_key[new_key] = new_mark
                            heapq.heappush(open_set, (new_cost, counter, new_key))
                            heap_pushes += 1
                            model_pushes += 1
                            counter += 1
                            if len(open_set) > max_heap_size:
                                max_heap_size = len(open_set)
                else:
                    for t in model_iter:
                        t_is_tau = transition_is_tau[t]
                        tau_cost_total = 0.0
                        # Use macro transition if this transition comes from marking_transition_map
                        if available_transition_map is not None and t in available_transition_map:
                            # This transition requires τ-path firing; include τ costs
                            tau_path = available_transition_map[t]
                            tau_path_len = len(tau_path)
                            if tau_cap is not None and tau_path_len > tau_cap:
                                macro_tau_paths_skipped_cap += 1
                                continue
                            tau_cost_total = tau_path_len * tau_cost_const
                            new_tau_run_len = 0
                            fire_key = (fire_cache_marking_key, t)
                            cached_fire = macro_fire_cache.get(fire_key)
                            if cached_fire is None:
                                fire_cache_misses += 1
                                new_mark = self._fire_macro_transition(marking, t)
                                if use_compact_state_keys:
                                    new_marking_id = _get_marking_id(new_mark)
                                    macro_fire_cache[fire_key] = (new_mark, new_marking_id)
                                else:
                                    new_marking_id = None
                                    macro_fire_cache[fire_key] = new_mark
                                macro_fire_calls += 1
                            else:
                                fire_cache_hits += 1
                                if use_compact_state_keys:
                                    new_mark, new_marking_id = cached_fire
                                else:
                                    new_mark = cached_fire
                                    new_marking_id = None
                        else:
                            # This is a directly enabled transition
                            if tau_cap is not None and t_is_tau and tau_run_len >= tau_cap:
                                tau_moves_skipped_cap += 1
                                continue
                            new_tau_run_len = tau_run_len + 1 if use_tau_cap and t_is_tau else 0
                            fire_key = (fire_cache_marking_key, t)
                            cached_fire = direct_fire_cache.get(fire_key)
                            if cached_fire is None:
                                fire_cache_misses += 1
                                new_mark = self._fire_transition(marking, t)
                                if use_compact_state_keys:
                                    new_marking_id = _get_marking_id(new_mark)
                                    direct_fire_cache[fire_key] = (new_mark, new_marking_id)
                                else:
                                    new_marking_id = None
                                    direct_fire_cache[fire_key] = new_mark
                                fire_transition_calls += 1
                            else:
                                fire_cache_hits += 1
                                if use_compact_state_keys:
                                    new_mark, new_marking_id = cached_fire
                                else:
                                    new_mark = cached_fire
                                    new_marking_id = None
                        move_type = 'tau' if t_is_tau else 'model'
                        c = tau_cost_const if t_is_tau else model_cost_const
                        new_cost = cost + tau_cost_total + c
                        if new_tau_run_len > max_tau_run_seen:
                            max_tau_run_seen = new_tau_run_len
                        if use_compact_state_keys:
                            new_key = (new_marking_id, timestamp, last_label_id, prefix_id, new_tau_run_len)
                        elif use_specialized_key_layout:
                            new_key = (new_mark.places, timestamp, last_label, path_prefix, new_tau_run_len)
                        else:
                            new_key = make_key(new_mark.places, timestamp, last_label, path_prefix, new_tau_run_len)

                        # Dist-on-push: only push if this is a better path
                        model_relaxations += 1
                        if new_cost < dist.get(new_key, INF):
                            if use_compact_state_keys:
                                new_base_key = (new_marking_id, timestamp, last_label_id, new_tau_run_len)
                            elif use_specialized_key_layout:
                                new_base_key = (new_mark.places, timestamp, last_label, new_tau_run_len)
                            else:
                                new_base_key = make_base_key(new_mark.places, timestamp, last_label, new_tau_run_len)
                            if not _apply_top_m(new_base_key, new_key, new_cost):
                                continue
                            dist[new_key] = new_cost
                            came_from[new_key] = (key, move_type, t.label or 'τ', c + tau_cost_total)
                            marking_by_key[new_key] = new_mark
                            heapq.heappush(open_set, (new_cost, counter, new_key))
                            heap_pushes += 1
                            model_pushes += 1
                            counter += 1
                            if len(open_set) > max_heap_size:
                                max_heap_size = len(open_set)

                # 2) Log moves (advance timestamp without firing any transition)
                if timestamp < n_ts:
                    # Determine which indices to consider for log moves
                    if restrict_log_moves:
                        # Restricted mode: only allow top-1 probability + parent's last_label
                        # This limits log moves to at most 2 options per timestamp
                        # Use raw (observed) probabilities for top-1.
                        top1_idx = int(raw_top1_by_ts[timestamp])
                        restricted_indices = [top1_idx]
                        if use_compact_state_keys:
                            last_idx = last_label_id
                        elif last_label is not None and last_label in label2idx:
                            last_idx = label2idx[last_label]
                        else:
                            last_idx = label_none_id
                        if last_idx >= 0:
                            if last_idx not in restricted_indices:
                                restricted_indices.append(last_idx)
                        iter_indices = restricted_indices
                    elif cand_idx is None:
                        iter_indices = range(n_acts)
                    else:
                        iter_indices = cand_idx

                    for idx in iter_indices:
                        log_edges_considered += 1
                        p_adj = float(prob_vec[idx])
                        # Filter out activities below threshold after adjustment
                        if p_adj < eps:
                            log_edges_below_eps += 1
                            continue
                        label = idx2label[int(idx)]
                        c = float(log_cost_vec[int(idx)])
                        # Switch penalty using bigram p(x_n | x_{n-1}) - use uncollapsed for within-run continuity
                        add_switch = 0.0
                        if use_compact_state_keys:
                            if switch_penalty_active and last_label_id >= 0 and int(idx) != last_label_id:
                                add_switch = switch_penalty_by_label_idx[last_label_id]
                        elif switch_penalty_active and last_label is not None and label != last_label:
                            add_switch = switch_penalty_by_label.get(last_label, 0.0)

                        new_cost = cost + c + add_switch
                        if use_compact_state_keys:
                            new_prefix_id = _extend_prefix_id(prefix_id, int(idx))
                            new_path_prefix = None
                        elif use_conditioning_context:
                            if path_prefix and path_prefix[-1] == label:
                                new_path_prefix = path_prefix
                            elif len(path_prefix) >= max_prefix_distinct_labels:
                                new_path_prefix = path_prefix[1:] + (label,)
                            else:
                                new_path_prefix = path_prefix + (label,)
                        else:
                            new_path_prefix = tuple()
                        if use_compact_state_keys:
                            new_key = (marking_id, next_timestamp, int(idx), new_prefix_id, 0)
                        elif use_specialized_key_layout:
                            new_key = (places, next_timestamp, label, new_path_prefix, 0)
                        else:
                            new_key = make_key(places, next_timestamp, label, new_path_prefix, 0)

                        # Dist-on-push: only push if this is a better path
                        log_relaxations += 1
                        if new_cost < dist.get(new_key, INF):
                            if use_compact_state_keys:
                                new_base_key = (marking_id, next_timestamp, int(idx), 0)
                            elif use_specialized_key_layout:
                                new_base_key = (places, next_timestamp, label, 0)
                            else:
                                new_base_key = make_base_key(places, next_timestamp, label, 0)
                            if not _apply_top_m(new_base_key, new_key, new_cost):
                                continue
                            dist[new_key] = new_cost
                            came_from[new_key] = (key, 'log', label, c + add_switch)
                            marking_by_key[new_key] = marking
                            heapq.heappush(open_set, (new_cost, counter, new_key))
                            heap_pushes += 1
                            log_pushes += 1
                            counter += 1
                            if len(open_set) > max_heap_size:
                                max_heap_size = len(open_set)

                # 3) Synchronous moves (labeled transitions that match softmax label; advance timestamp)
                sync_edges_considered += enabled_len
                for t, idx, t_label in sync_enabled:
                    # Allow parent's last_label for sync moves even if not in top-k
                    if cand_idx_set is not None and idx not in cand_idx_set:
                        if (
                            (last_label_id < 0 or idx != last_label_id)
                            if use_compact_state_keys
                            else (last_label is None or t_label != last_label)
                        ):
                            sync_edges_skipped_candidate += 1
                            continue
                    p_adj = float(prob_vec[idx])
                    # Filter out activities below threshold after adjustment
                    if p_adj < eps:
                        sync_edges_below_eps += 1
                        continue
                    # no observed-move restriction
                    c = float(sync_cost_vec[idx])

                    tau_cost_total = 0.0

                    # Use macro transition if this transition comes from marking_transition_map
                    if available_transition_map is not None and t in available_transition_map:
                        # This transition requires τ-path firing; include τ costs
                        tau_path = available_transition_map[t]
                        tau_path_len = len(tau_path)
                        if tau_cap is not None and tau_path_len > tau_cap:
                            macro_tau_paths_skipped_cap += 1
                            continue
                        tau_cost_total = tau_path_len * tau_cost_const
                        fire_key = (fire_cache_marking_key, t)
                        cached_fire = macro_fire_cache.get(fire_key)
                        if cached_fire is None:
                            fire_cache_misses += 1
                            new_mark = self._fire_macro_transition(marking, t)
                            if use_compact_state_keys:
                                new_marking_id = _get_marking_id(new_mark)
                                macro_fire_cache[fire_key] = (new_mark, new_marking_id)
                            else:
                                new_marking_id = None
                                macro_fire_cache[fire_key] = new_mark
                            macro_fire_calls += 1
                        else:
                            fire_cache_hits += 1
                            if use_compact_state_keys:
                                new_mark, new_marking_id = cached_fire
                            else:
                                new_mark = cached_fire
                                new_marking_id = None
                    else:
                        # This is a directly enabled transition
                        fire_key = (fire_cache_marking_key, t)
                        cached_fire = direct_fire_cache.get(fire_key)
                        if cached_fire is None:
                            fire_cache_misses += 1
                            new_mark = self._fire_transition(marking, t)
                            if use_compact_state_keys:
                                new_marking_id = _get_marking_id(new_mark)
                                direct_fire_cache[fire_key] = (new_mark, new_marking_id)
                            else:
                                new_marking_id = None
                                direct_fire_cache[fire_key] = new_mark
                            fire_transition_calls += 1
                        else:
                            fire_cache_hits += 1
                            if use_compact_state_keys:
                                new_mark, new_marking_id = cached_fire
                            else:
                                new_mark = cached_fire
                                new_marking_id = None

                    # Switch penalty using bigram p(x_n | x_{n-1}) - use uncollapsed for within-run continuity
                    add_switch = 0.0
                    if use_compact_state_keys:
                        if switch_penalty_active and last_label_id >= 0 and idx != last_label_id:
                            add_switch = switch_penalty_by_label_idx[last_label_id]
                    elif switch_penalty_active and last_label is not None and t_label != last_label:
                        add_switch = switch_penalty_by_label.get(last_label, 0.0)

                    new_cost = cost + tau_cost_total + c + add_switch
                    if use_compact_state_keys:
                        new_prefix_id = _extend_prefix_id(prefix_id, idx)
                        new_path_prefix = None
                    elif use_conditioning_context:
                        if path_prefix and path_prefix[-1] == t_label:
                            new_path_prefix = path_prefix
                        elif len(path_prefix) >= max_prefix_distinct_labels:
                            new_path_prefix = path_prefix[1:] + (t_label,)
                        else:
                            new_path_prefix = path_prefix + (t_label,)
                    else:
                        new_path_prefix = tuple()
                    if use_compact_state_keys:
                        new_key = (new_marking_id, next_timestamp, idx, new_prefix_id, 0)
                    elif use_specialized_key_layout:
                        new_key = (new_mark.places, next_timestamp, t_label, new_path_prefix, 0)
                    else:
                        new_key = make_key(new_mark.places, next_timestamp, t_label, new_path_prefix, 0)

                    # Dist-on-push: only push if this is a better path
                    sync_relaxations += 1
                    if new_cost < dist.get(new_key, INF):
                        if use_compact_state_keys:
                            new_base_key = (new_marking_id, next_timestamp, idx, 0)
                        elif use_specialized_key_layout:
                            new_base_key = (new_mark.places, next_timestamp, t_label, 0)
                        else:
                            new_base_key = make_base_key(new_mark.places, next_timestamp, t_label, 0)
                        if not _apply_top_m(new_base_key, new_key, new_cost):
                            continue
                        dist[new_key] = new_cost
                        came_from[new_key] = (key, 'sync', t_label, c + tau_cost_total + add_switch)
                        marking_by_key[new_key] = new_mark
                        heapq.heappush(open_set, (new_cost, counter, new_key))
                        heap_pushes += 1
                        sync_pushes += 1
                        counter += 1
                        if len(open_set) > max_heap_size:
                            max_heap_size = len(open_set)

                dist_len = len(dist)
                if dist_len > max_dist_size:
                    max_dist_size = dist_len
                marking_by_key_len = len(marking_by_key)
                if marking_by_key_len > max_marking_by_key_size:
                    max_marking_by_key_size = marking_by_key_len

                # Periodic heap compaction: remove stale entries
                iterations_since_compact += 1
                if iterations_since_compact >= COMPACT_THRESHOLD and len(open_set) > COMPACT_THRESHOLD:
                    if removed_since_compact >= COMPACT_MIN_REMOVED:
                        # Filter out stale entries (where heap cost != dist[key])
                        open_set = [(c, cnt, k) for c, cnt, k in open_set if dist.get(k, INF) == c]
                        heapq.heapify(open_set)
                        heap_compactions += 1
                        removed_since_compact = 0
                    iterations_since_compact = 0

            profile_no_path = True
            raise ValueError("No conforming path found for the partial trace.")
        finally:
            if profile_stats is not None:
                profile_stats["partial_calls_started"] = profile_stats.get("partial_calls_started", 0) + 1
                if profile_completed:
                    profile_stats["partial_calls_completed"] = profile_stats.get("partial_calls_completed", 0) + 1
                if profile_no_path:
                    profile_stats["partial_calls_no_path"] = profile_stats.get("partial_calls_no_path", 0) + 1
                profile_stats["partial_frames_total"] = profile_stats.get("partial_frames_total", 0) + int(n_ts)
                profile_stats["partial_max_frames"] = max(profile_stats.get("partial_max_frames", 0), int(n_ts))
                profile_stats["heap_pops"] = profile_stats.get("heap_pops", 0) + heap_pops
                profile_stats["heap_pushes"] = profile_stats.get("heap_pushes", 0) + heap_pushes
                profile_stats["stale_pops"] = profile_stats.get("stale_pops", 0) + stale_pops
                profile_stats["missing_marking_pops"] = profile_stats.get("missing_marking_pops", 0) + missing_marking_pops
                profile_stats["dominance_prunes"] = profile_stats.get("dominance_prunes", 0) + dominance_prunes
                profile_stats["states_expanded"] = profile_stats.get("states_expanded", 0) + states_expanded
                profile_stats["enabled_calls"] = profile_stats.get("enabled_calls", 0) + enabled_calls
                profile_stats["enabled_total"] = profile_stats.get("enabled_total", 0) + enabled_total
                profile_stats["enabled_scanned_transitions"] = profile_stats.get("enabled_scanned_transitions", 0) + enabled_scanned_transitions
                profile_stats["conditioning_calls"] = profile_stats.get("conditioning_calls", 0) + conditioning_calls
                profile_stats["candidate_selection_calls"] = profile_stats.get("candidate_selection_calls", 0) + candidate_selection_calls
                profile_stats["candidate_total"] = profile_stats.get("candidate_total", 0) + candidate_total
                profile_stats["model_edges_considered"] = profile_stats.get("model_edges_considered", 0) + model_edges_considered
                profile_stats["model_moves_skipped_restrict"] = profile_stats.get("model_moves_skipped_restrict", 0) + model_moves_skipped_restrict
                profile_stats["tau_moves_skipped_cap"] = profile_stats.get("tau_moves_skipped_cap", 0) + tau_moves_skipped_cap
                profile_stats["macro_tau_paths_skipped_cap"] = profile_stats.get("macro_tau_paths_skipped_cap", 0) + macro_tau_paths_skipped_cap
                profile_stats["max_tau_run_seen"] = max(profile_stats.get("max_tau_run_seen", 0), max_tau_run_seen)
                profile_stats["max_consecutive_tau_moves"] = tau_cap
                profile_stats["model_relaxations"] = profile_stats.get("model_relaxations", 0) + model_relaxations
                profile_stats["model_pushes"] = profile_stats.get("model_pushes", 0) + model_pushes
                profile_stats["log_edges_considered"] = profile_stats.get("log_edges_considered", 0) + log_edges_considered
                profile_stats["log_edges_below_eps"] = profile_stats.get("log_edges_below_eps", 0) + log_edges_below_eps
                profile_stats["log_relaxations"] = profile_stats.get("log_relaxations", 0) + log_relaxations
                profile_stats["log_pushes"] = profile_stats.get("log_pushes", 0) + log_pushes
                profile_stats["sync_edges_considered"] = profile_stats.get("sync_edges_considered", 0) + sync_edges_considered
                profile_stats["sync_edges_skipped_candidate"] = profile_stats.get("sync_edges_skipped_candidate", 0) + sync_edges_skipped_candidate
                profile_stats["sync_edges_below_eps"] = profile_stats.get("sync_edges_below_eps", 0) + sync_edges_below_eps
                profile_stats["sync_relaxations"] = profile_stats.get("sync_relaxations", 0) + sync_relaxations
                profile_stats["sync_pushes"] = profile_stats.get("sync_pushes", 0) + sync_pushes
                profile_stats["fire_transition_calls"] = profile_stats.get("fire_transition_calls", 0) + fire_transition_calls
                profile_stats["macro_fire_calls"] = profile_stats.get("macro_fire_calls", 0) + macro_fire_calls
                profile_stats["topm_rejects"] = profile_stats.get("topm_rejects", 0) + topm_rejects
                profile_stats["topm_evictions"] = profile_stats.get("topm_evictions", 0) + topm_evictions
                profile_stats["heap_compactions"] = profile_stats.get("heap_compactions", 0) + heap_compactions
                profile_stats["prob_vec_cache_hits"] = profile_stats.get("prob_vec_cache_hits", 0) + prob_vec_cache_hits
                profile_stats["prob_vec_cache_misses"] = profile_stats.get("prob_vec_cache_misses", 0) + prob_vec_cache_misses
                profile_stats["candidate_cache_hits"] = profile_stats.get("candidate_cache_hits", 0) + candidate_cache_hits
                profile_stats["candidate_cache_misses"] = profile_stats.get("candidate_cache_misses", 0) + candidate_cache_misses
                profile_stats["cost_vec_cache_hits"] = profile_stats.get("cost_vec_cache_hits", 0) + cost_vec_cache_hits
                profile_stats["cost_vec_cache_misses"] = profile_stats.get("cost_vec_cache_misses", 0) + cost_vec_cache_misses
                profile_stats["fire_cache_hits"] = profile_stats.get("fire_cache_hits", 0) + fire_cache_hits
                profile_stats["fire_cache_misses"] = profile_stats.get("fire_cache_misses", 0) + fire_cache_misses
                profile_stats["max_heap_size"] = max(profile_stats.get("max_heap_size", 0), max_heap_size)
                profile_stats["max_dist_size"] = max(profile_stats.get("max_dist_size", 0), max_dist_size)
                profile_stats["max_marking_by_key_size"] = max(profile_stats.get("max_marking_by_key_size", 0), max_marking_by_key_size)
                profile_stats["last_open_set_size"] = len(open_set)
                profile_stats["last_dist_size"] = len(dist)
                profile_stats["last_marking_by_key_size"] = len(marking_by_key)
                profile_stats["last_timestamp_seen"] = last_timestamp_seen


    def conformance_chunked(
        self,
        softmax_matrix: np.ndarray,
        initial_marking: 'Marking',
        cost_fn: Callable[[float, str], float],
        chunk_size: int = 10,
        eps: float = 1e-12,
        inline_progress: bool = False,
        progress_prefix: str = "",
        prob_dict_uncollapsed: Optional[Dict[Tuple[str, ...], Dict[str, float]]] = None,
        prob_dict_collapsed: Optional[Dict[Tuple[str, ...], Dict[str, float]]] = None,
        switch_penalty_weight: float = 0.0,
        use_state_caching: bool = True,
        merge_mismatched_boundaries: bool = True,
        # removed: restrict_to_observed_moves
        conditioning_alpha: Optional[float] = None,
        conditioning_combine_fn: Optional[Callable[[float, float, float], float]] = None,
        conditioning_n_prev_labels: int = 1,
        conditioning_interpolation_weights: Optional[List[float]] = None,
        conditioning_state_mode: str = "exact",
        conditioning_top_m: int = 3,
        candidate_top_p: Optional[float] = None,
        candidate_top_k: Optional[int] = None,
        candidate_min_k: int = 1,
        candidate_source: str = "conditioned",
        candidate_apply_to_sync: bool = True,
        restrict_log_moves: bool = False,
        restrict_model_moves_to_tau: bool = False,
        max_consecutive_tau_moves: Optional[int] = None,
        progress_log_interval_chunks: int = 0,
        profile_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process softmax_matrix in sequential chunks, calling partial_trace_conformance
        on each, and stitch together a global alignment and cost.
        
        Args:
            softmax_matrix: Probability matrix (n_activities, n_timestamps)
            initial_marking: Starting marking
            cost_fn: Cost function for moves
            chunk_size: Size of chunks to process
            eps: Minimum probability threshold - activities below this are filtered out
        """
        _, n_ts = softmax_matrix.shape
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if progress_log_interval_chunks < 0:
            raise ValueError("progress_log_interval_chunks must be non-negative")

        current_marking = initial_marking
        # Track last emitted label across chunks to apply switch penalty at boundaries
        current_last_label: Optional[str] = None
        total_cost = 0.0
        complete_alignment: List[Tuple[str, str]] = []
        chunk_results: List[Dict[str, Any]] = []

        total_start = time.perf_counter()

        # Helpers to get first/last predicted labels in a chunk alignment
        def _first_trace_label(alignment: List[Tuple[str, str, float]]) -> Optional[str]:
            for move_type, move_label, _ in alignment:
                if move_type in ('sync', 'log'):
                    return move_label
            return None

        def _last_trace_label(alignment: List[Tuple[str, str, float]]) -> Optional[str]:
            for move_type, move_label, _ in reversed(alignment):
                if move_type in ('sync', 'log'):
                    return move_label
            return None

        # Progress visualization uses nominal number of windows
        nominal_total_chunks = (n_ts + chunk_size - 1) // chunk_size
        display_chunk_idx = 0

        # Cache next-window computation to avoid recomputation when no merge occurs
        pending_result: Optional[Dict[str, Any]] = None
        pending_start_ts: Optional[int] = None
        pending_end_ts: Optional[int] = None
        pending_elapsed: Optional[float] = None
        pending_rate: Optional[float] = None

        def _log_chunk_progress(end_ts: int, chunk_elapsed: float, merged: bool) -> None:
            if progress_log_interval_chunks <= 0:
                return
            completed_chunks = len(chunk_results)
            if (
                completed_chunks == 1
                or completed_chunks % progress_log_interval_chunks == 0
                or end_ts >= n_ts
            ):
                elapsed_total = time.perf_counter() - total_start
                rate_total = (end_ts / elapsed_total) if elapsed_total > 0 else float("inf")
                prefix = f"{progress_prefix} " if progress_prefix else ""
                logger.info(
                    "%sconformance progress: chunk=%d/%d, frames=%d/%d, "
                    "elapsed=%.1fs, rate=%.2f frames/s, last_chunk=%.3fs, merged=%s",
                    prefix,
                    completed_chunks,
                    nominal_total_chunks,
                    end_ts,
                    n_ts,
                    elapsed_total,
                    rate_total,
                    chunk_elapsed,
                    merged,
                )

        start_ts = 0
        while start_ts < n_ts:
            display_chunk_idx += 1
            if inline_progress:
                try:
                    prefix = (progress_prefix + " ") if progress_prefix else ""
                    sys.stdout.write(f"\r{prefix}chunk {min(display_chunk_idx, nominal_total_chunks)}/{nominal_total_chunks}")
                    sys.stdout.flush()
                except Exception:
                    pass

            # Compute current window result (standalone), reusing cached result when possible
            end_ts1 = min(start_ts + chunk_size, n_ts)
            # Save state before chunk1 (for potential merge)
            state_before_chunk1_marking = current_marking
            state_before_chunk1_last_label = current_last_label

            if (
                pending_result is not None
                and pending_start_ts == start_ts
                and pending_end_ts == end_ts1
            ):
                # Reuse cached computation for this window
                result1 = pending_result
                c1_elapsed = pending_elapsed if pending_elapsed is not None else 0.0
                c1_steps = end_ts1 - start_ts
                c1_rate = pending_rate if pending_rate is not None else ((c1_steps / c1_elapsed) if c1_elapsed > 0 else float('inf'))
                # Clear cache after use
                pending_result = None
                pending_start_ts = None
                pending_end_ts = None
                pending_elapsed = None
                pending_rate = None
            else:
                chunk1 = softmax_matrix[:, start_ts:end_ts1]
                c1_start = time.perf_counter()
                result1 = self.partial_trace_conformance(
                    softmax_matrix=chunk1,
                    initial_marking=current_marking,
                    cost_fn=cost_fn,
                    eps=eps,
                    prob_dict_uncollapsed=prob_dict_uncollapsed,
                    prob_dict_collapsed=prob_dict_collapsed,
                    switch_penalty_weight=switch_penalty_weight,
                    initial_last_label=current_last_label,
                    state_cache=({} if use_state_caching else None),
                    conditioning_alpha=conditioning_alpha,
                    conditioning_combine_fn=conditioning_combine_fn,
                    conditioning_n_prev_labels=conditioning_n_prev_labels,
                    conditioning_interpolation_weights=conditioning_interpolation_weights,
                    conditioning_state_mode=conditioning_state_mode,
                    conditioning_top_m=conditioning_top_m,
                    candidate_top_p=candidate_top_p,
                    candidate_top_k=candidate_top_k,
                    candidate_min_k=candidate_min_k,
                    candidate_source=candidate_source,
                    candidate_apply_to_sync=candidate_apply_to_sync,
                    restrict_log_moves=restrict_log_moves,
                    restrict_model_moves_to_tau=restrict_model_moves_to_tau,
                    max_consecutive_tau_moves=max_consecutive_tau_moves,
                    profile_stats=profile_stats,
                )
                c1_elapsed = time.perf_counter() - c1_start
                c1_steps = end_ts1 - start_ts
                c1_rate = (c1_steps / c1_elapsed) if c1_elapsed > 0 else float('inf')

            # If there's no next window, accept chunk1 and finish
            if end_ts1 >= n_ts:
                total_cost += result1['total_cost']
                complete_alignment.extend(result1['alignment'])
                current_marking = result1['final_marking']
                current_last_label = _last_trace_label(result1['alignment'])

                chunk_results.append({
                    'chunk_index': len(chunk_results),
                    'start_timestamp': start_ts,
                    'end_timestamp': end_ts1,
                    'chunk_cost': result1['total_cost'],
                    'chunk_alignment_length': len(result1['alignment']),
                    'final_marking': result1['final_marking'],
                    'processing_seconds': c1_elapsed,
                    'processing_rate_steps_per_s': c1_rate,
                    'merged': False,
                })
                _log_chunk_progress(end_ts1, c1_elapsed, False)
                # Ensure no stale pending cache remains
                pending_result = None
                pending_start_ts = None
                pending_end_ts = None
                pending_elapsed = None
                pending_rate = None
                break

            # Look ahead one window and compare boundary labels
            last_label_c1 = _last_trace_label(result1['alignment'])

            end_ts2 = min(end_ts1 + chunk_size, n_ts)
            chunk2 = softmax_matrix[:, end_ts1:end_ts2]
            c2_start = time.perf_counter()
            result2 = self.partial_trace_conformance(
                softmax_matrix=chunk2,
                initial_marking=result1['final_marking'],
                cost_fn=cost_fn,
                eps=eps,
                prob_dict_uncollapsed=prob_dict_uncollapsed,
                prob_dict_collapsed=prob_dict_collapsed,
                switch_penalty_weight=switch_penalty_weight,
                initial_last_label=last_label_c1,
                state_cache=({} if use_state_caching else None),
                conditioning_alpha=conditioning_alpha,
                conditioning_combine_fn=conditioning_combine_fn,
                conditioning_n_prev_labels=conditioning_n_prev_labels,
                conditioning_interpolation_weights=conditioning_interpolation_weights,
                conditioning_state_mode=conditioning_state_mode,
                conditioning_top_m=conditioning_top_m,
                candidate_top_p=candidate_top_p,
                candidate_top_k=candidate_top_k,
                candidate_min_k=candidate_min_k,
                candidate_source=candidate_source,
                candidate_apply_to_sync=candidate_apply_to_sync,
                restrict_log_moves=restrict_log_moves,
                restrict_model_moves_to_tau=restrict_model_moves_to_tau,
                max_consecutive_tau_moves=max_consecutive_tau_moves,
                profile_stats=profile_stats,
            )
            c2_elapsed = time.perf_counter() - c2_start
            c2_steps = end_ts2 - end_ts1
            c2_rate = (c2_steps / c2_elapsed) if c2_elapsed > 0 else float('inf')

            first_label_c2 = _first_trace_label(result2['alignment'])

            # Decide merge on boundary mismatch (configurable)
            if merge_mismatched_boundaries and last_label_c1 is not None and first_label_c2 is not None and last_label_c1 != first_label_c2:
                # Merge the two windows and recompute on the combined subtrace
                merged_chunk = softmax_matrix[:, start_ts:end_ts2]
                m_start = time.perf_counter()
                merged_result = self.partial_trace_conformance(
                    softmax_matrix=merged_chunk,
                    initial_marking=state_before_chunk1_marking,
                    cost_fn=cost_fn,
                    eps=eps,
                    prob_dict_uncollapsed=prob_dict_uncollapsed,
                    prob_dict_collapsed=prob_dict_collapsed,
                    switch_penalty_weight=switch_penalty_weight,
                    initial_last_label=state_before_chunk1_last_label,
                    state_cache=({} if use_state_caching else None),
                    conditioning_alpha=conditioning_alpha,
                    conditioning_combine_fn=conditioning_combine_fn,
                    conditioning_n_prev_labels=conditioning_n_prev_labels,
                    conditioning_interpolation_weights=conditioning_interpolation_weights,
                    conditioning_state_mode=conditioning_state_mode,
                    conditioning_top_m=conditioning_top_m,
                    candidate_top_p=candidate_top_p,
                    candidate_top_k=candidate_top_k,
                    candidate_min_k=candidate_min_k,
                    candidate_source=candidate_source,
                    candidate_apply_to_sync=candidate_apply_to_sync,
                    restrict_log_moves=restrict_log_moves,
                    restrict_model_moves_to_tau=restrict_model_moves_to_tau,
                    max_consecutive_tau_moves=max_consecutive_tau_moves,
                    profile_stats=profile_stats,
                )
                m_elapsed = time.perf_counter() - m_start
                m_steps = end_ts2 - start_ts
                m_rate = (m_steps / m_elapsed) if m_elapsed > 0 else float('inf')

                total_cost += merged_result['total_cost']
                complete_alignment.extend(merged_result['alignment'])
                current_marking = merged_result['final_marking']
                current_last_label = _last_trace_label(merged_result['alignment'])

                chunk_results.append({
                    'chunk_index': len(chunk_results),
                    'start_timestamp': start_ts,
                    'end_timestamp': end_ts2,
                    'chunk_cost': merged_result['total_cost'],
                    'chunk_alignment_length': len(merged_result['alignment']),
                    'final_marking': merged_result['final_marking'],
                    'processing_seconds': m_elapsed,
                    'processing_rate_steps_per_s': m_rate,
                    'merged': True,
                    'merged_from': [start_ts, end_ts1],
                })
                _log_chunk_progress(end_ts2, m_elapsed, True)

                # Advance by two windows and clear any pending cache
                start_ts = end_ts2
                pending_result = None
                pending_start_ts = None
                pending_end_ts = None
                pending_elapsed = None
                pending_rate = None
            else:
                # Accept the first window and continue
                total_cost += result1['total_cost']
                complete_alignment.extend(result1['alignment'])
                current_marking = result1['final_marking']
                current_last_label = last_label_c1

                chunk_results.append({
                    'chunk_index': len(chunk_results),
                    'start_timestamp': start_ts,
                    'end_timestamp': end_ts1,
                    'chunk_cost': result1['total_cost'],
                    'chunk_alignment_length': len(result1['alignment']),
                    'final_marking': result1['final_marking'],
                    'processing_seconds': c1_elapsed,
                    'processing_rate_steps_per_s': c1_rate,
                    'merged': False,
                })
                _log_chunk_progress(end_ts1, c1_elapsed, False)

                # Cache the look-ahead result to avoid recomputing next iteration
                pending_result = result2
                pending_start_ts = end_ts1
                pending_end_ts = end_ts2
                pending_elapsed = c2_elapsed
                pending_rate = c2_rate

                # Move to next window
                start_ts = end_ts1

        total_elapsed = time.perf_counter() - total_start
        total_rate = (n_ts / total_elapsed) if total_elapsed > 0 else float('inf')
        # Finish the in-place progress line
        if inline_progress:
            try:
                sys.stdout.write("\r")
                sys.stdout.flush()
            except Exception:
                pass
        logger.info(
            f"Conformance total {n_ts} steps in {total_elapsed:.3f}s "
            f"({total_rate:.1f} steps/s) across {len(chunk_results)} chunks"
        )

        return {
            'alignment': complete_alignment,
            'total_cost': total_cost,
            'final_marking': current_marking,
            'chunk_results': chunk_results,
            'n_chunks': len(chunk_results),
            'original_matrix_shape': softmax_matrix.shape
        }

    def process_trace_conformance(
        self,
        softmax_matrix: np.ndarray,
        cost_fn: Callable[[float, str], float],
        chunk_size: int = 10,
        eps: float = 1e-12,
        inline_progress: bool = False,
        progress_prefix: str = "",
        prob_dict_uncollapsed: Optional[Dict[Tuple[str, ...], Dict[str, float]]] = None,
        prob_dict_collapsed: Optional[Dict[Tuple[str, ...], Dict[str, float]]] = None,
        switch_penalty_weight: float = 0.0,
        use_state_caching: bool = True,
        merge_mismatched_boundaries: bool = True,
        # removed: restrict_to_observed_moves
        conditioning_alpha: Optional[float] = None,
        conditioning_combine_fn: Optional[Callable[[float, float, float], float]] = None,
        conditioning_n_prev_labels: int = 1,
        conditioning_interpolation_weights: Optional[List[float]] = None,
        conditioning_state_mode: str = "exact",
        conditioning_top_m: int = 3,
        candidate_top_p: Optional[float] = None,
        candidate_top_k: Optional[int] = None,
        candidate_min_k: int = 1,
        candidate_source: str = "conditioned",
        candidate_apply_to_sync: bool = True,
        restrict_log_moves: bool = False,
        restrict_model_moves_to_tau: bool = False,
        max_consecutive_tau_moves: Optional[int] = None,
        progress_log_interval_chunks: int = 0,
        profile_stats: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], List[float]]:
        """
        Wrapper function to replace process_test_case_incremental using chunked_trace_conformance.

        This function maintains the same interface as process_test_case_incremental but uses
        the more efficient chunked conformance checking approach instead of beam search.

        Args:
            softmax_matrix: Softmax probability matrix (n_activities, n_timestamps)
            cost_fn: Cost function for moves
            chunk_size: Size of chunks to process iteratively
            eps: Minimum probability threshold - activities below this are filtered out

        Returns:
            Tuple[List[str], List[float]]: (predicted_sequence, move_costs)
        """
        if self.init_mark is None:
            raise ValueError("Model must have a valid initial marking (init_mark)")

        # Use chunked trace conformance
        result = self.conformance_chunked(
            softmax_matrix=softmax_matrix,
            initial_marking=self.init_mark,
            cost_fn=cost_fn,
            chunk_size=chunk_size,
            eps=eps,
            inline_progress=inline_progress,
            progress_prefix=progress_prefix,
            prob_dict_uncollapsed=prob_dict_uncollapsed,
            prob_dict_collapsed=prob_dict_collapsed,
            switch_penalty_weight=switch_penalty_weight,
            use_state_caching=use_state_caching,
            merge_mismatched_boundaries=merge_mismatched_boundaries,
            conditioning_alpha=conditioning_alpha,
            conditioning_combine_fn=conditioning_combine_fn,
            conditioning_n_prev_labels=conditioning_n_prev_labels,
            conditioning_interpolation_weights=conditioning_interpolation_weights,
            conditioning_state_mode=conditioning_state_mode,
            conditioning_top_m=conditioning_top_m,
            candidate_top_p=candidate_top_p,
            candidate_top_k=candidate_top_k,
            candidate_min_k=candidate_min_k,
            candidate_source=candidate_source,
            candidate_apply_to_sync=candidate_apply_to_sync,
            restrict_log_moves=restrict_log_moves,
            restrict_model_moves_to_tau=restrict_model_moves_to_tau,
            max_consecutive_tau_moves=max_consecutive_tau_moves,
            progress_log_interval_chunks=progress_log_interval_chunks,
            profile_stats=profile_stats,
        )

        # Extract sequence and costs from alignment
        predicted_sequence = []
        move_costs = []
        
        for move_type, move_label, move_cost in result['alignment']:
            # Only include moves that advance the trace (sync and log moves)
            if move_type in ['sync', 'log']:
                predicted_sequence.append(move_label)
                move_costs.append(move_cost)
        
        return predicted_sequence, move_costs
