"""
Core classes for Petri net modeling and process conformance checking.

This module provides the fundamental data structures and algorithms for:
- Petri net representation (places, transitions, arcs, markings)
- Reachability graph construction and exploration
- Synchronous product construction for alignment-based conformance checking
- A* and Dijkstra-based optimal alignment computation with probabilistic weights
- Support for partial conformance checking and trace recovery

Main Classes:
    Place, Transition, Arc: Basic Petri net components
    Marking: Token distribution representation
    PetriNet: Main Petri net class with reachability analysis
    SyncProduct: Specialized class for conformance checking alignments
    Graph, Node, Edge: Graph representation structures
    SearchNode: Search state for alignment algorithms

The module supports both classical and probabilistic conformance checking
approaches, including conditional probability-based cost functions and
n-gram smoothing for improved alignment quality.
"""

import numpy as np
import copy
import logging
from collections import deque, Counter
from heapq import heappush, heappop
from typing import Callable, Dict, List, Optional, Set, Tuple, Any, Union
from collections import defaultdict
import heapq

# Configure logger
logger = logging.getLogger(__name__)


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
    def __init__(
        self,
        marking: 'Marking',
        cost: float = float('inf'),
        ancestor: Optional['SearchNode'] = None,
        move_type: Optional[str] = None,
        move_label: Optional[str] = None,
        move_cost: float = 0.0,
        timestamp: int = 0,
    ):
        self.cost = cost
        self.ancestor = ancestor
        self.move_type = move_type
        self.move_label = move_label
        self.move_cost = move_cost
        self.marking = marking
        self.timestamp = timestamp

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


    def _find_available_transitions(self, mark_tuple: Tuple[int, ...]) -> List[Transition]:
        """
        Given a marking (as a tuple of token counts), return a list of transitions
        that are enabled (i.e., can fire) from this marking.

        If a `marking_transition_map` is available, this will return non-silent tau-reachable
        transitions. Otherwise, it returns only directly enabled transitions.

        Args:
            mark_tuple (Tuple[int, ...]): The current marking represented as a tuple of token counts.

        Returns:
            List[Transition]: A list of enabled Transition objects for the given marking.
        """
        if hasattr(self, "marking_transition_map") and self.marking_transition_map is not None:
            entry = self.marking_transition_map.get(mark_tuple)
            if entry and "available_transitions" in entry:
                # Return the list of non-silent transitions reachable via tau-moves.
                return list(entry["available_transitions"].keys())
            else:
                logger.warning(
                    f"Marking {mark_tuple} not found in marking_transition_map or missing 'available_transitions'. "
                    "Falling back to direct enabled transitions."
                )

        # Fallback to original behavior if map is not present or marking is not in map.
        return self._find_directly_enabled_transitions(mark_tuple)

    
    def __check_transition_prerequesits(self, transition: Transition, mark_tuple: Tuple[int, ...]) -> bool:
        """
        Check if the given transition is enabled under the current marking.

        Args:
            transition (Transition): The transition to check.
            mark_tuple (Tuple[int, ...]): The current marking as a tuple of token counts.

        Returns:
            bool: True if the transition is enabled (all input places have enough tokens), False otherwise.
        """
        for arc in transition.in_arcs:
            arc_weight = arc.weight
            source_idx = self.places_indices[arc.source.name]
            if mark_tuple[source_idx] < arc_weight:
                return False
            
        return True
            
    
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
                
        
    def conformance_checking(self, trace_model, hist_prob_dict=None, lamda=0.5):
        sync_prod = self.construct_synchronous_product(trace_model, self.cost_function)      
        return sync_prod.dijkstra_no_rg_construct(hist_prob_dict, lamda=lamda)
    

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

    def _fire_transition(self, mark, transition):
        """
        Fire transition with automatic optimization.
        """
        # Choose implementation based on finalization status
        if self._finalized and hasattr(transition, 'in_idx_weights'):
            # Normalize marking
            places = mark if isinstance(mark, tuple) else mark.places
            
            # Early exit for truly no-op transitions (no inputs AND no outputs)
            if len(transition.in_idx_weights) == 0 and len(transition.out_idx_weights) == 0:
                return Marking(places)
            
            # Fast firing
            new_places = list(places)
            
            # Consume tokens
            for idx, weight in transition.in_idx_weights:
                new_places[idx] -= weight
                if new_places[idx] < 0:
                    place_name = self.places[idx].name
                    raise ValueError(
                        f"Firing '{transition.name}' yields negative tokens at {place_name}: "
                        f"{places[idx]} - {weight} = {new_places[idx]}"
                    )
            
            # Produce tokens
            for idx, weight in transition.out_idx_weights:
                new_places[idx] += weight
            
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
    
      
    def compute_conditioned_weight(self, path_prefix, transition, prob_dict, max_length, lamda=0.5):
        if not prob_dict or not path_prefix or transition.label is None:
            return transition.weight
    
        transition_weight = transition.weight
        transition_label = transition.label
        path_prefix_tuple = tuple(path_prefix)
    
        def adjusted_weight(prefix):
            if transition_label in prob_dict[prefix]:
                return (1 - lamda) * (1 - prob_dict[prefix][transition_label]) + lamda * transition_weight
            return (1 - lamda) + lamda * transition_weight
    
        if path_prefix_tuple in prob_dict:
            return adjusted_weight(path_prefix_tuple)
    
        longest_prefix = self.find_longest_prefix(path_prefix_tuple, prob_dict, max_length)
        if longest_prefix:
            return adjusted_weight(longest_prefix)
    
        return 1  # Default cost for a non-sync move
    
    def find_longest_prefix(self, path_prefix, prob_dict, max_length):
        for i in range(min(len(path_prefix), max_length), 0, -1):
            sub_prefix = path_prefix[-i:]
            if sub_prefix in prob_dict:
                return sub_prefix
        return None

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
            
            for transition in enabled_transitions:
                if transition.label is None:  # τ transition
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
                else:  # Non-silent transition
                    if transition not in reachable_transitions or len(tau_path) < len(reachable_transitions[transition]):
                        reachable_transitions[transition] = tau_path
        
        return reachable_transitions


    def build_marking_transition_map(self, max_tau_depth: int = 100) -> Dict[Tuple[int, ...], Dict]:
        """Build complete marking-to-transition map and store on self."""
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
                
                # Add successors to queue
                try:
                    enabled_transitions = self._find_directly_enabled_transitions(current_marking)
                    for transition in enabled_transitions:
                        try:
                            successor = self._fire_transition(
                                Marking(current_marking), 
                                transition
                            )
                            if successor.places not in visited:
                                queue.append(successor.places)
                        except Exception as exc:
                            logger.warning(f"Could not fire {transition.name} from {current_marking}: {exc}")
                except Exception as exc:
                    logger.warning(f"Could not get transitions for marking {current_marking}: {exc}")
                        
            except Exception as exc:
                logger.warning(f"Could not compute τ-reachability for {current_marking}: {exc}")
                result[current_marking] = {"available_transitions": {}}
        
        self.marking_transition_map = result
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
        if hasattr(self, "marking_transition_map") and self.marking_transition_map:
            entry = self.marking_transition_map.get(marking_tuple)
            if entry and "available_transitions" in entry:
                return entry["available_transitions"]
        
        # Compute if not cached
        return self._compute_reachable_transitions_via_tau(marking_tuple, max_tau_depth)
    
    def get_tau_reachable_transitions_initial(self, max_tau_depth=100):
        """Get tau-reachable transitions for the initial marking."""
        return self.get_tau_reachable_transitions(self.init_mark, max_tau_depth)
    
    def get_tau_reachable_transitions_final(self, max_tau_depth=100):
        """Get tau-reachable transitions for the final marking."""
        if self.final_mark is None:
            raise ValueError("Final marking not set")
        return self.get_tau_reachable_transitions(self.final_mark, max_tau_depth)
    

    def dijkstra_no_rg_construct(
        self,
        prob_dict: Optional[Dict[Any, float]] = None,
        lambda_: float = 0.5,
        partial_conformance: bool = False,
        return_net_final_marking: bool = False,
        n_unique_final_markings: int = 1,
        overlap_size: int = 0,
        trace_activities_multiset: Optional[Dict[Any, int]] = None,
        use_heuristic_distance: bool = False,
        trace_recovery: bool = False,
        max_hist_len: Optional[int] = None,
    ) -> Tuple[Any, int]:
        """
        Perform Dijkstra-based search over synchronous product without constructing
        a replay graph.
        
        Expands minimal-distance nodes until `n_unique_final_markings` are found.
        
        Assumes helper methods return nodes with:
            - .marking.places: Tuple[...] as the state key
            - .dist: float as the path cost
        
        Returns
        -------
        Tuple[Any, int]
            Processed final node information and count of nodes opened during search.
        """
        # Initialize data structures
        open_heap: List[Any] = []
        best_distances: Dict[Tuple[Any, ...], float] = {}
        visited_markings: Set[Tuple[Any, ...]] = set()
        final_nodes: List[Any] = []
        final_markings_unique: Set[Tuple[Any, ...]] = set()
        nodes_opened: int = 0
        
        # Start search from initial node
        start = self._initialize_dijkstra_node(
            trace_activities_multiset, use_heuristic_distance
        )
        heappush(open_heap, start)
        
        while open_heap:
            current = heappop(open_heap)
            key = current.marking.places
            
            # Skip already processed markings
            if key in visited_markings:
                continue
                
            # Check if current node is a valid final state
            if self._is_dijkstra_final_node(
                current, partial_conformance, n_unique_final_markings, final_markings_unique
            ):
                final_nodes.append(current)
                if len(final_nodes) >= n_unique_final_markings:
                    break
                continue
            
            # Expand current node
            nodes_opened += 1
            for transition in self._find_available_transitions(key):
                successor = self._create_dijkstra_successor_node(
                    current, transition, prob_dict, lambda_, max_hist_len, use_heuristic_distance
                )
                successor_key = successor.marking.places
                
                # Add successor if it improves known distance
                if self._should_add_dijkstra_node(successor, best_distances):
                    best_distances[successor_key] = successor.dist
                    heappush(open_heap, successor)
            
            visited_markings.add(key)
        
        # Process and return results
        result = self._process_dijkstra_final_node(
            final_nodes,
            partial_conformance,
            overlap_size,
            trace_recovery,
            return_net_final_marking,
        )
        return result, nodes_opened


    def _initialize_dijkstra_node(self, trace_activities_multiset, use_heuristic_distance):
            """
            Initialize a new node for Dijkstra's algorithm.
            
            Args:
                trace_activities_multiset: Multiset of trace activities
                use_heuristic_distance: Boolean flag to determine if heuristic should be used
                
            Returns:
                A new search node initialized with the initial marking and calculated heuristic
            """
            # Calculate initial heuristic if enabled, otherwise use 0
            init_heuristic = (self.estimate_alignment_heuristic(self.init_mark, trace_activities_multiset) 
                            if use_heuristic_distance else 0)
            
            # Ensure trace_activities_multiset is a set if None
            trace_activities_multiset = trace_activities_multiset or set()
            
            # Create and return new search node
            return SearchNode(
                marking=self.init_mark,
                dist=0,
                trace_activities_multiset=trace_activities_multiset.copy(),
                heuristic_distance=init_heuristic,
                total_model_moves=0
            )

    def _is_dijkstra_final_node(self, node, partial_conformance):
        """
        Determines whether the given node is a final node according to the Dijkstra criteria.
        
        In partial conformance mode, it checks whether the tail portion of the node's marking
        (with length equal to the trace model's places) matches the trace model's final marking.
        If so, it extracts the model's marking from the node and records it.
        
        In full conformance mode, it checks whether the entire marking of the node matches the final marking.
        
        Args:
            node: The node to evaluate, which has a 'marking' attribute.
            partial_conformance (bool): Flag indicating whether partial conformance is used.
        
        Returns:
            bool: True if the node qualifies as a final node; otherwise, False.
        """
        if partial_conformance:
            # Define the length of the trace model's marking segment.
            trace_places_count = len(self.trace_model.places)
            # Extract the tail portion from the node's marking.
            node_tail_marking = node.marking.places[-trace_places_count:]
            # Compare with the trace model's final marking.
            if node_tail_marking == self.trace_model.final_mark.places:
                # Extract the model portion from the node's marking.
                model_marking = node.marking.places[:len(self.net.places)]
                return True
        else:
            # For full conformance, compare the entire marking.
            if node.marking.places == self.final_mark.places:
                return True
        return False
    

    def _create_dijkstra_successor_node(
        self,
        current_node: SearchNode,
        transition: Transition,
        prob_dict: Dict[Any, float],
        lamda: float,
        max_hist_len: int,
        use_heuristic_distance: bool,
        nodes_opened: int
    ) -> SearchNode:
        """
        Fire a transition in the synchronous product and build the corresponding
        Dijkstra search node.

        - Updates the marking by firing the transition.
        - Computes the conditioned transition weight.
        - Extends the path prefix if the transition has a label.
        - Computes heuristic distance and remaining activities.
        - Increments model/trace move counts based on the transition type.
        """
        # 1. Fire the transition and compute its cost
        new_marking = self._fire_transition(current_node.marking, transition)
        conditioned_weight = self.compute_conditioned_weight(
            current_node.path_prefix,
            transition,
            prob_dict,
            max_length=max_hist_len,
            lamda=lamda,
        )

        # 2. Build the new path prefix (only add label when present)
        new_prefix = (
            current_node.path_prefix + [transition.label]
            if transition.label is not None
            else current_node.path_prefix
        )

        # 3. Heuristic estimate and leftover trace activities
        heuristic_dist, leftover_multiset = self._compute_dijkstra_heuristic(
            current_node,
            transition,
            new_marking,
            use_heuristic_distance,
        )

        # 4. Update move counters
        is_model_move = transition.move_type in {"model", "sync"}
        is_trace_move = transition.move_type in {"trace", "sync"}
        new_model_moves = current_node.total_model_moves + int(is_model_move)
        new_trace_moves = current_node.total_trace_moves + int(is_trace_move)

        # 5. Construct and return the new search node
        return SearchNode(
            new_marking,
            dist=current_node.dist + conditioned_weight,
            ancestor=current_node,
            move_type=transition.move_type,
            move_label=transition.label,
            move_cost=conditioned_weight,
            timestamp=current_node.timestamp + 1
        )
    
    
    def _should_add_dijkstra_node(
        self,
        new_node: SearchNode,
        marking_distance: Dict[Any, float]
    ) -> bool:
        """
        Decide whether a newly generated Dijkstra node should be added to the
        open set. Returns True if:
        - We haven't seen this marking before, or
        - We've found a strictly shorter path to this marking.
        """
        places = new_node.marking.places
        previous_dist = marking_distance.get(places)
        return previous_dist is None or new_node.dist < previous_dist

    
    def _process_dijkstra_final_node(self, final_node):
        """
        Processes the final node obtained from the Dijkstra search to build the alignment path,
        calculate the total cost, and extract the final marking and node path.
    
        The function performs the following steps:
          1. Validates that the final node is not None.
          2. Builds the alignment path and computes the total distance using `_build_dijkstra_path`.
          3. Extracts the net's final marking (only considering places corresponding to the net).
          4. Constructs the complete node path from the initial node to the final node.
          5. Retrieves the count of nodes opened during the search.
    
        Args:
            final_node: The final node from the Dijkstra search. Must not be None.
    
        Returns:
            A tuple containing:
              - alignment: A list of Transition objects representing the alignment path.
              - total_distance: A float representing the total cost/distance of the path.
              - net_final_marking: A Marking object for the net's final marking (using only the net's places).
              - node_path: A list of node objects representing the complete path from the initial node to the final node.
              - nodes_opened: An integer count of nodes opened during the search.
    
        Raises:
            ValueError: If the provided final_node is None.
        """
        if final_node is None:
            raise ValueError("Final search node during Dijkstra search is None.")
    
        # Build the alignment path and compute the total distance.
        alignment, total_distance = self._build_dijkstra_path(final_node)
    
        # Extract the net's final marking (only include places corresponding to the net).
        net_final_marking = Marking(final_node.marking.places[:len(self.net.places)])
    
        # Reconstruct the complete node path from initial to final.
        node_path = []
        current = final_node
        while current:
            node_path.append(current)
            current = current.ancestor
        node_path.reverse()  # Now the path is from the initial node to the final node.
    
        nodes_opened = final_node.nodes_opened
    
        return alignment, total_distance, net_final_marking, nodes_opened
    

    def partial_trace_conformance(
        self,
        softmax_matrix: np.ndarray,
        initial_marking: 'Marking',
        cost_fn: Callable[[float, str], float],
        eps: float = 1e-12,
    ) -> Dict[str, Any]:
        """
        Compute a partial trace conformance alignment using Dijkstra/A*-style search.
        
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
        label2idx = {str(i): i for i in range(n_acts)}

        # Min-heap of (cost, node)
        open_set: List[Tuple[float, SearchNode]] = []
        start = SearchNode(marking=initial_marking, cost=0.0, timestamp=0)
        heapq.heappush(open_set, (0.0, start))

        # Best-known cost per (marking.places, timestamp)
        best: Dict[Tuple[Tuple[int, ...], int], float] = defaultdict(lambda: float('inf'))

        while open_set:
            cost, node = heapq.heappop(open_set)
            key = (node.marking.places, node.timestamp)
            if cost > best[key]:
                continue
            best[key] = cost

            # Goal reached: consumed all timestamps
            if node.timestamp == n_ts:
                return {
                    'alignment': node.reconstruct_path(),
                    'total_cost': node.cost,
                    'final_marking': node.marking
                }

            enabled = self._find_available_transitions(node.marking.places)

            # 1) Model moves (silent τ or labeled model moves; timestamp unchanged)
            for t in enabled:
                # Use macro transition if this transition comes from marking_transition_map
                if (hasattr(self, "marking_transition_map") and 
                    self.marking_transition_map is not None and
                    node.marking.places in self.marking_transition_map and
                    t in self.marking_transition_map[node.marking.places]["available_transitions"]):
                    # This transition requires tau-path firing
                    new_mark = self._fire_macro_transition(node.marking, t)
                else:
                    # This is a directly enabled transition
                    new_mark = self._fire_transition(node.marking, t)
                
                move_type = 'tau' if t.label is None else 'model'
                c = cost_fn(0.0, move_type)
                new_cost = cost + c
                new_key = (new_mark.places, node.timestamp)
                if new_cost < best.get(new_key, float('inf')):
                    heapq.heappush(open_set, (
                        new_cost,
                        SearchNode(
                            marking=new_mark,
                            cost=new_cost,
                            ancestor=node,
                            move_type=move_type,
                            move_label=t.label or 'τ',
                            move_cost=c,
                            timestamp=node.timestamp
                        )
                    ))

            # 2) Log moves (advance timestamp without firing any transition)
            if node.timestamp < n_ts:
                for label, idx in label2idx.items():
                    raw_p = softmax_matrix[idx, node.timestamp]
                    # Filter out activities below threshold (same as beam search)
                    if raw_p < eps:
                        continue
                    p = max(raw_p, 1e-12)  # Small epsilon for numerical stability
                    c = cost_fn(p, 'log')
                    new_cost = cost + c
                    new_key = (node.marking.places, node.timestamp + 1)
                    if new_cost < best.get(new_key, float('inf')):
                        heapq.heappush(open_set, (
                            new_cost,
                            SearchNode(
                                marking=node.marking,
                                cost=new_cost,
                                ancestor=node,
                                move_type='log',
                                move_label=label,
                                move_cost=c,
                                timestamp=node.timestamp + 1
                            )
                        ))

            # 3) Synchronous moves (labeled transitions that match softmax label; advance timestamp)
            for t in enabled:
                if t.label is None or t.label not in label2idx:
                    continue
                idx = label2idx[t.label]
                raw_p = softmax_matrix[idx, node.timestamp]
                # Filter out activities below threshold (same as beam search)
                if raw_p < eps:
                    continue
                p = max(raw_p, 1e-12)  # Small epsilon for numerical stability
                c = cost_fn(p, 'sync')
                new_cost = cost + c
                
                # Use macro transition if this transition comes from marking_transition_map
                if (hasattr(self, "marking_transition_map") and 
                    self.marking_transition_map is not None and
                    node.marking.places in self.marking_transition_map and
                    t in self.marking_transition_map[node.marking.places]["available_transitions"]):
                    # This transition requires tau-path firing
                    new_mark = self._fire_macro_transition(node.marking, t)
                else:
                    # This is a directly enabled transition
                    new_mark = self._fire_transition(node.marking, t)
                
                new_key = (new_mark.places, node.timestamp + 1)
                if new_cost < best.get(new_key, float('inf')):
                    heapq.heappush(open_set, (
                        new_cost,
                        SearchNode(
                            marking=new_mark,
                            cost=new_cost,
                            ancestor=node,
                            move_type='sync',
                            move_label=t.label,
                            move_cost=c,
                            timestamp=node.timestamp + 1
                        )
                    ))

        raise ValueError("No conforming path found for the partial trace.")


    def conformance_chunked(
        self,
        softmax_matrix: np.ndarray,
        initial_marking: 'Marking',
        cost_fn: Callable[[float, str], float],
        chunk_size: int = 10,
        eps: float = 1e-12,
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
        n_acts, n_ts = softmax_matrix.shape
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        current_marking = initial_marking
        total_cost = 0.0
        complete_alignment: List[Tuple[str, str]] = []
        chunk_results: List[Dict[str, Any]] = []

        for chunk_idx, start_ts in enumerate(range(0, n_ts, chunk_size)):
            end_ts = min(start_ts + chunk_size, n_ts)
            chunk = softmax_matrix[:, start_ts:end_ts]

            result = self.partial_trace_conformance(
                softmax_matrix=chunk,
                initial_marking=current_marking,
                cost_fn=cost_fn,
                eps=eps
            )

            # Accumulate
            total_cost += result['total_cost']
            complete_alignment.extend(result['alignment'])
            current_marking = result['final_marking']

            chunk_results.append({
                'chunk_index': chunk_idx,
                'start_timestamp': start_ts,
                'end_timestamp': end_ts,
                'chunk_cost': result['total_cost'],
                'chunk_alignment_length': len(result['alignment']),
                'final_marking': result['final_marking']
            })

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
            eps=eps
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




