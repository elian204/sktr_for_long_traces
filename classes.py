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
    search_node_new: Search state for alignment algorithms

The module supports both classical and probabilistic conformance checking
approaches, including conditional probability-based cost functions and
n-gram smoothing for improved alignment quality.
"""

import numpy as np
import copy
from collections import deque
from typing import Dict, List, Set, Tuple, Any
import logging

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


class search_node_new:
    def __init__(self, marking, dist=np.inf, ancestor=None, transition_to_ancestor=None, path_prefix=None,
                 trace_activities_multiset=None, heuristic_distance=None, total_model_moves=0):
        self.dist = dist
        self.ancestor = ancestor
        self.transition_to_ancestor = transition_to_ancestor
        self.marking = marking
        self.path_prefix = path_prefix if path_prefix is not None else []
        self.trace_activities_multiset = trace_activities_multiset
        # Initialize heuristic_distance with a default of 0 if not provided
        self.heuristic_distance = heuristic_distance if heuristic_distance is not None else 0
        self.total_model_moves = total_model_moves
        
    def __lt__(self, other):
        # First compare based on the sum of dist and heuristic_distance
        if (self.dist + self.heuristic_distance) == (other.dist + other.heuristic_distance):
            # If they are equal, compare based on total_model_moves (larger first)
            return self.total_model_moves > other.total_model_moves
        return (self.dist + self.heuristic_distance) < (other.dist + other.heuristic_distance)


    def __repr__(self):
        return f'Node: {self.marking}, dist: {self.dist}, heuristic: {self.heuristic_distance}'    
    
    
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
     
    
    def _find_available_transitions(self, mark_tuple: Tuple[int, ...]) -> List[Transition]:
        """
        Given a marking (as a tuple of token counts), return a list of transitions
        that are enabled (i.e., can fire) from this marking.

        Args:
            mark_tuple (Tuple[int, ...]): The current marking represented as a tuple of token counts.

        Returns:
            List[Transition]: A list of enabled Transition objects for the given marking.
        """
        
        available_transitions = []
        for transition in self.transitions:
            if self.__check_transition_prerequesits(transition, mark_tuple):
                available_transitions.append(transition)
                
        return available_transitions

    
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
        return sync_prod._dijkstra_no_rg_construct(hist_prob_dict, lamda=lamda)
    

    def _fire_transition(self, mark, transition):
        '''Input: Mark object or tuple, Transition object
        Output: Marking object''' 

        # Check if mark is a tuple or an instance of Marking, and get the places accordingly
        if isinstance(mark, tuple):
            places = mark
        elif isinstance(mark, Marking):  # Assuming Marking is a class you've defined
            places = mark.places
        else:
            raise TypeError("Expected mark to be either a tuple or Marking instance")

        subtract_mark = [0] * len(places)
        for arc in transition.in_arcs:
            place_idx = self.places_indices[arc.source.name]
            subtract_mark[place_idx] -= arc.weight
        
        add_mark = [0] * len(places)
        for arc in transition.out_arcs:
            place_idx = self.places_indices[arc.target.name]
            add_mark[place_idx] += arc.weight
  
        new_mark = tuple([sum(x) for x in zip(places, subtract_mark, add_mark)])
        for elem in new_mark:
            if elem < 0:
                print(f'The original mark was: {mark}, subtracting: {subtract_mark}, adding: {add_mark}, \
resulting in: {new_mark}, during transition: {transition.name}')

        new_mark_obj = Marking(new_mark)
        return new_mark_obj

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

    # ---------------------------------------------------------------------------
    #  Helper – canonical converter
    # ---------------------------------------------------------------------------
    def _ensure_marking_object(self, mark):
        """
        Convert `mark` (tuple or Marking) to a Marking instance.

        Parameters
        ----------
        mark : Tuple[int, ...] | Marking
        """
        if isinstance(mark, tuple):
            return Marking(mark)
        if hasattr(mark, "places"):
            return mark         # already a Marking
        raise TypeError(f"Expected tuple or Marking, got {type(mark)}")


    # ---------------------------------------------------------------------------
    #  τ-path exploration for a single marking
    # ---------------------------------------------------------------------------
    def _compute_reachable_transitions_via_tau(
        self,
        marking_places: Tuple[int, ...],
        max_tau_depth: int = 100
    ) -> Dict["Transition", Tuple["Transition", ...]]:
        """
        Return all non-silent transitions reachable from `marking_places`
        together with the shortest τ-path enabling each of them.

        Returns
        -------
        Mapping[Transition, Tuple[tau1, tau2, ...]]
        Empty tuple ⇒ transition is directly enabled.
        """
        if max_tau_depth <= 0:
            raise ValueError("max_tau_depth must be positive")

        reachable: Dict = {}                               # Transition → τ-path
        queue = deque([(marking_places, tuple())])         # (mark, current τ-path)
        visited: Dict[Tuple[int, ...], int] = {}           # Marking → shortest τ-len

        while queue:
            current_mark, tau_path = queue.popleft()

            if visited.get(current_mark, float("inf")) <= len(tau_path):
                continue
            visited[current_mark] = len(tau_path)

            try:
                available = self._find_available_transitions(current_mark)
            except Exception as exc:
                logger.warning("Could not get transitions for %s: %s", current_mark, exc)
                continue

            # record non-silent transitions
            for t in available:
                if t.label is not None:                    # non-silent
                    if t not in reachable or len(tau_path) < len(reachable[t]):
                        reachable[t] = tau_path

            # expand along τ moves
            for t in available:
                if t.label is None:                        # τ
                    if len(tau_path) + 1 > max_tau_depth:
                        continue
                    try:
                        successor = self._fire_transition(
                            self._ensure_marking_object(current_mark), t
                        ).places
                    except Exception as exc:
                        logger.warning("Could not fire τ %s: %s", t.name, exc)
                        continue

                    if visited.get(successor, float("inf")) > len(tau_path) + 1:
                        queue.append((successor, tau_path + (t,)))

        return reachable


    # ---------------------------------------------------------------------------
    #  Main: build and STORE full marking → transition map
    # ---------------------------------------------------------------------------
    def build_marking_transition_map(
        self,
        max_tau_depth: int = 100
    ) -> Dict[Tuple[int, ...], Dict]:
        """
        Compute the map and store it on `self` as `self.marking_transition_map`.
        """
        if self.init_mark is None:
            raise ValueError("Initial marking must be set")
        if max_tau_depth <= 0:
            raise ValueError("max_tau_depth must be positive")

        visited: Set[Tuple[int, ...]] = set()
        queue   = deque([self.init_mark.places])
        result: Dict[Tuple[int, ...], Dict] = {}

        while queue:
            marking = queue.popleft()
            if marking in visited:
                continue

            visited.add(marking)
            result[marking] = {
                "available_transitions": self._compute_reachable_transitions_via_tau(
                    marking, max_tau_depth
                )
            }

            for t in self._find_available_transitions(marking):
                try:
                    succ = self._fire_transition(
                        self._ensure_marking_object(marking), t
                    )
                    if succ.places not in visited:
                        queue.append(succ.places)
                except Exception as exc:
                    logger.warning("Could not fire %s from %s: %s", t.name, marking, exc)

        # -------- NEW: store on the object ------------------------------------
        self.marking_transition_map = result
        return result
