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
from heapq import heappush, heappop
from collections import deque
from typing import Dict, List, Set, Tuple, Union, Optional, Any


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
     
    
    def _find_available_transitions(self, mark_tuple):
        '''Input: tuple
           Output: list'''
        
        available_transitions = []
        for transition in self.transitions:
            if self.__check_transition_prerequesits(transition, mark_tuple):
                available_transitions.append(transition)
                
        return available_transitions

    
    def __check_transition_prerequesits(self, transition, mark_tuple):
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


class SyncProduct(PetriNet):
   
    def __init__(self, net, trace_model, cost_function=None, init_mark=None, final_mark=None):
           
        super().__init__()  
        net.assign_model_transitions_move_type()
        trace_model.assign_trace_transitions_move_type()
        
        self.net = net
        self.trace_model = trace_model
        self.places = self.net.places + self.trace_model.places
        self.transitions = self.net.transitions + self.trace_model.transitions
        self.arcs = self.net.arcs + self.trace_model.arcs

        self.cost_function = lambda x: 0 if cost_function is None else cost_function(x)

        new_sync_transitions = self.net._generate_all_sync_transitions(self.trace_model, cost_function)
        self.add_transitions_with_arcs(new_sync_transitions)
        self.update_sync_product_trans_names()
        
        self.places_indices = {place.name: idx for idx, place in enumerate(self.places)}
        self.transitions_indices = {transition.name: idx for idx, transition in enumerate(self.transitions)}        
    
        self.init_mark = Marking(self.net.init_mark.places + self.trace_model.init_mark.places) if init_mark is None else init_mark
        self.final_mark = Marking(self.net.final_mark.places + self.trace_model.final_mark.places) if final_mark is None else final_mark


    def update_sync_product_trans_names(self, sync_product=None):
        if sync_product is None:
            transitions = self.transitions
        else:
            transitions = sync_product.transitions

        for trans in transitions:
            if trans.move_type == 'model':
                if not hasattr(self.net, '_sync_names_updated') or not self.net._sync_names_updated:
                    trans.name = f'(>>, {trans.name})'
            elif trans.move_type == 'trace':
                trans.name = f'({trans.name}, >>)'
            else:
                trans.name = f'({trans.name}, {trans.name})'

        if not hasattr(self.net, '_sync_names_updated'):
            self.net._sync_names_updated = True

        transitions_indices = {transitions[i].name: i for i in range(len(transitions))}

        if sync_product is not None:
            sync_product.transitions_indices = transitions_indices
            return sync_product
        else:
            self.transitions_indices = transitions_indices


    def construct_synchronous_product(self, trace_model, cost_function, net_init_mark=None, net_final_mark=None):
        return SyncProduct(net=self, trace_model=trace_model, cost_function=cost_function,
                           net_init_mark=net_init_mark, net_final_mark=net_final_mark)


    def _dijkstra_no_rg_construct(
        self,
        prob_dict: Optional[Dict[Tuple[str, ...], Dict[str, float]]] = None,
        alpha: float = 0.5,
        partial_conformance: bool = False,
        return_net_final_marking: bool = False,
        n_unique_final_markings: int = 1,
        overlap_size: int = 0,
        trace_activities_multiset: Optional[Set[str]] = None,
        use_heuristic_distance: bool = False,
        trace_recovery: bool = False,
        use_cond_probs: bool = False,
        lambdas: Optional[Tuple[float, ...]] = None,
        use_ngram_smoothing: bool = False
    ) -> Tuple[Union[List[Transition], List[Tuple]], int]:
        """
        Perform Dijkstra's algorithm without reachability graph construction.
        
        This method implements a variant of Dijkstra's algorithm for finding optimal
        alignments between process models and traces, with support for probabilistic
        weights and various optimization strategies.
        
        Parameters
        ----------
        prob_dict : dict, optional
            Probability dictionary for conditional weights
        alpha : float, default=0.5
            Blending factor for probability calculations
        partial_conformance : bool, default=False
            Whether to perform partial conformance checking
        return_net_final_marking : bool, default=False
            Whether to return the net final marking
        n_unique_final_markings : int, default=1
            Number of unique final markings to find
        overlap_size : int, default=0
            Size of overlap for partial conformance
        trace_activities_multiset : set, optional
            Multiset of activities in the trace
        use_heuristic_distance : bool, default=False
            Whether to use A* heuristic
        trace_recovery : bool, default=False
            Whether to recover full alignment trace
        use_cond_probs : bool, default=False
            Whether to use conditional probabilities
        lambdas : tuple, optional
            N-gram weights for smoothing
        use_ngram_smoothing : bool, default=False
            Whether to use n-gram smoothing
            
        Returns
        -------
        tuple
            (results, nodes_opened) where results depend on parameters
        """
        search_state = self._initialize_search_state()
        init_node = self._initialize_dijkstra_node(
            trace_activities_multiset,
            use_heuristic_distance
        )
        
        self._add_node_to_heap(search_state['heap'], init_node)
        
        while search_state['heap'] and not self._is_search_complete(
            search_state, n_unique_final_markings
        ):
            current_node = self._get_next_node(search_state)
            
            if self._should_skip_node(current_node, search_state):
                continue
            
            if self._process_potential_final_node(
                current_node, search_state, partial_conformance, 
                n_unique_final_markings
            ):
                continue
            
            self._expand_node(
                current_node, search_state, prob_dict, lambdas,
                use_cond_probs, alpha, use_heuristic_distance,
                use_ngram_smoothing
            )
        
        results = self._process_dijkstra_final_nodes(
            search_state['final_nodes'],
            partial_conformance,
            overlap_size,
            trace_recovery,
            return_net_final_marking
        )
        
        return results, search_state['nodes_opened']
    
    def _initialize_search_state(self) -> Dict[str, Any]:
        """Initialize the search state with empty data structures."""
        return {
            'heap': [],
            'marking_distance_dict': {},
            'visited_markings': set(),
            'final_nodes': [],
            'final_markings_unique': set(),
            'nodes_opened': 0
        }
    
    def _add_node_to_heap(self, heap: List[search_node_new], node: search_node_new) -> None:
        """Add a node to the priority queue."""
        heappush(heap, node)
    
    def _get_next_node(self, search_state: Dict[str, Any]) -> search_node_new:
        """Get the next node from the priority queue."""
        return heappop(search_state['heap'])
    
    def _should_skip_node(
        self, 
        node: search_node_new, 
        search_state: Dict[str, Any]
    ) -> bool:
        """Check if a node should be skipped."""
        return tuple(node.marking.places) in search_state['visited_markings']
    
    def _is_search_complete(
        self, 
        search_state: Dict[str, Any], 
        n_unique_final_markings: int
    ) -> bool:
        """Check if the search is complete."""
        return len(search_state['final_nodes']) >= n_unique_final_markings
    
    def _process_potential_final_node(
        self,
        node: search_node_new,
        search_state: Dict[str, Any],
        partial_conformance: bool,
        n_unique_final_markings: int
    ) -> bool:
        """
        Process a node that might be a final node.
        
        Returns
        -------
        bool
            True if node was final and processed, False otherwise
        """
        if self._is_dijkstra_final_node(
            node, 
            partial_conformance, 
            n_unique_final_markings, 
            search_state['final_markings_unique']
        ):
            search_state['final_nodes'].append(node)
            return True
        return False
    
    def _expand_node(
        self,
        current_node: search_node_new,
        search_state: Dict[str, Any],
        prob_dict: Optional[Dict[Tuple[str, ...], Dict[str, float]]],
        lambdas: Optional[Tuple[float, ...]],
        use_cond_probs: bool,
        alpha: float,
        use_heuristic_distance: bool,
        use_ngram_smoothing: bool
    ) -> None:
        """Expand a node by generating all successors."""
        search_state['nodes_opened'] += 1
        
        available_transitions = self._find_available_transitions(
            current_node.marking.places
        )
        
        for transition in available_transitions:
            new_node = self._create_dijkstra_successor_node(
                current_node, transition, prob_dict, lambdas,
                use_cond_probs, alpha, use_heuristic_distance,
                use_ngram_smoothing
            )
            
            self._update_search_state_with_node(new_node, search_state)
        
        search_state['visited_markings'].add(tuple(current_node.marking.places))
    
    def _update_search_state_with_node(
        self,
        new_node: search_node_new,
        search_state: Dict[str, Any]
    ) -> None:
        """Update search state with a new node if it should be added."""
        if self._should_add_dijkstra_node(
            new_node, 
            search_state['marking_distance_dict']
        ):
            marking_key = tuple(new_node.marking.places)
            search_state['marking_distance_dict'][marking_key] = new_node.dist
            self._add_node_to_heap(search_state['heap'], new_node)
    
    def _initialize_dijkstra_node(
        self,
        trace_activities_multiset: Optional[Set[str]],
        use_heuristic_distance: bool
    ) -> search_node_new:
        """
        Initialize the starting node for Dijkstra's algorithm.
        
        Parameters
        ----------
        trace_activities_multiset : set of str, optional
            Multiset of activities in the trace
        use_heuristic_distance : bool
            Whether to use A* heuristic
            
        Returns
        -------
        search_node_new
            Initial search node
        """
        trace_activities = trace_activities_multiset or set()
        init_heuristic = 0
        
        if use_heuristic_distance:
            init_heuristic = self.estimate_alignment_heuristic(
                self.init_mark, 
                trace_activities
            )
        
        return search_node_new(
            marking=self.init_mark,
            dist=0,
            trace_activities_multiset=trace_activities.copy(),
            heuristic_distance=init_heuristic,
            total_model_moves=0
        )
    
    def _is_dijkstra_final_node(
        self,
        node: search_node_new,
        partial_conformance: bool,
        n_unique_final_markings: int,
        final_markings_unique: Set[Tuple[int, ...]]
    ) -> bool:
        """
        Check if a node represents a final state.
        
        Parameters
        ----------
        node : search_node_new
            Current search node
        partial_conformance : bool
            Whether performing partial conformance check
        n_unique_final_markings : int
            Number of unique final markings found
        final_markings_unique : set
            Set of unique final markings
            
        Returns
        -------
        bool
            True if node is a final state
        """
        if partial_conformance:
            trace_marking = node.marking.places[-len(self.trace_model.places):]
            if trace_marking == self.trace_model.final_mark.places:
                model_marking = tuple(node.marking.places[:len(self.net.places)])
                if model_marking not in final_markings_unique:
                    final_markings_unique.add(model_marking)
                    return True
        else:
            return node.marking.places == self.final_mark.places
        
        return False
    
    def _create_dijkstra_successor_node(
        self,
        current_node: search_node_new,
        transition: Transition,
        prob_dict: Optional[Dict[Tuple[str, ...], Dict[str, float]]],
        lambdas: Optional[Tuple[float, ...]],
        use_cond_prob: bool,
        alpha: float,
        use_heuristic_distance: bool,
        use_ngram_smoothing: bool
    ) -> search_node_new:
        """
        Create a successor node by firing a transition.
        
        Parameters
        ----------
        current_node : search_node_new
            Current search node
        transition : Transition
            Transition to fire
        prob_dict : dict, optional
            Probability dictionary for conditional weights
        lambdas : tuple of float, optional
            N-gram weights
        use_cond_prob : bool
            Whether to use conditional probabilities
        alpha : float
            Blending factor for probabilities
        use_heuristic_distance : bool
            Whether to use A* heuristic
        use_ngram_smoothing : bool
            Whether to use n-gram smoothing
            
        Returns
        -------
        search_node_new
            New successor node
        """
        new_marking = self._fire_transition(current_node.marking, transition)
        
        transition_weight = self.compute_conditioned_weight(
            path_prefix=current_node.path_prefix,
            transition=transition,
            prob_dict=prob_dict,
            lambdas=lambdas,
            use_cond_prob=use_cond_prob,
            alpha=alpha,
            use_ngram_smoothing=use_ngram_smoothing
        )
        
        new_path_prefix = self._update_path_prefix(
            current_node.path_prefix,
            transition
        )
        
        heuristic_distance, leftover_activities = self._compute_dijkstra_heuristic(
            current_node,
            transition,
            new_marking,
            use_heuristic_distance
        )
        
        total_model_moves = self._update_model_moves(
            current_node.total_model_moves,
            transition
        )
        
        return search_node_new(
            marking=new_marking,
            dist=current_node.dist + transition_weight,
            ancestor=current_node,
            transition_to_ancestor=transition,
            heuristic_distance=heuristic_distance,
            path_prefix=new_path_prefix,
            trace_activities_multiset=leftover_activities,
            total_model_moves=total_model_moves
        )
    
    def _update_path_prefix(
        self,
        current_prefix: List[str],
        transition: Transition
    ) -> List[str]:
        """Update path prefix based on transition type."""
        if transition.move_type in {'trace', 'sync'}:
            return current_prefix + [transition.label]
        return current_prefix
    
    def _update_model_moves(
        self,
        current_total: int,
        transition: Transition
    ) -> int:
        """Update total model moves counter."""
        if transition.move_type in {'model', 'sync'}:
            return current_total + 1
        return current_total
    
    def _compute_dijkstra_heuristic(
        self,
        current_node: search_node_new,
        transition: Transition,
        new_marking: Marking,
        use_heuristic_distance: bool
    ) -> Tuple[float, Set[str]]:
        """
        Compute heuristic distance for A* search.
        
        Parameters
        ----------
        current_node : search_node_new
            Current search node
        transition : Transition
            Transition being fired
        new_marking : Marking
            New marking after firing
        use_heuristic_distance : bool
            Whether to compute heuristic
            
        Returns
        -------
        tuple
            (heuristic_distance, leftover_trace_activities)
        """
        if not use_heuristic_distance:
            return 0, set()
        
        leftover_activities = current_node.trace_activities_multiset.copy()
        
        if transition.move_type in {'trace', 'sync'}:
            leftover_activities = self.subtract_activities(
                leftover_activities,
                transition.label
            )
        
        heuristic_distance = self.estimate_alignment_heuristic(
            new_marking,
            leftover_activities
        )
        
        return heuristic_distance, leftover_activities
    
    def _should_add_dijkstra_node(
        self,
        new_node: search_node_new,
        marking_distance_dict: Dict[Tuple[int, ...], float]
    ) -> bool:
        """Check if a node should be added to the search space."""
        marking_key = tuple(new_node.marking.places)
        return (marking_key not in marking_distance_dict or
                marking_distance_dict[marking_key] > new_node.dist)
    
    def _process_dijkstra_final_nodes(
        self,
        final_nodes: List[search_node_new],
        partial_conformance: bool,
        overlap_size: int,
        trace_recovery: bool,
        return_net_final_marking: bool
    ) -> Union[List[Transition], List[Tuple]]:
        """
        Process final nodes to extract alignment results.
        
        Parameters
        ----------
        final_nodes : list of search_node_new
            List of final nodes
        partial_conformance : bool
            Whether performing partial conformance
        overlap_size : int
            Size of overlap in trace model
        trace_recovery : bool
            Whether to recover alignment trace
        return_net_final_marking : bool
            Whether to return final marking
            
        Returns
        -------
        list
            Either list of transitions or list of tuples
        """
        if trace_recovery and final_nodes:
            return self._recover_alignment_trace(final_nodes[0])
        
        final_trace_marking = self._create_final_trace_marking(overlap_size)
        results = []
        
        for node in final_nodes:
            path = self._build_alignment_path(node)
            
            if return_net_final_marking:
                model_marking = Marking(node.marking.places[:len(self.net.places)])
                results.append((path, node.dist, model_marking, node.heuristic_distance))
            else:
                results.append((path, node.dist, node.heuristic_distance))
        
        return results
    
    def _create_final_trace_marking(self, overlap_size: int) -> List[int]:
        """Create final trace marking for partial conformance."""
        marking = [0] * len(self.trace_model.places)
        marking[len(self.trace_model.places) - overlap_size - 1] = 1
        return marking
    
    def _recover_alignment_trace(self, node: search_node_new) -> List[Transition]:
        """Recover the sequence of transitions from final node."""
        alignment = []
        current = node
        
        while current.ancestor:
            alignment.append(current.transition_to_ancestor)
            current = current.ancestor
        
        return alignment[::-1]
    
    def _build_alignment_path(self, node: search_node_new) -> List[Transition]:
        """Build the complete alignment path from a final node."""
        path = []
        current = node
        
        while current.ancestor:
            path.append(current.transition_to_ancestor)
            current = current.ancestor
        
        return path[::-1]
    
    def compute_conditioned_weight(
        self,
        path_prefix: Union[List[str], Tuple[str, ...]],
        transition: Transition,
        prob_dict: Optional[Dict[Tuple[str, ...], Dict[str, float]]] = None,
        lambdas: Optional[Tuple[float, ...]] = None,
        use_cond_prob: bool = False,
        alpha: float = 0.5,
        use_ngram_smoothing: bool = False,
    ) -> float:
        """
        Compute conditional weight of a transition.
        
        Supports both n-gram smoothing and prefix search approaches for
        computing conditional probabilities based on path history.
        
        Parameters
        ----------
        path_prefix : list or tuple of str
            Sequence of activities leading to current transition
        transition : Transition
            Transition to compute weight for
        prob_dict : dict, optional
            Probability dictionary for conditional weights
        lambdas : tuple of float, optional
            Weights for n-gram lengths (required for n-gram smoothing)
        use_cond_prob : bool
            Whether to use conditional probabilities
        alpha : float
            Blending factor between original and conditional weights
        use_ngram_smoothing : bool
            Whether to use n-gram smoothing approach
            
        Returns
        -------
        float
            Conditioned weight for the transition
            
        Raises
        ------
        ValueError
            If required parameters are missing or invalid
        """
        if not use_cond_prob or prob_dict is None:
            return transition.weight
        
        if isinstance(prob_dict, dict) and not prob_dict:
            raise ValueError("Cannot compute conditional probabilities with empty prob_dict")
        
        # Return static weight for non-synchronous moves
        if transition.move_type in {'trace', 'model'} or transition.label is None:
            return transition.weight
        
        prefix_tuple = tuple(path_prefix)
        
        if use_ngram_smoothing:
            return self._compute_ngram_based_weight(
                prefix_tuple, transition, prob_dict, lambdas, alpha
            )
        else:
            return self._compute_prefix_based_weight(
                prefix_tuple, transition, prob_dict, alpha
            )
    
    def _compute_ngram_based_weight(
        self,
        prefix_tuple: Tuple[str, ...],
        transition: Transition,
        prob_dict: Dict[Tuple[str, ...], Dict[str, float]],
        lambdas: Tuple[float, ...],
        alpha: float
    ) -> float:
        """Compute weight using n-gram smoothing approach."""
        if lambdas is None:
            raise ValueError("Lambdas must be provided for n-gram smoothing")
        
        prob = self._compute_ngram_probability(
            prefix_tuple, transition.label, prob_dict, lambdas
        )
        blended_prob = (1 - alpha) * prob + alpha * transition.prob
        return self.cost_function(blended_prob)
    
    def _compute_prefix_based_weight(
        self,
        prefix_tuple: Tuple[str, ...],
        transition: Transition,
        prob_dict: Dict[Tuple[str, ...], Dict[str, float]],
        alpha: float
    ) -> float:
        """Compute weight using prefix search approach."""
        prob = self._compute_prefix_search_probability(
            prefix_tuple, transition.label, prob_dict
        )
        cost_prob = 1 - prob
        return (1 - alpha) * cost_prob + alpha * transition.weight
    
    def _compute_ngram_probability(
        self,
        path_prefix_tuple: Tuple[str, ...],
        transition_label: str,
        prob_dict: Dict[Tuple[str, ...], Dict[str, float]],
        lambdas: Tuple[float, ...]
    ) -> float:
        """
        Compute probability using n-gram smoothing.
        
        Parameters
        ----------
        path_prefix_tuple : tuple of str
            Sequence of activities in path prefix
        transition_label : str
            Label of current transition
        prob_dict : dict
            N-gram probability dictionary
        lambdas : tuple of float
            Weights for different n-gram lengths
            
        Returns
        -------
        float
            Computed probability
        """
        if not lambdas:
            raise ValueError("Lambdas list must be provided for n-gram approach")
        
        if not path_prefix_tuple:
            return prob_dict.get((), {}).get(transition_label, 0)
        
        total_weighted_prob = 0.0
        total_lambda_weight = 0.0
        max_n = min(len(path_prefix_tuple), len(lambdas))
        
        for n in range(1, max_n + 1):
            prefix_n_gram = path_prefix_tuple[-n:]
            prob = prob_dict.get(prefix_n_gram, {}).get(transition_label, 0)
            lambda_weight = lambdas[n - 1]
            
            total_weighted_prob += lambda_weight * prob
            total_lambda_weight += lambda_weight
        
        if total_lambda_weight == 0:
            return 0
        
        return total_weighted_prob / total_lambda_weight
    
    def _compute_prefix_search_probability(
        self,
        path_prefix_tuple: Tuple[str, ...],
        transition_label: str,
        prob_dict: Dict[Tuple[str, ...], Dict[str, float]]
    ) -> float:
        """
        Compute probability using prefix search.
        
        Parameters
        ----------
        path_prefix_tuple : tuple of str
            Sequence of activities in path prefix
        transition_label : str
            Label of current transition
        prob_dict : dict
            Prefix probability dictionary
            
        Returns
        -------
        float
            Computed probability
        """
        if not path_prefix_tuple:
            return 0
        
        # Check exact prefix match
        if path_prefix_tuple in prob_dict:
            return prob_dict[path_prefix_tuple].get(transition_label, 0)
        
        # Find longest matching prefix
        longest_prefix = self.find_longest_prefix(path_prefix_tuple, prob_dict)
        if longest_prefix:
            return prob_dict[longest_prefix].get(transition_label, 0)
        
        return 0
    
    def find_longest_prefix(
        self,
        path_prefix_tuple: Tuple[str, ...],
        prob_dict: Dict[Tuple[str, ...], Any],
        max_length: Optional[int] = None
    ) -> Optional[Tuple[str, ...]]:
        """
        Find longest prefix that exists in dictionary.
        
        Parameters
        ----------
        path_prefix_tuple : tuple of str
            Complete path to search in
        prob_dict : dict
            Dictionary to search for prefixes
        max_length : int, optional
            Maximum prefix length to consider
            
        Returns
        -------
        tuple or None
            Longest existing prefix, or None if not found
        """
        max_len = len(path_prefix_tuple)
        if max_length is not None:
            max_len = min(max_len, max_length)
        
        for length in range(max_len, 0, -1):
            prefix = path_prefix_tuple[-length:]
            if prefix in prob_dict:
                return prefix
        
        return None