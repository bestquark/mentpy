"""The graph_state module"""

from typing import Optional, List, Tuple

import numpy as np
import scipy as scp
import networkx as nx

import pennylane as qml


class GraphState:
    """The GraphState class that deals with operations and manipulations of graph states
    
    :group: states
    """

    def __init__(
        self,
        graph: nx.graph,
        input_state: Optional[np.ndarray] = None,
        input_nodes: Optional[np.ndarray] = None,
        output_nodes: Optional[np.ndarray] = None,
        flow: Optional[np.ndarray] = None,
    ) -> None:
        """Initializes a graph state"""
        self.graph = graph
        self.input_state = input_state
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.flow = flow


def lc_reduce(state: GraphState):
    """Reduce graph state
    
    :group: states
    """
    raise NotImplementedError


def merge(
    state1: GraphState, state2: GraphState, indices_tuple: List[Tuple]
) -> GraphState:
    """Merge two graph states into a larger graph state
    
    :group: states
    """
    raise NotImplementedError
