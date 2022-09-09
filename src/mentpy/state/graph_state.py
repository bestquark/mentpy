"""The graph_state module"""

from functools import cached_property
from typing import Optional, List, Tuple, Union

import numpy as np
import scipy as scp
import networkx as nx
import mentpy as mtp

import pennylane as qml


class GraphState:
    r"""The GraphState class that deals with operations and manipulations of graph states

    Examples
    --------
    Create a 1D cluster state :math:`|G>` of five qubits

    g = nx.Graph()
    g.add_edges_from([(0,1), (1,2), (2,3), (3, 4)])
    state = mtp.GraphState(g, input_nodes=[0], output_nodes=[4])

    :group: states
    """

    def __init__(
        self,
        graph: Union[nx.Graph, nx.DiGraph],
        input_state: np.ndarray = np.array([]),
        input_nodes: np.ndarray = np.array([]),
        output_nodes: np.ndarray = np.array([]),
    ) -> None:
        """Initializes a graph state"""
        self.graph = graph
        self.input_state = input_state
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

    @cached_property
    def outputc(self):
        r"""Returns :math:`O^c`, the complement of output nodes."""
        return [v for v in self.graph.nodes() if v not in self.output_nodes]

    @cached_property
    def inputc(self):
        r"""Returns :math:`I^c`, the complement of input nodes."""
        return [v for v in self.graph.nodes() if v not in self.input_nodes]


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


def entanglement_entropy(
    state: GraphState, subRegionA: List, subRegionB: Optional[List] = None
):
    """Calculates the entanglement entropy between subRegionA and subRegionB
    of state. If subRegionB is None, then :python:`subRegionB = set(state.graph.nodes()) - set(subRegionA)`
    by default."""

    G = state.graph.copy()

    # minimum_cut requires the capacity kwarg.
    nx.set_edge_attributes(G, 1, name="capacity")
    if subRegionB is None:
        subRegionB = set(state.graph.nodes()) - set(subRegionA)

    # Allow subregions. These are merged into a supernode to calculate
    # the minimum cut between them.
    if isinstance(subRegionA, List):
        for v in subRegionA:
            G = nx.contracted_nodes(G, subRegionA[0], v)
        subRegionA = subRegionA[0]
    if isinstance(subRegionB, List):
        for v in subRegionA:
            G = nx.contracted_nodes(G, subRegionB[0], v)
        subRegionB = subRegionB[0]

    return nx.minimum_cut(G, subRegionA, subRegionB)[0]
