# Author: Luis Mantilla
# Github: BestQuark
"""The graph_state module"""

from functools import cached_property, reduce
from typing import Optional, List, Tuple, Callable, Union, Any

import numpy as np
import scipy as scp
import networkx as nx

from mentpy.states.graphstate import GraphState
from mentpy.states.flow import find_gflow, find_cflow, find_flow, check_if_flow

__all__ = ["MBQCState", "draw", "merge", "hstack", "vstack"]


class MBQCState:
    r"""The MBQCGraph class that deals with operations and manipulations of graph states
    Args
    ----
    graph: mp.GraphState
        The graph state of the MBQC state.
    input_nodes: list
        The input nodes of the MBQC state.
    output_nodes: list
        The output nodes of the MBQC state.
    wires: Optional[list]
        The wires of the MBQC state.

    Examples
    --------
    Create a 1D cluster state :math:`|G>` of five qubits

    .. ipython:: python

        g = mp.GraphState()
        g.add_edges_from([(0,1), (1,2), (2,3), (3, 4)])
        state = mp.MBQCState(g, input_nodes=[0], output_nodes=[4])


    See Also
    --------
    :class:`mp.GraphState`

    Group
    -----
    states
    """

    def __init__(
        self,
        graph: GraphState,
        input_nodes: np.ndarray = np.array([]),
        output_nodes: np.ndarray = np.array([]),
        flow: Optional[Callable] = None,
        partial_order: Optional[callable] = None,
        measurement_order: Optional[List[int]] = None,
        gflow: Optional[Callable] = None,
    ) -> None:
        """Initializes a graph state"""
        self._graph = graph

        # check input and output nodes are in graph. If not, raise error with the node(s) that are not in the graph
        if not all([v in self.graph.nodes for v in input_nodes]):
            raise ValueError(
                f"Input nodes {input_nodes} are not in the graph. Graph nodes are {self.graph.nodes}"
            )
        if not all([v in self.graph.nodes for v in output_nodes]):
            raise ValueError(
                f"Output nodes {output_nodes} are not in the graph. Graph nodes are {self.graph.nodes}"
            )
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes

        if (flow is None) or (partial_order is None):
            flow, partial_order, depth = find_cflow(graph, input_nodes, output_nodes)

        elif (flow is not None) and (partial_order is not None):
            check_if_flow(graph, input_nodes, output_nodes, flow, partial_order)

        self._flow = flow
        self._partial_order = partial_order

        if gflow is None:
            if flow is None:
                gflow, gpartial_order, depth = find_gflow(
                    graph, input_nodes, output_nodes
                )
            else:
                gflow, gpartial_order = flow, partial_order

        # TODO: check if given gflow it is definitely gflow

        self.gflow = gflow

        if measurement_order is None and flow is not None:
            measurement_order = self.calculate_order()

        self._depth = depth
        self._measurement_order = measurement_order

    def __repr__(self) -> str:
        """Return the representation of the current graph state"""
        return (
            f"MBQCState with {self.graph.number_of_nodes()} nodes and "
            f"{self.graph.number_of_edges()} edges"
        )

    def __len__(self) -> int:
        """Return the number of nodes in the GraphStateCircuit"""
        return len(self.graph)

    # if an attribute is not found, look for it in the graph
    def __getattr__(self, name):
        return getattr(self.graph, name)

    @property
    def graph(self) -> GraphState:
        r"""Return the graph of the resource state."""
        return self._graph

    @property
    def input_nodes(self) -> np.ndarray:
        r"""Return the input nodes of the MBQC circuit."""
        return self._input_nodes

    @property
    def output_nodes(self) -> np.ndarray:
        r"""Return the output nodes of the MBQC circuit."""
        return self._output_nodes

    @property
    def flow(self) -> Callable:
        r"""Return the flow function of the MBQC circuit."""
        return self._flow

    @property
    def partial_order(self) -> Callable:
        r"""Return the partial order function of the MBQC circuit."""
        return self._partial_order

    @property
    def depth(self) -> int:
        r"""Return the depth of the MBQC circuit."""
        return self._depth

    @property
    def measurement_order(self) -> List[int]:
        r"""Return the measurement order of the MBQC circuit."""
        return self._measurement_order

    @measurement_order.setter
    def measurement_order(self, measurement_order: List[int]) -> None:
        r"""Set the measurement order of the MBQC circuit."""
        if not _check_measurement_order(measurement_order, self.partial_order):
            raise ValueError(f"Invalid measurement order {measurement_order}.")
        self._measurement_order = measurement_order

    @cached_property
    def outputc(self):
        r"""Returns :math:`O^c`, the complement of output nodes."""
        return [v for v in self.graph.nodes() if v not in self.output_nodes]

    @cached_property
    def inputc(self):
        r"""Returns :math:`I^c`, the complement of input nodes."""
        return [v for v in self.graph.nodes() if v not in self.input_nodes]

    def calculate_order(self):
        r"""Returns the order of the measurements"""
        n = len(self.graph)
        mat = np.zeros((n, n))

        for indi, i in enumerate(list(self.graph.nodes())):
            for indj, j in enumerate(list(self.graph.nodes())):
                if self.partial_order(i, j):
                    mat[indi, indj] = 1

        sum_mat = np.sum(mat, axis=1)
        order = np.argsort(sum_mat)[::-1]

        # turn order into labels of graph
        order = [list(self.graph.nodes())[i] for i in order]

        return order

    def add_edge(self, u, v):
        r"""Adds an edge between nodes u and v"""
        self.graph.add_edge(u, v)
        # try resetting self with new graph, if it fails, remove the edge
        try:
            self.__init__(self.graph, self.input_nodes, self.output_nodes)
        except Exception as e:
            self.graph.remove_edge(u, v)
            raise ValueError(f"Cannot add edge between {u} and {v}.\n" + str(e))

    def add_edges_from(self, edges, **kwargs):
        r"""Adds edges from a list of tuples"""
        new_graph = self.graph.copy()
        new_graph.add_edges_from(edges, **kwargs)
        try:
            self.__init__(
                new_graph,
                self.input_nodes,
                self.output_nodes,
                self.flow,
                self.partial_order,
            )
        except Exception as e:
            raise ValueError(f"Cannot add edges {edges}.\n" + str(e))


def _check_measurement_order(measurement_order: List, partial_order: Callable):
    r"""Checks that the measurement order is valid"""
    for i in range(len(measurement_order)):
        for j in range(i + 1, len(measurement_order)):
            if partial_order(measurement_order[i], measurement_order[j]):
                return False
    return True


def merge(state1: MBQCState, state2: MBQCState, concat_indices=[]) -> MBQCState:
    """Merge two graph states into a larger graph state. This is, the input and
    output of the new MBQC state will depend on the concat_indices."""

    for (i, j) in concat_indices:
        if i not in state1.output_nodes or j not in state2.input_nodes:
            raise ValueError(f"Cannot merge states at indices {i} and {j}")

    graph = nx.disjoint_union(state1.graph, state2.graph)
    input_nodes = state1.input_nodes + [
        i + len(state1.graph) for i in state2.input_nodes
    ]
    output_nodes = state1.output_nodes + [
        i + len(state1.graph) for i in state2.output_nodes
    ]

    for (i, j) in concat_indices:
        input_nodes.remove(j + len(state1.graph))
        output_nodes.remove(i)
        graph.add_edge(i, j + len(state1.graph))

    return MBQCState(graph, input_nodes, output_nodes)


def vstack(states):
    """Vertically stack a list of graph states into a larger graph state. This is,
    the input of the new MBQC state is the input of the first state, and the output
    is the output of the last state.

    Group
    -----
    states
    """
    if len(states) == 0:
        raise ValueError("Cannot vertically stack an empty list of states.")
    if len(states) == 1:
        return states[0]
    return reduce(_vstack2, states)


def hstack(states):
    """Horizontally stack a list of graph states into a larger graph state. This is,
    the input of the new MBQC state is the input of the first state, and the output
    is the output of the last state.

    Group
    -----
    states
    """
    if len(states) == 0:
        raise ValueError("Cannot horizontally stack an empty list of states.")
    if len(states) == 1:
        return states[0]
    return reduce(_hstack2, states)


def _vstack2(state1: MBQCState, state2: MBQCState) -> MBQCState:
    """Vertically stack two graph states into a larger graph state. This is,
    the input of the new MBQC state is both the input of the first and second
    state, and the output is the output of the first and second state.

    Group
    -----
    states
    """
    graph = nx.disjoint_union(state1.graph, state2.graph)
    input_nodes = state1.input_nodes + [
        i + len(state1.graph) for i in state2.input_nodes
    ]
    output_nodes = state1.output_nodes + [
        i + len(state1.graph) for i in state2.output_nodes
    ]

    # TODO: Compute flow and partial order
    return MBQCState(graph, input_nodes, output_nodes)


def _hstack2(state1: MBQCState, state2: MBQCState) -> MBQCState:
    """Horizontally stack two graph states into a larger graph state. This is,
    the input of the new MBQC state is the input of the first state, and the
    output is the output of the second state.

    Args
    ----
    state1: MBQCState
        The first state to stack.
    state2: MBQCState
        The second state to stack.

    Group
    -----
    states
    """
    # check that the size of the output of the first state is the same as the
    # size of the input of the second state
    if len(state1.output_nodes) != len(state2.input_nodes):
        raise ValueError(
            "The output of the first state must be the same size as the input "
            "of the second state."
        )

    # create new graph
    graph = nx.disjoint_union(state1.graph, state2.graph)

    nodes1 = list(state1.graph.nodes())
    nodes2 = list(state2.graph.nodes())

    for i in range(len(state1.output_nodes)):
        graph.add_edge(
            nodes1.index(state1.output_nodes[i]),
            nodes2.index(state2.input_nodes[i]) + len(nodes1),
        )

    input_nodes = [nodes1.index(i) for i in state1.input_nodes]
    output_nodes = [nodes1.index(j) + len(nodes1) for j in state2.output_nodes]

    # TODO: Compute flow and partial order
    return MBQCState(graph, input_nodes, output_nodes)


def draw(state: Union[MBQCState, GraphState], fix_wires=None, **kwargs):
    """Draws mbqc circuit with flow.

    TODO: Add support for graphs without flow, but with gflow

    Group
    -----
    states
    """
    node_colors = {}
    for i in state.graph.nodes():
        if i in state.input_nodes:
            node_colors[i] = "#CCCCCC"
        elif i in state.output_nodes:
            node_colors[i] = "#ADD8E6"
        else:
            node_colors[i] = "#FFBD59"

    # options = {'node_color': '#FFBD59'}
    options = {"node_color": [node_colors[node] for node in state.graph.nodes()]}
    options.update(kwargs)

    if isinstance(state, GraphState):
        nx.draw(state, **options)

    elif isinstance(state, MBQCState):
        fixed_nodes = state.input_nodes + state.output_nodes
        position_xy = {}
        for indx, p in enumerate(state.input_nodes):
            position_xy[p] = (0, -2 * indx)

        separation = len(state.outputc) // len(state.output_nodes)
        if fix_wires is not None:
            for wire in fix_wires:
                if len(wire) + 2 > separation:
                    separation = len(wire) + 2
        for indx, p in enumerate(state.output_nodes):
            position_xy[p] = (2 * (separation) - 2, -2 * indx)

        if fix_wires is not None:
            x = [list(x) for x in fix_wires]

            fixed_nodes += sum(x, [])

            for indw, wire in enumerate(fix_wires):
                for indx, p in enumerate(wire):
                    if p != "*":
                        position_xy[p] = (2 * (indx + 1), -2 * indw)

        # remove all '*' from fixed_nodes
        fixed_nodes = [x for x in fixed_nodes if x != "*"]

        node_pos = nx.spring_layout(
            state.graph, pos=position_xy, fixed=fixed_nodes, k=1 / len(state.graph)
        )
        nx.draw(state.graph, pos=node_pos, **options)
        if state.flow is not None:
            nx.draw(_graph_with_flow(state), pos=node_pos, **options)


def _graph_with_flow(state):
    """Return digraph with flow (but does not have all CZ edges!)"""
    g = state.graph
    gflow = nx.DiGraph()
    gflow.add_nodes_from(g.nodes())
    for node in state.outputc:
        gflow.add_edge(node, state.flow(node))
    return gflow
