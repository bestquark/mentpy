# Author: Luis Mantilla
# Github: BestQuark
"""
This is the common_ansatz module. 
It has several common ansatzes that can be used for MBQC algorithms
"""

from typing import List
from mentpy.states.graphstate import GraphState
from mentpy.states.mbqcstate import MBQCState, hstack

__all__ = ["linear_cluster", "many_wires"]


def linear_cluster(n) -> MBQCState:
    r"""Returns a linear cluster state of n qubits.
    Args
    ----
    n: int
        The number of qubits in the cluster state.

    Returns
    -------
    The linear cluster state of n qubits.

    Examples
    --------
    Create a 1D cluster state :math:`|G>` of five qubits
    .. ipython:: python

        g = mp.templates.linear_cluster(5)
        print(g)

    Group
    -----
    states
    """
    g = GraphState()
    g.add_edges_from([(i, i + 1) for i in range(n - 1)])
    gs = MBQCState(g, input_nodes=[0], output_nodes=[n - 1])
    return gs


def many_wires(n_wires: List) -> MBQCState:
    r"""Returns a graph state with many wires.
    Args
    ----
    n_wires: List
        A list of the number of qubits in each wire.

    Returns
    -------
    The graph state with many wires.

    Examples
    --------
    Create a graph state with three wires of 2, 3, and 4 qubits respectively
    .. ipython:: python

        g = mp.templates.many_wires([2, 3, 4])
        print(g)

    Group
    -----
    states
    """
    # require n_wires to be a list of integers greater than 1
    if not all([isinstance(n, int) and n > 1 for n in n_wires]):
        raise ValueError("n_wires must be a list of integers greater than 1")

    g = GraphState()
    for i, n in enumerate(n_wires):
        g.add_edges_from(
            [(j + sum(n_wires[:i]), j + sum(n_wires[:i]) + 1) for j in range(n - 1)]
        )

    # input nodes are the first qubit in each wire and output nodes are the last qubit in each wire
    gs = MBQCState(
        g,
        input_nodes=[sum(n_wires[:i]) for i in range(len(n_wires))],
        output_nodes=[sum(n_wires[: i + 1]) - 1 for i in range(len(n_wires))],
    )
    return gs


def grid_cluster(n, m) -> MBQCState:
    r"""Returns a grid cluster state of n x m qubits.
    Args
    ----
    n: int
        The number of rows in the cluster state.
    m: int
        The number of columns in the cluster state.

    Returns
    -------
    The grid cluster state of n x m qubits.

    Examples
    --------
    Create a 2D cluster state :math:`|G>` of five qubits

    .. ipython:: python

        g = mp.templates.grid_cluster(2, 3)
        print(g)

    Group
    -----
    states
    """
    g = GraphState()
    # add edges between rows
    n_wires = [n] * m
    for i, n in enumerate(n_wires):
        g.add_edges_from(
            [(j + sum(n_wires[:i]), j + sum(n_wires[:i]) + 1) for j in range(n - 1)]
        )

    # add edges between columns
    for i in range(m - 1):
        g.add_edges_from([(i * n + j, (i + 1) * n + j) for j in range(n)])

    gs = MBQCState(
        g,
        input_nodes=[sum(n_wires[:i]) for i in range(len(n_wires))],
        output_nodes=[sum(n_wires[: i + 1]) - 1 for i in range(len(n_wires))],
    )
    return gs


def muta(n_wires, n_layers):
    """This is the Multiple Triangle Ansatz (MuTA) template."""

    SIZE_TRIANGLE = 4

    big_graph = None
    for wire in range(n_wires):

        g = many_wires([SIZE_TRIANGLE] * n_wires)

        for connect in range(n_wires):
            if connect != wire:
                g.add_edge(SIZE_TRIANGLE * wire + 1, SIZE_TRIANGLE * connect)
                g.add_edge(SIZE_TRIANGLE * wire + 1, SIZE_TRIANGLE * connect + 2)

        if big_graph is None:
            big_graph = g
        else:
            big_graph = hstack((big_graph, g))

    bigger_graph = None
    for layer in range(n_layers):
        if bigger_graph is None:
            bigger_graph = big_graph
        else:
            bigger_graph = hstack((bigger_graph, big_graph))

    return big_graph
