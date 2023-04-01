# Author: Luis Mantilla
# Github: BestQuark
"""
This is the common_ansatz module. 
It has several common ansatzes that can be used for MBQC algorithms
"""

from typing import List
from mentpy.states.graphstate import GraphState
from mentpy.states.mbqcstate import MBQCState, hstack


def linear_cluster(n, **kwargs) -> MBQCState:
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
        @savefig linear_cluster.png width=1000px
        mp.draw(g)

    Group
    -----
    templates
    """
    g = GraphState()
    g.add_edges_from([(i, i + 1) for i in range(n - 1)])
    gs = MBQCState(g, input_nodes=[0], output_nodes=[n - 1], **kwargs)
    return gs


def many_wires(n_wires: List, **kwargs) -> MBQCState:
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
        @savefig many_wires.png width=1000px
        mp.draw(g)

    Group
    -----
    templates
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
        **kwargs,
    )
    return gs


def grid_cluster(n, m, **kwargs) -> MBQCState:
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
        @savefig grid_cluster.png width=1000px
        mp.draw(g)

    Group
    -----
    templates
    """
    g = GraphState()
    # add edges between rows
    n_wires = [m] * n
    for i, m in enumerate(n_wires):
        g.add_edges_from(
            [(j + sum(n_wires[:i]), j + sum(n_wires[:i]) + 1) for j in range(m - 1)]
        )

    # add edges between columns
    for i in range(n - 1):
        g.add_edges_from([(i * m + j, (i + 1) * m + j) for j in range(m)])

    gs = MBQCState(
        g,
        input_nodes=[sum(n_wires[:i]) for i in range(len(n_wires))],
        output_nodes=[sum(n_wires[: i + 1]) - 1 for i in range(len(n_wires))],
        **kwargs,
    )
    return gs


def muta(n_wires, n_layers, **kwargs):
    """This is the Multiple Triangle Ansatz (MuTA) template.

    Args
    ----
    n_wires: int
        The number of wires in the graph state.
    n_layers: int
        The number of layers in the graph state.

    Returns
    -------
    The graph state with the MuTA template.

    Examples
    --------
    Create a MuTA ansatz with 3 wires and 2 layers

    .. ipython:: python

        g = mp.templates.muta(3, 2)
        @savefig muta.png width=1000px
        mp.draw(g, figsize=(16,5))

    Group
    -----
    templates
    """

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

    return bigger_graph
