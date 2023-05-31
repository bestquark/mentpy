# Author: Luis Mantilla
# Github: BestQuark
"""
This is the common_ansatz module. 
It has several common ansatzes that can be used for MBQC algorithms
"""

from typing import List
from mentpy.operators import Ment
from mentpy.mbqc.states.graphstate import GraphState
from mentpy.mbqc.mbqcircuit import MBQCircuit, hstack, merge


def linear_cluster(n, **kwargs) -> MBQCircuit:
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
    gs = MBQCircuit(g, input_nodes=[0], output_nodes=[n - 1], **kwargs)
    return gs


def many_wires(n_wires: List, **kwargs) -> MBQCircuit:
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
    gs = MBQCircuit(
        g,
        input_nodes=[sum(n_wires[:i]) for i in range(len(n_wires))],
        output_nodes=[sum(n_wires[: i + 1]) - 1 for i in range(len(n_wires))],
        **kwargs,
    )
    return gs


def grid_cluster(n, m, periodic=False, **kwargs) -> MBQCircuit:
    r"""Returns a grid cluster state of n x m qubits.

    Args
    ----
    n: int
        The number of rows in the cluster state.
    m: int
        The number of columns in the cluster state.
    periodic: bool
        If True, the returned state will be a cylinder.

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

    if periodic and n > 1:
        # add edges between first and last row
        for j in range(m):
            g.add_edge(j, (n - 1) * m + j)

    gs = MBQCircuit(
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
    options = {
        "restrict-trainable": True,
    }

    SIZE_TRIANGLE = 5

    big_graph = None
    for wire in range(n_wires):
        g = many_wires([SIZE_TRIANGLE] * n_wires)
        if options["restrict-trainable"]:
            g.trainable_nodes = list(
                set(g.trainable_nodes) - set([i - 1 for i in g.output_nodes])
            )
            # TODO! Make soemthing with this...

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

    # TODO: I think this is ending in a Hadamard rotated state (odd)
    # It might need a padding of 1 extra qubit in each wire.
    return bigger_graph


def spturb(n_qubits: int, n_layers: int, periodic=False, **kwargs):
    """This is the Symmetry Protected Topological Perturbator Ansatz (SPTurb) template.

    Args
    ----
    n_qubits: int
        The number of qubits in the SPT state.
    n_layers: int
        The number of layers in the graph state.
    periodic: bool
        Whether to use periodic boundary conditions.

    Returns
    -------
    The graph state with the SPTurb template.

    Group
    -----
    templates
    """
    if n_qubits < 4:
        raise ValueError("n_qubits must be greater than 4")

    gr = many_wires([5, 2, 5]).graph
    gr.add_edge(2, 6)
    gr.add_edge(9, 6)
    sym_block1 = MBQCircuit(
        gr,
        input_nodes=[0, 7],
        output_nodes=[4, 11],
        measurements={5: Ment(plane="XY")},
        default_measurement=Ment(plane="X"),
    )
    sym_block2 = MBQCircuit(
        gr,
        input_nodes=[0, 7],
        output_nodes=[4, 11],
        measurements={
            5: Ment(plane="XY"),
            1: Ment(plane="Y"),
            3: Ment(plane="Y"),
            8: Ment(plane="Y"),
            10: Ment(plane="Y"),
        },
        default_measurement=Ment(plane="X"),
    )
    spt_ansatz = many_wires(
        [3] * n_qubits,
        measurements={3 * i + 1: Ment() for i in range(n_qubits)},
        default_measurement=Ment(plane="X"),
    )

    n_blocks = n_qubits if periodic else n_qubits - 2

    for layer in range(n_layers):
        if layer != 0:
            spt_ansatz = hstack(
                (
                    spt_ansatz,
                    many_wires(
                        [3] * n_qubits,
                        measurements={3 * i + 1: Ment() for i in range(n_qubits)},
                        default_measurement=Ment(plane="X"),
                    ),
                )
            )
        for m in range(2):
            for i in range(n_blocks):
                n1 = spt_ansatz.output_nodes[i]
                n2 = spt_ansatz.output_nodes[(i + 2) % n_qubits]
                if m % 2 == 0:
                    spt_ansatz = merge(spt_ansatz, sym_block1, [(n1, 0), (n2, 7)])
                else:
                    spt_ansatz = merge(spt_ansatz, sym_block2, [(n1, 0), (n2, 7)])

    return spt_ansatz
