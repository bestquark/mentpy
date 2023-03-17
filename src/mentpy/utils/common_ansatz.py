"""
This is the common_ansatz module. 
It has several common ansatzes that can be used for MBQC algorithms
"""

import networkx as nx
from typing import List, Tuple, Optional, Union
from mentpy.state import GraphState

def linear_cluster(n):
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
        g = mtp.linear_cluster(5)
        print(g)
    """
    g = nx.Graph()
    g.add_edges_from([(i, i + 1) for i in range(n - 1)])
    gs = GraphState(g, input_nodes=[0], output_nodes=[n - 1])
    return gs

def many_wires(n_wires: List):
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
        g = mtp.many_wires([2, 3, 4])
        print(g)
    """
    # require n_wires to be a list of integers greater than 1
    if not all([isinstance(n, int) and n > 1 for n in n_wires]):
        raise ValueError(
            "n_wires must be a list of integers greater than 1"
        )
    
    g = nx.Graph()
    for i, n in enumerate(n_wires):
        g.add_edges_from([(j + sum(n_wires[:i]), j + sum(n_wires[:i]) + 1) for j in range(n - 1)])
    
    # input nodes are the first qubit in each wire and output nodes are the last qubit in each wire
    gs = GraphState(g, input_nodes=[sum(n_wires[:i]) for i in range(len(n_wires))], output_nodes=[sum(n_wires[:i + 1]) - 1 for i in range(len(n_wires))])
    return gs