# Author: Luis Mantilla
# Github: BestQuark
import numpy as np
from mentpy.operators import PauliOp
import networkx as nx
from typing import Optional, List

__all__ = ["GraphState", "entanglement_entropy"]


class GraphState(nx.Graph):
    """A graph state class that inherits from networkx.Graph.

    Examples
    --------
    Create a 1D cluster state :math:`|G>` of five qubits

    .. ipython:: python

        g = mp.GraphState()
        g.add_edges_from([(0,1), (1,2), (2,3), (3, 4)])
        print(g)

    See Also
    --------
    :class:`mentpy.mbqc.MBQCircuit`

    Group
    -----
    states
    """

    def __init__(self, *args, **kwargs):
        """Initialize a graph state. See networkx.Graph for more information."""
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f"GraphState with {self.number_of_nodes()} nodes and {self.number_of_edges()} edges."

    def __len__(self):
        return self.number_of_nodes()

    def __eq__(self, other):
        return nx.is_isomorphic(self, other)

    def index_mapping(self):
        """Return a mapping of the nodes to their indices."""
        return {v: i for i, v in enumerate(self.nodes())}

    def stabilizers(self):
        """
        Generate the stabilizers of a graph state.

        Examples
        --------
        Calculate the stabilizers of a 1D cluster state :math:`|G>` of five qubits

        .. ipython:: python
            :okwarning:

            g = mp.GraphState()
            g.add_edges_from([(0,1), (1,2), (2,3), (3, 4)])
            print(g.stabilizers())
        """
        return _get_stabilizers(self)


def lc_reduce(graph: GraphState):
    """Reduce graph state

    Group
    -----
    states
    """
    raise NotImplementedError


# TODO: Check if this is correct.
def entanglement_entropy(
    state: GraphState, subRegionA: List, subRegionB: Optional[List] = None
) -> float:
    """Calculate the entanglement entropy of a subregion of a graph state."""

    G = state.copy()

    # minimum_cut requires the capacity kwarg.
    nx.set_edge_attributes(G, 1, name="capacity")
    if subRegionB is None:
        subRegionB = set(state.nodes()) - set(subRegionA)

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


def _get_stabilizers(graph: GraphState) -> List[PauliOp]:
    """Generate the stabilizers of a graph state.

    Group
    -----
    states
    """
    z_mat = nx.adjacency_matrix(graph).todense()
    x_mat = np.eye(graph.number_of_nodes(), dtype=int)

    return PauliOp(np.hstack((x_mat, z_mat)))
