from mentpy.mbqc import MBQCircuit

import networkx as nx
import itertools


class FlowSpace:
    r"""The flow space graph of a MBQCGraph.

    Each node corresponds to a possible graph over ``len(graph_state)`` qubits.
    Each edge between nodes represent going from one graph to another via adding or removing edges.

    .. note::  ``flow_space()`` will only work for MBQCGraph with less
    than 8 qubits.
    """

    def __init__(self, graph_state: MBQCircuit, allow_any_size_graphs: bool = False):
        """Creates the flow graph space of a graph state circuit."""

        if len(graph_state) > 7 and (not allow_any_size_graphs):
            raise UserWarning(
                f"Expected a graph_state of size 7 or less, but {len(graph_state)} "
                "was given."
            )

        self.number_nodes = len(graph_state)
        self.input_nodes = graph_state.input_nodes
        self.output_nodes = graph_state.output_nodes
        total_graph, wf_graph, wof_graph = self.all_graphs_graph()
        self.total_graph_space = total_graph
        self.flow_graph_space = wf_graph
        self.no_flow_graph_space = wof_graph

    def __repr__(self) -> str:
        r"""Returns the representation of the flow space"""
        return (
            f"Flow space over {self.number_nodes} nodes with {self.input_nodes} input nodes "
            f"and {self.output_nodes} output nodes."
        )

    def generator_all_graphs(self):
        """Returns a generator that generates all possible graphs for :math:`n` ordered nodes."""
        n = self.number_nodes
        total_n = int(n * (n - 1) / 2)
        completegraph = nx.complete_graph(n)
        its = itertools.product([0, 1], repeat=total_n)
        for j in its:
            g = nx.Graph()
            g.add_nodes_from(list(range(n)))
            edgs = [edg for i, edg in enumerate(list(completegraph.edges())) if j[i]]
            g.add_edges_from(edgs)
            yield g

    def all_graphs_graph(self):
        """Returns a tuple with three graphs.

        The first graph is the graph corresponding to all possible graphs.
        The second is the subgraph of all graphs with flow. The third is the
        subgraph of all graphs without flow."""

        graphs_list = list(self.generator_all_graphs())
        graph_space = nx.Graph()
        wflow = []
        woflow = []
        for ind, g in enumerate(graphs_list):
            gs = MBQCircuit(
                g, input_nodes=self.input_nodes, output_nodes=self.output_nodes
            )

            fl = gs.flow

            if fl is not None:
                graph_space.add_node(ind, flow=True, mbqc_circuit=gs)
                wflow.append(ind)
            else:
                graph_space.add_node(ind, flow=False, mbqc_circuit=gs)
                woflow.append(ind)

        for ind1, g1 in enumerate(graphs_list):
            for ind2, g2 in enumerate(graphs_list):
                if ind1 < ind2:
                    if len(set(g2.edges()).symmetric_difference(set(g1.edges()))) == 1:
                        graph_space.add_edge(ind1, ind2)

        return graph_space, graph_space.subgraph(wflow), graph_space.subgraph(woflow)
