import numpy as np
import cirq
import networkx as nx

from scipy.spatial import distance
from scipy.special import kl_div, rel_entr

import pennylane as qml

from mentpy.mbqc import MBQCircuit
from mentpy.simulators import PatternSimulator
from mentpy.utils.lc_equivalence import are_lc_equivalent
from mentpy.utils.generate_data import generate_haar_random_states
from mentpy.utils.flow_space import FlowSpace


def haar_probability_density_of_fidelities(F: float, n_qubits: int):
    r"""Returns the probability density function of fidelities
    :math:`P_{Haar}(F) = (N-1)(1-F)^{N-2}` where :math:`N = 2^{n}` is
    the dimension of the Hilbert space.

    Args
    ----
    F (float): Fidelity. Must be between :math:`0` and :math:`1`.
    n_qubits (int): Number of qubits. Must be greater than or equal to 1.

    Returns
    -------
    :math:`P_{Haar}(F) = (N-1)(1-F)^{N-2}`  where  :math:`N = 2^{n}`
    """
    N = int(2**n_qubits)
    return (N - 1) * ((1 - F) ** (N - 2))


def expressivity_with_histogram(
    graph_state_circuit: MBQCircuit,
    n_samples: int = 10000,
    n_bins: int = 1000,
    method: str = "KL",
):
    r"""Returns the expressivity calculated using the Kullback-Leiber entropy

    Args
    ----
    graph_state_circuit: Graph state circuit for which we will calculate the expressivity
    n_samples: Number of samples that we will use to estimate the expressivity
    n_bins: Number of bins of the histogram
    method: 'KL' for Kullback-Leibler divergence or 'RE' for relative entropy.

    Returns
    -------
    expressivity (float): Expressivity measure for graph_state_circuit
    """
    expressivity = -1
    # TODO: Add covering number method to calculate expressivity (?)
    if method not in ["KL", "RE", "JS"]:
        raise UserWarning("Unsupported method for calculating expressivity")

    samples_from_circuit = sample_probability_density_of_fidelities(
        graph_state_circuit, n_samples=n_samples
    )
    digitized_data, edges_hist = np.histogram(
        samples_from_circuit, bins=n_bins, density=True, range=(0, 1)
    )

    midpoints = 0.5 * (edges_hist[:-1] + edges_hist[1:])
    haar_prob_fun = lambda fid: haar_probability_density_of_fidelities(
        fid, len(graph_state_circuit.output_nodes)
    )

    digitized_haar = haar_prob_fun(np.array(midpoints))

    discrete_prob_data = digitized_data * np.diff(edges_hist)
    discrete_prob_haar = digitized_haar * np.diff(edges_hist)

    if method == "KL":
        expressivity = sum(kl_div(discrete_prob_data, discrete_prob_haar))
    elif method == "RE":
        expressivity = sum(rel_entr(discrete_prob_data, discrete_prob_haar))
    elif method == "JS":
        expressivity = distance.jensenshannon(discrete_prob_data, discrete_prob_haar)

    return expressivity


def sample_probability_density_of_fidelities(
    graph_state_circuit: MBQCircuit, n_samples=1000, backend="pennylane"
):
    r"""Calculates samples of the probability of fidelities of the given graph state circuit"""

    pattern_simulator = PatternSimulator(graph_state_circuit)
    random_input_states = generate_haar_random_states(
        len(graph_state_circuit.input_nodes), n_samples
    )
    fidelities = []

    if backend == "pennylane":
        qmlcircuit = pattern_simulator.graphstate_to_circuit()
        for random_st in random_input_states:
            random_pattern = (
                2 * np.pi * np.random.rand(pattern_simulator.max_measure_number)
            )
            final_state = qmlcircuit(random_pattern, output="density")
            fid = qml.math.fidelity(np.kron(random_st.T, random_st), final_state)
            fidelities.append(fid)

    elif backend == "cirq":
        for random_st in random_input_states:
            random_pattern = (
                2 * np.pi * np.random.rand(pattern_simulator.max_measure_number)
            )
            pattern_simulator.reset(input_state=random_st)

            _ = pattern_simulator.measure_pattern(random_pattern)
            final_state = pattern_simulator.current_sim_state
            fidelities.append(cirq.fidelity(random_st, final_state))

    else:
        print(f"Unsupported backend {backend}")

    return fidelities


def digraph_expressivity_of_flow_space(flow_space: FlowSpace, method="KL", **kwargs):
    """Returns digraph given the expressivity of a ``FlowSpace`` object.

    Args
    ----
        flow_space (FlowSpace): Graph for which we will calculate the expressivity
        method (str): 'KL' for Kullback-Leiber entropy, 'RE' for relative entropy.
    """

    expr_digraph = nx.DiGraph()
    for node in flow_space.flow_graph_space.nodes():
        graph_circuit = flow_space.flow_graph_space.nodes[node]["mbqc_circuit"]
        expr = expressivity_with_histogram(graph_circuit, method=method, **kwargs)
        expr_digraph.add_node(node, expr=expr, mbqc_circuit=graph_circuit)

    for edge in flow_space.flow_graph_space.edges():
        i, j = edge
        expri, exprj = expr_digraph.nodes[i]["expr"], expr_digraph.nodes[j]["expr"]
        if expri <= exprj:
            expr_digraph.add_edge(j, i)
        else:
            expr_digraph.add_edge(i, j)

    return expr_digraph


def lc_cluster_flowspace(deg_graph, sanity_check=True):
    """Cluster flow graph in lc_equivalent graphs."""
    independent_groups = {}
    independent_groups_ind = {}
    for node in deg_graph.nodes():
        curr_g = deg_graph.nodes[node]["mbqc_circuit"].graph
        if not independent_groups:
            independent_groups[0] = [curr_g]
            independent_groups_ind[0] = [node]
        else:
            added = False
            for k, g in independent_groups.items():
                if are_lc_equivalent(g[0], curr_g)[0]:
                    added = True
                    independent_groups[k].append(curr_g)
                    independent_groups_ind[k].append(node)
                    if sanity_check:
                        for n in g:
                            if not are_lc_equivalent(n, curr_g)[0]:
                                raise RuntimeError(
                                    "Graph not lc equivalent with all group"
                                )
            if not added:
                new_key = max(independent_groups) + 1
                independent_groups[new_key] = [curr_g]
                independent_groups_ind[new_key] = [node]

    return independent_groups_ind, independent_groups


def draw_digraph_flow_space(digraph_expr: nx.digraph, **kwargs):
    """Draws the expressivity digraph of flow space"""
    pospos = nx.spring_layout(digraph_expr.to_undirected())
    nx.draw(digraph_expr, node_size=55, pos=pospos, **kwargs)
