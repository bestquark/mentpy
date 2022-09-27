import numpy as np
import cirq
import networkx as nx
from mentpy import GraphStateCircuit, PatternSimulator
from mentpy.measurement import pattern_simulator
from mentpy.utils.generate_data import generate_random_input_states
from mentpy.utils.flow_space import FlowSpace


def haar_probability_density_of_fidelities(F: float, n_qubits: int, n_bins):
    r"""Returns the probability density function of fidelities
    :math:`P_{Haar}(F) = (N-1)(1-F)^{N-2}` where :math:`N = 2^{n}` is
    the dimension of the Hilbert space.

    Args
    ----
    F (float): Fidelity. Must be between 0 and 1.
    n_qubits (int): Number of qubits. Must be greater than or equal to 1.
    n_bins (int): Number of bins used in the histogram

    Returns
    -------
    :math:`P_{Haar}(F) = (N-1)(1-F)^{N-2}` where :math:`N = 2^{n}`
    """
    N = int(2**n_qubits)
    return ((N - 1) * ((1 - F) ** (N - 2)))/n_bins


def expressivity_using_relative_entropy(graph_state_circuit: GraphStateCircuit, n_samples=10000, n_bins=1000):
    r"""Returns the expressivity calculated using relative entropy"""
    samples_from_circuit = sample_probability_density_of_fidelities(
        graph_state_circuit, n_samples=n_samples
    )
    haar_prob_fun = lambda fid: haar_probability_density_of_fidelities(
        fid, len(graph_state_circuit.output_nodes), n_bins
    )
    # TODO:

def expressivity_using_KL(
    graph_state_circuit: GraphStateCircuit, n_samples=10000, n_bins=1000
):
    r"""Returns the expressivity calculated using the Kullback-Leiber entropy"""
    samples_from_circuit = sample_probability_density_of_fidelities(
        graph_state_circuit, n_samples=n_samples
    )
    haar_prob_fun = lambda fid: haar_probability_density_of_fidelities(
        fid, len(graph_state_circuit.output_nodes), n_bins
    )

    # TODO: Calculate expressivity here!

def expressivity_with_histogram(graph_state_circuit: GraphStateCircuit, n_samples=10000, n_bins=1000, method='KL'):
    r"""Returns the expressivity calculated using the Kullback-Leiber entropy"""
    
    samples_from_circuit = sample_probability_density_of_fidelities(
        graph_state_circuit, n_samples=n_samples
    )
    haar_prob_fun = lambda fid: haar_probability_density_of_fidelities(
        fid, len(graph_state_circuit.output_nodes),n_bins
    )

def sample_probability_density_of_fidelities(
    graph_state_circuit: GraphStateCircuit, n_samples=1000
):
    r"""Calculates samples of the probability of fidelities of the given graph state circuit"""

    pattern_simulator = PatternSimulator(graph_state_circuit)
    random_input_states = generate_random_input_states(
        pattern_simulator.state, n_samples
    )
    fidelities = []
    for random_st in random_input_states:
        pattern_simulator.reset(input_state=random_st)
        random_pattern = (
            2 * np.pi * np.random.rand(pattern_simulator.max_measure_number)
        )
        _ = pattern_simulator.measure_pattern(random_pattern)
        final_state = pattern_simulator.current_sim_state
        fidelities.append(cirq.fidelity(random_st, final_state))

    return fidelities

def digraph_expressivity_of_flow_space(flow_space: FlowSpace, method = 'KL', **kwargs):
    """Returns digraph given the expressivity of a ``FlowSpace`` object.
    
    Args
    ----
    flow_space (FlowSpace): Graph for which we will calculate the expressivity
    method (str): 'KL' for Kullback-Leiber entropy, 'RE' for relative entropy.
    
    Kwargs
    ------
    n_samples (int): Number of samples to calculate fidelity histogram. 10000 is default.
    n_bins (int): Number of bins of fidelity histogram. 1000 is default.
    """
    if method=='KL':
        expressivity_function = expressivity_using_KL
    elif method=='RE':
        expressivity_function = expressivity_using_relative_entropy

    #TODO: Add covering number method to calculate expressivity
    else:
        raise UserWarning("Unsupported method for calculating expressivity")
    
    expr_digraph = nx.DiGraph()
    for node in flow_space.flow_graph_space.nodes():
        graph_circuit = flow_space.flow_graph_space[node]['mbqc_circuit']
        expr = expressivity_function(graph_circuit, **kwargs)
        expr_digraph.add_node(node, expr = expr, mbqc_circuit = graph_circuit)
    
    for edge in flow_space.flow_graph_space.edges():
        i,j = edge
        expri, exprj = expr_digraph.node[i]['expr'], expr_digraph.node[j]['expr']
        if expri<=exprj:
            expr_digraph.add_edge(j,i)
        else:
            expr_digraph.add_edge(i,j)
    
    return expr_digraph
        


                
