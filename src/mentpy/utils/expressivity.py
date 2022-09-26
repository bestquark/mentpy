import numpy as np
import cirq
from mentpy import GraphStateCircuit, PatternSimulator
from mentpy.measurement import pattern_simulator
from mentpy.utils import generate_random_input_states


def haar_probability_density_of_fidelities(F: float, n: int):
    r"""Returns the probability density function of fidelities
    :math:`P_{Haar}(F) = (N-1)(1-F)^{N-2}` where :math:`N = 2^{n}` is
    the dimension of the Hilbert space.

    Args
    ----
    F (float): Fidelity. Must be between 0 and 1.
    n (int): Number of qubits. Must be greater than or equal to 1.

    Returns
    -------
    :math:`P_{Haar}(F) = (N-1)(1-F)^{N-2}` where :math:`N = 2^{n}`
    """
    N = int(2**n)
    return (N - 1) * ((1 - F) ** (N - 2))


def expressivity_using_relative_entropy():
    r"""Returns the expressivity calculated using relative entropy"""
    # TODO:


def expressivity_using_KL(
    graph_state_circuit: GraphStateCircuit, n_samples=10000, n_bins=1000
):
    r"""Returns the expressivity calculated using the Kullback-Leiber entropy"""
    samples_from_circuit = sample_probability_density_of_fidelities(
        graph_state_circuit, n_samples=n_samples
    )
    haar_prob_fun = lambda fid: haar_probability_density_of_fidelities(
        fid, len(graph_state_circuit.output_nodes)
    )

    # TODO: Calculate expressivity here!


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
