from mentpy import PatternSimulator


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


def expressivity_using_KL():
    r"""Returns the expressivity calculated using the Kullback-Leiber entropy"""
    # TODO:


def calculate_probability_density_of_fidelities(
    pattern_simulator: PatternSimulator, samples=1000
):
    r"""Calculates the probability of fidelities of a given graph state circuit"""
