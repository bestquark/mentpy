import numpy as np
import cirq

from mentpy import GraphStateCircuit


def generate_random_input_states(
    mbqc_circuit: GraphStateCircuit, n_samples: int = 1
) -> list:
    r"""Generates ``n_samples`` random states for the given ``mbqc_circuit``
    sampled using the Haar measure."""

    if n_samples < 1:
        raise UserWarning(
            f"n_samples expected to be greater than or equal to 1, "
            f"but {n_samples} was given."
        )

    n_qubits = len(mbqc_circuit.input_nodes)
    return [generate_haar_random_state(n_qubits) for _ in range(n_samples)]


def generate_haar_random_state(n_qubits: int) -> np.ndarray:
    r"""Makes one Haar random state over n_qubits"""

    zero_list = n_qubits * [cirq.KET_ZERO.state_vector()]
    ket_zeros = cirq.kron(*zero_list)
    haar_random_u = cirq.testing.random_special_unitary(dim=int(2**n_qubits))
    return (haar_random_u @ ket_zeros.T).T
