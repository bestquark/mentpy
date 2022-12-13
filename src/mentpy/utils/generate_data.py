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
    return [generate_haar_random_states(n_qubits) for _ in range(n_samples)]


def _generate_haar_random_state(n_qubits: int) -> np.ndarray:
    r"""Makes one Haar random state over n_qubits"""

    zero_list = n_qubits * [cirq.KET_ZERO.state_vector()]
    ket_zeros = cirq.kron(*zero_list)
    haar_random_u = cirq.testing.random_special_unitary(dim=int(2**n_qubits))
    return (haar_random_u @ ket_zeros.T).T[0]

def generate_haar_random_states(n_qubits: int, n_samples:int = 1) -> np.ndarray:
    r"""Makes one Haar random state over n_qubits"""

    if n_samples == 1:
        return _generate_haar_random_state(n_qubits)
    else:
        return [_generate_haar_random_state(n_qubits) for _ in range(n_samples)]

def random_special_unitary(n_qubits : int):
    """Returns a random special unitary in ``n_qubits`` sampled from the Haar distribution."""
    return cirq.testing.random_special_unitary(dim=int(2**n_qubits))

def random_train_test_data_for_unitary(unitary: np.ndarray, n_samples: int, test_size: float = 0.3) -> tuple:
    r"Return random training and test data (input, target) for a given unitary gate ``unitary``."
    n_qubits = int(np.log2(unitary.shape[0]))
    random_inputs = generate_haar_random_states(n_qubits, n_samples = n_samples)
    random_targets = [(unitary @ st.T).T for st in random_inputs]
    return _train_test_split(random_inputs, random_targets, test_size=test_size)

def _train_test_split(inputs, targets, test_size: float = 0.3) -> tuple:
    r"Split the data into training and test sets."
    n_samples = len(inputs)
    n_test_samples = int(n_samples * test_size)
    n_train_samples = n_samples - n_test_samples
    return (inputs[:n_train_samples], targets[:n_train_samples]), (inputs[n_train_samples:], targets[n_train_samples:])

def random_training_data_for_unitary(mbqc_circuit: GraphStateCircuit,  n_samples: int, unitary: np.ndarray) -> tuple:
    r"Return random training data (input, target) for a given unitary gate ``unitary``."
    random_inputs = generate_random_input_states(mbqc_circuit, n_samples = n_samples)
    random_targets = [(unitary @ st.T).T for st in random_inputs]
    return random_inputs, random_targets

