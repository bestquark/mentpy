import numpy as np
import cirq  # TODO: remove this dependency

CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

CSGate = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1j]])

HGate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

PauliX = np.array([[0, 1], [1, 0]])

PauliY = np.array([[0, -1j], [1j, 0]])

PauliZ = np.array([[1, 0], [0, -1]])

TGate = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])

SGate = np.array([[1, 0], [0, 1j]])


# TODO: Remove dependency on cirq here
def random_su(n_qubits: int):
    """Returns a random special unitary in ``n_qubits`` sampled from the Haar distribution."""
    return cirq.testing.random_special_unitary(dim=int(2**n_qubits))


def swap_qubits(state_vector, i, j):
    n_qubits = int(np.log2(len(state_vector)))

    # Generate a swap matrix
    swap_matrix = np.zeros((2, 2), dtype=int)
    swap_matrix[0, 1] = swap_matrix[1, 0] = 1

    # Generate the identity matrix for other qubits
    identity = np.identity(2)

    # Construct the full swap operator
    swap_operator = None
    for k in range(n_qubits):
        if k == i or k == j:
            current_operator = swap_matrix
        else:
            current_operator = identity

        if swap_operator is None:
            swap_operator = current_operator
        else:
            swap_operator = np.kron(swap_operator, current_operator)

    # Apply the swap operator to the state vector
    swapped_state_vector = swap_operator @ state_vector
    return swapped_state_vector
