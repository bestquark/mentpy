import numpy as np
import cirq

CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

CS = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1j]])

H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

PauliX = np.array([[0, 1], [1, 0]])

PauliY = np.array([[0, -1j], [1j, 0]])

PauliZ = np.array([[1, 0], [0, -1]])

T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])


def random_su(n_qubits: int):
    """Returns a random special unitary in ``n_qubits`` sampled from the Haar distribution."""
    return cirq.testing.random_special_unitary(dim=int(2**n_qubits))
