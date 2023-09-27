# Copyright (C) [2023] Luis Mantilla
#
# This program is released under the GNU GPL v3.0 or later.
# See <https://www.gnu.org/licenses/> for details.
"""Gates module."""
import numpy as np
from scipy.stats import unitary_group


CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

CSGate = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1j]])

HGate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

PauliX = np.array([[0, 1], [1, 0]])

PauliY = np.array([[0, -1j], [1j, 0]])

PauliZ = np.array([[1, 0], [0, -1]])

TGate = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])

SGate = np.array([[1, 0], [0, 1j]])


def random_su(n_qubits: int):
    """Returns a random special unitary in ``n_qubits`` sampled from the Haar distribution."""
    U = unitary_group.rvs(2**n_qubits)
    detU = np.linalg.det(U)
    U = U / np.power(detU, 1 / (2**n_qubits))
    return U


def swap_qubits(state_vector, i, j):
    """Return state vector after swapping qubits i and j."""

    n_qubits = int(np.log2(len(state_vector)))

    swap_matrix = np.zeros((2, 2), dtype=int)
    swap_matrix[0, 1] = swap_matrix[1, 0] = 1

    identity = np.identity(2)
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

    swapped_state_vector = swap_operator @ state_vector
    return swapped_state_vector


def arbitrary_qubit_gate(u, i, n):
    """
    Implement an arbitrary single qubit gate u on qubit i in a system of n qubits
    """
    op = 1
    for k in range(0, n):
        if k == i:
            op = np.kron(op, u)
        else:
            op = np.kron(op, np.eye(2))
    return op


def swap_ij(i, j, n):
    """
    Generate the swap operator for qubits i and j in a system of n qubits
    """
    assert i < n and j < n
    op1, op2, op3, op4 = np.ones(4)
    for k in range(n):
        if k == i or k == j:
            op1 = np.kron(op1, np.kron(np.array([[1], [0]]).T, np.array([[1], [0]])))
            op4 = np.kron(op4, np.kron(np.array([[0], [1]]).T, np.array([[0], [1]])))
        else:
            op1 = np.kron(op1, np.eye(2))
            op4 = np.kron(op4, np.eye(2))

        if k == i:
            op2 = np.kron(op2, np.kron(np.array([[1], [0]]).T, np.array([[0], [1]])))
            op3 = np.kron(op3, np.kron(np.array([[0], [1]]).T, np.array([[1], [0]])))
        elif k == j:
            op2 = np.kron(op2, np.kron(np.array([[0], [1]]).T, np.array([[1], [0]])))
            op3 = np.kron(op3, np.kron(np.array([[1], [0]]).T, np.array([[0], [1]])))
        else:
            op2 = np.kron(op2, np.eye(2))
            op3 = np.kron(op3, np.eye(2))
    return op1 + op2 + op3 + op4


def cnot_ij(i, j, n):
    """
    Generate the CNOT operator for qubits i and j in a system of n qubits
    """
    op1, op2, op3, op4 = np.ones(4)
    for k in range(1, n + 1):
        if k == i or k == j:
            op1 = np.kron(op1, np.kron(np.array([[1], [0]]).T, np.array([[1], [0]])))
        else:
            op1 = np.kron(op1, np.eye(2))
        if k == i:
            op2 = np.kron(op2, np.kron(np.array([[1], [0]]).T, np.array([[1], [0]])))
            op3 = np.kron(op3, np.kron(np.array([[0], [1]]).T, np.array([[0], [1]])))
            op4 = np.kron(op4, np.kron(np.array([[0], [1]]).T, np.array([[0], [1]])))
        elif k == j:
            op2 = np.kron(op2, np.kron(np.array([[0], [1]]).T, np.array([[0], [1]])))
            op3 = np.kron(op3, np.kron(np.array([[1], [0]]).T, np.array([[0], [1]])))
            op4 = np.kron(op4, np.kron(np.array([[0], [1]]).T, np.array([[1], [0]])))
        else:
            op2 = np.kron(op2, np.eye(2))
            op3 = np.kron(op3, np.eye(2))
            op4 = np.kron(op4, np.eye(2))

    return op1 + op2 + op3 + op4


def controlled_z(i, j, n):
    """
    Controlled z gate between qubits i and j.
    n is the total number of qubits
    """
    assert i < n and j < n
    op1, op2 = 1, 2
    for k in range(0, n):
        op1 = np.kron(op1, np.eye(2))
        if k in [i, j]:
            op2 = np.kron(
                op2,
                np.kron(np.conjugate(np.array([[0], [1]]).T), np.array([[0], [1]])),
            )
        else:
            op2 = np.kron(op2, np.eye(2))
    return op1 - op2
