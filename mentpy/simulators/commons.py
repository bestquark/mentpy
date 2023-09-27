# Copyright (C) [2023] Luis Mantilla
#
# This program is released under the GNU GPL v3.0 or later.
# See <https://www.gnu.org/licenses/> for details.
"""A module to store common functions used in the simulators."""

import numpy as np
import math


def arbitrary_qubit_gate(u, i, n):
    """
    Single qubit gate u acting on qubit i
    n is the number of qubits
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
    Swaps qubit i with qubit j
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


def partial_trace(rho, indices):
    """
    Partial trace of state rho over some indices
    """
    x, y = rho.shape
    n = int(math.log(x, 2))
    r = len(indices)
    sigma = np.zeros((int(x / (2**r)), int(y / (2**r))))
    for m in range(0, 2**r):
        qubits = format(m, "0" + f"{r}" + "b")
        ptrace = 1
        for k in range(0, n):
            if k in indices:
                idx = indices.index(k)
                if qubits[idx] == "0":
                    ptrace = np.kron(ptrace, np.array([[1], [0]]))
                elif qubits[idx] == "1":
                    ptrace = np.kron(ptrace, np.array([[0], [1]]))
            else:
                ptrace = np.kron(ptrace, np.eye(2))
        sigma = sigma + np.conjugate(ptrace.T) @ rho @ (ptrace)
    return sigma
