# Copyright (C) [2023] Luis Mantilla
#
# This program is released under the GNU GPL v3.0 or later.
# See <https://www.gnu.org/licenses/> for details.
import numpy as np
import math

__all__ = [
    "partial_trace",
    "pure2density",
    "partial_trace_pure_state",
    "partial_trace_mixed_state",
]


def pure2density(psi):
    """
    Convert a pure state to a density matrix.

    Note
    ----
    Given a pure state :math:`|\psi\rangle`, this function returns its density matrix
    representation :math:`|\psi\\rangle\langle\psi|`.

    Args
    ----
    psi: np.ndarray
        The pure state represented as a state vector.

    Returns
    -------
    np.ndarray
        The density matrix representation of the given pure state.

    Group
    -----
    quantum_operations
    """
    return np.outer(psi, np.conj(psi).T)


def partial_trace_pure_state(psi, indices):
    """
    Partial trace over specified indices for pure states.

    Note
    ----
    This function assumes the input is a pure quantum state.

    Args
    ----
    psi: np.ndarray
        The pure state represented as a state vector.
    indices: list[int]
        The indices over which the partial trace should be computed.

    Returns
    -------
    np.ndarray
        Resulting state after performing the partial trace.

    Group
    -----
    quantum_operations
    """
    num_qubits = int(np.log2(psi.shape[0]))
    remaining_qubits = sorted(set(range(num_qubits)) - set(indices))
    tensor = psi.reshape([2] * num_qubits).transpose(remaining_qubits + indices)
    traced_tensor = tensor.reshape([-1] + [2] * len(indices)).sum(
        axis=tuple(range(1, 1 + len(indices)))
    )
    traced_tensor = traced_tensor.reshape(-1)
    traced_tensor /= np.linalg.norm(traced_tensor)
    return traced_tensor


def partial_trace_mixed_state(rho, indices):
    """
    Partial trace over specified indices for mixed states.

    Note
    ----
    This function assumes the input is a mixed quantum state represented by a density matrix.

    Args
    ----
    rho: np.ndarray
        The mixed state represented as a density matrix.
    indices: list[int]
        The indices over which the partial trace should be computed.

    Returns
    -------
    np.ndarray
        Resulting state after performing the partial trace.

    Group
    -----
    quantum_operations
    """
    zero = np.array([[1], [0]])
    one = np.array([[0], [1]])
    n = int(math.log2(rho.shape[0]))
    r = len(indices)
    sigma = np.zeros((2 ** (n - r), 2 ** (n - r)), dtype=complex)
    for m in range(2**r):
        qubits = format(m, "0" + f"{r}" + "b")
        ptrace = 1
        for k in range(n):
            if k in indices:
                idx = indices.index(k)
                if qubits[idx] == "0":
                    ptrace = np.kron(ptrace, zero)
                else:
                    ptrace = np.kron(ptrace, one)
            else:
                ptrace = np.kron(ptrace, np.eye(2))
        sigma += np.conjugate(ptrace.T) @ rho @ ptrace
    return sigma


def partial_trace(data, indices):
    """
    Perform a partial trace over specified indices for a quantum state.

    Note
    ----
    The function determines the type of the input data (pure or mixed state)
    and delegates the actual computation to one of the specific functions.

    Args
    ----
    data: np.ndarray
        The quantum state, either a state vector (pure state) or a density matrix (mixed state).
    indices: list[int]
        The indices over which the partial trace should be computed.

    Returns
    -------
    np.ndarray
        Resulting state after performing the partial trace.

    See Also
    --------
    :func:`partial_trace_pure_state`, :func:`partial_trace_mixed_state`

    Group
    -----
    quantum_operations
    """
    if len(data.shape) == 1:  # Pure state
        return partial_trace_pure_state(data, indices)
    elif len(data.shape) == 2:  # Mixed state
        return partial_trace_mixed_state(data, indices)
    else:
        raise ValueError("Invalid input shape for quantum state.")
