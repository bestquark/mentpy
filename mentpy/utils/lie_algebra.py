from itertools import combinations

import numpy as np
from mentpy import MBQCircuit, PauliOp
import galois

import copy

GF = galois.GF(2)


def binary_gaussian_elimination(A, b):
    rows, cols = A.shape
    augmented_matrix = np.hstack((A, b.reshape(-1, 1)))

    row = 0
    for col in range(cols):
        # Find pivot in current column
        for pivot_row in range(row, rows):
            if augmented_matrix[pivot_row, col]:
                break
        else:
            continue

        # Swap rows if needed
        if pivot_row != row:
            augmented_matrix[[row, pivot_row]] = augmented_matrix[[pivot_row, row]]

        # Zero out below pivot
        for i in range(row + 1, rows):
            if augmented_matrix[i, col]:
                augmented_matrix[i] ^= augmented_matrix[row]

        row += 1

        if row >= rows:
            break

    # Back-substitution
    x = np.zeros(cols, dtype=int)
    for row in range(min(rows, cols) - 1, -1, -1):
        col = np.argmax(augmented_matrix[row, :cols])
        x[col] = augmented_matrix[row, -1] ^ np.sum(
            augmented_matrix[row, col + 1 : cols] * x[col + 1 : cols]
        )

    return x


def _constrains(j: int, state: MBQCircuit, stabilizers: PauliOp):
    """Creates the A matrix for the constraints on the stabilizers"""

    if j not in state.outputc:
        raise ValueError("j should be in outputc")

    mo = state.measurement_order
    mapping = state.index_mapping()
    mo = [mapping[i] for i in mo]
    j = mapping[j]

    A = []
    b = []
    for k in mo:
        indk = mo.index(k)
        indj = mo.index(j)
        if indk <= indj:
            A.append(stabilizers[k].matrix[0, : len(mo)])
            b.append(np.zeros(1, dtype=int))

    for k in [mapping[i] for i in state.outputc]:
        A.append(stabilizers[k].matrix[0, len(mo) :])
        p = 0 if k != j else 1
        b.append(np.array([p]))

    return GF(np.vstack(A)), GF(np.vstack(b))


def _check_solution(A, b, x):
    x = x.reshape(-1, 1)
    return np.linalg.norm(A @ x - b) == 0


def _find_solution(j: int, state: MBQCircuit, stabilizers: PauliOp):
    """Finds the solution of the system of constraints"""
    A, b = _constrains(j, state, stabilizers)
    sol = GF(binary_gaussian_elimination(A, b))

    if not _check_solution(A, b, sol):
        raise ValueError("Solution not found for j: " + str(j))

    op = PauliOp("I" * len(state.measurement_order))
    for i in range(len(sol)):
        if sol[i] == 1:
            op = op.commutator(stabilizers[i])
    return op


def calculate_complete_gens_lie_algebra(state: MBQCircuit):
    """Calculates the Pauli operators for the Lie algebra of a given state"""
    graph_stabs = state.stabilizers()

    lieAlgebra = None
    for i in state.outputc:
        op = _find_solution(i, state, graph_stabs)
        if lieAlgebra is None:
            lieAlgebra = op
        else:
            lieAlgebra.append(op)

    return lieAlgebra


def calculate_gens_lie_algebra(state: MBQCircuit):
    """Calculates the generators of the Lie algebra of a given state"""
    mapping = {
        i: j
        for i, j in zip(state.measurement_order, range(len(state.measurement_order)))
    }
    graph_stabs = state.stabilizers()

    lieAlgebra = None
    for i in state.outputc:
        op = _find_solution(i, state, graph_stabs)
        if lieAlgebra is None:
            lieAlgebra = op
        else:
            lieAlgebra.append(op)

    output_ops = lieAlgebra.get_subset([mapping[i] for i in state.output_nodes])
    return remove_repeated_ops(output_ops)


# def lie_algebra_completion_old(generators: PauliOp, max_iter: int = 1000):
#     """Completes a given set of Pauli operators to a basis of the Lie algebra"""
#     lieAlg = copy.deepcopy(generators)
#     already_commutated = []
#     iter = 0
#     while iter < max_iter:
#         iter += 1
#         new_lieAlg = None
#         for i, j in combinations(range(len(lieAlg)), 2):
#             if (i, j) not in already_commutated:
#                 already_commutated.append((i, j))
#                 if lieAlg[i].symplectic_prod(lieAlg[j])[0][0] != 0:
#                     if new_lieAlg is None:
#                         new_lieAlg = lieAlg[i].commutator(lieAlg[j])
#                     else:
#                         new_lieAlg.append(lieAlg[i].commutator(lieAlg[j]))

#         if new_lieAlg is None:
#             break
#         else:
#             lieAlg.append(new_lieAlg)
#             new_lieAlg = remove_repeated_ops(lieAlg)
#             if len(new_lieAlg) == len(lieAlg):
#                 break

#         lieAlg = new_lieAlg

#     if iter >= max_iter - 1:
#         raise ValueError("Max iterations reached")

#     return lieAlg


def lie_algebra_completion(generators: PauliOp, max_iter: int = 1000):
    """Completes a given set of Pauli operators to a basis of the Lie algebra"""
    lieAlg = copy.deepcopy(generators)
    already_commutated = set()
    n = len(lieAlg)
    queue = [(i, j) for i, j in combinations(range(n), 2)]
    iter = 0
    while iter < max_iter and queue:
        iter += 1
        i, j = queue.pop(0)
        if (i, j) not in already_commutated:
            already_commutated.add((i, j))
            newOp = lieAlg[i].commutator(lieAlg[j])
            if newOp not in lieAlg:
                lieAlg.append(newOp)
                queue.extend((len(lieAlg) - 1, k) for k in range(len(lieAlg) - 1))

    if iter >= max_iter - 1 and queue:
        raise ValueError("Max iterations reached")

    # check IIII is in lieAlg
    n_qubits = int(lieAlg.matrix.shape[1] // 2)
    if PauliOp("I" * n_qubits) not in lieAlg:
        lieAlg.append(PauliOp("I" * n_qubits))

    return lieAlg


def calculate_lie_algebra(state: MBQCircuit, max_iter: int = 10000):
    """Calculates the Lie algebra of a given state"""
    generators = calculate_gens_lie_algebra(state)
    return lie_algebra_completion(generators, max_iter=max_iter)


def remove_repeated_ops(ops: PauliOp):
    """Removes repeated Pauli operators from a set"""
    ops = copy.deepcopy(ops)
    unrep_ops = ops[0]
    for op in ops[1:]:
        if op not in unrep_ops:
            unrep_ops.append(op)
    return unrep_ops


# dimension of su(n)
def dim_su(n):
    """Calculates the dimension of :math:`Su(n)`"""
    return int(n**2 - 1)


def dim_so(n):
    """Calculates the dimension of :math:`So(n)`"""
    return int(n * (n - 1) // 2)


def dim_sp(n):
    """Calculates the dimension of :math:`Sp(n)`"""
    assert n % 2 == 0, "n must be even"
    n = n // 2
    return int(n * (2 * n + 1))
