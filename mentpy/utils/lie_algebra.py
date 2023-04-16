from itertools import combinations

import numpy as np
from mentpy import MBQCircuit, PauliOp
import galois

GF = galois.GF(2)


def _constrains(j, state, stabilizers):
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
            A.append(stabilizers[k].mat[0, mo])
            b.append(np.zeros(1, dtype=int))

    for k in [mapping[i] for i in state.outputc]:
        A.append(stabilizers[k].mat[0, np.array(mo) + len(mo)])
        p = 0 if k != mapping[j] else 1
        b.append(np.array([p]))

    return GF(np.vstack(A)), GF(np.vstack(b))


def _check_solution(A, b, x):
    print(A.shape)
    print(A)
    print(x.shape)
    print(b.shape)
    print(b)
    return np.linalg.norm(A @ (x.T) - b) == 0


def _find_solution(j, state, stabilizers):
    """Finds the solution of the system of constraints"""
    A, b = _constrains(j, state, stabilizers)
    sys = np.hstack((A, b))
    rr = sys.row_reduce()
    sol = rr[:, -1]

    if not _check_solution(A, b, sol):
        raise ValueError("Solution not found for j: " + str(j))

    op = PauliOp("I" * len(state.measurement_order))
    for i in range(len(sol)):
        if sol[i] == 1:
            op = op.commutator(stabilizers[i])
    return op


def calculate_lie_algebra(state: MBQCircuit):
    """Calculates the Lie algebra of a given state"""
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
            lieAlgebra = lieAlgebra.append(op)

    return lieAlgebra.get_subset([mapping[i] for i in state.output_nodes])


def lie_algebra_completion(generators: PauliOp, max_iter: int = 100):
    """Completes a given set of Pauli operators to a basis of the Lie algebra"""
    lieAlg = generators
    already_commutated = []
    iter = 0
    while iter < max_iter:
        iter += 1
        new_lieAlg = None
        for i, j in combinations(range(len(lieAlg)), 2):
            if (i, j) not in already_commutated:
                already_commutated.append((i, j))
                if lieAlg[i].symplectic_prod(lieAlg[j])[0][0] != 0:
                    if new_lieAlg is None:
                        new_lieAlg = lieAlg[i].commutator(lieAlg[j])
                    else:
                        new_lieAlg.append(lieAlg[i].commutator(lieAlg[j]))

        if new_lieAlg is None:
            break
        else:
            new_lieAlg = remove_repeated_ops(lieAlg.append(new_lieAlg))
            if len(new_lieAlg) == len(lieAlg):
                break

        lieAlg = new_lieAlg

    return lieAlg


def remove_repeated_ops(ops: PauliOp):
    """Removes repeated Pauli operators from a set"""
    unrep_ops = ops[0]
    for op in ops[1:]:
        if op not in unrep_ops:
            unrep_ops.append(op)
    return unrep_ops
