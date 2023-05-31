import numpy as np
from typing import Union, List
import galois
import copy

__all__ = ["PauliOp"]

GF = galois.GF(2)


class PauliOp:
    """Class for representing Pauli operators as matrices and strings.

    Parameters
    ----------
    op: Union[np.ndarray, str, List[str]]
        The Pauli operator to be represented. Can be a matrix, a string, or a list of strings.

    Examples
    --------
    Create a Pauli operator from a matrix

    .. ipython:: python

        op = mp.PauliOp(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]))
        print(op)

    Create a Pauli operator from a string

    .. ipython:: python

        op = mp.PauliOp('XIZ;ZII;IIZ;IZI')
        print(op)

    Create a Pauli operator from a list of strings

    .. ipython:: python

        op = mp.PauliOp(['XIZ', 'ZII', 'IIZ', 'IZI'])
        print(op)

    Group
    -----
    operators
    """

    def __init__(self, op: Union[np.ndarray, str, List[str]]):
        """Initialize a PauliOp object."""

        if isinstance(op, np.ndarray):
            if op.shape[1] % 2 != 0:
                raise ValueError(
                    "Tableau representation must have an even number of columns"
                )
            self.matrix = GF(op)
            self.txt = self._mat_to_txt(op)

        elif isinstance(op, str):
            op = op.replace(" ", "")
            if op[-1] == ";":
                op = op[:-1]

            op_list = op.split(";")

            if not all([len(op) == len(op_list[0]) for op in op_list]):
                raise ValueError("All Pauli operators must be the same length")

            self.txt = op.replace(";", "\n")
            self.matrix = self._txt_to_mat(op_list)

        elif isinstance(op, list):
            if not all([len(pauliop) == len(op[0]) for pauliop in op]):
                raise ValueError("All Pauli operators must be the same length")

            self.txt = "\n".join(op)
            self.matrix = self._txt_to_mat(op)

        else:
            raise ValueError(
                "PauliOp must be initialized with a string or a numpy array"
            )

        if self.matrix.shape[0] == 1:
            self._array = [self]
        else:
            self._array = [PauliOp(op) for op in self.txt.split("\n") if op != ""]

    def __repr__(self):
        return self.txt

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, key):
        paulis = self._array[key]
        if isinstance(key, slice):
            return PauliOp(";".join([pauli.txt for pauli in paulis]))
        return paulis

    def __contains__(self, item: "PauliOp"):
        for row in item.matrix:
            if not np.any((self.matrix == row).all(axis=1)):
                return False
        return True

    def __hash__(self):
        return hash(str(self.matrix))

    def __eq__(self, other: "PauliOp"):
        return np.all(self.matrix == other.matrix)

    def _mat_to_txt(self, op):
        n_qubits = op.shape[1] // 2
        txt = ""
        for i in range(op.shape[0]):
            for j in range(op.shape[1] // 2):
                if op[i, j] == 1 and op[i, j + n_qubits] == 0:
                    txt += "X"
                elif op[i, j] == 0 and op[i, j + n_qubits] == 1:
                    txt += "Z"
                elif op[i, j] == 1 and op[i, j + n_qubits] == 1:
                    txt += "Y"
                else:
                    txt += "I"
                if j == n_qubits - 1 and i != op.shape[0] - 1:
                    txt += "\n"
        return txt

    def _txt_to_mat(self, op_list):
        n_qubits = len(op_list[0])
        n_ops = len(op_list)

        mat = np.zeros((n_ops, 2 * n_qubits), dtype=int)

        for i, op in enumerate(op_list):
            for j, char in enumerate(op):
                if char == "X":
                    mat[i, j] = 1
                elif char == "Z":
                    mat[i, j + n_qubits] = 1
                elif char == "Y":
                    mat[i, j] = 1
                    mat[i, j + n_qubits] = 1
        return GF(mat)

    def commutator(self, other) -> "PauliOp":
        """Returns the commutator of two Pauli operators."""
        new_matrix = GF(self.matrix) + GF(other.matrix)
        return PauliOp(new_matrix)

    def symplectic_prod(self, other):
        """Returns the symplectic product of two Pauli operators."""
        x1 = self.matrix[:, : self.matrix.shape[1] // 2]
        x2 = other.matrix[:, : other.matrix.shape[1] // 2]
        z1 = self.matrix[:, self.matrix.shape[1] // 2 :]
        z2 = other.matrix[:, other.matrix.shape[1] // 2 :]
        return x1 @ z2.T + z1 @ x2.T

    def append(self, other):
        """Appends a Pauli operator to the end of another Pauli operator.

        Examples
        --------
        .. ipython:: python

            op1 = mp.PauliOp('XIZ;IZI')
            op2 = mp.PauliOp('XZZ')
            op1.append(op2)
            print(op1)
        """
        new_mat = np.vstack((self.matrix, other.matrix))
        self.__init__(new_mat)

    def get_subset(self, indices):
        """Returns a subset of the Pauli operator.

        Parameters
        ----------
        indices: List[int]
            The indices of the Pauli operators to be returned.

        Examples
        --------
        .. ipython:: python

            op = mp.PauliOp('XIZ;ZII;IIZ;IZI')
            print(op.get_subset([0, 2]))
        """
        inds = indices.copy()
        if max(inds) >= self.matrix.shape[1] // 2:
            raise ValueError("Index out of range")

        inds += [i + self.matrix.shape[1] // 2 for i in inds]
        return PauliOp(self.matrix[:, inds])
