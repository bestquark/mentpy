import numpy as np
import pytest
from mentpy import PauliOp


def test_pauli_op_init_from_matrix():
    op = PauliOp(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0]]))
    assert op.txt == "XI\nIX\nIZ\nYI"


def test_pauli_op_init_from_string():
    op = PauliOp("XIZ;ZII;IIZ;YYY")
    print(op.matrix)
    assert np.array_equal(
        op.matrix,
        np.array(
            [
                [1, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        ),
    )


def test_pauli_op_init_from_list():
    op = PauliOp(["XIZ", "ZII", "IIZ", "YYY"])
    assert np.array_equal(
        op.matrix,
        np.array(
            [
                [1, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        ),
    )


def test_pauli_op_init_invalid_input():
    with pytest.raises(ValueError):
        PauliOp(42)


def test_pauli_op_getitem():
    op = PauliOp("XIZ;ZII;IIZ;IZI")
    assert op[0].txt == "XIZ"
    assert op[1].txt == "ZII"
    assert op[2].txt == "IIZ"
    assert op[3].txt == "IZI"


def test_pauli_op_get_subset():
    op = PauliOp("XIZ;ZII;IIZ;IZI")
    subset_op = op.get_subset([0, 2])
    assert subset_op.txt == "XZ\nZI\nIZ\nII"


def test_pauli_op_append():
    op1 = PauliOp("XIZ;IZI")
    op2 = PauliOp("XZZ")
    op1.append(op2)
    assert op1.txt == "XIZ\nIZI\nXZZ"


def test_pauli_op_contains():
    op1 = PauliOp("XIZ;IZI")
    op2 = PauliOp("XIZ")
    assert op2 in op1
