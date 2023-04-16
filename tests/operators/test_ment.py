import numpy as np
import pytest
import mentpy as mp


def test_ment_initialization():
    ment = mp.Ment(0, "X")
    assert ment.plane == "X"
    assert ment.angle == 0


def test_ment_repr():
    ment = mp.Ment(None, "X")
    assert repr(ment) == "Ment(0, X)"

    ment = mp.Ment(np.pi / 4, "XY")
    assert repr(ment) == f"Ment({round(np.pi / 4, 4)}, XY)"


def test_ment_is_trainable():
    ment = mp.Ment(None, "X")
    assert ment.is_trainable() == False

    ment = mp.Ment(None, "YZ")
    assert ment.is_trainable() == True

    ment = mp.Ment(np.pi / 4, "XY")
    assert ment.is_trainable() == False


def test_ment_matrix():

    ment = mp.Ment(np.pi / 4, "XY")
    assert np.allclose(
        ment.matrix(),
        np.cos(np.pi / 4) * mp.gates.PauliX + np.sin(np.pi / 4) * mp.gates.PauliY,
    )

    ment = mp.Ment(0, "X")
    assert np.allclose(ment.matrix(), mp.gates.PauliX)

    ment = mp.Ment(0, "Y")
    assert np.allclose(ment.matrix(), mp.gates.PauliY)

    ment = mp.Ment(0, "Z")
    assert np.allclose(ment.matrix(), mp.gates.PauliZ)


def test_ment_invalid_plane():
    with pytest.raises(ValueError):
        mp.Ment(None, "A")


def test_ment_invalid_angle():
    with pytest.raises(ValueError):
        mp.Ment(np.pi, "X")
