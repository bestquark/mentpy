import pennylane as qml


def fidelity(a, b):
    """Borrows the fidelity function from PennyLane."""
    return qml.math.fidelity(a, b)
