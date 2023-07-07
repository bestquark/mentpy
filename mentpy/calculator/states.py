import numpy as np


def pure2density(psi):
    """Convert a pure state to a density matrix."""
    return np.outer(psi, np.conj(psi).T)
