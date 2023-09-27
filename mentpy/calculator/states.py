# Copyright (C) [2023] Luis Mantilla
#
# This program is released under the GNU GPL v3.0 or later.
# See <https://www.gnu.org/licenses/> for details.
import numpy as np


def pure2density(psi):
    """Convert a pure state to a density matrix."""
    return np.outer(psi, np.conj(psi).T)
