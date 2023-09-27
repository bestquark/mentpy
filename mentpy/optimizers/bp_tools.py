# Copyright (C) [2023] Luis Mantilla
#
# This program is released under the GNU GPL v3.0 or later.
# See <https://www.gnu.org/licenses/> for details.
"""A module to study barren plateaus in MBQC."""
import numpy as np
from mentpy.gradients import grad


def compute_gradient_variance(f, x, estimate_gradient, num_samples=10, **kwargs):
    """Compute the variance of the gradient of a function f."""
    grads = []
    for _ in range(num_samples):
        grads.append(estimate_gradient(f, x, **kwargs))
    grads = np.array(grads)
    return np.var(grads, axis=0)
