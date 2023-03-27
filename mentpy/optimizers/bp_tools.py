import numpy as np
from mentpy.gradients import estimate_gradient


def compute_gradient_variance(f, x, estimate_gradient, num_samples=10, **kwargs):
    """Compute the variance of the gradient of a function f."""
    grads = []
    for _ in range(num_samples):
        grads.append(estimate_gradient(f, x, **kwargs))
    grads = np.array(grads)
    return np.var(grads, axis=0)
