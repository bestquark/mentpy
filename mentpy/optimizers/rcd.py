from mentpy.optimizers.base_optimizer import BaseOptimizer
from mentpy.gradients import estimate_gradient

import numpy as np
import random


class RCDOptimizer(BaseOptimizer):
    """Class for the random coordinate descent optimizer.

    Args
    ----
    step_size : float, optional
        The initial step size of the optimizer, by default 0.1
    adaptive : bool, optional
        Whether to use an adaptive step size, by default False

    Examples
    --------
    Create a random coordinate descent optimizer

    .. ipython:: python

        opt = mp.optimizers.RCDOptimizer()
        print(opt)

    Group
    -----
    optimizers
    """

    def __init__(self, step_size=0.1, adaptive=False) -> None:
        """Initialize the random coordinate descent optimizer."""
        self.step_size = step_size
        self.adaptive = adaptive

    def optimize(self, f, x0, num_iters=100, callback=None, verbose=False, **kwargs):
        """Optimize a function f using the random coordinate descent optimizer."""
        x = x0
        coord_iters = np.zeros(len(x))

        for i in range(num_iters):
            # Random coordinate descent optimizer
            coord_idx = random.randint(0, len(x) - 1)
            coord_iters[coord_idx] += 1
            delta = np.zeros_like(x)
            delta[coord_idx] = 1e-5
            partial_gradient = (f(x + delta) - f(x - delta)) / (2 * delta[coord_idx])

            current_step_size = self.step_size
            if self.adaptive:
                current_step_size /= np.sqrt(coord_iters[coord_idx])

            x[coord_idx] -= current_step_size * partial_gradient

            if callback is not None:
                callback(x, i)
            if verbose:
                print(f"Iteration {i+1} of {num_iters}: {x} with value {f(x)}")
        return x

    def update_step_size(self, x, i, factor=0.99):
        """Update the step size of the optimizer."""
        self.step_size = self.step_size * factor

    def optimize_and_gradient_norm(
        self, f, x0, num_iters=100, callback=None, verbose=False, **kwargs
    ):
        """Optimize a function f using the random coordinate descent optimizer."""
        x = x0
        coord_iters = np.zeros(len(x))
        norm = []

        for i in range(num_iters):
            # Random coordinate descent optimizer
            coord_idx = random.randint(0, len(x) - 1)
            coord_iters[coord_idx] += 1
            delta = np.zeros_like(x)
            delta[coord_idx] = 1e-5
            partial_gradient = (f(x + delta) - f(x - delta)) / (2 * delta[coord_idx])

            current_step_size = self.step_size
            if self.adaptive:
                current_step_size /= np.sqrt(coord_iters[coord_idx])

            x[coord_idx] -= current_step_size * partial_gradient

            norm.append(np.linalg.norm(partial_gradient))

            if callback is not None:
                callback(x, i)
            if verbose:
                print(f"Iteration {i+1} of {num_iters}: {x} with value {f(x)}")
        return x, norm

    def reset(self, *args, **kwargs):
        pass
