from mentpy.optimizers.base_optimizer import BaseOptimizer
from mentpy.gradients import estimate_gradient

import numpy as np


class SGDOptimizer(BaseOptimizer):
    """Class for the SGD optimizer.

    Args
    ----
    step_size : float, optional
        The step size of the optimizer, by default 0.1
    momentum : float, optional
        The momentum of the optimizer, by default 0.9
    nesterov : bool, optional
        Whether to use Nesterov momentum, by default False

    Examples
    --------
    Create an SGD optimizer

    .. ipython:: python

        opt = mp.optimizers.SGDOptimizer()
        print(opt)

    See Also
    --------
    :class:`mp.optimizers.AdamOptimizer`

    Group
    -----
    optimizers
    """

    def __init__(self, step_size=0.1, momentum=0.0, nesterov=False) -> None:
        """Initialize the SGD optimizer."""
        self.step_size = step_size
        self.momentum = momentum
        self.nesterov = nesterov

    def optimize(self, f, x0, num_iters=100, callback=None, verbose=False, **kwargs):
        """Optimize a function f using the SGD optimizer."""
        v = np.zeros(len(x0))
        x = x0

        for i in range(num_iters):
            # SGD Optimizer
            g = estimate_gradient(f, x, **kwargs)
            v = self.momentum * v - self.step_size * g
            if self.nesterov:
                x = x + self.momentum * v - self.step_size * g
            else:
                x = x + v
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
        """Optimize a function f using the SGD optimizer."""
        v = np.zeros(len(x0))
        x = x0
        norm = []
        for i in range(num_iters):
            # SGD Optimizer
            g = estimate_gradient(f, x, **kwargs)
            norm.append(np.linalg.norm(g))
            v = self.momentum * v - self.step_size * g
            if self.nesterov:
                x = x + self.momentum * v - self.step_size * g
            else:
                x = x + v
            if callback is not None:
                callback(x, i)
            if verbose:
                print(f"Iteration {i+1} of {num_iters}: {x} with value {f(x)}")
        return x, norm

    def reset(self, *args, **kwargs):
        pass
