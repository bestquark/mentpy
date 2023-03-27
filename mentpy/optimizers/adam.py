import numpy as np
from mentpy.gradients import estimate_gradient

from mentpy.optimizers.base_optimizer import BaseOptimizer


class AdamOptimizer(BaseOptimizer):
    """Class for the Adam optimizer.

    Parameters
    ----------
    step_size : float, optional
        The step size of the optimizer, by default 0.1
    b1 : float, optional
        The first moment decay rate, by default 0.9
    b2 : float, optional
        The second moment decay rate, by default 0.999
    eps : float, optional
        A small number to avoid division by zero, by default 10**-8

    Examples
    --------
    Create an Adam optimizer

    .. ipython:: python

        opt = mp.optimizers.AdamOptimizer()
        print(opt)


    See Also
    --------
    :class:`mp.optimizers.SGDOptimizer`

    Group
    -----
    optimizers
    """

    def __init__(self, step_size=0.1, b1=0.9, b2=0.999, eps=10**-8) -> None:
        """Initialize the Adam optimizer."""
        self.step_size = step_size
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def optimize(self, f, x0, num_iters=100, callback=None, verbose=False, **kwargs):
        """Optimize a function f using the Adam optimizer."""
        m = np.zeros(len(x0))
        v = np.zeros(len(x0))
        x = x0

        for i in range(num_iters):
            # Adam Optimizer
            g = estimate_gradient(f, x, **kwargs)
            m = (1 - self.b1) * g + self.b1 * m  # First  moment estimate.
            v = (1 - self.b2) * (g**2) + self.b2 * v  # Second moment estimate.
            mhat = m / (1 - self.b1 ** (i + 1))  # Bias correction.
            vhat = v / (1 - self.b2 ** (i + 1))
            x = x - self.step_size * mhat / (np.sqrt(vhat) + self.eps)
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
        """Optimize a function f using the Adam optimizer."""
        m = np.zeros(len(x0))
        v = np.zeros(len(x0))
        x = x0
        norm = np.zeros(num_iters)

        for i in range(num_iters):
            # Adam Optimizer
            g = estimate_gradient(f, x, **kwargs)
            m = (1 - self.b1) * g + self.b1 * m
            v = (1 - self.b2) * (g**2) + self.b2 * v
            mhat = m / (1 - self.b1 ** (i + 1))
            vhat = v / (1 - self.b2 ** (i + 1))
            x = x - self.step_size * mhat / (np.sqrt(vhat) + self.eps)
            norm[i] = np.linalg.norm(g)
            if callback is not None:
                callback(x, i)
            if verbose:
                print(f"Iteration {i+1} of {num_iters}: {x} with value {f(x)}")

        return x, norm

    def reset(self):
        """Reset the optimizer."""
        pass
