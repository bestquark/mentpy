"""Module to calculate gradients using the parameter shift rule."""
import numpy as np


def psr_gradient(cost, x, shift=1.5):
    """Calculate the gradient of a cost function using the parameter shift rule.

    Args:
        cost (callable): Cost function to calculate the gradient of.
        x (array): Input to the cost function.
        shift (float, optional): Shift to use in the parameter shift rule. Defaults to 1.5.

    Returns:
        array: Gradient of the cost function.
    """
    grad = np.zeros(len(x))
    for i in range(len(x)):
        grad[i] = (
            cost(x + shift * np.eye(len(x))[i]) - cost(x - shift * np.eye(len(x))[i])
        ) / (2 * shift)
    return grad


def psr_hessian(cost, x, shift=1.5):
    """Calculate the Hessian of a cost function using the parameter shift rule.

    Args:
        cost (callable): Cost function to calculate the Hessian of.
        x (array): Input to the cost function.
        shift (float, optional): Shift to use in the parameter shift rule. Defaults to 1.5.

    Returns:
        array: Hessian of the cost function.
    """
    hess = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            hess[i, j] = (
                cost(x + shift * np.eye(len(x))[i] + shift * np.eye(len(x))[j])
                - cost(x + shift * np.eye(len(x))[i] - shift * np.eye(len(x))[j])
                - cost(x - shift * np.eye(len(x))[i] + shift * np.eye(len(x))[j])
                + cost(x - shift * np.eye(len(x))[i] - shift * np.eye(len(x))[j])
            ) / (4 * shift**2)
    return hess
