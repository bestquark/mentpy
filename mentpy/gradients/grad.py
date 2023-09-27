# Copyright (C) [2023] Luis Mantilla
#
# This program is released under the GNU GPL v3.0 or later.
# See <https://www.gnu.org/licenses/> for details.
"""Module that contains functions to calculate gradients of cost functions."""
import numpy as np
from ._finite_difference import fd_gradient, fd_hessian
from ._parameter_shift import psr_gradient, psr_hessian

__all__ = ["get_gradient", "get_hessian"]


def get_gradient(cost, x, method="parameter-shift", *args, **kwargs):
    """Calculate the gradient of a cost function.

    Args:
        cost (callable): Cost function to calculate the gradient of.
        x (array): Input to the cost function.
        method (str, optional): Method to use to calculate the gradient. Defaults to 'parameter-shift'.

    Returns:
        array: Gradient of the cost function.
    """

    match method:
        case "parameter-shift" | "psr" | "parametershift":
            return psr_gradient(cost, x, *args, **kwargs)
        case "finite-differences" | "fd" | "finitedifferences":
            return fd_gradient(cost, x, *args, **kwargs)
        case _:
            raise UserWarning(
                f"Expected method to be 'parameter-shift' or 'finite-difference' but {method} was given"
            )


def get_hessian(cost, x, method="parameter-shift", *args, **kwargs):
    """Calculate the Hessian of a cost function.

    Args:
        cost (callable): Cost function to calculate the Hessian of.
        x (array): Input to the cost function.
        method (str, optional): Method to use to calculate the Hessian. Defaults to 'parameter-shift'.

    Returns:
        array: Hessian of the cost function.
    """

    match method:
        case "parameter-shift" | "psr" | "parametershift":
            return psr_hessian(cost, x, *args, **kwargs)
        case "finite-differences" | "fd" | "finitedifferences":
            return fd_hessian(cost, x, *args, **kwargs)
        case _:
            raise UserWarning(
                f"Expected method to be 'parameter-shift' or 'finite-difference' but {method} was given"
            )
