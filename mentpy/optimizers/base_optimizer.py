# Copyright (C) [2023] Luis Mantilla
#
# This program is released under the GNU GPL v3.0 or later.
# See <https://www.gnu.org/licenses/> for details.
"""A base class for optimizers.""" ""
import abc


class BaseOptimizer(abc.ABC):
    """Base class for optimizers.

    Note
    ----
    This class should not be used directly. Instead, use one of the subclasses.

    See Also
    --------
    :class:`mp.optimizers.SGDOptimizer`, :class:`mp.optimizers.AdamOptimizer`

    Group
    -----
    optimizers
    """

    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def step(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def optimize(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def optimize_and_gradient_norm(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def reset(self, *args, **kwargs):
        pass
