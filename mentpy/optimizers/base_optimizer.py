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
    def optimize(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def optimize_and_gradient_norm(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def reset(self, *args, **kwargs):
        pass
