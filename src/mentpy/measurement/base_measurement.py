"""Base measurement"""

from abc import ABCMeta, abstractmethod
import numpy as np
from typing import Tuple, Callable

from mentpy.state import GraphState

# from mentpy.measurement import pattern


class BaseMeasurement(metaclass=ABCMeta):
    """Base class for measurements

    :group: measurements
    """

    def __init__(self, state: GraphState, qubit: int):
        """Initialize a base measurement"""
        self.state = state
        self.qubit = qubit

    @abstractmethod
    def measure(self, pattern: np.ndarray) -> Tuple:
        """Measure the given pattern"""
        raise NotImplementedError

    @abstractmethod
    def onequbit_measure(self, op, qubit) -> Tuple:
        """Measure one qubit with operator op and return tuple
        containing state and measurement outcome"""
        raise NotImplementedError
