"""Base measurement"""

from abc import ABCMeta, abstractmethod
import numpy as np
from typing import Tuple

from mentpy.state import GraphState

# from mentpy.measurement import pattern


class BaseMeasurement(metaclass=ABCMeta):
    """Base class for measurements

    :group: measurements
    """

    def __init__(self, state: GraphState):
        """Initialize a base measurement"""
        self.state = state

    @abstractmethod
    def measure(self, pattern: np.ndarray) -> Tuple:
        """Measure the given pattern"""
        raise NotImplementedError
