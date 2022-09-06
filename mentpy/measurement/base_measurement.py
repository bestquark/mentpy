"""Base measurement"""

from abc import ABCMeta, abstractmethod
from mentpy.state import GraphState


class BaseMeasurement(metaclass=ABCMeta):
    """Base class for measurements"""

    def __init__(self, state: GraphState):
        """Initialize a base measurement"""
        self.state = state
