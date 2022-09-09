from graphviz import Graph
import numpy as np

from mentpy.measurement import BaseMeasurement
from mentpy.state import GraphState

from typing import List, Tuple, Callable


class GeneralMeasurement(BaseMeasurement):
    def __init__(self, state: GraphState, flow: Callable, top_order: np.ndarray):
        """Initializes GeneralMeasurement object"""
        super().__init__(state, flow)

    def measure(self, pattern: np.ndarray) -> Tuple:
        """Measures the given pattern"""

    def onequbit_measure(self, op, qubit) -> Tuple:
        """Measures qubit with op."""
