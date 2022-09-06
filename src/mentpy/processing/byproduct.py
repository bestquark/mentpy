"""This is the byproduct module used to calculate the byproduct of a graph state with a given flow."""

import numpy as np

from mentpy.state import GraphState
from mentpy.measurement import BaseMeasurement

class ByProduct():
    """Calculate the ByProduct operator of a given graph state with flow
    :group: processing
    """

    def __init__(self, graph_state : GraphState) -> None:
        """Initialize the ByProduct operator"""
        self.graph_state = graph_state
        

    def correct(self, measurement_pattern , measurement_outcome : np.ndarray):
        """Correct the measurement outcomes given a measurement pattern"""
        raise NotImplementedError