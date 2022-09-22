import numpy as np

from mentpy.measurement import BaseMeasurement
from mentpy.state import GraphStateCircuit

from typing import List, Tuple, Callable


class GeneralMeasurement(BaseMeasurement):
    """General measurements class

    :group: measurements
    """

    def __init__(self, state: GraphStateCircuit, flow: Callable, top_order: np.ndarray):
        """Initializes GeneralMeasurement object"""
        super().__init__(state, flow, top_order)

    def measure(self, pattern: np.ndarray) -> Tuple:
        """Measures the given pattern"""

    def onequbit_measure(self, op, qubit) -> Tuple:
        """Measures qubit with op."""


# TODO: Implement twirling algorithm from https://arxiv.org/pdf/quant-ph/0609052.pdf !!
