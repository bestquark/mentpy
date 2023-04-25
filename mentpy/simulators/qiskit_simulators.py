from typing import Union, List, Tuple, Optional
import numpy as np

from mentpy.mbqc.mbqcircuit import MBQCircuit
from mentpy.simulators.base_simulator import BaseSimulator


class QiskitSimulator(BaseSimulator):
    """Simulator for measuring patterns of MBQC circuits.
    Note
    ----
    This is a placeholder for the Qiskit simulator. It is not yet implemented.

    Args
    ----
    mbqcircuit: mp.MBQCircuit
        The MBQC circuit used for the simulation.
    simulator: str
        The simulator to use.
    input_state: np.ndarray
        The input state of the simulator.

    See Also
    --------
    :class:`mp.PatternSimulator`, :class:`mp.PennylaneSimulator`

    Group
    -----
    simulators
    """

    def __init__(self, mbqcircuit: MBQCircuit, input_state: np.ndarray = None) -> None:
        super().__init__(mbqcircuit, input_state)

        raise NotImplementedError

    def measure(self, angle: float, plane: str = "XY", **kwargs):
        raise NotImplementedError

    def run(
        self, angles: List[float], planes: Union[List[str], str] = "XY", **kwargs
    ) -> Tuple[List[int], np.ndarray]:
        raise NotImplementedError

    def reset(self, input_state: np.ndarray = None):
        raise NotImplementedError
