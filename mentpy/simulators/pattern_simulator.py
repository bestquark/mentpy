from typing import Union, List, Tuple, Optional
import numpy as np

from mentpy.states.mbqcstate import MBQCState
from mentpy.simulators.base_simulator import BaseSimulator
from mentpy.simulators.pennylane_simulator import *
from mentpy.simulators.cirq_simulator import *
from mentpy.simulators.np_simulator_dm import *

__all__ = ["PatternSimulator"]


class PatternSimulator:
    """Simulator for measuring patterns of MBQC states.

    Parameters
    ----------
    mbqcstate: mp.MBQCState
        The MBQC state used for the simulation.
    simulator: str
        The simulator to use. Currently only 'pennylane-default.qubit' is supported.

    See Also
    --------
    :class:`mp.PennylaneSimulator`, :class:`mp.CirqSimulator`

    Group
    -----
    simulators
    """

    def __init__(
        self,
        mbqcstate: MBQCState,
        input_state: np.ndarray = None,
        simulator="pennylane",
        *args,
        **kwargs,
    ) -> None:

        supported_simulators = {
            "pennylane": PennylaneSimulator,
            "cirq": CirqSimulator,
            "numpy-dm": NumpySimulatorDM,
            "numpy": NumpySimulator,
        }

        if simulator not in supported_simulators:
            raise ValueError(
                f"Simulator {simulator} not supported. Supported simulators are {supported_simulators.keys()}"
            )

        if input_state is None:
            input_state = 1
            for i in range(len(mbqcstate.input_nodes)):
                input_state = np.kron(input_state, np.array([1, 1]) / np.sqrt(2))

        self.simulator = supported_simulators[simulator](
            mbqcstate, input_state, *args, **kwargs
        )

    def __getattr__(self, name):
        return getattr(self.simulator, name)

    def __call__(
        self, angles: List[float], planes: Union[List[str], str] = "XY", **kwargs
    ):
        return self.measure_pattern(angles, planes, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.simulator!r})"

    def measure(self, angle: float, plane: str = "XY", **kwargs):
        return self.simulator.measure(angle, plane, **kwargs)

    def measure_pattern(
        self, angles: List[float], planes: Union[List[str], str] = "XY", **kwargs
    ) -> Tuple[List[int], np.ndarray]:

        return self.simulator.measure_pattern(angles, planes, **kwargs)

    def reset(self, input_state: np.ndarray = None):
        return self.simulator.reset(input_state)
