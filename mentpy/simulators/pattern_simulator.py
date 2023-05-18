from typing import Union, List, Tuple, Optional
import numpy as np

from mentpy.mbqc.mbqcircuit import MBQCircuit
from mentpy.simulators.base_simulator import BaseSimulator
from mentpy.simulators.pennylane_simulator import *
from mentpy.simulators.np_simulator_dm import *
from mentpy.simulators.np_simulator_sv import *

# from mentpy.simulators.cirq_simulator import *
# from mentpy.simulators.qiskit_simulator import *

__all__ = ["PatternSimulator"]


class PatternSimulator:
    """Simulator for measuring patterns of MBQC circuits.

    Parameters
    ----------
    mbqcircuit: mp.MBQCircuit
        The MBQC circuit used for the simulation.
    backend: str
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
        mbqcircuit: MBQCircuit,
        input_state: np.ndarray = None,
        backend="pennylane",
        *args,
        **kwargs,
    ) -> None:
        supported_backends = {
            "pennylane": PennylaneSimulator,
            # "cirq": CirqSimulator,
            # "qiskit": QiskitSimulator,
            "numpy-dm": NumpySimulatorDM,
            "numpy-sv": NumpySimulatorSV,
        }

        backend = backend.lower()
        if backend not in supported_backends:
            raise ValueError(
                f"Backend {backend} not supported. Supported backends are {supported_backends.keys()}"
            )

        if input_state is None:
            input_state = 1
            for i in range(len(mbqcircuit.input_nodes)):
                input_state = np.kron(input_state, np.array([1, 1]) / np.sqrt(2))
        else:
            if not isinstance(input_state, np.ndarray):
                raise TypeError(
                    f"Input state must be a numpy array, not {type(input_state)}"
                )

        self.simulator = supported_backends[backend](
            mbqcircuit, input_state, *args, **kwargs
        )

    def __getattr__(self, name):
        return getattr(self.simulator, name)

    def __call__(self, angles: List[float], **kwargs):
        return self.run(angles, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.simulator!r})"

    def measure(self, angle: float, **kwargs):
        return self.simulator.measure(angle, **kwargs)

    def run(self, angles: List[float], **kwargs) -> Tuple[List[int], np.ndarray]:
        return self.simulator.run(angles, **kwargs)

    def reset(self, input_state: np.ndarray = None):
        return self.simulator.reset(input_state)
