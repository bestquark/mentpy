import abc
import numpy as np
from typing import Union, List, Tuple, Optional

from mentpy.mbqc.mbqcircuit import MBQCircuit


class BaseSimulator(abc.ABC):
    """Base class for simulators.

    Note
    ----
    This class should not be used directly. Instead, use one of the subclasses.

    Args
    ----
    mbqcircuit: mp.MBQCircuit
        The MBQC circuit used for the simulation.
    input_state: np.ndarray
        The input state of the simulator.

    See Also
    --------
    :class:`mp.PatternSimulator`, :class:`mp.PennylaneSimulator`, :class:`mp.CirqSimulator`

    Group
    -----
    simulators
    """

    def __init__(
        self,
        mbqcircuit: MBQCircuit,
        input_state: np.ndarray = None,
    ) -> None:
        self._mbqcirc = mbqcircuit
        self._input_state = input_state
        self._outcomes = {}

    @property
    def mbqcircuit(self) -> MBQCircuit:
        """The MBQC circuit used for the simulation."""
        return self._mbqcirc

    @property
    def input_state(self) -> np.ndarray:
        """The input state of the simulator."""
        return self._input_state

    @input_state.setter
    def input_state(self, input_state: np.ndarray):
        """Sets the input state of the simulator."""
        self._input_state = input_state

    @property
    def outcomes(self) -> dict:
        """The outcomes of the simulation."""
        return self._outcomes

    @outcomes.setter
    def outcomes(self, outcomes: dict):
        """Sets the outcomes of the simulation."""
        self._outcomes = outcomes

    def __call__(self, angles: List[float], **kwargs):
        return self.run(angles, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} for {self.mbqcircuit}"

    @abc.abstractmethod
    def measure(self, angle: float, **kwargs):
        """Measures the state of the system.

        Parameters
        ----------
        angle: float
            The angle of measurement.
        """
        pass

    @abc.abstractmethod
    def run(self, parameters: List[float], **kwargs) -> Tuple[List[int], np.ndarray]:
        """Measures the state of the system.

        Parameters
        ----------
        parameters: List[float]
            The parameters of the MBQC circuit (if any).
        """
        pass

    @abc.abstractmethod
    def reset(self, input_state=None):
        """Resets the simulator to the initial state."""
        pass
