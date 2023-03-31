import abc
import numpy as np
from typing import Union, List, Tuple, Optional

from mentpy.states.mbqcstate import MBQCState


class BaseSimulator(abc.ABC):
    """Base class for simulators.

    Note
    ----
    This class should not be used directly. Instead, use one of the subclasses.

    Args
    ----
    mbqcstate: mp.MBQCState
        The MBQC state used for the simulation.
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
        mbqcstate: MBQCState,
        input_state: np.ndarray = None,
        trainable_nodes: List = None,
    ) -> None:
        self._mbqcstate = mbqcstate
        self._input_state = input_state
        self._trainable_nodes = trainable_nodes

    @property
    def mbqcstate(self) -> MBQCState:
        """The MBQC state used for the simulation."""
        return self._mbqcstate

    @property
    def input_state(self) -> np.ndarray:
        """The input state of the simulator."""
        return self._input_state

    @input_state.setter
    def input_state(self, input_state: np.ndarray):
        """Sets the input state of the simulator."""
        self._input_state = input_state

    @property
    def trainable_nodes(self) -> List:
        """The nodes that are trainable."""
        return self._trainable_nodes

    def __call__(
        self, angles: List[float], planes: Union[List[str], str] = "XY", **kwargs
    ):
        return self.measure_pattern(angles, planes, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} for {self.mbqcstate}"

    @abc.abstractmethod
    def measure(self, angle: float, plane: str = "XY", **kwargs):
        """Measures the state of the system.

        Parameters
        ----------
        angle: float
            The angle of measurement.
        plane: str
            The plane of measurement.
        """
        pass

    @abc.abstractmethod
    def measure_pattern(
        self, angles: List[float], planes: Union[List[str], str] = "XY", **kwargs
    ) -> Tuple[List[int], np.ndarray]:
        """Measures the state of the system.

        Parameters
        ----------
        angles: List[float]
            The angles of measurement.
        planes: List[str]
            The planes of measurement.
        """
        pass

    @abc.abstractmethod
    def reset(self, input_state=None):
        """Resets the simulator to the initial state."""
        pass
