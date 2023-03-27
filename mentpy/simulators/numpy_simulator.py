from typing import Union, List, Tuple, Optional
import numpy as np

from mentpy.states.mbqcstate import MBQCState
from mentpy.simulators.base_simulator import BaseSimulator


class NumpySimulator(BaseSimulator):
    def __init__(self, mbqcstate: MBQCState, input_state: np.ndarray = None) -> None:
        super().__init__(mbqcstate, input_state)

    def measure(self, angle: float, plane: str = "XY", **kwargs):
        raise NotImplementedError

    def measure_pattern(
        self, angles: List[float], planes: Union[List[str], str] = "XY", **kwargs
    ) -> Tuple[List[int], np.ndarray]:
        raise NotImplementedError

    def reset(self, input_state: np.ndarray = None):
        raise NotImplementedError
