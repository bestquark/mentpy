from typing import Optional, Union, Callable
import numpy as np
import warnings

from .gates import PauliX, PauliY, PauliZ
from .ment import Ment

class ControlMent(Ment):

    def __init__(
        self,
        condition: Optional[Union[bool, Callable[..., bool]]] =  None,
        true_angle: Optional[Union[int, float, tuple, str]] = None,
        true_plane: Optional[str] = "XY",
        false_angle: Optional[Union[int, float, tuple, str]] = 0,
        false_plane: Optional[str] = "X",
    ):
        """Controlled measurement operator."""
        super().__init__(angle=false_angle, plane=false_plane)
        true_ment = Ment(angle=true_angle, plane=true_plane)
        self._condition = condition
        self._true_ment = true_ment
    
    def __repr__(self) -> str:
        if self.condition:
            return self._true_ment.__repr__()
        else:
            return super().__repr__()
        
    @property
    def condition(self) -> bool:
        if isinstance(self._condition, bool):
            return self._condition
        elif isinstance(self._condition, Callable):
            return self._condition()
    
    @condition.setter
    def condition(self, condition):
        if not isinstance(condition, (bool, Callable[..., bool])):
            raise TypeError(
                f"Invalid argument type. Expected bool or Callable[..., bool] but got {type(condition)}"
            )
        self._condition = condition
    
    @property
    def angle(self):
        if self.condition:
            return self._true_ment.angle
        else:
            return super().angle
    
    @property
    def plane(self):
        if self.condition:
            return self._true_ment.plane
        else:
            return super().plane
    
    @property
    def is_trainable(self):
        return super().is_trainable() or self._true_ment.is_trainable()
    
    def copy(self):
        return ControlMent(self.condition, self._true_ment.angle, self._true_ment.plane,
                           self.angle, self.plane)

    def matrix(self, angle):
        """Return the matrix of the controlled measurement operator."""
        if self.condition:
            return self._true_ment.matrix(angle)
        else:
            return super().matrix(angle)