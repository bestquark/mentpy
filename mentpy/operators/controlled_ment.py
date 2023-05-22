from typing import Optional, Union, Callable
import numpy as np
import warnings

from .gates import PauliX, PauliY, PauliZ
from .ment import Ment, MentOutcome


class ControlMent(Ment):
    def __init__(
        self,
        condition: Optional[Union[bool, MentOutcome]] = None,
        true_angle: Optional[Union[int, float, tuple, str]] = None,
        true_plane: Optional[str] = "XY",
        false_angle: Optional[Union[int, float, tuple, str]] = 0,
        false_plane: Optional[str] = "X",
    ):
        """Controlled measurement operator."""
        super().__init__(angle=false_angle, plane=false_plane)
        true_ment = Ment(angle=true_angle, plane=true_plane)
        self._true_ment = true_ment
        self._condition = condition

    def __repr__(self) -> str:
        return (
            f"ControlMent(False: {super().__repr__()}, True: {repr(self._true_ment)})"
        )

    @property
    def condition(self) -> bool:
        if isinstance(self._condition, bool):
            return lambda x: self._condition
        elif isinstance(self._condition, MentOutcome):
            return self._condition

    @condition.setter
    def condition(self, condition):
        if not isinstance(condition, (bool, MentOutcome)):
            raise TypeError(
                f"Invalid argument type. Expected bool or MentOutcome but got {type(condition)}"
            )
        self._condition = condition

    @property
    def angle(self, *args, **kwargs):
        if args == () and kwargs == {}:
            if isinstance(self._condition, bool):
                if self._condition:
                    return self._true_ment.angle
                else:
                    return super().angle
            elif self._true_ment.angle is None or super().angle is None:
                return None
            else:
                return super().angle
        else:
            if self.condition(*args, **kwargs):
                return self._true_ment.angle
            else:
                return super().angle

    @property
    def plane(self, *args, **kwargs):
        if args == () and kwargs == {}:
            if self._true_ment.plane is None or super().plane is None:
                return None
            else:
                return super().plane
        else:
            if self.condition(*args, **kwargs):
                return self._true_ment.plane
            else:
                return super().plane

    @property
    def is_trainable(self):
        return super().is_trainable() or self._true_ment.is_trainable()

    def copy(self):
        return ControlMent(
            self.condition,
            self._true_ment.angle,
            self._true_ment.plane,
            self.angle,
            self.plane,
        )

    def matrix(self, angle: float | None = None, *args, **kwargs):
        """Return the matrix of the controlled measurement operator."""
        if self.condition(*args, **kwargs):
            return self._true_ment.matrix(angle, *args, **kwargs)
        else:
            return super().matrix(angle, *args, **kwargs)

    def get_povm(self, angle: float | None = None, *args, **kwargs):
        if self.condition(*args, **kwargs):
            return self._true_ment.get_povm(angle, *args, **kwargs)
        else:
            return super().get_povm(angle, *args, **kwargs)


ControlledMent = ControlMent
