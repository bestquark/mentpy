from typing import Optional, Union
import numpy as np

from .gates import PauliX, PauliY, PauliZ


class Ment:
    """Measurement operator.

    Args
    ----
    angle: float
        The angle of the measurement. Only used if plane is "XY", "XZ", or "YZ".
    plane: str
        The plane of the measurement. Can be "XY", "XZ", "YZ", "X", "Y", "Z".
    """

    def __init__(
        self,
        angle: Optional[Union[int, float, str]] = None,
        plane: Optional[str] = "XY",
    ):
        """Measurement operator."""

        if isinstance(angle, (int, float)) or angle is None:
            angle = float(angle) if angle is not None else None
            plane = plane if plane is not None else "XY"
        elif isinstance(angle, str):
            temp_plane = angle
            if isinstance(plane, (int, float)):
                angle = float(plane)
            else:
                angle = None
            plane = temp_plane
        else:
            raise TypeError(
                f"Invalid argument type. Expected float or str but got {type(angle)}"
            )

        plane = plane.upper()
        allowd_planes = ["XY", "XZ", "YZ", "X", "Y", "Z"]
        if plane not in allowd_planes:
            raise ValueError(f"Plane {plane} is not supported.")

        if plane in ["X", "Y", "Z"]:
            if angle is not None and angle != 0:
                raise ValueError(f"Plane {plane} does not support angle.")
            else:
                angle = 0

        self._plane = plane
        self._angle = angle

    def __repr__(self):
        theta = round(self.angle, 4) if self.angle is not None else "θ"
        return f"Ment({theta}, {self.plane})"

    @property
    def plane(self):
        return self._plane

    @property
    def angle(self):
        return self._angle
    
    @angle.setter
    def angle(self, angle):
        self._angle = angle
    
    def set_angle(self, angle):
        # return self.__class__(angle, self.plane)
        self.angle = angle
        return self

    def is_trainable(self):
        "Returns True if the measurement is trainable."
        return self.angle is None and self.plane in ["XY", "XZ", "YZ"]

    def matrix(self, angle: Optional[float] = None):
        "Returns the matrix representation of the measurement."
        if self.angle is None and angle is None:
            raise ValueError("Measurement is trainable, please provide an angle.")
        elif self.angle is not None and angle is not None:
            raise ValueError(f"Measurement has a fixed angle of {round(self.angle, 4)}")
        elif self.angle is not None:
            angle = self.angle

        if self.plane == "XY":
            matrix = np.cos(angle) * PauliX + np.sin(angle) * PauliY 

        elif self.plane == "XZ":
            matrix = np.cos(angle) * PauliX + np.sin(angle) * PauliZ

        elif self.plane == "YZ":
            matrix = np.cos(angle) * PauliY + np.sin(angle) * PauliZ

        elif self.plane in ["X", "Y", "Z"]:
            matrices = {"X": PauliX, "Y": PauliY, "Z": PauliZ}
            matrix = matrices[self.plane]

        return matrix


Measurement = Ment
