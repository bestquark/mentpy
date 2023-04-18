from typing import Optional, Union
import numpy as np
import warnings

from .gates import PauliX, PauliY, PauliZ


class Ment:
    """Measurement operator.

    Args
    ----
    angle: float
        The angle of the measurement. Only used if plane is "XY", "XZ", "YZ", or "XYZ".
        If plane is "XYZ", the input should be a tuple of two angles.
    plane: str
        The plane of the measurement. Can be "XY", "XZ", "YZ", "XYZ", "X", "Y", "Z".
    """

    def __init__(
        self,
        angle: Optional[Union[int, float, tuple, str]] = None,
        plane: Optional[str] = "XY",
    ):
        """Measurement operator."""

        if isinstance(angle, (int, float, tuple)) or angle is None:
            angle = angle if angle is not None else None
            plane = plane if plane is not None else "XY"
        elif isinstance(angle, str):
            temp_plane = angle
            if isinstance(plane, (int, float, tuple)):
                angle = plane
            else:
                angle = None
            plane = temp_plane
        else:
            raise TypeError(
                f"Invalid argument type. Expected float or str but got {type(angle)}"
            )

        plane = plane.upper()
        allowd_planes = ["XY", "XZ", "YZ", "XYZ", "X", "Y", "Z"]
        if plane not in allowd_planes:
            raise ValueError(f"Plane {plane} is not supported.")
        elif plane == "XYZ":
            warnings.warn("Plane XYZ might be unstable. Use at your own risk.")

        if plane in ["X", "Y", "Z"]:
            if angle is not None and angle != 0:
                raise ValueError(f"Plane {plane} does not support angle.")
            else:
                angle = 0

        self._plane = plane
        self._angle = angle

    def __repr__(self):
        theta = round(self.angle, 4) if isinstance(self.angle, (int, float)) else "θ"
        theta = (
            (round(self.angle[0], 4), round(self.angle[1], 4))
            if isinstance(self.angle, tuple)
            else theta
        )
        return f"Ment({theta}, {self.plane})"

    @property
    def plane(self):
        return self._plane

    @property
    def angle(self):
        return self._angle

    def set_angle(self, angle):
        "Sets the angle of the measurement."
        self._angle = angle
        return self

    def copy(self):
        "Returns a copy of the measurement."
        return Ment(self.angle, self.plane)

    def is_trainable(self):
        "Returns True if the measurement is trainable."
        return self.angle is None and self.plane in ["XY", "XZ", "YZ", "XYZ"]

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

        elif self.plane == "XYZ":
            if isinstance(angle, tuple):
                angle1, angle2 = angle
            else:
                raise TypeError(
                    f"Invalid argument type. Expected tuple but got {type(angle)}"
                )
            matrix = (
                np.cos(angle1) * np.cos(angle2) * PauliX
                + np.sin(angle1) * np.cos(angle2) * PauliY
                + np.sin(angle2) * PauliZ
            )

        elif self.plane in ["X", "Y", "Z"]:
            matrices = {"X": PauliX, "Y": PauliY, "Z": PauliZ}
            matrix = matrices[self.plane]

        return matrix


Measurement = Ment
