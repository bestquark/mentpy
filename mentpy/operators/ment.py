from typing import Optional
import numpy as np

from .gates import PauliX, PauliY, PauliZ


class Ment:
    """Measurement operator.

    Args
    ----
    plane: str
        The plane of the measurement. Can be "XY", "XZ", "YZ", "X", "Y", "Z".
    angle: float
        The angle of the measurement. Only used if plane is "XY", "XZ", or "YZ".
    """

    def __init__(self, angle: Optional[float] = None, plane: str = "XY"):
        """Measurement operator."""
        plane = plane.upper()
        allowd_planes = ["XY", "XZ", "YZ", "X", "Y", "Z"]
        if plane not in allowd_planes:
            raise ValueError(f"Plane {plane} is not supported.")

        if plane in ["X", "Y", "Z"] and (angle is not None and angle != 0):
            raise ValueError(f"Plane {plane} does not support angle.")

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

    def is_trainable(self):
        "Returns True if the measurement is trainable."
        return self.angle is None and self.plane in ["XY", "XZ", "YZ"]

    def matrix(self, angle: Optional[float] = None):
        "Returns the matrix representation of the measurement."
        if self.angle is None and angle is None:
            raise ValueError("Measurement is trainable, please provide an angle.")
        elif self.angle is not None and angle is not None:
            raise ValueError("Measurement is not trainable.")
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
