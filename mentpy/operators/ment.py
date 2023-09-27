# Copyright (C) [2023] Luis Mantilla
#
# This program is released under the GNU GPL v3.0 or later.
# See <https://www.gnu.org/licenses/> for details.
"""A module for measurement operators."""
from typing import Optional, Union, Callable, Any
import numpy as np
import warnings

from .gates import PauliX, PauliY, PauliZ


class MentOutcome:
    """Measurement outcome class."""

    def __init__(self, outcome: Callable[..., bool], node_id=None, cond_nodes=None):
        if isinstance(outcome, (bool, int)):
            outcome_value = bool(outcome % 2)
            outcome = lambda *args, **kwargs: outcome_value
        self._outcome = outcome
        self._node_id = node_id
        if isinstance(outcome, MentOutcome):
            self._cond_nodes = outcome.cond_nodes
        else:
            self._cond_nodes = (
                cond_nodes
                if cond_nodes is not None
                else (set([node_id]) if node_id is not None else set())
            )

    @property
    def node_id(self):
        return self._node_id

    @node_id.setter
    def node_id(self, node_id):
        self._node_id = node_id

    @property
    def cond_nodes(self):
        return self._cond_nodes

    def __repr__(self) -> str:
        return f"Measurement Outcome"

    def __call__(self, *args, **kwargs):
        try:
            return self._outcome(*args, **kwargs)
        except:
            raise UserWarning("Could not evaluate callable at given")

    def _binary_operation(self, operation, other):
        if isinstance(other, (bool, int)):
            return MentOutcome(
                lambda x: bool(operation(self._outcome(x), other) % 2),
                cond_nodes=self._cond_nodes,
            )
        elif isinstance(other, MentOutcome):
            return MentOutcome(
                lambda x: bool(operation(self._outcome(x), other._outcome(x)) % 2),
                cond_nodes=self._cond_nodes | other._cond_nodes,
            )
        elif isinstance(other, Callable):
            return MentOutcome(
                lambda x: bool(operation(self._outcome(x), other(x)) % 2),
                cond_nodes=self._cond_nodes,
            )
        else:
            raise TypeError(f"Invalid type {type(other)}")

    def __mul__(self, other):
        return self._binary_operation(lambda x, y: x * y, other)

    def __add__(self, other):
        return self._binary_operation(lambda x, y: x + y, other)

    def __sub__(self, other):
        return self._binary_operation(lambda x, y: x - y, other)

    def __truediv__(self, other):
        return self._binary_operation(lambda x, y: x / y, other)

    def __floordiv__(self, other):
        return self._binary_operation(lambda x, y: x // y, other)

    def __mod__(self, other):
        return self._binary_operation(lambda x, y: x % y, other)

    def __pow__(self, other):
        return self._binary_operation(lambda x, y: x**y, other)

    def __eq__(self, other):
        return self._binary_operation(lambda x, y: x == y, other)

    def __ne__(self, other):
        return self._binary_operation(lambda x, y: x != y, other)

    def __lt__(self, other):
        return self._binary_operation(lambda x, y: x < y, other)

    def __le__(self, other):
        return self._binary_operation(lambda x, y: x <= y, other)

    def __gt__(self, other):
        return self._binary_operation(lambda x, y: x > y, other)

    def __ge__(self, other):
        return self._binary_operation(lambda x, y: x >= y, other)

    def __and__(self, other):
        return self._binary_operation(lambda x, y: x and y, other)

    def __or__(self, other):
        return self._binary_operation(lambda x, y: x or y, other)

    def __xor__(self, other):
        return self._binary_operation(lambda x, y: x ^ y, other)

    def __invert__(self):
        return MentOutcome(lambda x: not self._outcome(x))


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
        self._node_id = -1
        self._outcome = MentOutcome(lambda x: x[self._node_id])

    def __repr__(self):
        theta = round(self._angle, 4) if isinstance(self._angle, (int, float)) else "θ"
        theta = (
            (round(self._angle[0], 4), round(self._angle[1], 4))
            if isinstance(self._angle, tuple)
            else theta
        )
        return f"Ment({theta}, {self._plane})"

    @property
    def plane(self):
        return self._plane

    @property
    def angle(self):
        return self._angle

    @property
    def outcome(self) -> MentOutcome:
        return self._outcome

    @property
    def node_id(self) -> Any:
        return self._node_id

    @node_id.setter
    def node_id(self, node_id: Any):
        self._node_id = node_id
        self._outcome = MentOutcome(lambda x: x[self._node_id], self._node_id)

    def set_angle(self, angle):
        "Sets the angle of the measurement."
        self._angle = angle
        return self

    def copy(self):
        "Returns a copy of the measurement."
        return Ment(self._angle, self._plane)

    def is_trainable(self):
        "Returns True if the measurement is trainable."
        return self._angle is None and self._plane in ["XY", "XZ", "YZ", "XYZ"]

    def matrix(self, angle: Optional[float] = None, *args, **kwargs):
        "Returns the matrix representation of the measurement."
        if self._angle is None and angle is None:
            raise ValueError("Measurement is trainable, please provide an angle.")
        elif self._angle is not None and angle is not None:
            if self._angle != angle:
                raise ValueError(
                    f"Measurement has a fixed angle of {round(self._angle, 4)}"
                )
        elif self._angle is not None:
            angle = self._angle

        match self._plane:
            case "XY":
                matrix = np.cos(angle) * PauliX + np.sin(angle) * PauliY
            case "X" | "Y" | "Z":
                matrices = {"X": PauliX, "Y": PauliY, "Z": PauliZ}
                matrix = matrices[self._plane]
            case "XZ":
                matrix = np.cos(angle) * PauliX + np.sin(angle) * PauliZ
            case "YZ":
                matrix = np.cos(angle) * PauliY + np.sin(angle) * PauliZ
            case "XYZ":
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

        return matrix

    def get_povm(self, angle: Optional[float] = None, *args, **kwargs):
        """Returns the POVM representation of the measurement."""
        mat = self.matrix(angle, *args, **kwargs)
        m0 = (np.eye(2) + mat) / 2
        m1 = (np.eye(2) - mat) / 2
        return m0, m1


Measurement = Ment
