from typing import Union, List, Tuple, Optional

import numpy as np
import math
import networkx as nx

from mentpy.operators import Ment
from mentpy.mbqc.mbqcircuit import MBQCircuit
from mentpy.simulators.base_simulator import BaseSimulator

# COMMON GATES
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]])
Pi8 = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])

# COMMON QUANTUM STATES
q_zero = np.array([1, 0])
qubit_plus = H @ q_zero


class NumpySimulatorSV(BaseSimulator):
    """A vector state simulator that uses numpy to simulate the quantum circuit."""

    def __init__(
        self, mbqcircuit: MBQCircuit, input_state: np.ndarray = None, **kwargs
    ) -> None:
        super().__init__(mbqcircuit, input_state)

        self.window_size = kwargs.pop("window_size", 1)
        self.schedule = kwargs.pop("schedule", None)
        self.force0 = kwargs.pop("force0", True)

        if not self.force0:
            raise NotImplementedError("Numpy simulator does not support force0=False.")

        # Only support XY Measurements
        for node in mbqcircuit.graph.nodes:
            if mbqcircuit[node] is not None:
                if mbqcircuit[node].plane not in ["X", "Y", "XY"]:
                    raise ValueError(
                        f"Node {node} has plane {mbqcircuit[node].plane}, but only XY plane is supported."
                    )

        # TODO: FIND SCHEDULE IF NOT PROVIDED
        if self.schedule is not None:
            self.schedule_measure = [
                i for i in self.schedule if i not in mbqcircuit.output_nodes
            ]
        elif mbqcircuit.measurement_order is not None:
            # remove output nodes from the measurement order
            self.schedule_measure = [
                i
                for i in mbqcircuit.measurement_order
                if i not in mbqcircuit.output_nodes
            ]
            self.schedule = mbqcircuit.measurement_order
            if self.window_size == 1 and mbqcircuit.flow is not None:
                self.window_size = len(mbqcircuit.input_nodes) + 1
        else:
            raise ValueError(
                "Schedule must be provided for numpy simulator as the MBQCircuit does not have a flow."
            )

        self.input_state = input_state

        n_qubits_input = len(mbqcircuit.input_nodes)

        if n_qubits_input > self.window_size:
            raise ValueError(
                f"Input state has {n_qubits_input} qubits, but window size is set to {self.window_size}."
                " Input state must have at most as many qubits as the window size minus one."
            )

        if self.window_size > len(self.schedule_measure):
            raise ValueError(
                f"Window size is set to {self.window_size}, but schedule only has {len(self.schedule_measure)} measurements."
            )

        self.current_measurement = 0

        for i in range(self.window_size - n_qubits_input):
            self.input_state = np.kron(self.input_state, qubit_plus)

        self.qstate = self.input_state
        # get subgraph of the first window_size nodes
        self.subgraph = self.mbqcircuit.graph.subgraph(
            self.schedule[: self.window_size]
        )

        self.initial_czs = np.eye(2**self.window_size)

        # apply cz gates to neighboring qubits
        for node in self.subgraph.nodes:
            for neighbour in self.subgraph.neighbors(node):
                # avoid repeated application of cz gates
                if node < neighbour:
                    indx = self.current_simulated_nodes().index(node)
                    indy = self.current_simulated_nodes().index(neighbour)
                    cz = self.controlled_z(indx, indy, self.window_size)
                    # self.qstate = cz @ self.qstate @ np.conj(cz).T
                    self.initial_czs = cz @ self.initial_czs

        self.qstate = self.initial_czs @ self.qstate

    def current_simulated_nodes(self) -> List[int]:
        """Returns the nodes that are currently simulated."""
        return self.schedule[
            self.current_measurement : self.current_measurement + self.window_size
        ]

    def current_number_simulated_nodes(self) -> int:
        """Returns the number of nodes that are currently simulated."""
        n = self.window_size
        return min(n, len(self.mbqcircuit) - self.current_measurement)

    def measure(self, angle: float) -> Tuple:

        if self.current_measurement >= len(self.schedule_measure):
            raise ValueError("No more measurements to be done.")

        current_ment = self.mbqcircuit[
            self.schedule_measure[self.current_measurement]
        ].copy()
        self.qstate, outcome = self.measure_ment(
            current_ment.set_angle(angle), 0, force0=self.force0
        )

        self.current_measurement += 1
        self.qstate = self.partial_trace_pure_state(self.qstate, [0])

        if self.current_measurement + self.window_size <= len(
            self.mbqcircuit.graph.nodes
        ):
            self.qstate = np.kron(self.qstate, qubit_plus)
            new_qubit = self.current_simulated_nodes()[-1]

            # get neighbours of new qubit
            neighbours = nx.neighbors(self.mbqcircuit.graph, new_qubit)

            # do cz between new qubit and neighbours
            indx_new = self.current_simulated_nodes().index(new_qubit)
            for neighbour in neighbours:
                if neighbour in self.current_simulated_nodes():
                    indxn = self.current_simulated_nodes().index(neighbour)
                    cz = self.controlled_z(indxn, indx_new, self.window_size)
                    self.qstate = cz @ self.qstate @ np.conj(cz.T)

        return self.qstate, outcome

    def run(
        self, angles: List[float], output_form="dm"
    ) -> Tuple[List[int], np.ndarray]:
        """Measures the quantum state in the given pattern.

        Args:
            angles (List[float]): List of angles to be used for the measurements.
            output_form (str): Output form of the quantum state. Can be 'dm' for density matrix or 'sv' for statevector.
        """

        if len(angles) != len(self.mbqcircuit.trainable_nodes):
            raise ValueError(
                f"Number of angles ({len(angles)}) does not match number of trainable nodes ({len(self.mbqcircuit.trainable_nodes)})."
            )

        for i in self.schedule_measure:
            if i in self.mbqcircuit.trainable_nodes:
                angle = angles[self.mbqcircuit.trainable_nodes.index(i)]
                plane = self.mbqcircuit[self.mbqcircuit.trainable_nodes.index(i)].plane
            else:
                plane = self.mbqcircuit[i].plane
                if plane == "X":
                    angle = 0
                elif plane == "Y":
                    angle = np.pi / 2
                else:
                    raise ValueError(
                        f"Plane {plane} is not supported for numpy simulator."
                    )

            self.qstate, outcome = self.measure(angle)

        # check if output nodes have a measurement, if so, measure them
        for i in self.mbqcircuit.output_nodes:
            if isinstance(self.mbqcircuit[i], Ment):
                self.qstate, outcome = self.measure_ment(
                    self.mbqcircuit[i], i, force0=self.force0
                )
        if output_form.lower() == "dm" or output_form.lower() == "densitymatrix":
            return np.outer(self.qstate, np.conj(self.qstate).T)
        elif output_form.lower() == "sv" or output_form.lower() == "statevector":
            return self.qstate
        else:
            raise ValueError(f"Output form {output_form} is not supported.")

    def reset(self, input_state: np.ndarray = None):
        """Resets the simulator to the initial state."""
        self.current_measurement = 0

        if input_state is not None:
            self.input_state = input_state
            for i in range(self.window_size - len(self.mbqcircuit.input_nodes)):
                self.input_state = np.kron(self.input_state, qubit_plus)

        self.qstate = self.input_state

        self.qstate = self.initial_czs @ self.qstate @ np.conj(self.initial_czs).T

    def arbitrary_qubit_gate(self, u, i, n):
        """
        Single qubit gate u acting on qubit i
        n is the number of qubits
        """
        op = 1
        for k in range(0, n):
            if k == i:
                op = np.kron(op, u)
            else:
                op = np.kron(op, np.eye(2))
        return op

    def swap_ij(self, i, j, n):
        """
        Swaps qubit i with qubit j
        """
        assert i < n and j < n
        op1, op2, op3, op4 = np.ones(4)
        for k in range(n):
            if k == i or k == j:
                op1 = np.kron(
                    op1, np.kron(np.array([[1], [0]]).T, np.array([[1], [0]]))
                )
                op4 = np.kron(
                    op4, np.kron(np.array([[0], [1]]).T, np.array([[0], [1]]))
                )
            else:
                op1 = np.kron(op1, np.eye(2))
                op4 = np.kron(op4, np.eye(2))

            if k == i:
                op2 = np.kron(
                    op2, np.kron(np.array([[1], [0]]).T, np.array([[0], [1]]))
                )
                op3 = np.kron(
                    op3, np.kron(np.array([[0], [1]]).T, np.array([[1], [0]]))
                )
            elif k == j:
                op2 = np.kron(
                    op2, np.kron(np.array([[0], [1]]).T, np.array([[1], [0]]))
                )
                op3 = np.kron(
                    op3, np.kron(np.array([[1], [0]]).T, np.array([[0], [1]]))
                )
            else:
                op2 = np.kron(op2, np.eye(2))
                op3 = np.kron(op3, np.eye(2))
        return op1 + op2 + op3 + op4

    def partial_trace_pure_state(self, psi, indices):
        """Partial trace of a pure state over some indices. It is asumed that the indices
        over which the trace is taken are unentangled with the rest of the rest of the system."""
        n = int(np.log2(len(psi)))
        indices_to_keep = [x for x in range(n) if x not in indices]
        preserved_dim = 2 ** len(indices_to_keep)

        # Reshape the state vector into a tensor with shape (2, 2, ..., 2)
        reshaped_psi = psi.reshape([2] * n)

        # Move the preserved dimensions to the front and traced dimensions to the back
        reshaped_psi = np.moveaxis(
            reshaped_psi, indices_to_keep, range(len(indices_to_keep))
        )

        # Select the appropriate elements from the reshaped tensor
        index_slice = [0] * len(
            indices
        )  # assuming that the traced qubits are in state |0>
        reshaped_psi = reshaped_psi[tuple(index_slice)]

        # Reshape back into a vector with shape (preserved_dim,)
        output_state = reshaped_psi.reshape((preserved_dim,))

        return output_state

    def measure_ment(self, ment: Ment, i, force0=True):
        """
        Measures a ment
        """
        if ment.plane not in ["X", "Y", "XY"]:
            raise ValueError(
                f"Plane {ment.plane} is not supported for state vector numpy simulator."
            )

        op = ment.matrix()
        if op is None:
            raise ValueError(f"Ment has no matrix representation at qubit {i}")

        p0 = (np.eye(2) + op) / 2
        p1 = (np.eye(2) - op) / 2
        p1_extended = self.arbitrary_qubit_gate(
            p1, i, self.current_number_simulated_nodes()
        )
        p0_extended = self.arbitrary_qubit_gate(
            p0, i, self.current_number_simulated_nodes()
        )

        prob0 = np.abs(np.vdot(self.qstate, p0_extended @ self.qstate)) ** 2
        prob1 = np.abs(np.vdot(self.qstate, p1_extended @ self.qstate)) ** 2

        if not force0:
            outcome = np.random.choice([0, 1], p=[prob0, prob1] / (prob0 + prob1))
        else:
            outcome = 0

        if outcome == 0:
            self.qstate = p0_extended @ self.qstate
            self.qstate /= np.linalg.norm(self.qstate)
        else:
            self.qstate = p1_extended @ self.qstate
            self.qstate /= np.linalg.norm(self.qstate)

        return self.qstate, outcome

    def controlled_z(self, i, j, n):
        """
        Controlled z gate between qubits i and j.
        n is the total number of qubits
        """
        assert i < n and j < n
        op1, op2 = 1, 2
        for k in range(0, n):
            op1 = np.kron(op1, np.eye(2))
            if k in [i, j]:
                op2 = np.kron(
                    op2,
                    np.kron(np.conjugate(np.array([[0], [1]]).T), np.array([[0], [1]])),
                )
            else:
                op2 = np.kron(op2, np.eye(2))
        return op1 - op2

    def cnot_ij(self, i, j, n):
        """
        CNOT gate with
        j: target qubit
        n: number of qubits
        """
        op1, op2, op3, op4 = np.ones(4)
        for k in range(1, n + 1):
            if k == i or k == j:
                op1 = np.kron(
                    op1, np.kron(np.array([[1], [0]]).T, np.array([[1], [0]]))
                )
            else:
                op1 = np.kron(op1, np.eye(2))
            if k == i:
                op2 = np.kron(
                    op2, np.kron(np.array([[1], [0]]).T, np.array([[1], [0]]))
                )
                op3 = np.kron(
                    op3, np.kron(np.array([[0], [1]]).T, np.array([[0], [1]]))
                )
                op4 = np.kron(
                    op4, np.kron(np.array([[0], [1]]).T, np.array([[0], [1]]))
                )
            elif k == j:
                op2 = np.kron(
                    op2, np.kron(np.array([[0], [1]]).T, np.array([[0], [1]]))
                )
                op3 = np.kron(
                    op3, np.kron(np.array([[1], [0]]).T, np.array([[0], [1]]))
                )
                op4 = np.kron(
                    op4, np.kron(np.array([[0], [1]]).T, np.array([[1], [0]]))
                )
            else:
                op2 = np.kron(op2, np.eye(2))
                op3 = np.kron(op3, np.eye(2))
                op4 = np.kron(op4, np.eye(2))

        return op1 + op2 + op3 + op4
