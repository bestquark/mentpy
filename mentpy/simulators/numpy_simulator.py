from typing import Union, List, Tuple, Optional

import numpy as np
import math
import networkx as nx

from mentpy.states.mbqcstate import MBQCState
from mentpy.simulators.base_simulator import BaseSimulator

import pennylane as qml

# COMMON GATES
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]])
Pi8 = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])

# COMMON QUANTUM STATES
q_zero = np.array([1, 0])
qubit_plus = H @ q_zero


class NumpySimulator(BaseSimulator):
    """Simulator that uses numpy to simulate the quantum circuit."""

    def __init__(
        self, mbqcstate: MBQCState, input_state: np.ndarray = None, **kwargs
    ) -> None:
        super().__init__(mbqcstate, input_state)

        self.window_size = kwargs.pop("window_size", 1)
        self.schedule = kwargs.pop("schedule", None)
        self.force0 = kwargs.pop("force0", True)

        if not self.force0:
            raise NotImplementedError("Numpy simulator does not support force0=False.")

        # TODO: FIND SCHEDULE IF NOT PROVIDED
        if mbqcstate.measurement_order is not None:
            # remove output nodes from the measurement order
            self.schedule_measure = [
                i
                for i in mbqcstate.measurement_order
                if i not in mbqcstate.output_nodes
            ]
            self.schedule = mbqcstate.measurement_order
            if self.window_size == 1 and mbqcstate.flow is not None:
                self.window_size = len(mbqcstate.input_nodes) + 1
        elif self.schedule is not None:
            self.schedule_measure = self.schedule
        else:
            raise ValueError(
                "Schedule must be provided for numpy simulator as the MBQCState does not have a flow."
            )

        self.input_state = input_state

        n_qubits_input = len(mbqcstate.input_nodes)

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

        self.qstate = self.pure2density(self.input_state)
        # get subgraph of the first window_size nodes
        self.subgraph = self.mbqcstate.graph.subgraph(self.schedule[: self.window_size])

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

        self.qstate = self.initial_czs @ self.qstate @ np.conj(self.initial_czs).T

        self.proyectors_x = self.get_proyectors(0, 0)
        self.proyectors_y = self.get_proyectors(np.pi / 2, 0)

    def current_simulated_nodes(self) -> List[int]:
        """Returns the nodes that are currently simulated."""
        return self.schedule[
            self.current_measurement : self.current_measurement + self.window_size
        ]

    def measure(self, angle: float, plane: str = "XY") -> Tuple:
        if plane != "XY" and plane != "X" and plane != "Y":
            raise NotImplementedError("Only XY plane is supported for numpy simulator.")

        if self.current_measurement >= len(self.schedule_measure):
            raise ValueError("No more measurements to be done.")

        self.qstate, outcome = self.measure_angle(angle, 0, force0=self.force0)

        self.current_measurement += 1
        self.qstate = self.partial_trace(self.qstate, [0])

        if self.current_measurement + self.window_size <= len(
            self.mbqcstate.graph.nodes
        ):
            self.qstate = np.kron(self.qstate, self.pure2density(qubit_plus))
            new_qubit = self.current_simulated_nodes()[-1]

            # get neighbours of new qubit
            neighbours = nx.neighbors(self.mbqcstate.graph, new_qubit)

            # do cz between new qubit and neighbours
            indx_new = self.current_simulated_nodes().index(new_qubit)
            for neighbour in neighbours:
                if neighbour in self.current_simulated_nodes():
                    indxn = self.current_simulated_nodes().index(neighbour)
                    cz = self.controlled_z(indxn, indx_new, self.window_size)
                    self.qstate = cz @ self.qstate @ np.conj(cz.T)

        return self.qstate, outcome

    def measure_pattern(
        self, angles: List[float], planes: Union[List[str], str] = "XY"
    ) -> Tuple[List[int], np.ndarray]:
        """Measures the quantum state in the given pattern."""
        if isinstance(planes, str):
            planes = [planes] * len(angles)

        if len(angles) != len(self.mbqcstate.trainable_nodes):
            raise ValueError(
                f"Number of angles ({len(angles)}) does not match number of trainable nodes ({len(self.mbqcstate.trainable_nodes)})."
            )

        for i in self.schedule_measure:
            if i in self.mbqcstate.trainable_nodes:
                angle = angles[self.mbqcstate.trainable_nodes.index(i)]
                plane = planes[self.mbqcstate.trainable_nodes.index(i)]
            else:
                plane = self.mbqcstate.planes[i]
                if plane == "X":
                    angle = 0
                elif plane == "Y":
                    angle = np.pi / 2
                else:
                    raise ValueError(
                        f"Plane {plane} is not supported for numpy simulator."
                    )

            self.qstate, outcome = self.measure(angle, plane)

        return self.qstate

    def reset(self, input_state: np.ndarray = None):
        """Resets the simulator to the initial state."""
        self.current_measurement = 0

        if input_state is not None:
            self.input_state = input_state
            for i in range(self.window_size - len(self.mbqcstate.input_nodes)):
                self.input_state = np.kron(self.input_state, qubit_plus)

        self.qstate = self.pure2density(self.input_state)

        self.qstate = self.initial_czs @ self.qstate @ np.conj(self.initial_czs).T

    def arbitrary_qubit_gate(self, u, i, n):
        """
        Single qubit gate u acting on qubit i
        n is the number of qubits
        """
        op = 1
        for k in range(n):
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

    def partial_trace(self, rho, indices):
        """
        Partial trace of state rho over some indices
        """
        x, y = rho.shape
        n = int(math.log(x, 2))
        r = len(indices)
        sigma = np.zeros((int(x / (2**r)), int(y / (2**r))))
        for m in range(0, 2**r):
            qubits = format(m, "0" + f"{r}" + "b")
            ptrace = 1
            for k in range(0, n):
                if k in indices:
                    idx = indices.index(k)
                    if qubits[idx] == "0":
                        ptrace = np.kron(ptrace, np.array([[1], [0]]))
                    elif qubits[idx] == "1":
                        ptrace = np.kron(ptrace, np.array([[0], [1]]))
                else:
                    ptrace = np.kron(ptrace, np.eye(2))
            sigma = sigma + np.conjugate(ptrace.T) @ rho @ (ptrace)
        return sigma

    def measure_angle(self, angle, i, force0=False):  # PF: made rho optional argument
        """
        Measures qubit i of state rho with an angle
        """
        rho = self.qstate
        n = self.window_size
        n_qubits = min(n, len(self.mbqcstate) - self.current_measurement)
        cond1 = n == n_qubits
        if angle == 0 and cond1:
            pi0, pi1 = self.proyectors_x
        elif np.isclose(angle, np.pi / 2, atol=1e-3) and cond1:
            pi0, pi1 = self.proyectors_y
        else:
            pi0, pi1 = self.get_proyectors(angle, i, n_qubits=n_qubits, force0=force0)

        prob0 = np.real(np.trace(rho @ pi0))

        if not force0:
            prob1 = np.around(
                np.real(np.trace(rho @ pi1)), 10
            )  # PF: round to deal with deterministic outcomes (0 and 1 can be numerically outside of [0,1])
            measurement = np.random.choice([0, 1], p=[prob0, prob1] / (prob0 + prob1))
        elif force0:
            measurement = 0

        if measurement == 0:
            rho = pi0 @ rho @ np.conj(pi0.T) / prob0
        elif measurement == 1:
            rho = pi1 @ rho @ np.conj(pi1.T) / prob1

        return rho, measurement

    def get_proyectors(self, angle, i, n_qubits=None, force0=False):
        """
        Returns the proyectors for the measurement of qubit i with angle
        """
        n = n_qubits or self.window_size
        pi0 = 1
        pi1 = 1
        pi0op = np.array([[1, np.exp(-angle * 1j)], [np.exp(angle * 1j), 1]]) / 2

        for k in range(0, n):
            if k == i:
                pi0 = np.kron(pi0, pi0op)
            else:
                pi0 = np.kron(pi0, np.eye(2))

        if not force0:
            pi1 = 1
            pi1op = np.array([[1, -np.exp(-angle * 1j)], [-np.exp(angle * 1j), 1]]) / 2
            for k in range(0, n):
                if k == i:
                    pi1 = np.kron(pi1, pi1op)
                else:
                    pi1 = np.kron(pi1, np.eye(2))

        return pi0, pi1

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

    def pure2density(self, psi):
        """
        Input: quantum state
        Output: corresponding density matrix
        """
        return np.outer(psi, np.conj(psi).T)
