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


class NumpySimulatorDM(BaseSimulator):
    """A density matrix simulator that uses numpy to simulate the quantum circuit."""

    # TODO: CALCULATE OPTIMAL WINDOW SIZE AUTOMATICALLY
    def __init__(
        self, mbqcircuit: MBQCircuit, input_state: np.ndarray = None, **kwargs
    ) -> None:
        super().__init__(mbqcircuit, input_state)

        self.window_size = kwargs.pop("window_size", 1)
        self.schedule = kwargs.pop("schedule", None)
        self.force0 = kwargs.pop("force0", True)

        if not self.force0:
            raise NotImplementedError("Numpy simulator does not support force0=False.")

        # TODO: FIND SCHEDULE IF NOT PROVIDED
        if self.schedule is not None:
            self.schedule_measure = [
                i for i in self.schedule if i not in mbqcircuit.measurements.values()
            ]
        elif mbqcircuit.measurement_order is not None:
            # remove output nodes from the measurement order
            self.schedule_measure = [
                i
                for i in mbqcircuit.measurement_order
                if i not in mbqcircuit.quantum_output_nodes
            ]
            self.schedule = mbqcircuit.measurement_order
            if self.window_size == 1 and mbqcircuit.flow is not None:
                self.window_size = len(mbqcircuit.input_nodes) + 1
        else:
            raise ValueError(
                "Schedule must be provided for numpy simulator as the MBQCircuit does not have a flow."
            )

        input_state = self.reorder_qubits(
            input_state,
            self.mbqcircuit.input_nodes,
            self.schedule[: len(self.mbqcircuit.input_nodes)],
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

        self.qstate = self.pure2density(self.input_state)
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

        self.qstate = self.initial_czs @ self.qstate @ np.conj(self.initial_czs).T
        self.outcomes = {}

    def current_simulated_nodes(self) -> List[int]:
        """Returns the nodes that are currently simulated."""
        return self.schedule[
            self.current_measurement : self.current_measurement + self.window_size
        ]

    def current_number_simulated_nodes(self) -> int:
        """Returns the number of nodes that are currently simulated."""
        n = self.window_size
        return min(n, len(self.mbqcircuit) - self.current_measurement)

    def measure(self, angle: float, mode="sample") -> Tuple:
        if self.current_measurement >= len(self.schedule_measure):
            raise ValueError("No more measurements to be done.")

        current_ment = self.mbqcircuit[
            self.schedule_measure[self.current_measurement]
        ].copy()
        self.qstate, outcome = self.measure_ment(
            current_ment, angle, 0, force0=self.force0, mode=mode
        )
        # check if qstate has nan
        if np.isnan(self.qstate).any():
            raise ValueError(
                "qstate has nan, you might want to increase the window size"
            )
        self.current_measurement += 1

        self.qstate = self.partial_trace(self.qstate, [0])

        if self.current_measurement + self.window_size <= len(
            self.mbqcircuit.graph.nodes
        ):
            self.qstate = np.kron(self.qstate, self.pure2density(qubit_plus))
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

    def run(self, angles: List[float], mode="sample") -> Tuple[List[int], np.ndarray]:
        """Measures the quantum state in the given pattern."""

        if len(angles) != len(self.mbqcircuit.trainable_nodes):
            raise ValueError(
                f"Number of angles ({len(angles)}) does not match number of trainable nodes ({len(self.mbqcircuit.trainable_nodes)})."
            )

        for i in self.schedule_measure:
            if i in self.mbqcircuit.trainable_nodes:
                angle = angles[self.mbqcircuit.trainable_nodes.index(i)]
            else:
                angle = self.mbqcircuit[i].angle

            self.qstate, outcome = self.measure(angle, mode)
            self.outcomes[i] = outcome

        current_output_order = [
            i for i in self.schedule if i not in self.schedule_measure
        ]
        if self.mbqcircuit.quantum_output_nodes != current_output_order:
            self.qstate = self.reorder_qubits(
                self.qstate, current_output_order, self.mbqcircuit.quantum_output_nodes
            )

        # # check if output nodes have a measurement, if so, measure them
        # for i in self.mbqcircuit.output_nodes:
        #     if isinstance(self.mbqcircuit[i], Ment):
        #         self.qstate, outcome = self.measure_ment(
        #             self.mbqcircuit[i], self.mbqcircuit[i].angle, i, force0=self.force0
        #         )
        #         self.outcomes[i] = outcome

        return self.qstate

    def reset(self, input_state: np.ndarray = None):
        """Resets the simulator to the initial state."""
        self.current_measurement = 0

        if input_state is not None:
            input_state = self.reorder_qubits(
                input_state,
                self.mbqcircuit.input_nodes,
                self.schedule[: len(self.mbqcircuit.input_nodes)],
            )
            self.input_state = input_state
            for i in range(self.window_size - len(self.mbqcircuit.input_nodes)):
                self.input_state = np.kron(self.input_state, qubit_plus)

        self.qstate = self.pure2density(self.input_state)

        self.qstate = self.initial_czs @ self.qstate @ np.conj(self.initial_czs).T
        self.outcomes = {}

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

    def measure_ment(self, ment: Ment, angle, i, force0=False, mode="sample"):
        """
        Measures a ment
        """
        op = ment.matrix(angle, self.outcomes)
        if op is None:
            raise ValueError(f"Ment has no matrix representation at qubit {i}")

        p0, p1 = ment.get_povm(angle, self.outcomes)
        p0_extended = self.arbitrary_qubit_gate(
            p0, i, self.current_number_simulated_nodes()
        )
        p1_extended = self.arbitrary_qubit_gate(
            p1, i, self.current_number_simulated_nodes()
        )

        prob0 = np.real(np.trace(self.qstate @ p0_extended))
        prob1 = np.real(np.trace(self.qstate @ p1_extended))

        COND = (mode == "expectation" or mode == "exp") and ment.plane == "Z"

        if not force0 or ment.plane == "Z":
            if COND:
                outcome = prob1 / (prob0 + prob1)
            else:
                outcome = np.random.choice([0, 1], p=[prob0, prob1] / (prob0 + prob1))
        else:
            if prob0 < 1e-4:
                outcome = 1  # important when measuring Z's -- should make pretty
            else:
                outcome = 0

        if not COND:
            if outcome == 0:
                self.qstate = p0_extended @ self.qstate @ np.conj(p0_extended).T / prob0
            else:
                self.qstate = p1_extended @ self.qstate @ np.conj(p1_extended).T / prob1

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

    def pure2density(self, psi):
        """
        Input: quantum state
        Output: corresponding density matrix
        """
        return np.outer(psi, np.conj(psi).T)

    def find_swaps(self, source, target):
        assert set(source) == set(
            target
        ), f"Both lists must have the same elements, but source={source} and target={target}"

        swaps = []
        source = list(source)  # Make a copy to avoid modifying the original list

        for i, target_element in enumerate(target):
            if source[i] != target_element:
                j = source.index(target_element, i + 1)
                source[i], source[j] = source[j], source[i]  # Swap elements
                swaps.append((i, j))

        return swaps

    def reorder_qubits(self, state, current_order, target_order):
        """
        Reorders the qubits in the given order.
        """
        new_state = state.copy()
        type_state = "dm"
        if len(new_state.shape) == 1:
            type_state = "pure"

        swaps = self.find_swaps(current_order, target_order)
        for i, j in swaps:
            swap_op = self.swap_ij(i, j, len(current_order))
            if type_state == "pure":
                new_state = swap_op @ new_state
            else:
                new_state = swap_op @ new_state @ np.conj(swap_op).T
        return new_state
