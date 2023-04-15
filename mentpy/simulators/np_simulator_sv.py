from typing import Union, List, Tuple, Optional

import numpy as np
import math
import networkx as nx

from mentpy.states.mbqcstate import MBQCState
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


class NumpySimulator(BaseSimulator):
    """Simulator that uses numpy to simulate the quantum circuit"""

    def __init__(
        self, mbqcstate: MBQCState, input_state: np.ndarray = None, **kwargs
    ) -> None:
        super().__init__(mbqcstate, input_state)

        self.window_size = kwargs.pop("window_size", 1)
        self.schedule = kwargs.pop("schedule", None)
        self.force0 = kwargs.pop("force0", True)

        if not self.force0:
            raise NotImplementedError("Numpy simulator does not support force0=False.")

        self.qstate = self.input_state
        # TODO: FIND SCHEDULE IF NOT PROVIDED
        if self.schedule is not None:
            self.schedule_measure = [
                i for i in self.schedule if i not in mbqcstate.output_nodes
            ]
        elif mbqcstate.measurement_order is not None:
            # remove output nodes from the measurement order
            self.schedule_measure = [
                i
                for i in mbqcstate.measurement_order
                if i not in mbqcstate.output_nodes
            ]
            self.schedule = mbqcstate.measurement_order
            if self.window_size == 1 and mbqcstate.flow is not None:
                self.window_size = len(mbqcstate.input_nodes) + 1
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
