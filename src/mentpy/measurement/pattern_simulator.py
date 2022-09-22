from abc import ABCMeta, abstractmethod
import numpy as np
import cirq
from typing import Union, Callable, List, Optional

from mentpy.state import GraphState
from mentpy.state import find_flow


class PatternSimulator:
    """Abstract class for simulating measurement patterns

    :group: measurements
    """

    def __init__(
        self,
        state: GraphState,
        simulator: cirq.SimulatorBase = cirq.Simulator(),
        flow: Optional[Callable] = None,
        top_order: Optional[np.ndarray] = None,
    ):
        """Initializes Pattern object"""
        self.state = state
        self.measure_number = 0

        if flow is None:
            flow, top_order = find_flow(state)

        self.flow = flow
        self.top_order = top_order
        self.simulator = simulator

        self.total_simu = len(state.input_nodes) + 1

        self.qubit_register = cirq.LineQubit.range(self.total_simu)

        self.current_sim_graph = state.graph.subgraph(self.current_sim_ind).copy()

        # these atributes can only be updated in measure and measure_pattern
        self.current_sim_state = self.append_plus_state(
            state.input_state, self.current_sim_graph.edges()
        )

        self.max_measure_number = len(state.outputc)
        self.state_rank = len(self.current_sim_state.shape)
        self.measurement_outcomes = {}

    @property
    def current_sim_ind(self):
        r"""Returns the current simulated indices"""
        return self.top_order[
            self.measure_number : self.measure_number + self.total_simu
        ]

    @property
    def simind2qubitind(self):
        r"""Returns a dictionary to translate from simulated indices (eg. [6, 15, 4]) to qubit
        indices (eg. [1, 3, 2])"""
        return {q: ind for ind, q in enumerate(self.current_sim_ind)}

    @property
    def qubitind2simind(self):
        r"""Returns a dictionary to translate from qubit indices (eg. [1, 3, 2]) to simulated 
        indices (eg. [6, 15, 4])"""
        return {ind: q for ind, q in enumerate(self.current_sim_ind)}

    

    def append_plus_state(self, psi, cz_neighbors):
        r"""Return :math:`\prod_{Neigh} CZ_{ij} |\psi \rangle \otimes |+\rangle`"""

        augmented_state = cirq.kron(psi, cirq.KET_PLUS.state_vector())
        result = self._run_short_circuit(
            self.entangling_moment(cz_neighbors), augmented_state
        )
        return result.final_state_vector

    def _run_short_circuit(self, moment, init_state):
        """Runs a short circuit with moment ``moment`` and initial state ``init_state``."""

        circ = cirq.Circuit()
        circ.append(cirq.I.on_each(self.qubit_register))
        circ.append(moment())
        return self.simulator.simulate(circ, initial_state=init_state)

    def entangling_moment(self, cz_neighbors):
        r"""Entangle cz_neighbors"""

        def czs_moment():
            for i, j in cz_neighbors:
                qi, qj = self.simind2qubitind[i], self.simind2qubitind[j]
                yield cirq.CZ(self.qubit_register[qi], self.qubit_register[qj])

        return czs_moment

    def measurement_moment(self, angle, qindex):
        """Return a measurement moment of qubit at qindex with angle ``angle``."""

        def measure_moment():
            qi = self.qubit_register[qindex]
            yield cirq.Rz(rads=angle).on(qi)
            yield cirq.H(qi)
            yield cirq.measure(qi)

        return measure_moment

    def measure(self, angle, correct_for_outcome = False):
        """Measure next qubit in the given topological order"""

        outcome = None

        if self.measure_number < self.max_measure_number:
            ind_to_measure = self.current_sim_ind[0]
            curr_ind_to_measure = self.simind2qubitind[ind_to_measure]
            angle_moment = self.measurement_moment(angle, curr_ind_to_measure)
            result = self._run_short_circuit(angle_moment, self.current_sim_state)
            tinds = [self.simind2qubitind[j] for j in self.current_sim_ind[1:]]
            outcome = result.measurements[f"q({curr_ind_to_measure})"]
            self.measurement_outcomes[ind_to_measure] = outcome

            if correct_for_outcome:
                self.correct_measurement_outcome(ind_to_measure, outcome)

            # update this if density matrix??
            self.current_sim_state = cirq.partial_trace_of_state_vector_as_mixture(
                result.final_state_vector, keep_indices=tinds
            )[0][1]

            self.measure_number += 1

        else:
            raise UserWarning(
                "All qubits have been measured. Consider reseting the state using self.reset()"
            )

        return outcome

    def entangle_and_measure(self, angle, **kwargs):
        """First, entangles, and then, measures the qubit lowest in the topological ordering
        and entangles the next plus state"""

        outcome = None

        if self.measure_number < self.max_measure_number:
            self.current_sim_graph = self.state.graph.subgraph(self.current_sim_ind).copy()
            # these atributes can only be updated in measure and measure_pattern
            self.current_sim_state = self.append_plus_state(
                self.current_sim_graph,
                self.current_sim_graph.edges(self.current_sim_ind[-1]),
            )
            outcome = self.measure(angle, **kwargs)
        else:
            raise UserWarning(
                "All qubits have been measured. Consider reseting the state using self.reset()"
            )

        return outcome

    def correct_measurement_outcome(self, qubit, outcome):
        r"""Correct for measurement angle by multiplying by stabilizer
        :math:`X_{f(i)} \prod_{j \in N(f(i))} Z_j`"""
        if outcome == 1:
            fqubit = self.flow(qubit)
            stab_moment = self.stabilizer_moment(self.simind2qubitind[fqubit])
            corrected_state = self._run_short_circuit(stab_moment, self.current_sim_state)
            self.current_sim_state = corrected_state
    
    def stabilizer_moment(self, qindex):
        r"""Returns the moment that applies a stabilizer :math:`X_{i} \prod_{j \in N(i)} Z_j"""

        def stabilizer_circuit():
            
            yield cirq.X(self.qubit_register[qindex])
            for qj in self.current_sim_graph.neighbors(self.qubitind2simind[qindex]):
                qj = self.qubit_register[self.simind2qubitind[qj]]
                yield cirq.Z(qj)
    
        return stabilizer_circuit
        
    
    def get_adapted_angle(self, angle, qubit):
        r"""Calculates the adapted angle at qubit ``qubit``."""
        # TODO!!

    def measure_pattern(self, pattern: Union[np.ndarray, dict]):
        """Measures in the pattern specified by the given list. Return the quantum state obtained
        after the measurement pattern.

        Args:
            pattern: dict specifying the operator (value) to be measured at qubit :math:`i` (key)
        """
        if isinstance(pattern, dict):
            pattern = [pattern[q] for q in self.top_order]

        for ind, angle in enumerate(pattern):
            if ind == 0:
                # extra qubit already entangled at initialization (because nodes in I can have edges)
                self.measure(angle)
            else:
                self.entangle_and_measure(angle)

        return self.measurement_outcomes, self.current_sim_state

    def reset(self):
        """Resets the state to run another simulation."""
        self.__init__(
            self.state, self.simulator, flow=self.flow, top_order=self.top_order
        )
