from turtle import position
import warnings
from abc import ABCMeta, abstractmethod
import numpy as np
import cirq
from typing import Union, Callable, List, Optional, Any
import networkx as nx
import pennylane as qml
from mentpy import GraphState
from mentpy import find_flow


class PatternSimulator:
    """Abstract class for simulating measurement patterns

    :group: measurements
    """

    def __init__(
        self,
        state: GraphState,
        simulator: cirq.SimulatorBase = cirq.Simulator(),
        flow: Optional[Callable] = None,
        partial_order: Optional[callable] = None,
        measurement_order: Optional[List[int]] = None,
        input_state: Optional[np.ndarray] = None,
        trace_in_middle = False,
        device = "default.qubit"
    ):
        """Initializes Pattern object"""
        self.state = state
        self.measure_number = 0
        self.trace_in_middle = trace_in_middle

        if (flow is None) or (partial_order is None):
            flow, partial_order = find_flow(state)

        self.flow = flow
        self.partial_order = partial_order
        
        if measurement_order is None:
            measurement_order = self.calculate_order()

        self.measurement_order = measurement_order

        self.simulator = simulator

        self.total_simu = len(state.input_nodes) + 1

        self.qubit_register = cirq.LineQubit.range(self.total_simu)

        self.current_sim_graph = state.graph.subgraph(self.current_sim_ind).copy()

        # these atributes can only be updated in measure and measure_pattern
        if input_state is None:
            input_state = state.input_state
        else:
            pass  # TODO: Check size of input state is right

        if self.trace_in_middle:
            self.current_sim_state = self.append_plus_state(
                input_state, self.current_sim_graph.edges()
            )
        else:
            self.current_sim_state = state.input_state
            self.qubit_register = cirq.LineQubit.range(len(self.state.graph))

        self.max_measure_number = len(state.outputc)
        self.state_rank = len(self.current_sim_state.shape)
        self.measurement_outcomes = {}
        self._circuit = self.graphstate_to_circuit(device=device)
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self._circuit(*args, **kwds)

    def __repr__(self) -> str:
        return f"PatternSimulator of a graph state with {len(self.state.graph)} nodes"

    @property
    def current_sim_ind(self):
        r"""Returns the current simulated indices"""
        return self.measurement_order[
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

    def calculate_order(self):
        r"""Returns the order of the measurements"""
        n = len(self.state.graph)
        mat = np.zeros((n,n))

        for indi, i in enumerate(list(self.state.graph.nodes())):
            for indj, j in enumerate(list(self.state.graph.nodes())):
                if self.partial_order(i, j):
                    mat[indi,indj] = 1

        sum_mat = np.sum(mat, axis=1) 
        order = np.argsort(sum_mat)[::-1]

        # turn order into labels of graph
        order = [list(self.state.graph.nodes())[i] for i in order]

        return order

    def append_plus_state(self, psi, cz_neighbors):
        r"""Return :math:`\prod_{Neigh} CZ_{ij} |\psi \rangle \otimes |+\rangle`"""

        augmented_state = cirq.kron(psi, cirq.KET_PLUS.state_vector())
        result = self._run_short_circuit(
            self.entangling_moment(cz_neighbors), augmented_state
        )
        return result.final_state_vector

    def _run_short_circuit(self, moment, init_state, id_on_qubits=None):
        """Runs a short circuit with moment ``moment`` and initial state ``init_state``."""

        circ = cirq.Circuit()
        if id_on_qubits is None:
            circ.append(cirq.I.on_each(self.qubit_register))
        else:
            for qq in id_on_qubits:
                circ.append(cirq.I.on(qq))

        if isinstance(moment, list):
            for m in moment:
                circ.append(m())
        else: circ.append(moment())
        return self.simulator.simulate(circ, initial_state=init_state)

    def entangling_moment(self, cz_neighbors):
        r"""Entangle cz_neighbors"""

        def czs_moment():
            for i, j in cz_neighbors:
                if self.trace_in_middle:
                    qi, qj = self.simind2qubitind[i], self.simind2qubitind[j]
                else:
                    qi, qj = i, j
                yield cirq.CZ(self.qubit_register[qi], self.qubit_register[qj])

        return czs_moment

    def measurement_moment(self, angle, qindex, key=None):
        """Return a measurement moment of qubit at qindex with angle ``angle``."""

        def measure_moment():
            qi = self.qubit_register[qindex]
            yield cirq.Rz(rads=angle).on(qi)
            yield cirq.H(qi)
            yield cirq.measure(qi, key=key)

        return measure_moment

    def _measure(self, angle, correct_for_outcome=False):
        """Measure next qubit in the given topological order"""

        outcome = None

        if self.measure_number < self.max_measure_number:
            ind_to_measure = self.current_sim_ind[0]
            tinds = [self.simind2qubitind[j] for j in self.current_sim_ind[1:]]

            curr_ind_to_measure = self.simind2qubitind[ind_to_measure]
            angle_moment = self.measurement_moment(angle, curr_ind_to_measure)
            result = self._run_short_circuit(angle_moment, self.current_sim_state)
            self.current_sim_state = result.final_state_vector
            outcome = result.measurements[f"q({curr_ind_to_measure})"]
            self.measurement_outcomes[ind_to_measure] = outcome

            # update this if density matrix??
            # --------------------

            # For state vectors (neet to generalize to density matrices)
            partial_trace = cirq.partial_trace_of_state_vector_as_mixture(
                result.final_state_vector, keep_indices=tinds
            )

            self.current_sim_graph.remove_node(ind_to_measure)

            if len(partial_trace) > 1:
                raise RuntimeError("State evolution is not unitary.")

            self.current_sim_state = partial_trace[0][1]

            if correct_for_outcome:
                self.correct_measurement_outcome(ind_to_measure, outcome)

            # ------------------

            self.measure_number += 1

        else:
            raise UserWarning(
                "All qubits have been measured. Consider reseting the state using self.reset()"
            )

        return outcome

    def _entangle_and_measure(self, angle, **kwargs):
        """First, entangles, and then, measures the qubit lowest in the topological ordering
        and entangles the next plus state"""

        outcome = None

        if self.measure_number < self.max_measure_number:
            self.current_sim_graph = self.state.graph.subgraph(
                self.current_sim_ind
            ).copy()
            # these atributes can only be updated in measure and measure_pattern
            self.current_sim_state = self.append_plus_state(
                self.current_sim_state,
                self.current_sim_graph.edges(self.current_sim_ind[-1]),
            )
            outcome = self._measure(angle, **kwargs)
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
            pad = [
                self.qubit_register[self.simind2qubitind[pi]]
                for pi in self.current_sim_graph.nodes()
            ]
            corrected_state = self._run_short_circuit(
                stab_moment, self.current_sim_state, id_on_qubits=pad
            )
            self.current_sim_state = corrected_state.final_state_vector

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

    def measure(self, angle, **kwargs):
        r"""Return outcome of measurement of corresponding qubit in flow with angle."""
        if self.measure_number == 0:
            self._measure(angle, **kwargs)
        elif self.measure_number < self.max_measure_number:
            self._entangle_and_measure(angle, **kwargs)
        else:
            raise UserWarning(
                "All measurable qubits have been measured."
                " Consider reseting the MBQCGraph"
            )

    def measure_pattern(
        self,
        pattern: Union[np.ndarray, dict],
        input_state=None,
        correct_for_outcome=False,
    ):
        """Measures in the pattern specified by the given list. Return the quantum state obtained
        after the measurement pattern.

        Args:
            pattern: dict specifying the operator (value) to be measured at qubit :math:`i` (key)
        """

        if len(pattern) != self.max_measure_number:
            raise UserWarning(
                f"Pattern should be of size {self.max_measure_number}, "
                f"but {len(pattern)} was given."
            )

        if self.measure_number != 0:
            warnings.warn(f"Graph state was measured before, so it was reset.")
            self.reset()
 
        # Simulates with input_state as input
        if input_state is not None:
            self.reset(input_state)

        # Makes the dictionary pattern into a list
        if isinstance(pattern, dict):
            pattern = [pattern[q] for q in self.measurement_order]

        if self.trace_in_middle:
            for ind, angle in enumerate(pattern):
                if ind == 0:
                    # extra qubit already entangled at initialization (because nodes in I can have edges)
                    self._measure(angle, correct_for_outcome=correct_for_outcome)
                else:
                    self._entangle_and_measure(
                        angle, correct_for_outcome=correct_for_outcome
                    )

        else:
            notrace_moment = self._measure_pattern_notrace_moment(pattern)
            in_state = self._make_input_state_notrace(self.current_sim_state)
            result = self._run_short_circuit(notrace_moment, init_state = in_state)
            self.current_sim_state = cirq.partial_trace_of_state_vector_as_mixture(
                result.final_state_vector, keep_indices=self.state.output_nodes)[0][1]
            self.measurement_outcomes = {q : result.measurements[f'q{q}'] for q in self.state.outputc }

        return self.measurement_outcomes, self.current_sim_state
    
    def _make_input_state_notrace(self, input_state):
        r"""Returns the quantum state :math:`|\psi\rangle \otimes |+\rangle^n`."""
        inputc_state = len(self.state.inputc) * [cirq.KET_PLUS.state_vector()]
        in_state = [input_state] + inputc_state
        return cirq.kron(*in_state)

    def _measure_pattern_notrace_moment(self, angles):
        """Measures in the pattern specified by the given list. Return the quantum state obtained
        after the measurement pattern.

        Args:
            pattern: dict specifying the operator (value) to be measured at qubit :math:`i` (key)
        """
        measure_correct_moments = []
        entangle_moment = self.entangling_moment(self.state.graph.edges())
        measure_correct_moments.append(entangle_moment)
        for angle, q in zip(angles, self.measurement_order):
            mm = self.measurement_moment(angle, q, key=f"q{q}") # measure moment
            measure_correct_moments.append(mm)
            cm = self._correction_moment_notrace(q)
            measure_correct_moments.append(cm)

        return measure_correct_moments

    
    def _correction_moment_notrace(self, q):
        """Returns the correction moment for qubit q"""
        fqubit = self.flow(q)
        def stabilizer_circuit():
            yield cirq.X(self.qubit_register[fqubit]).with_classical_controls(f'q{q}')
            for qj in self.state.graph.neighbors(fqubit):
                qj = self.qubit_register[qj]
                yield cirq.Z(qj).with_classical_controls(f'q{q}')
        return stabilizer_circuit
            

    def reset(self, input_state=None):
        """Resets the state to run another simulation."""
        self.__init__(
            self.state,
            self.simulator,
            flow=self.flow,
            measurement_order=self.measurement_order,
            input_state=input_state,
            trace_in_middle=self.trace_in_middle
        )
        
    def graphstate_to_circuit(self, device = "default.qubit") -> qml.QNode:
        """Converts a graph state mbq"""
        gs = self.state
        gr = gs.graph
        N = gr.number_of_nodes()
        dev = qml.device(device, wires=N)
        @qml.qnode(dev)
        def circuit(param, output = 'expval', st = None):
            assert len(param) == N or len(param)==N-len(gs.output_nodes), f"Length of param is {len(param)}, but expected {N} or {N-len(gs.output_nodes)}."
            if len(param)!=N:
                param = np.append(param, np.zeros_like(gs.output_nodes))
            input_v = st if st is not None else gs.input_state[0]
            qml.QubitStateVector(input_v, wires=gs.input_nodes)
            for j in gs.inputc:
                qml.Hadamard(j)
            for i,j in gr.edges():
                qml.CZ(wires=[i,j])
            
            topord_no_output = [x for x in self.measurement_order if (x not in gs.output_nodes)]
            for indx,p in zip(topord_no_output, param[:len(gs.outputc)]):
                qml.RZ(p, wires = indx)
                qml.Hadamard(wires= indx)
                m_0 = qml.measure(indx)
                qml.cond(m_0, qml.PauliX)(wires=self.flow(indx))
                for neigh in gr.neighbors(self.flow(indx)):
                    if neigh!=indx and (self.measurement_order.index(neigh)>self.measurement_order.index(indx)):
                        qml.cond(m_0, qml.PauliZ)(wires=neigh)
            
            if output == 'expval':
                for indx,p in zip(gs.output_nodes, param[len(gs.outputc):]):
                    qml.RZ(p, wires = indx)
                return [qml.expval(qml.PauliX(j)) for j in gs.output_nodes]
            elif output == 'sample':
                for indx,p in zip(gs.output_nodes, param[len(gs.outputc):]):
                    qml.RZ(p, wires = indx)
                return [qml.sample(qml.PauliX(j)) for j in gs.output_nodes]
            elif output == 'density':
                return qml.density_matrix(gs.output_nodes)
        return circuit



def draw_mbqc_circuit(circuit: PatternSimulator, fix_wires = None, **kwargs):
    """Draws mbqc circuit with flow.
    
    :groups: measurements
    """
    options = {'node_color': '#FFBD59'}
    options.update(kwargs)

    fixed_nodes = circuit.state.input_nodes + circuit.state.output_nodes
    position_xy = {}
    for indx, p in enumerate(circuit.state.input_nodes):
        position_xy[p] = (0, -2*indx)
    
    separation = len(circuit.state.outputc)//len(circuit.state.output_nodes)
    if fix_wires is not None:
        for wire in fix_wires:
            if len(wire)+2>separation:
                separation =len(wire)+2
    for indx, p in enumerate(circuit.state.output_nodes):
        position_xy[p] = (2*(separation), -2*indx)
    

    if fix_wires is not None:
        x = [list(x) for x in fix_wires]
        fixed_nodes += sum(x, [])
        for indw, wire in enumerate(fix_wires):
            for indx, p in enumerate(wire):
                position_xy[p] = (2*(indx+1), -2*indw)

    node_pos = nx.spring_layout(circuit.state.graph, pos=position_xy, fixed = fixed_nodes, k=1/len(circuit.state.graph))
    nx.draw(circuit.state.graph, pos = node_pos, **options)
    nx.draw(_graph_with_flow(circuit), pos = node_pos, **options)

def _graph_with_flow(circuit: PatternSimulator):
    """Return digraph with flow (but does not have all CZ edges!)"""
    g = circuit.state.graph
    gflow = nx.DiGraph()
    gflow.add_nodes_from(g.nodes)
    for node in circuit.state.outputc:
        gflow.add_edge(node, circuit.flow(node))
    return gflow


