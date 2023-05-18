from typing import Union, List, Tuple, Optional

import numpy as np

from mentpy.simulators.base_simulator import BaseSimulator
from mentpy.mbqc.mbqcircuit import MBQCircuit

import pennylane as qml
import networkx as nx

__all__ = ["PennylaneSimulator"]


class PennylaneSimulator(BaseSimulator):
    """Simulator for measuring patterns of MBQC circuits.

    Args
    ----
    mbqcircuit: mp.MBQCircuit
        The MBQC circuit used for the simulation.
    input_state: np.ndarray
        The input state of the simulator.

    See Also
    --------
    :class:`mp.PatternSimulator`, :class:`mp.CirqSimulator`

    Group
    -----
    simulators
    """

    def __init__(
        self, mbqcircuit: MBQCircuit, input_state: np.ndarray, *args, **kwargs
    ) -> None:
        self.circuit = mbqcircuit_to_circuit(
            mbqcircuit,
            kwargs.pop("device", "default.qubit"),
            kwargs.pop("circuit_noise", None),
            p=kwargs.pop("p", 0),
        )
        super().__init__(mbqcircuit, input_state)

        if len(mbqcircuit.controlled_nodes) > 0:
            raise NotImplementedError(
                "Controlled nodes are not supported with the PennyLane simulator."
            )

    def measure(self, angle: float, plane: str = "XY"):
        raise NotImplementedError

    def run(self, angles: List[float], **kwargs) -> Tuple[List[int], np.ndarray]:
        if len(angles) != len(self.mbqcircuit.trainable_nodes):
            raise ValueError(
                f"Number of angles ({len(angles)}) does not match number of trainable nodes ({len(self.mbqcircuit.trainable_nodes)})."
            )

        # extend angles to all nodes

        extended_angles = []

        if len(self.mbqcircuit.trainable_nodes) != len(self.mbqcircuit.outputc):
            for i in self.mbqcircuit.outputc:
                if i in self.mbqcircuit.trainable_nodes:
                    angle = angles[self.mbqcircuit.trainable_nodes.index(i)]
                else:
                    plane = self.mbqcircuit.planes[i]
                    if plane == "X":
                        angle = 0
                    elif plane == "Y":
                        angle = np.pi / 2
                    elif plane == "XY":
                        angle = self.mbqcircuit[i].angle
                    else:
                        raise ValueError(
                            f"Plane {plane} is not supported for pennylane simulator."
                        )

                extended_angles.append(angle)
        else:
            extended_angles = angles

        return self.circuit(extended_angles, st=self.input_state, **kwargs)

    def reset(self, input_state=None):
        if input_state is None:
            input_state = self.input_state
        self.input_state = input_state


def mbqcircuit_to_circuit(
    gsc, device="default.qubit", circuit_noise=None, *args, **kwargs
):
    """Converts a MBQCircuit to a PennyLane circuit."""
    gr = gsc.graph
    N = gr.number_of_nodes()
    dev = qml.device(device, wires=N)

    @qml.qnode(dev)
    def circuit(param, output="density", st=None):
        if output != "density":
            assert (
                len(param) == N
            ), f"Length of param is {len(param)}, but expected {N}."
        else:
            assert len(param) == N - len(
                gsc.output_nodes
            ), f"Length of param is {len(param)}, but expected {N-len(gsc.output_nodes)}."
        input_v = st
        qml.QubitStateVector(input_v, wires=gsc.input_nodes)

        for j in gsc.inputc:
            qml.Hadamard(j)
        for i, j in gr.edges():
            qml.CZ(wires=[i, j])

        if circuit_noise is not None:
            for i in range(N):
                if circuit_noise == "depolarizing":
                    qml.DepolarizingChannel(wires=i, *args, **kwargs)
                elif circuit_noise == "amplitude_damping":
                    qml.AmplitudeDamping(wires=i, *args, **kwargs)
                elif circuit_noise == "phase_damping":
                    qml.PhaseDamping(wires=i, *args, **kwargs)
                elif circuit_noise == "phase_flip":
                    qml.PhaseFlip(wires=i, *args, **kwargs)
                elif circuit_noise == "generalized_amplitude_damping":
                    qml.GeneralizedAmplitudeDamping(wires=i, *args, **kwargs)
                else:
                    raise ValueError(f"Unrecognized circuit noise: {circuit_noise}")

        topord_no_output = [
            x for x in gsc.measurement_order if (x not in gsc.output_nodes)
        ]
        for indx in topord_no_output:
            p = param[gsc.outputc.index(indx)]

            qml.RZ(-p, wires=indx)
            qml.Hadamard(wires=indx)
            m_0 = qml.measure(indx)
            qml.cond(m_0, qml.PauliX)(wires=gsc.flow(indx))
            for neigh in gr.neighbors(gsc.flow(indx)):
                if neigh != indx and (
                    gsc.measurement_order.index(neigh)
                    > gsc.measurement_order.index(indx)
                ):
                    qml.cond(m_0, qml.PauliZ)(wires=neigh)

        # if output == "expval":
        #     for indx, p in zip(gsc.output_nodes, param[len(gsc.outputc) :]):
        #         qml.RZ(-p, wires=indx)
        #     return [qml.expval(qml.PauliX(j)) for j in gsc.output_nodes]
        # elif output == "sample":
        #     for indx, p in zip(gsc.output_nodes, param[len(gsc.outputc) :]):
        #         qml.RZ(-p, wires=indx)
        #     return [qml.sample(qml.PauliX(j)) for j in gsc.output_nodes]
        if output == "density":
            return qml.density_matrix(gsc.output_nodes)

    return circuit
