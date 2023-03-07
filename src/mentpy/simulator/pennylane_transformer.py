import pennylane as qml
import networkx as nx

def graphstate_to_circuit(gsc):
    """Converts a graph state mbq"""
    gs = gsc.state
    gr = gs.graph
    N = gr.number_of_nodes()
    dev = qml.device("default.qubit", wires=N)
    @qml.qnode(dev)
    def circuit(param, output = 'expval', st = None):
        if output != 'density':
            assert len(param) == N, f"Length of param is {len(param)}, but expected {N}."
        else:
            assert len(param) == N-2, f"Length of param is {len(param)}, but expected {N-2}."
        input_v = st if st is not None else gs.input_state[0]
        qml.QubitStateVector(input_v, wires=gs.input_nodes)
        for j in gs.inputc:
            qml.Hadamard(j)
        for i,j in gr.edges():
            qml.CZ(wires=[i,j])
        
        topord_no_output = [x for x in gsc.measurement_order if (x not in gs.output_nodes)]
        for indx,p in zip(topord_no_output, param[:len(gs.outputc)]):
            qml.RZ(p, wires = indx)
            qml.Hadamard(wires= indx)
            m_0 = qml.measure(indx)
            qml.cond(m_0, qml.PauliX)(wires=gsc.flow(indx))
            for neigh in gr.neighbors(gsc.flow(indx)):
                if neigh!=indx and (gsc.measurement_order.index(neigh)>gsc.measurement_order.index(indx)):
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