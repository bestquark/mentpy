"""This is the Flow module. It deals with the flow of a given graph state"""
import math
import numpy as np
import networkx as nx

from mentpy.state import GraphState
from typing import List

import galois


def find_flow(state: GraphState, sanity_check=True):
    r"""Finds the generalized flow of graph state if allowed.

    Implementation of https://arxiv.org/pdf/quant-ph/0603072.pdf.

    Returns
    -------
    The flow function ``flow`` and the partial order function.

    Examples
    --------
    Find the flow of a graph state :math:`|G\rangle`.

    .. ipython:: python

        g = nx.Graph()
        g.add_edges_from([(0,1), (1,2), (2,3), (3, 4)])
        state = mtp.MBQCGraph(g, input_nodes = [0], output_nodes = [4])
        flow, partial_order = mtp.find_flow(state)
        print("Flow of node 1: ", flow(1))
        print("Partial order: ", partial_order)

    :group: states
    """
    n_input, n_output = len(state.input_nodes), len(state.output_nodes)
    if n_input != n_output:
        raise ValueError(
            f"Cannot find flow. Input ({n_input}) and output ({n_output}) nodes have different size."
        )
    
    update_labels = False
    # check if labels of graph are integers going from 0 to n-1 and if not, create a mapping
    if not all([i in state.graph.nodes for i in range(len(state.graph))]):
        mapping = {v: i for i, v in enumerate(state.graph.nodes)}
        inverse_mapping = {i: v for i, v in enumerate(state.graph.nodes)}
        # create a copy of the object state
        old_state = state
        new_graph = nx.relabel_nodes(state.graph.copy(), mapping)
        inp, outp = [mapping[v] for v in state.input_nodes], [mapping[v] for v in state.output_nodes]

        state = GraphState(new_graph, input_nodes = inp, output_nodes = outp)
        update_labels = True

    tau = _build_path_cover(state)
    if tau:
        f, P, L = _get_chain_decomposition(state, tau)
        sigma = _compute_suprema(state, f, P, L)

        if sigma is not None:
            int_flow = _flow_from_array(state, f)
            vertex2index = {v: index for index, v in enumerate(state.input_nodes)}
            def int_partial_order(x,y):
                return sigma[vertex2index[int(P[y])], int(x)] <= L[y]

            # if labels were updated, update them back
            if update_labels:
                state = old_state
                flow = lambda v: inverse_mapping[int_flow(mapping[v])]
                partial_order = lambda x,y: int_partial_order(mapping[x], mapping[y])
            else:
                flow = int_flow
                partial_order = int_partial_order

            state_flow = (flow, partial_order)
            if sanity_check:
                if not _check_if_flow(state, flow, partial_order):
                    raise RuntimeError(
                        "Sanity check found that flow does not satisfy flow conditions."
                    )
            return state_flow

        else: 
            raise UserWarning("The given state does not have a flow.")
    else:
        raise UserWarning("Could not find a flow for the given state.")


def _flow_from_array(state: GraphState, f: List):
    """Create a flow function from a given array f"""

    def flow(v):
        if v in state.outputc:
            return int(f[v])
        else:
            raise UserWarning(f"The node {v} is not in domain of the flow.")

    return flow


def _get_chain_decomposition(state: GraphState, C: nx.DiGraph):
    """Gets the chain decomposition"""
    P = np.zeros(len(state.graph))
    L = np.zeros(len(state.graph))
    f = {v: 0 for v in set(state.graph) - set(state.output_nodes)}
    for i in state.input_nodes:
        v, l = i, 0
        while v not in state.output_nodes:
            f[v] = int(next(C.successors(v)))
            P[v] = i
            L[v] = l
            v = int(f[v])
            l += 1
        P[v], L[v] = i, l
    return (f, P, L)


def _compute_suprema(state: GraphState, f, P, L):
    """Compute suprema

    status: 0 if none, 1 if pending, 2 if fixed.
    """
    (sup, status) = _init_status(state, P, L)
    for v in set(state.graph.nodes()) - set(state.output_nodes):
        if status[v] == 0:
            (sup, status) = _traverse_infl_walk(state, f, sup, status, v)

        if status[v] == 1:
            return None

    return sup


def _traverse_infl_walk(state: GraphState, f, sup, status, v):
    """Compute the suprema by traversing influencing walks
    
    status: 0 if none, 1 if pending, 2 if fixed.
    """
    status[v] = 1
    vertex2index = {v: index for index, v in enumerate(state.input_nodes)}

    for w in list(state.graph.neighbors(f[v])) + [f[v]]:
        if w != v:
            if status[w] == 0:
                (sup, status) = _traverse_infl_walk(state, f, sup, status, w)
            if status[w] == 1:
                return (sup, status)
            else:
                for i in state.input_nodes:
                    if sup[vertex2index[i], v] > sup[vertex2index[i], w]:
                        sup[vertex2index[i], v] = sup[vertex2index[i], w]
    status[v] = 2
    return sup, status


def _init_status(state: GraphState, P, L):
    """Initialize the supremum function

    status: 0 if none, 1 if pending, 2 if fixed.
    """
    sup = np.zeros((len(state.input_nodes), len(state.graph.nodes())))
    vertex2index = {v: index for index, v in enumerate(state.input_nodes)}
    status = np.zeros(len(state.graph.nodes()))
    for v in state.graph.nodes():
        for i in state.input_nodes:
            if i == P[v]:
                sup[vertex2index[i], v] = L[v]
            else:
                sup[vertex2index[i], v] = len(state.graph.nodes())

        status[v] = 2 if v in state.output_nodes else 0

    return sup, status


def _build_path_cover(state: GraphState):
    """Builds a path cover

    status: 0 if 'fail', 1 if 'success'
    """
    fam = nx.DiGraph()
    visited = np.zeros(state.graph.number_of_nodes())
    iter = 0
    for i in state.input_nodes:
        iter += 1
        (fam, visited, status) = _augmented_search(state, fam, iter, visited, i)
        if not status:
            return status

    if not len(set(state.graph.nodes) - set(fam.nodes())):
        return fam

    return 0


def _augmented_search(state: GraphState, fam: nx.DiGraph, iter: int, visited, v):
    """Does an augmented search

    status: 0 if 'fail', 1 if 'success'
    """
    visited[v] = iter
    if v in state.output_nodes:
        return (fam, visited, 1)
    if (
        (v in fam.nodes())
        and (v not in state.input_nodes)
        and (visited[next(fam.predecessors(v))] < iter)
    ):
        (fam, visited, status) = _augmented_search(
            state, fam, iter, visited, next(fam.predecessors(v))
        )
        if status:
            fam = fam.remove_edge(next(fam.predecessors(v)), v)
            return (fam, visited, 1)

    for w in state.graph.neighbors(v):
        if (
            (visited[w] < iter)
            and (w not in state.input_nodes)
            and (not fam.has_edge(v, w))
        ):
            if w not in fam.nodes():
                (fam, visited, status) = _augmented_search(state, fam, iter, visited, w)
                if status:
                    fam.add_edge(v, w)
                    return (fam, visited, 1)
            elif visited[next(fam.predecessors(w))] < iter:
                (fam, visited, status) = _augmented_search(
                    state, fam, iter, visited, next(fam.predecessors(w))
                )
                if status:
                    fam.remove_edge(next(fam.predecessors(w)), w)
                    fam.add_edge(v, w)
                    return (fam, visited, 1)

    return (fam, visited, 0)


def _check_if_flow(state: GraphState, flow, partial_order) -> bool:
    """Checks if flow satisfies conditions on state."""
    conds = True
    for i in state.outputc:
        nfi = list(state.graph.neighbors(flow(i)))
        c1 = i in nfi
        c2 = partial_order(i, flow(i))
        c3 = math.prod(
            [partial_order(i, k) for k in set(nfi) - {i}]
        )
        conds = conds * c1 * c2 * c3
        if not c1:
            print(f"Condition 1 failed for node {i}. {i} not in {nfi}")
        if not c2:
            print(f"Condition 2 failed for node {i}. {i} ≮ {flow(i)}")
        if not c3:
            print(f"Condition 3 failed for node {i}.")
            for k in set(nfi) - {i}:
                if not partial_order(i, k):
                    print(f"{i} ≮ {k}")
    return conds

### This section implements causal flow

def causal_flow(state: GraphState) -> object:
    """Finds the causal flow of a ``MBQCGraph`` if it exists.
    Retrieved from https://arxiv.org/pdf/0709.2670v1.pdf.
    """
    l = {}

    for v in state.output_nodes:
        l[v] = 0
    
    result, flow = causal_flow_aux(state, set(state.input_nodes), set(state.output_nodes), set(state.output_nodes)-set(state.input_nodes), 1, l)
    
    if not result:
        raise UserWarning("No causal flow exists for this graph.")

    # return flow and partial order
    return lambda x: flow[x] , lambda u, v: l[u] > l[v]

def causal_flow_aux(state: GraphState, inputs, outputs, C, k, l) -> object:
    """Aux function for causal_flow"""
    V = set(state.graph.nodes())
    out_prime = set()
    C_prime = set()
    g = {}

    for _, v in enumerate(C):
        # get intersection of neighbors of v and (V \ output nodes
        intersection = set(state.graph.neighbors(v)) & (V - outputs)
        if len(intersection)==1:
            u = intersection.pop()
            g[u] = v
            l[v] = k
            out_prime.add(u)
            C_prime.add(v)
    
    if len(out_prime) == 0:
        if set(outputs) == V:
            return True, l
        else:
            return False, l
    else:
        return causal_flow_aux(state, inputs, outputs | out_prime, (C - C_prime) | ((out_prime & V) - inputs), k+1, l)

### This section is for GFlow ###

# def gflow(state: MBQCGraph) -> object:
#     """Finds the generalized flow of a ``MBQCGraph`` if it exists.
#    Retrieved from https://arxiv.org/pdf/0709.2670v1.pdf.
#    """
#     gamma = nx.adjacency_matrix(state.graph).toarray()
#     l = {}
#     g = {}

#     for v in state.output_nodes:
#         l[v] = 0
    
    
#     result, flow, g = gflowaux(set(state.graph.nodes()), gamma, set(state.input_nodes), set(state.output_nodes), 1, l, g)

#     return lambda x: flow[x], lambda u, v: l[u] > l[v], lambda u: g[u]

# def gflowaux(V, gamma, inputs, outputs, k, l, g) -> object:
#     """Aux function for gflow"""
#     GF = galois.GF(2)
#     node_mapping = {i: node for i, node in enumerate(V)}

#     C = set()
#     for u in V-outputs:
#         # submatrix = gamma[np.ix_(list(V-outputs), list(outputs - inputs))]
#         # get submatrix with rows corresponding to V-outputs and columns corresponding to outputs - inputs 
#         # but use the node_mapping to get the correct indices
#         submatrix = np.zeros((len(V-outputs), len(outputs - inputs)), dtype=int)
#         for i, v in enumerate(V-outputs):
#             for j, w in enumerate(outputs - inputs):
#                 submatrix[i, j] = gamma[node_mapping[v], node_mapping[w]]

#         iu = np.zeros(len(V-outputs), dtype=int)
#         iu[list(V-outputs).index(u)] = 1
#         # use node_mapping to get the correct index
        
#         submatrix, iu = GF(submatrix), GF(np.array([iu]).T)
#         #solve submatrix @ X = iu using galois in mod2
#         sys = np.hstack((submatrix, iu))
#         row_reduced = sys.row_reduce()
        
#         # FILL HERE TO GET SOLUTION

        

#         # check if solution is a valid solution
#         if np.linalg.norm(submatrix @ solution - iu) <= 1e-5:
#             l[u] = k
#             C.add(u)
#             g[u] = solution
    
#     if len(C) == 0:
#         return True, l, g
#     else:
#         return gflowaux(V, gamma, inputs, outputs | C, k+1, l, g)


# def _check_if_gflow(state: MBQCGraph, gflow, partial_order) -> bool:
#     """Checks if gflow satisfies conditions on state."""
    
def gflow(state: GraphState) -> object:
    """Finds the generalized flow of a ``MBQCGraph`` if it exists.
   Retrieved from https://arxiv.org/pdf/0709.2670v1.pdf.
   """
    gamma = nx.adjacency_matrix(state.graph).toarray()
    l = {}
    g = {}

    for v in state.output_nodes:
        l[v] = 0
    
    
    result, flow, g = gflowaux(state, gamma, set(state.input_nodes), set(state.output_nodes)-set(state.input_nodes), 1, l, g)

    return lambda x: flow[x], lambda u, v: l[u] > l[v], lambda u: g[u]

def gflowaux(state: GraphState, gamma, inputs, outputs, k, l, g) -> object:
    """Aux function for gflow"""
    out_prime = set()
    GF = galois.GF(2)
    V = set(state.graph.nodes())
    C = set()
    for u in V - outputs:
        submatrix = np.zeros((len(V-outputs), len(outputs)), dtype=int)
        b = np.zeros((len(V-outputs), 1), dtype=int)
        b[list(V-outputs).index(u)] = 1
        syst = np.hstack((submatrix, b))
        row_reduced = syst.row_reduce()
        solution = np.zeros((len(outputs), 1), dtype=int)
        for i, v in enumerate(outputs):
            if row_reduced[i, i] == 1:
                solution[i] = row_reduced[i, -1]
        
        solution, b = GF(solution), GF(b)
        if np.linalg.norm(submatrix @ solution - b) <= 1e-5:
            l[u] = k
            C.add(u)
            g[u] = solution
    
    if len(C) == 0:
        if set(outputs) == V:
            return True, l, g
        else:
            return False, l, g
    
    else:
        return gflowaux(state, gamma, inputs, outputs | C, k+1, l, g)

