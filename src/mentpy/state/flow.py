"""This is the Flow module. It deals with the flow of a given graph state"""

import numpy as np
import networkx as nx

from mentpy.state import GraphState


def find_flow(state: GraphState):
    r"""Finds the generalized flow of graph state if allowed. Otherwise returns None

    Implementation of https://arxiv.org/pdf/quant-ph/0603072.pdf.

    Examples
    --------
    Find the flow of a graph state :math:`|G\rangle`.

    .. ipython:: python

        g = nx.Graph()
        g.add_edges_from([(0,1), (1,2), (2,3), (3, 4)])
        state = mtp.GraphState(g, input_nodes = [0], output_nodes = [4])
        flow = mtp.find_flow(state)
        print(flow)

    :group: states
    """
    state_flow = None
    n_input, n_output = len(state.input_nodes), len(state.output_nodes)
    if n_input != n_output:
        raise ValueError(
            f"Cannot find flow. Input ({n_input}) and output ({n_output}) nodes have different size."
        )

    tau = _build_path_cover(state)
    if tau:
        f, P, L = _get_chain_decomposition(state, tau)
        sigma = _compute_suprema(state, f, P, L)

        if sigma is not None:
            state_flow = f

    return state_flow


def _get_chain_decomposition(state: GraphState, C: nx.DiGraph):
    """Gets the chain decomposition"""
    P = np.zeros(len(state.graph))
    L = np.zeros(len(state.graph))
    f = np.zeros(len(set(state.graph) - set(state.output_nodes)))
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
    """Compute the suprema by traversing influencing walks"""
    status[v] = 1
    for w in state.graph.neighbors(v):
        if w == f[v] and w != v:
            if status[w] == 0:
                (sup, status) = _traverse_infl_walk(state, f, sup, status, w)
            if status[w] == 1:
                return (sup, status)
            else:
                for i in state.input_nodes:
                    if sup[i, v] > sup[i, w]:
                        sup[i, v] = sup[i, w]
    status[v] = 2
    return sup, status


def _init_status(state: GraphState, P, L):
    """Initialize the supremum function

    status: 0 if none, 1 if pending, 2 if fixed.
    """
    sup = np.zeros((len(state.input_nodes), len(state.graph.nodes())))
    status = np.zeros(len(state.graph.nodes()))
    for v in state.graph.nodes():
        for i in state.input_nodes:
            if i == P[v]:
                sup[i, v] = L[v]
            else:
                sup[i, v] = len(state.graph.nodes())

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


def check_if_flow(state: GraphState, flow):
    """Checks if flow satisfies conditions on state."""
