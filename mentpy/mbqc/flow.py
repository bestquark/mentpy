# Author: Luis Mantilla
# Github: BestQuark
"""This is the Flow module. It deals with the flow of a given graph state"""
import math
import numpy as np
import networkx as nx

from mentpy.mbqc import GraphState
from typing import Any, List
import warnings

import galois


GF = galois.GF(2)


def binary_gaussian_elimination(A, b):
    rows, cols = A.shape
    augmented_matrix = np.hstack((A, b.reshape(-1, 1)))

    row = 0
    for col in range(cols):
        # Find pivot in current column
        for pivot_row in range(row, rows):
            if augmented_matrix[pivot_row, col]:
                break
        else:
            continue

        # Swap rows if needed
        if pivot_row != row:
            augmented_matrix[[row, pivot_row]] = augmented_matrix[[pivot_row, row]]

        # Zero out below pivot
        for i in range(row + 1, rows):
            if augmented_matrix[i, col]:
                augmented_matrix[i] ^= augmented_matrix[row]

        row += 1

        if row >= rows:
            break

    # Back-substitution
    x = np.zeros(cols, dtype=int)
    for row in range(min(rows, cols) - 1, -1, -1):
        col = np.argmax(augmented_matrix[row, :cols])
        x[col] = augmented_matrix[row, -1] ^ np.sum(
            augmented_matrix[row, col + 1 : cols] * x[col + 1 : cols]
        )

    return x


class Flow:
    """This class deals with the flow of a given graph state"""

    def __init__(self, graph: GraphState, input_nodes, output_nodes):
        self.graph = graph
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        flow_function, partial_order, depth, layers = self.find_flow()
        self.func = flow_function
        self.partial_order = partial_order
        self.depth = depth
        self.layers_dict = layers
        if layers is not None:
            ord_layers = [
                [n for n, l in layers.items() if l == j]
                for j in range(max(layers.values()) + 1)
            ]
            self.layers = ord_layers[::-1]

            # This can be optimized!!
            order = [item for sublist in self.layers for item in sublist]
            for i in self.input_nodes[::-1]:
                order.remove(i)
                order.insert(0, i)
            self.measurement_order = order
        else:
            self.layers = None
            self.measurement_order = None

    def __repr__(self):
        return f"Flow(n={self.graph.number_of_nodes()})"

    def __call__(self, node):
        return self.func(node)

    def find_flow(self):
        # include gflow and pflow
        flow_stuff = find_cflow(self.graph, self.input_nodes, self.output_nodes)
        if flow_stuff[0] is None:
            flow_stuff = find_gflow(self.graph, self.input_nodes, self.output_nodes)
            if flow_stuff[0] is None:
                # TODO: implement pflow
                # flow_stuff = find_pflow(self.graph, self.input_nodes, self.output_nodes)
                pass

        return flow_stuff

    def adapt_angles(self, angles, outcomes):
        raise NotImplementedError

    def adapt_angle(self, angle, node, previous_outcomes):
        raise NotImplementedError


### This section implements causal flow -- Runs in time O(min(m, kn))


def find_cflow(graph: GraphState, input_nodes, output_nodes) -> object:
    """Finds the causal flow of a ``MBQCGraph`` if it exists.
    Retrieved from https://arxiv.org/pdf/0709.2670v1.pdf.
    """
    if len(input_nodes) != len(output_nodes):
        raise ValueError(
            f"Cannot find flow or gflow. Input ({len(input_nodes)}) and output ({len(output_nodes)}) nodes have different size."
        )

    l = {}
    g = {}
    past = {}
    C_set = set()

    for v in graph.nodes():
        l[v] = 0
        past[v] = 0

    for v in set(output_nodes) - set(input_nodes):
        past[v] = len(
            set(graph.neighbors(v)) & (set(graph.nodes() - set(output_nodes)))
        )
        if past[v] == 1:
            C_set = C_set.union({v})

    flow, ln = causal_flow_aux(
        graph, set(input_nodes), set(output_nodes), C_set, past, 1, g, l
    )

    if len(flow) != len(graph.nodes()) - len(output_nodes):
        return None, None, None, None

    return lambda x: flow[x], lambda u, v: ln[u] > ln[v], max(flow.values()), ln


def causal_flow_aux(graph: GraphState, inputs, outputs, C, past, k, g, l) -> object:
    """Aux function for causal_flow"""
    V = set(graph.nodes())
    C_prime = set()

    for _, v in enumerate(C):
        intersection = set(graph.neighbors(v)) & (V - outputs)
        if len(intersection) == 1:
            u = intersection.pop()
            g[u] = v
            l[u] = k
            outputs.add(u)
            if u not in inputs:
                past[u] = len(set(graph.neighbors(u)) & (V - outputs))
                if past[u] == 1:
                    C_prime.add(u)
            for w in set(graph.neighbors(u)):
                if past[w] > 0:
                    past[w] -= 1
                    if past[w] == 1:
                        C_prime.add(w)

    if len(C_prime) == 0:
        return g, l

    else:
        return causal_flow_aux(
            graph,
            inputs,
            outputs,
            C_prime,
            past,
            k + 1,
            g,
            l,
        )


### This section implements generalized flow -- Runs in time O(n^4)


def find_gflow(graph: GraphState, input_nodes, output_nodes) -> object:
    """Finds the generalized flow of a ``MBQCGraph`` if it exists.
    Retrieved from https://arxiv.org/pdf/0709.2670v1.pdf.
    """
    gamma = nx.adjacency_matrix(graph).toarray()

    l = {}
    g = {}

    for v in output_nodes:
        l[v] = 0

    result, gn, ln = gflowaux(
        graph,
        gamma,
        set(input_nodes),
        set(output_nodes),
        1,
        g,
        l,
    )

    if result == False:
        warnings.warn("No gflow exists for this graph.", UserWarning, stacklevel=2)
        return None, None, None, None

    return lambda x: gn[x], lambda u, v: ln[u] > ln[v], max(ln.values()), ln


def gflowaux(graph: GraphState, gamma, inputs, outputs, k, g, l) -> object:
    """Aux function for gflow"""

    mapping = graph.index_mapping()
    V = set(graph.nodes())
    C = set()
    vmol = list(V - outputs)
    for u in vmol:
        submatrix = np.zeros((len(vmol), len(outputs - inputs)), dtype=int)
        for i, v in enumerate(vmol):
            for j, w in enumerate(outputs - inputs):
                submatrix[i, j] = gamma[mapping[v], mapping[w]]

        b = np.zeros((len(vmol), 1), dtype=int)
        b[vmol.index(u)] = 1
        submatrix = GF(submatrix)
        b = GF(b)
        solution = GF(binary_gaussian_elimination(submatrix, b)).reshape(-1, 1)

        if np.linalg.norm(submatrix @ solution - b) <= 1e-5:
            l[u] = k
            C.add(u)
            g[u] = solution

    if len(C) == 0:
        if set(outputs) == V:
            return True, g, l
        else:
            return False, g, l

    else:
        return gflowaux(graph, gamma, inputs, outputs | C, k + 1, g, l)


## This section implements PauliFlow. Currently not working.


def find_pflow(
    graph: GraphState, input_nodes, output_nodes, basis="XY", testing=False
) -> object:
    """Implementation of pauli flow algorithm in https://arxiv.org/pdf/2109.05654v1.pdf"""

    if not testing:
        raise NotImplementedError("This algorithm is not yet implemented.")

    if type(basis) == str:
        basis = {v: basis for v in graph.nodes()}
    elif type(basis) != dict:
        raise TypeError("Basis must be a string or a dictionary.")

    lx = set()
    ly = set()
    lz = set()
    d = {}
    p = {}

    gamma = nx.adjacency_matrix(graph).toarray()

    for v in graph.nodes():
        if v in output_nodes:
            d[v] = 0
        if basis[v] == "X":
            lx = lx.add(v)
        elif basis[v] == "Y":
            ly = ly.add(v)
        elif basis[v] == "Z":
            lz = lz.add(v)

    return pflowaux(graph, gamma, input_nodes, basis, set(), output_nodes, 0, d, p)


def pflowaux(graph: GraphState, gamma, inputs, plane, A, B, k, d, p) -> object:
    """Aux function for pflow"""
    C = set()
    mapping = graph.index_mapping()
    for u in set(graph.nodes()) - set(B):
        submatrix1, submatrix2, submatrix3 = None, None, None
        solution1, solution2, solution3 = None, None, None
        if plane[u] in ["XY", "X", "Y"]:
            submatrix1 = 0  # TODO
            solution1 = 0  # TODO
        if plane[u] in ["XZ", "X", "Z"]:
            submatrix2 = 0  # TODO
            solution2 = 0  # TODO
        if plane[u] in ["YZ", "Y", "Z"]:
            submatrix3 = 0  # TODO
            solution3 = 0  # TODO

        if (
            (solution1 is not None)
            or (solution2 is not None)
            or (solution3 is not None)
        ):
            C.add(u)
            sol = solution1 or solution2 or solution3
            p[u] = sol
            d[u] = k

    if len(C) == 0 and k > 0:
        if set(B) == set(graph.nodes()):
            return True, p, d
        else:
            return False, set(), set()
    else:
        B = B.union(C)
        return pflowaux(graph, gamma, inputs, plane, B, B, k + 1, d, p)


## Finds flow of a graph state. This is deprecated and will be removed in the future.


def find_flow(graph: GraphState, input_nodes, output_nodes, sanity_check=True):
    r"""Finds the generalized flow of graph state if allowed.

    Implementation of https://arxiv.org/pdf/quant-ph/0603072.pdf.

    Returns
    -------
    The flow function ``flow`` and the partial order function.

    Group
    -----
    states
    """
    # raise deprecated warning
    warnings.warn(
        "The function find_flow is deprecated. Use find_cflow instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    n_input, n_output = len(input_nodes), len(output_nodes)
    inp = input_nodes
    outp = output_nodes
    if n_input != n_output:
        raise ValueError(
            f"Cannot find flow or gflow. Input ({n_input}) and output ({n_output}) nodes have different size."
        )

    update_labels = False
    # check if labels of graph are integers going from 0 to n-1 and if not, create a mapping
    if not all([i in graph.nodes for i in range(len(graph))]):
        mapping = {v: i for i, v in enumerate(graph.nodes)}
        inverse_mapping = {i: v for i, v in enumerate(graph.nodes)}
        # create a copy of the object state
        old_state = graph
        new_graph = nx.relabel_nodes(graph.copy(), mapping)
        inp, outp = [mapping[v] for v in input_nodes], [
            mapping[v] for v in output_nodes
        ]

        graph = GraphState(new_graph)
        update_labels = True

    tau = _build_path_cover(graph, inp, outp)
    if tau:
        f, P, L = _get_chain_decomposition(graph, inp, outp, tau)
        sigma = _compute_suprema(graph, inp, outp, f, P, L)

        if sigma is not None:
            int_flow = _flow_from_array(graph, inp, outp, f)
            vertex2index = {v: index for index, v in enumerate(inp)}

            def int_partial_order(x, y):
                return sigma[vertex2index[int(P[y])], int(x)] <= L[y]

            # if labels were updated, update them back
            if update_labels:
                graph = old_state
                flow = lambda v: inverse_mapping[int_flow(mapping[v])]
                partial_order = lambda x, y: int_partial_order(mapping[x], mapping[y])
            else:
                flow = int_flow
                partial_order = int_partial_order

            state_flow = (flow, partial_order)
            if sanity_check:
                if not check_if_flow(graph, inp, outp, flow, partial_order):
                    raise RuntimeError(
                        "Sanity check found that flow does not satisfy flow conditions."
                    )
            return state_flow

        else:
            warnings.warn(
                "The given state does not have a flow.", UserWarning, stacklevel=2
            )
            return None, None
    else:
        warnings.warn(
            "Could not find a flow for the given state.", UserWarning, stacklevel=2
        )
        return None, None


def _flow_from_array(graph: GraphState, input_nodes, output_nodes, f: List):
    """Create a flow function from a given array f"""

    def flow(v):
        if v in [v for v in graph.nodes() if v not in output_nodes]:
            return int(f[v])
        else:
            raise UserWarning(f"The node {v} is not in domain of the flow.")

    return flow


def _get_chain_decomposition(
    graph: GraphState, input_nodes, output_nodes, C: nx.DiGraph
):
    """Gets the chain decomposition"""
    P = np.zeros(len(graph))
    L = np.zeros(len(graph))
    f = {v: 0 for v in set(graph) - set(output_nodes)}
    for i in input_nodes:
        v, l = i, 0
        while v not in output_nodes:
            f[v] = int(next(C.successors(v)))
            P[v] = i
            L[v] = l
            v = int(f[v])
            l += 1
        P[v], L[v] = i, l
    return (f, P, L)


def _compute_suprema(graph: GraphState, input_nodes, output_nodes, f, P, L):
    """Compute suprema

    status: 0 if none, 1 if pending, 2 if fixed.
    """
    (sup, status) = _init_status(graph, input_nodes, output_nodes, P, L)
    for v in set(graph.nodes()) - set(output_nodes):
        if status[v] == 0:
            (sup, status) = _traverse_infl_walk(
                graph, input_nodes, output_nodes, f, sup, status, v
            )

        if status[v] == 1:
            return None

    return sup


def _traverse_infl_walk(
    graph: GraphState, input_nodes, output_nodes, f, sup, status, v
):
    """Compute the suprema by traversing influencing walks

    status: 0 if none, 1 if pending, 2 if fixed.
    """
    status[v] = 1
    vertex2index = {v: index for index, v in enumerate(input_nodes)}

    for w in list(graph.neighbors(f[v])) + [f[v]]:
        if w != v:
            if status[w] == 0:
                (sup, status) = _traverse_infl_walk(
                    graph, input_nodes, output_nodes, f, sup, status, w
                )
            if status[w] == 1:
                return (sup, status)
            else:
                for i in input_nodes:
                    if sup[vertex2index[i], v] > sup[vertex2index[i], w]:
                        sup[vertex2index[i], v] = sup[vertex2index[i], w]
    status[v] = 2
    return sup, status


def _init_status(graph: GraphState, input_nodes: List, output_nodes: List, P, L):
    """Initialize the supremum function

    status: 0 if none, 1 if pending, 2 if fixed.
    """
    sup = np.zeros((len(input_nodes), len(graph.nodes())))
    vertex2index = {v: index for index, v in enumerate(input_nodes)}
    status = np.zeros(len(graph.nodes()))
    for v in graph.nodes():
        for i in input_nodes:
            if i == P[v]:
                sup[vertex2index[i], v] = L[v]
            else:
                sup[vertex2index[i], v] = len(graph.nodes())

        status[v] = 2 if v in output_nodes else 0

    return sup, status


def _build_path_cover(graph: GraphState, input_nodes: List, output_nodes: List):
    """Builds a path cover

    status: 0 if 'fail', 1 if 'success'
    """
    fam = nx.DiGraph()
    visited = np.zeros(graph.number_of_nodes())
    iter = 0
    for i in input_nodes:
        iter += 1
        (fam, visited, status) = _augmented_search(
            graph, input_nodes, output_nodes, fam, iter, visited, i
        )
        if not status:
            return status

    if not len(set(graph.nodes) - set(fam.nodes())):
        return fam

    return 0


def _augmented_search(
    graph: GraphState,
    input_nodes: List,
    output_nodes: List,
    fam: nx.DiGraph,
    iter: int,
    visited,
    v,
):
    """Does an augmented search

    status: 0 if 'fail', 1 if 'success'
    """
    visited[v] = iter
    if v in output_nodes:
        return (fam, visited, 1)
    if (
        (v in fam.nodes())
        and (v not in input_nodes)
        and (visited[next(fam.predecessors(v))] < iter)
    ):
        (fam, visited, status) = _augmented_search(
            graph,
            input_nodes,
            output_nodes,
            fam,
            iter,
            visited,
            next(fam.predecessors(v)),
        )
        if status:
            try:
                fam = fam.remove_edge(next(fam.predecessors(v)), v)
                return (fam, visited, 1)
            except:
                return (fam, visited, 0)

    for w in graph.neighbors(v):
        try:
            if (
                (visited[w] < iter)
                and (w not in input_nodes)
                and (not fam.has_edge(v, w))
            ):
                if w not in fam.nodes():
                    (fam, visited, status) = _augmented_search(
                        graph, input_nodes, output_nodes, fam, iter, visited, w
                    )
                    if status:
                        fam.add_edge(v, w)
                        return (fam, visited, 1)
                elif visited[next(fam.predecessors(w))] < iter:
                    (fam, visited, status) = _augmented_search(
                        graph,
                        input_nodes,
                        output_nodes,
                        fam,
                        iter,
                        visited,
                        next(fam.predecessors(w)),
                    )
                    if status:
                        fam.remove_edge(next(fam.predecessors(w)), w)
                        fam.add_edge(v, w)
                        return (fam, visited, 1)
        except:
            return (fam, visited, 0)

    return (fam, visited, 0)


def check_if_flow(
    graph: GraphState, input_nodes: List, output_nodes: List, flow, partial_order
) -> bool:
    """Checks if flow satisfies conditions on state."""
    conds = True
    for i in [v for v in graph.nodes() if v not in output_nodes]:
        nfi = list(graph.neighbors(flow(i)))
        c1 = i in nfi
        c2 = partial_order(i, flow(i))
        c3 = math.prod([partial_order(i, k) for k in set(nfi) - {i}])
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
