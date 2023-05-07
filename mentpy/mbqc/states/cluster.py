# TODO: Add AKLT state class inheritance from graph state
import networkx as nx


class ClusterState(nx.Graph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError
