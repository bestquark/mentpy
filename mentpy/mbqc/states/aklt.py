# TODO: Add AKLT state class
import networkx as nx


class AKLTState(nx.Graph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError
