from typing import Set


class Edge:
    def __init__(self,
                 from_state: State,
                 to_state: State,
                 rate: float):
        self._from_state = from_state
        self._to_state = to_state
        self._rate = rate

class State:

    def __init__(self,
                 name: str,
                 in_edges: Set[Edge] = set(),
                 out_edges: Set[Edge] = set()):
        self._name = name
        self._in_edges = in_edges
        self._out_edges = out_edges

    def add_incoming(self, edge: Edge):
        self._in_edges.add(edge)

    def add_outgoing(self, edge: Edge):
        self._out_edges.add(edge)


def compile_matrix
