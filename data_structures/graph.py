from typing import List, Tuple

import collections
from data_structures import nodes


class Vertex(nodes.Node):

    def __init__(self, value):
        super().__init__(value)
        self.visited = False
        self.num_in_degrees = 0

    def __eq__(self, other):
        return self.value == other.value

    def mark_visited(self):
        self.visited = True

    def in_degrees(self):
        return self.num_in_degrees

    def add_in_degree(self):
        self.num_in_degrees += 1


class Edge:

    def __init__(self, v: Vertex, u: Vertex, weight: float):
        self.v = v
        self.u = u
        self.weight = weight


class Graph:

    def __init__(self):
        self.graph = collections.defaultdict(list)

    def add_edges(self, edges: List[Tuple[Vertex, Vertex]]):
        """."""
        for v, u in edges:
            self.add_edge(v, u)

    def add_edge(self, v: Vertex, u: Vertex):
        """."""
        u.add_in_degree()
        self.graph[v].append(u)

    def bfs(self, start: Vertex):
        """Return a list of nodes in BFS order."""
        result = []
        queue = list()

        queue.append(start)
        while queue:
            curr = queue.pop(0)
            curr.mark_visited()
            for u in self.graph[curr]:
                if not u.visited:
                    queue.append(u)
            result.append(curr)
        return result

    def dfs(self, start: Vertex):
        """Return a list of nodes in DFS order."""
        result = []
        stack = list()

        stack.append(start)
        while stack:
            curr = stack.pop()
            curr.mark_visited()
            for u in self.graph[curr]:
                if not u.visited:
                    stack.append(u)
            result.append(curr)
        return result
