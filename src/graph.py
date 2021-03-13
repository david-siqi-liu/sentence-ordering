from itertools import permutations


class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.graph = [[None for _ in range(num_vertices)]
                      for _ in range(num_vertices)]

    def addEdge(self, u, v, w):
        self.graph[u][v] = w

    def max_flow(self, start):
        if start:
            candidate_paths = [[start] + list(p) for p in list(
                permutations(list(range(0, start)) + list(range(start + 1, self.num_vertices)), self.num_vertices - 1))]
        else:
            candidate_paths = [list(p) for p in list(
                permutations(list(range(0, self.num_vertices)), self.num_vertices))]
        best_path, max_weight = None, float('-inf')
        for path in candidate_paths:
            weight = self.get_path_weight(path)
            if weight > max_weight:
                best_path = path
                max_weight = weight
        return best_path, max_weight

    def get_path_weight(self, path):
        weight = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            assert self.graph[u][v] != None
            weight += self.graph[u][v]
        return weight
