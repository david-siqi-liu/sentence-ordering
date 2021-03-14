from itertools import permutations


class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.graph = [[None for _ in range(num_vertices)]
                      for _ in range(num_vertices)]

    def add_edge(self, u, v, w):
        self.graph[u][v] = w

    def max_flow(self, start):
        if start != None:
            candidate_paths = [[start] + list(p) for p in list(
                permutations(list(range(0, start)) + list(range(start + 1, self.num_vertices)), self.num_vertices - 1))]
        else:
            candidate_paths = [list(p) for p in list(
                permutations(list(range(0, self.num_vertices)), self.num_vertices))]
        best_path, max_pos_count, max_weight = None, \
            float('-inf'), float('-inf')
        for path in candidate_paths:
            weight, pos_count = self.get_path_weight_and_pos_count(path)
            if pos_count > max_pos_count:
                best_path, max_pos_count, max_weight = path, pos_count, weight
            elif (pos_count == max_pos_count) and (weight > max_weight):
                best_path, max_weight = path, weight
        return best_path, max_weight

    def get_path_weight_and_pos_count(self, path):
        weight, pos_count = 0, 0
        for i, u in enumerate(path[:-1]):
            for v in path[i + 1:]:
                w = self.graph[u][v]
                assert w != None
                weight += w
                if w > 0:
                    pos_count += 1
        return weight, pos_count

    def greedy(self, start):
        if start != None:
            return self.get_greedy_path_and_weight([start], 0)
        best_path, max_weight = None, float('-inf')
        for s in range(self.num_vertices):
            path, weight = self.get_greedy_path_and_weight([s], 0)
            if weight > max_weight:
                best_path, max_weight = path, weight
        return best_path, max_weight

    def get_greedy_path_and_weight(self, curr_path, curr_weight):
        if len(curr_path) == self.num_vertices:
            return curr_path, curr_weight
        u = curr_path[-1]
        best_v, best_w = None, float('-inf')
        for v in range(self.num_vertices):
            if v not in curr_path:
                w = self.graph[u][v]
                assert w != None
                if w > best_w:
                    best_w = w
                    best_v = v
        new_path = curr_path + [best_v]
        new_weight = curr_weight + best_w
        return self.get_greedy_path_and_weight(new_path, new_weight)
