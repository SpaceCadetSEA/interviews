from collections import defaultdict, Counter, deque

from data_structures.nodes import Node


def estimate_water_flow(heights):
    num_rows = len(heights)
    num_cols = len(heights[0])

    pacific_ocean = set()
    atlantic_ocean = set()

    # row == 0 or col == 0 -> pacific
    # row == num_rows - 1 or col == num_cols - 1 -> atlantic
    for row in range(num_rows):
        dfs(heights, row, 0, pacific_ocean)
        dfs(heights, row, num_cols - 1, atlantic_ocean)

    for col in range(num_cols):
        dfs(heights, 0, col, pacific_ocean)
        dfs(heights, num_rows - 1, col, atlantic_ocean)

    return list(pacific_ocean.intersection(atlantic_ocean))


def dfs(heights, row, col, ocean_set):
    ocean_set.add((row, col))

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions:
        # find valid nodes to add to set
        new_row, new_col = row + dr, col + dc

        # we could have already visited node...
        if (new_row, new_col) in ocean_set:
            continue
        # can we visit it, is it out of bounds?
        if (
            new_row > len(heights) - 1
            or new_row < 0
            or new_col > len(heights[0]) - 1
            or new_col < 0
        ):
            continue
        # can we visit it, is it taller than its neighbor?
        if heights[new_row][new_col] < heights[row][col]:
            continue

        dfs(heights, new_row, new_col, ocean_set)

    return ocean_set


def clone(root):
    already_created = {}
    return _clone_rec(root, already_created)


def _clone_rec(curr_node, already_created):
    if curr_node is None:
        return None

    new_node = Node(curr_node.data)
    already_created[curr_node] = new_node

    for neighbor in curr_node.neighbors:
        completed = already_created.get(neighbor)
        if not completed:
            new_node.neighbors.append(_clone_rec(neighbor, already_created))
        else:
            new_node.neighbors.append(completed)
    return new_node


def valid_tree(n, edges):
    if len(edges) != n - 1:
        return False

    adjacency_list = [[] for _ in range(n)]

    for edge in edges:
        adjacency_list[edge[0]].append(edge[1])
        adjacency_list[edge[1]].append(edge[0])

    visited_nodes = set([0])
    stack = [0]
    while stack:
        curr_node = stack.pop()
        for neighbor in adjacency_list[curr_node]:
            if neighbor not in visited_nodes:
                visited_nodes.add(neighbor)
                stack.append(neighbor)

    return len(visited_nodes) == n


if __name__ == "__main__":
    print(valid_tree(5, [[1, 0], [2, 1], [3, 2], [4, 3]]))
