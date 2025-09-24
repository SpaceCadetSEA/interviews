"""
⬅️ We have provided a union_find.py file under the "Files" tab
of this widget. You can use this file to build your solution.
"""

from data_structures.union_find import UnionFind


def num_islands(grid):
    rows = len(grid)
    cols = len(grid[0])
    union = UnionFind(grid)

    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] == "1":
                grid[row][col] = "0"

                if row + 1 < rows and grid[row + 1][col] == "1":
                    union.union(row * cols + col, (row + 1) * cols + col)
                if col + 1 < cols and grid[row][col + 1] == "1":
                    union.union(row * cols + col, row * cols + (col + 1))

    return union.get_count()


def longest_consecutive_sequence(nums):
    # the preprocessing allows us to find num + 1 when we iterate over the list
    # of nums to perform the union-find algorithm.
    parents = {num: num for num in nums}
    size = {num: 1 for num in nums}
    max_len = 1

    for num in nums:
        if num + 1 in parents:
            # find parents of two and union
            root_x = find(num, parents)
            root_y = find(num + 1, parents)
            if root_x != root_y:
                if size[root_x] < size[root_y]:
                    root_x, root_y = root_y, root_x
                parents[root_y] = root_x
                size[root_x] += size[root_y]
                max_len = max(max_len, size[root_x])
    return max_len


def find(num, parent):
    if num != parent[num]:
        parent[num] = find(parent[num], parent)
    return parent[num]


class UnionFind:

    # Constructor
    def __init__(self, n):
        self.parent = []
        for i in range(n + 1):
            self.parent.append(i)

    # Function to find which subset a particular element belongs to
    def find(self, v):
        if self.parent[v] != v:
            self.parent[v] = self.find(self.parent[v])
        return self.parent[v]

    # Function to join two subsets into a single subset
    def union(self, x, y):
        p1, p2 = self.find(x), self.find(y)
        self.parent[p1] = p2


def count_components(n, edges):
    union_find = UnionFind(n)

    for x, y in edges:
        union_find.union(x, y)

    for i in range(n):
        union_find.find(i)

    return len(set(union_find.parent))


if __name__ == "__main__":
    count_components(0, [])
