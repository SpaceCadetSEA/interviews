from typing import List
from collections import deque
from math import inf
from data_structures.trees import BinaryTree, TreeNode


"""
FUNCTION backtrack(solution, options):
  IF solution is complete:
    processSolution(solution)
    RETURN

  FOR each option IN options:
    IF option is valid:
      makeChoice(option)  # Choose the option
      backtrack(solution + [option], remaining options)  # Recur with new choice
      undoChoice(option)  # Backtrack
"""


def word_search(grid: List[List[str]], word: str) -> bool:
    m = len(grid)
    n = len(grid[0])

    word_found = False
    visited_nodes = [[0 for col in range(n)] for row in range(m)]

    for row in range(m):
        for col in range(n):
            if grid[row][col] == word[0] and not word_found:
                visited_nodes[row][col] = 1
                word_found = _word_search_rec(grid, word[1:], (row, col), visited_nodes)
                visited_nodes[row][col] = 0
    return word_found


def _word_search_rec(
    grid: List[List[str]], curr_word: str, curr_pos: List[int], visited: List[List[int]]
) -> bool:
    # base case
    if len(curr_word) == 0:
        return True

    # use curr_pos to find all neighboring nodes that we haven't visited
    visited[curr_pos[0]][curr_pos[1]] = 1
    options = find_valid_options(len(grid), len(grid[0]), curr_pos, visited)
    word_found = False

    for option in options:
        if grid[option[0]][option[1]] == curr_word[0] and not word_found:
            word_found = _word_search_rec(grid, curr_word[1:], option, visited)
            visited[curr_pos[0]][curr_pos[1]] = 0
    return word_found


def find_valid_options(m, n, curr_pos, visited):
    # up down left right
    options = []
    row = curr_pos[0]
    col = curr_pos[1]
    if row > 0:
        if visited[row - 1][col] != 1:
            options.append([row - 1, col])
    if row < m - 1:
        if visited[row + 1][col] != 1:
            options.append([row + 1, col])
    if col > 0:
        if visited[row][col - 1] != 1:
            options.append([row, col - 1])
    if col < n - 1:
        if visited[row][col + 1] != 1:
            options.append([row, col + 1])
    return options


def total_n_queens(n):
    cols = set()
    diagonals = set()
    anti_diagonals = set()
    return _solve_rec(n, 0, diagonals, anti_diagonals, cols)


def _solve_rec(n, row, diagonals, anti_diagonals, cols):
    if row == n:
        return 1

    solutions = 0

    for col in range(n):
        curr_diagonal = row - col
        curr_anti_diagonal = row + col

        if (
            col in cols
            or curr_diagonal in diagonals
            or curr_anti_diagonal in anti_diagonals
        ):
            continue

        cols.add(col)
        diagonals.add(curr_diagonal)
        anti_diagonals.add(curr_anti_diagonal)

        solutions += _solve_rec(n, row + 1, diagonals, anti_diagonals, cols)

        cols.remove(col)
        diagonals.remove(curr_diagonal)
        anti_diagonals.remove(curr_anti_diagonal)

    return solutions


def flood_fill(grid, sr, sc, target):
    if grid[sr][sc] == target:
        return grid
    return _flood_fill_rec(grid, grid[sr][sc], sr, sc, target)


def _flood_fill_rec(grid, curr_color, sr, sc, target):
    if grid[sr][sc] != curr_color:
        return grid

    grid[sr][sc] = target
    coords = [[sr, sc - 1], [sr, sc + 1], [sr - 1, sc], [sr + 1, sc]]
    for row, col in coords:
        if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
            grid = _flood_fill_rec(grid, curr_color, row, col, target)

    return grid


def rob(root):
    return max(_rob_rec(root))


def _rob_rec(root):
    if root == None or root.data == None:
        return [0, 0]
    
    left_values = _rob_rec(root.left)
    right_values = _rob_rec(root.right)
    include = root.data + left_values[1] + right_values[1]
    exclude = max(left_values) + max(right_values)
    
    return [include, exclude]


if __name__ == "__main__":
    # do stuff
    values = [5, 8, 17, 12, 11, 2, 4]
    tree_nodes = [TreeNode(val) for val in values]
    tree = BinaryTree(tree_nodes)
    print(rob(tree.root))
