"""
problem:

board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

Given word = "ABCCED", return true.
Given word = "SEE", return true.
Given word = "ABCB", return false.
"""


def contains_word(board, word):
    num_rows = len(board)
    num_cols = len(board[0])

    if not word and board:
        return True
    if num_rows == 0 and word:
        return False

    visited = [[False for _ in range(num_cols)] for _ in range(num_rows)]
    curr_i, curr_j, found = 0, 0, False

    while (curr_i < num_rows and curr_j < num_cols) and not found:
        if board[curr_i][curr_j] == word[0]:
            found = dfs_finder(
                board, word[1:], num_rows, num_cols, curr_i, curr_j, visited
            )
        curr_j += 1
        if curr_j == num_cols:
            curr_i += 1
            curr_j = 0
    return found


def dfs_finder(board, word, n_rows, n_cols, curr_i, curr_j, visited):
    visited[curr_i][curr_j] = True

    if not word:
        return True

    found = False
    neighbors = get_neighbors(curr_i, curr_j, n_rows, n_cols)
    while neighbors and not found:
        i, j = neighbors.pop(0)
        if not visited[i][j] and board[i][j] == word[0]:
            found = dfs_finder(
                board, word[1:], n_rows, n_cols, i, j, visited
            )
    return found


def get_neighbors(curr_i, curr_j, n_rows, n_cols):
    results = []
    for i in range(curr_i - 1, curr_i + 2):
        if 0 <= i < n_rows and i != curr_i:
            results.append((i, curr_j))
    for j in range(curr_j - 1, curr_j + 2):
        if 0 <= j < n_cols and j != curr_j:
            results.append((curr_i, j))
    return results


if __name__ == '__main__':
    board = [
        ['A', 'B', 'C', 'E'],
        ['S', 'F', 'C', 'S'],
        ['A', 'D', 'E', 'E']
    ]

    word = 'ABCCED'
    assert contains_word(board, word)

    word = 'SEE'
    assert contains_word(board, word)

    word = 'ABCB'
    assert not contains_word(board, word)
