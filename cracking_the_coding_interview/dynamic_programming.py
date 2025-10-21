from typing import List

"""
Triple Step:

A child is running up a staircase with n steps and can hop either 1 step, 
2 steps, or 3 steps at a time. Implement a method to count how many possible 
ways the child can run up the stairs.
"""


def triple_steps(n: int):
    if n < 0:
        return 0
    if n == 0:
        return 1
    if n == 1:
        return 1
    if n == 2:
        return 2

    res = [0] * (n + 1)

    res[0] = 1
    res[1] = 1
    res[2] = 2

    for i in range(3, n + 1):
        res[i] = res[i - 1] + res[i - 2] + res[i - 3]
    return res[n]


"""
Robot in a Grid

Imagine a robot sitting on the upper left corner of grid with r rows and 
c columns. The robot can only move in two directions, right and down, but 
certain cells are "off limits" such that the robot cannot step on them. 

Design an algorithm to find a path for the robot from the top left to the 
bottom right.
"""


def robot_rock(grid: List[List[int]]):
    # Presume a 0 means we can traverse, and a 1 means we cannot.
    # Looks like a recursive backtracking problem.

    # We'll start at the top cell and identify the valid cells we can traverse
    # into.

    # Then we'll try a DFS traversal of these cells. If we encounter a 1 in the
    # cell, then we cannot enter it and have to back up and try a different
    # path.

    # 1. identify valid cells
    # 2. navigate to cell & continue
    # 3. backtrack and try different path
    if not grid or not grid[0] or grid[0][0] == 1:
        return False

    def robot_rock_rec(row, col, seen, path):
        if row == len(grid) - 1 and col == len(grid[0]) - 1:
            path.append((row, col))
            return True

        if (row, col) in seen:
            return False

        seen.add((row, col))
        path.append((row, col))

        if (
            row + 1 < len(grid) and 
            grid[row + 1][col] != 1 and 
            robot_rock_rec(row + 1, col, seen, path)
        ):
            return True
        
        if (
            col + 1 < len(grid[0]) and
            grid[row][col + 1] != 1 and
            robot_rock_rec(row, col + 1, seen, path)
        ):
            return True
        
        path.pop()
        return False

    seen = set()
    path = []
    robot_rock_rec(0, 0, seen, path)
    return path


def magic_index(arr: List[int]):
    n = len(arr)
    start, end = 0, n - 1

    def bisearch(start, end):
        if start >= end:
            return -1
        mid = end - start // 2
        if arr[mid] == mid:
            return mid
        else:
            if arr[mid] < mid:
                return bisearch(mid + 1, end)
            else:
                return bisearch(start, mid - 1)
            
    return bisearch(start, end)


if __name__ == "__main__":
    grid = [
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0],
    ]

    print(robot_rock(grid))
