from collections import deque


def estimate_water_flow(heights):
    num_rows = len(heights)
    num_cols = len(heights[0])
    flow_to_pacific = set()
    flow_to_atlantic = set()

    for i in range(num_rows):
        flow_to_pacific = dfs(heights, i, 0, flow_to_pacific)
        flow_to_atlantic = dfs(heights, i, num_cols - 1, flow_to_atlantic)

    for i in range(1, num_cols):
        flow_to_pacific = dfs(heights, 0, i, flow_to_pacific)
        flow_to_atlantic = dfs(heights, num_rows - 1, i - 1, flow_to_atlantic)

    return list(flow_to_pacific.intersection(flow_to_atlantic))
    # DFS


def dfs(heights, row, col, ocean_flow):
    ocean_flow.add((row, col))
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions:
        new_row = row + dr
        new_col = col + dc

        if (
            0 > new_row
            or new_row > len(heights) - 1
            or 0 > new_col
            or new_col > len(heights[0]) - 1
        ):
            continue

        if (new_row, new_col) in ocean_flow:
            continue

        if heights[new_row][new_col] < heights[row][col]:
            continue

        dfs(heights, new_row, new_col, ocean_flow)


if __name__ == "__main__":
    print(
        estimate_water_flow(
            [
                [1, 2, 2, 3, 5],
                [3, 2, 3, 4, 4],
                [2, 4, 5, 3, 1],
                [6, 7, 1, 4, 5],
                [5, 1, 1, 2, 4],
            ]
        )
    )
