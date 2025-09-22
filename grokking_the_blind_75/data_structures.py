from collections import deque
import itertools
from typing import List

from data_structures.linked_list import LinkedList, display, ListNode


def fold_linked_list(head: ListNode) -> ListNode:
    if not head:
        return head
    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    prev = None
    curr = slow
    while curr:  # while we have nodes to process, e.g. curr != None
        next = curr.next
        curr.next = prev
        prev = curr
        curr = next

    first = head
    second = prev

    while second.next:
        # save the next nodes
        first_next = first.next
        second_next = second.next

        # insert the current node from 2 after the current node from 1
        second.next = first.next
        first.next = second

        # update pointers for next iteration
        first = first_next
        second = second_next
    return head


def reverse_linked_list(head: ListNode) -> ListNode:
    if not head:
        return head

    prev = None
    curr = head

    while curr:
        next = curr.next
        curr.next = prev
        prev = curr
        curr = next

    return prev


def reverse_k_groups(head, k):
    dummy = ListNode(0)
    dummy.next = head
    ptr = dummy

    while ptr:
        # Keep track of the current position
        tracker = ptr
        # Traverse k nodes to check if there are enough nodes to reverse
        for _ in range(k):
            # If there are not enough nodes to reverse, break out of the loop
            if tracker == None:
                break
            tracker = tracker.next

        if tracker == None:
            break

        # Reverse the current group of k nodes
        previous = None
        current = ptr.next
        for _ in range(k):
            # temporarily store the next node
            next = current.next
            current.next = previous
            previous = current
            current = next

        # Connect the reversed group to the rest of the linked list
        last_node_of_reversed_group = ptr.next
        last_node_of_reversed_group.next = current
        ptr.next = previous
        ptr = last_node_of_reversed_group

    return dummy.next


def valid_parens(string: str) -> bool:
    stack = []
    parens = {"}": "{", ")": "(", "]": "["}

    for s in string:  # O(N)
        if s not in parens.keys():
            stack.append(s)
        else:
            if len(stack) == 0:
                return False
            curr_parens = stack.pop()
            if curr_parens != parens[s]:
                return False

    if len(stack) > 0:
        return False
    return True


def remove_duplicates(string: str) -> str:
    stack = []
    for s in string:
        if not stack:
            stack.append(s)
        else:
            last = stack[-1]
            if s == last:
                stack.pop()
            else:
                stack.append(s)
    return "".join(stack)


def calculator(expression: str) -> int:
    number = 0
    sign_value = 1
    result = 0
    operations_stack = []

    for c in expression:
        if c.isdigit():
            number = number * 10 + int(c)
        if c in "+-":
            result += number * sign_value
            sign_value = -1 if c == "-" else 1
            number = 0
        elif c == "(":
            operations_stack.append(result)
            operations_stack.append(sign_value)
            result = 0
            sign_value = 1

        elif c == ")":
            result += sign_value * number
            pop_sign_value = operations_stack.pop()
            result *= pop_sign_value

            second_value = operations_stack.pop()
            result += second_value
            number = 0

    return result + number * sign_value


def set_matrix_zeroes(mat: List[List[int]]) -> List[List[int]]:
    frow = fcol = False
    m = len(mat)
    n = len(mat[0])

    for i in range(m):
        for j in range(n):  # O(M x N)
            if mat[i][0] == 0:
                frow = True
            if mat[0][j] == 0:
                fcol = True

    for i in range(1, m):
        for j in range(1, n):
            if mat[i][j] == 0:
                mat[0][j] = 0
                mat[i][0] = 0

    for i in range(1, m):
        if mat[i][0] == 0:
            for j in range(1, n):
                mat[i][j] = 0

    for j in range(1, n):
        if mat[0][j] == 0:
            for i in range(1, m):
                mat[i][j] = 0

    if frow:
        for i in range(m):
            mat[i][0] = 0
    if fcol:
        for j in range(n):
            mat[0][j] = 0

    return mat


def rotate_image(matrix: List[List[int]]) -> List[List[int]]:
    n = len(matrix)
    if n < 2:
        return matrix

    for row in range(n - 1 // 2):  # O(N^2)
        for col in range(row, n - 1 - row):
            top_left = (row, col)
            top_right = (col, n - 1 - row)
            bottom_right = (n - 1 - row, n - 1 - col)
            bottom_left = (n - 1 - col, row)

            # swap top left and top right
            (matrix[top_left[0]][top_left[1]], matrix[top_right[0]][top_right[1]]) = (
                matrix[top_right[0]][top_right[1]],
                matrix[top_left[0]][top_left[1]],
            )
            # swap top left and bottom right
            (
                matrix[top_left[0]][top_left[1]],
                matrix[bottom_right[0]][bottom_right[1]],
            ) = (
                matrix[bottom_right[0]][bottom_right[1]],
                matrix[top_left[0]][top_left[1]],
            )
            # swap top left and bottom left
            (
                matrix[top_left[0]][top_left[1]],
                matrix[bottom_left[0]][bottom_left[1]],
            ) = (
                matrix[bottom_left[0]][bottom_left[1]],
                matrix[top_left[0]][top_left[1]],
            )
    return matrix


def spiral_order(matrix: List[List[int]]) -> List[int]:
    rows, cols = len(matrix), len(matrix[0])
    row, col = 0, -1
    direction = 1
    result = []

    while rows > 0 and cols > 0:  # O(M * N)
        for _ in range(cols):
            col += direction
            result.append(matrix[row][col])
        rows -= 1

        for _ in range(rows):
            row += direction
            result.append(matrix[row][col])
        cols -= 1

        direction *= -1

    return result


def find_exit_column(grid: List[List[int]]) -> List[int]:
    rows, cols = len(grid), len(grid[0])
    result = [0 for _ in range(cols)]

    col = 0

    while col < cols:
        curr_col = col
        for row in range(rows):
            if grid[row][curr_col] == -1:
                if curr_col == 0:
                    result[col] = -1
                    break
                else:
                    prev_col = curr_col - 1
                    if grid[row][prev_col] == 1:
                        result[col] = -1
                        break
                    else:
                        curr_col -= 1
            else:
                if curr_col == cols - 1:
                    result[col] = -1
                    break
                else:
                    next_col = curr_col + 1
                    if grid[row][next_col] == -1:
                        result[col] = -1
                        break
                    else:
                        curr_col += 1
        if result[col] != -1:
            result[col] = curr_col
        col += 1

    return result


# def find_exit_column(grid):
#     result = [-1] * len(grid[0])
#     for col in range(len(grid[0])):
#         current_col = col
#         for row in range(len(grid)):
#             next_col = current_col + grid[row][current_col]
#             if (
#                 next_col < 0
#                 or next_col > len(grid[0]) - 1
#                 or grid[row][current_col] != grid[row][next_col]
#             ):
#                 break
#             if row == len(grid) - 1:
#                 result[col] = next_col
#             current_col = next_col
#     return result


def count_unguarded(
    m: int, n: int, guards: List[List[int]], walls: List[List[int]]
) -> int:
    """
    Leetcode problem:
    https://leetcode.com/problems/count-unguarded-cells-in-the-grid/description/

    key insight is to minimize the iteration over the matrix when calculating
    the safe spaces... my initial implementation which did a depth first
    approach visiting all other rows and cols next to the guard cell.

    changing the pattern to identify the directions you can travel and
    iterating over them for each guard decreased the runtime from 3s to 400ms
    """
    matrix = [[0 for _ in range(n)] for _ in range(m)]
    for row, col in walls:
        matrix[row][col] = 1
    for row, col in guards:
        matrix[row][col] = 1

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for row, col in guards:
        for dr, dc in directions:
            curr_row, curr_col = row + dr, col + dc
            while 0 <= curr_row < m and 0 <= curr_col < n:
                if matrix[curr_row][curr_col] == 1:
                    break
                if matrix[curr_row][curr_col] == 0:
                    matrix[curr_row][curr_col] = 2
                curr_row += dr
                curr_col += dc

    free_spaces = sum(1 for r in range(m) for c in range(n) if matrix[r][c] == 0)

    return free_spaces


def totalStrength(strength: List[int]) -> int:
    """
    Leetcode problem: https://leetcode.com/problems/sum-of-total-strength-of-wizards/

    Naive solution exceeds time limit due to O(N^2) runtime.
    Need to investigate prefixsums and monotonic stacks to get the correct
    solution.

    (more of a math problem than a coding question)
    """
    MOD = 10**9 + 7
    n = len(strength)

    # i, i:i+1, ... i:n-1
    power = []
    subsets = sub_lists(strength)

    for wizards in subsets:
        power.append(min(wizards) * sum(wizards))

    # i = 0
    # while i < n:  # O(N)
    #     for j in range(n - i):  # O(N)
    #         wizard_sublist = strength[i:i+j+1]
    #         power = min(wizard_sublist) * sum(wizard_sublist)
    #         sets_of_wizards.append(power)
    #     i += 1

    return sum(wizards) % MOD


def sub_lists(xs):
    n = len(xs)
    indices = list(range(n + 1))
    for i, j in itertools.combinations(indices, 2):
        yield xs[i:j]


def binary_search_matrix(nums, target):
    if len(nums) == 0:
        return False
    m = len(nums)
    n = len(nums[0])
    start, end = 0, m * n - 1
    while start < end:
        mid = (end + start) // 2
        # translating row and col from the midpoint value
        row = mid // m
        col = mid % m
        if nums[row][col] == target:
            return True
        if nums[row][col] < target:
            start = mid + 1
        else:
            end = mid - 1
    return False


def shortest_bridge(grid: List[List[int]]) -> int:
    """
    Leetcode medium: https://leetcode.com/problems/shortest-bridge/

    Use a combination of depth first search to find the first island.

    Then we do breadth first search by enqueueing every node we've seen so far
    and finding all of their neighbors. We also track the number of levels of
    BFS as our answer. When we find another 1, we can exit the function and
    return the current level.
    """
    n = len(grid)
    if n == 2:
        return 1

    # find the dimensions of the first island.
    visited = set()
    stack = []

    # find the first 1 in the matrix for the first island.
    row, col = 0, 0
    while len(visited) == 0:
        if grid[row][col] == 1:
            stack.append((row, col))
            visited.add((row, col))
        elif col + 1 == n:
            row += 1
            col = 0
        else:
            col += 1

    # perform breadth first search to identify the entire island
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    while stack:
        node = stack.pop()
        for dr, dc in directions:
            new_row = node[0] + dr
            new_col = node[1] + dc
            if (
                (new_row, new_col) not in visited
                and 0 <= new_row < n
                and 0 <= new_col < n
                and grid[new_row][new_col] == 1
            ):
                visited.add((new_row, new_col))
                stack.append((new_row, new_col))

    queue = deque(list(visited))

    level = 0
    while queue:
        new_queue = deque()
        for row, col in queue:
            for dr, dc in directions:
                new_row = row + dr
                new_col = col + dc
                if (
                    (new_row, new_col) not in visited
                    and 0 <= new_row < n
                    and 0 <= new_col < n
                ):
                    if grid[new_row][new_col] == 1:
                        return level
                    else:
                        new_queue.append((new_row, new_col))
                        visited.add((new_row, new_col))
        queue = new_queue
        level += 1

    return level


def taskSchedulerII(tasks: List[int], space: int) -> int:
    """
    Leetcode problem: https://leetcode.com/problems/task-scheduler-ii/description/
    """
    # maintain a dict that has the task id and the day we can next process
    # this task
    task_manager = {}
    curr_day = 0
    for task in tasks:
        # If task was done before and cooldown hasn't passed, wait
        if task in task_manager and curr_day <= task_manager[task] + space:
            curr_day = task_manager[task] + space + 1
        else:
            curr_day += 1
        task_manager[task] = curr_day  # Update last day we did this task
    return curr_day


if __name__ == "__main__":
    linked_list = LinkedList([1, 2, 3, 4, 5])
    # display(reverse_k_groups(linked_list.head, 2))
    # print(calculator("12 - (6 + 2) + 5"))
    print(taskSchedulerII([5, 8, 8, 5], 2))
