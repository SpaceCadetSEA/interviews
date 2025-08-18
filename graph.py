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


"""
FUNCTION topologicalSort(graph):
  # Initialize in-degree map
  inDegree = {node: 0 FOR each node in graph}

  # Calculate in-degrees of all nodes
  FOR node IN graph:
    # Find the neighbors of each node in the given graph
    FOR neighbor IN graph[node]:
      inDegree[neighbor] += 1

  # Add nodes with 0 in-degree to the queue
  queue = new Queue()
  FOR node IN inDegree:
    IF inDegree[node] == 0:
      queue.enqueue(node)

  topologicalOrder = []

  # Process nodes in topological order
  WHILE queue is not empty:
    current = queue.dequeue()
    topologicalOrder.append(current)

    # Reduce in-degree of neighbors and enqueue if they become 0
    FOR neighbor IN graph[current]:
      inDegree[neighbor] -= 1
      IF inDegree[neighbor] == 0:
        queue.enqueue(neighbor)

  # Check for cycles (If not all nodes are processed, a cycle exists)
  IF length of topologicalOrder != length of graph:
    RETURN "Cycle detected, topological sort not possible"

  RETURN topologicalOrder
"""


def alien_order(words):
    c = Counter({c: 0 for word in words for c in word})
    adjacency_list = defaultdict(set)

    for word1, word2 in zip(words, words[1:]):
        for w1, w2 in zip(word1, word2):
            if w1 != w2:
                if w1 not in adjacency_list:
                    adjacency_list[w1].add(w2)
                    c[w2] += 1
                break
        else:
            if len(word2) < len(word1):
                return ""

    result = ""
    queue = deque([node for node in c if c[node] == 0])
    while queue:
        node = queue.popleft()
        result += node
        for child in adjacency_list[node]:
            c[child] -= 1
            if c[child] == 0:
                queue.append(child)

    if len(result) < len(c):
        return ""

    return result


def can_finish(num_courses, prerequisites):
    """
    Given a number of college courses and their required prerequisites,
    return True if the student can take all the courses, False otherwise.
    
    This does not represent topological sort correctly... see below.
    """
    adj_list = {i: [] for i in range(num_courses)}
    for c, prereq in prerequisites:
        adj_list[prereq].append(c)
        
    stack = [c for c, p in adj_list.items() if not p]
    classes = []
    while stack:
        prereq = stack.pop()
        classes.append(prereq)
        can_take = [c for c, p in adj_list.items() if prereq in p]
        for c in can_take:
            adj_list[c].remove(prereq)
            if not adj_list[c]:
                stack.append(c)
    
    return len(classes) == num_courses


def can_finish_topological(num_courses, prerequisites):
    """
    Given a number of college courses and their required prerequisites,
    return True if the student can take all the courses, False otherwise.
    
    This version does use Topological Sort using Kahn's Algorithm. The key
    difference is the use of the count of in-degrees for each vertex.
    
    - Counts incoming edges (prerequisites) for each course.
    - Starts with courses having zero prerequisites.
    - Iteratively removes courses from the queue, reducing the prerequisite 
        count for dependent courses.
    - Checks if all courses can be taken by comparing the number of processed 
        courses to num_courses.
    """
    count = Counter({i: 0 for i in range(num_courses)})

    for c, _ in prerequisites:
        count[c] += 1

    queue = deque([c for c in count if count[c] == 0])
    classes = []
    while queue:
        prereq = queue.popleft()
        # add to our class list
        classes.append(prereq)
        # decrement requirements
        can_take = [c for c, p in prerequisites if p == prereq]
        for c in can_take:
            count[c] -= 1
            if not count[c]:
                queue.append(c)
    return num_courses == len(classes)


if __name__ == "__main__":
    print(can_finish(5, [[1, 0], [2, 1], [3, 2], [4, 3]]))
