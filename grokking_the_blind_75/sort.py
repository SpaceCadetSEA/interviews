from collections import deque, defaultdict, Counter

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


def verify_alien_dictionary(words, order):
    order_to_priority = {char: i for i, char in enumerate(order)}

    for word1, word2 in zip(words, words[1:]):
        for w1, w2 in zip(word1, word2):
            if order_to_priority[w1] == order_to_priority[w2]:
                continue
            elif order_to_priority[w1] < order_to_priority[w2]:
                break
            else:
                return False
        else:
            if len(word2) < len(word1):
                return False

    return True


def find_compilation_order(dependencies):
    vertex_set = set()
    for c, p in dependencies:
        vertex_set.add(c)
        vertex_set.add(p)

    adj_list = {v: [] for v in vertex_set}
    in_edge_map = {v: 0 for v in vertex_set}

    for c, p in dependencies:
        adj_list[p].append(c)
        in_edge_map[c] += 1

    queue = deque([p for p in in_edge_map if in_edge_map[p] == 0])

    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in adj_list[node]:
            in_edge_map[child] -= 1
            if in_edge_map[child] == 0:
                queue.append(child)

    if len(order) != len(vertex_set):
        return []
    return order


"""
FUNCTION cyclicSort(arr):
  index = 0

  WHILE index < length of arr:
    # Get the correct index for the current element
    correctIndex = arr[index] - 1

    # Swap if the element is not in its correct position
    IF arr[index] != arr[correctIndex]:
      swap(arr[index], arr[correctIndex])
    ELSE:
      index += 1  # Move to the next index when no swap is needed
"""


def find_missing_number(nums):
    i = 0
    while i < len(nums):
        if nums[i] > len(nums) - 1 or nums[i] == i:
            i += 1
        else:
            curr_val = nums[i]
            to_swap = nums[curr_val]
            nums[i] = to_swap
            nums[curr_val] = curr_val
    for i in range(len(nums)):
        if nums[i] != i:
            return i
    return len(nums)


def smallest_missing_positive_integer(nums):
    """
    Cyclic Sort -- O(N)
    """
    i = 0
    while i < len(nums):
        correct_idx = nums[i] - 1
        if 0 <= correct_idx < len(nums) and nums[i] != nums[correct_idx]:
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
        else:
            i += 1

    for i in range(len(nums)):
        if nums[i] != i + 1:
            return i + 1
    return len(nums) + 1


def find_corrupt_pair(nums):
    # perform cyclic sort -- [1, n]
    i = 0
    while i < len(nums):
        correct_idx = nums[i] - 1
        if 0 <= correct_idx < len(nums) and nums[i] != nums[correct_idx]:
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
        else:
            i += 1

    # iterate over sorted list...
    for i in range(len(nums)):
        if nums[i] != i + 1:
            return [i + 1, nums[i]]


if __name__ == "__main__":
    print(find_corrupt_pair([3, 1, 2, 5, 2]))
