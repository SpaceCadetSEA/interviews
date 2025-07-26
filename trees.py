from typing import List
from collections import deque
from math import inf
from data_structures.trees import BinaryTree, TreeNode


def level_order_traversal(root):
    if not root:
        return "None"

    current_queue = deque()
    next_queue = deque()
    result = ""

    current_queue.append(root)
    while current_queue:
        current_node = current_queue.popleft()
        result += f"{current_node.data}"

        if current_node.left:
            next_queue.append(current_node.left)
        if current_node.right:
            next_queue.append(current_node.right)

        if not current_queue:
            if next_queue:
                result += " : "
                current_queue, next_queue = next_queue, current_queue
        else:
            result += ", "
    return result


def word_ladder(src: str, dest: str, words: List[str]) -> int:
    word_set = set(words)
    if dest not in word_set:
        return 0
    q = deque()
    number_steps = 0

    q.append(src)
    while q:
        number_steps += 1
        size = len(q)
        for _ in range(size):
            curr_word = q.popleft()

            for i in range(len(curr_word)):
                alpha = "abcdefghijklmnopqrstuvwxyz"
                for c in alpha:
                    temp = list(curr_word)
                    temp[i] = c
                    temp = "".join(temp)

                    if temp == dest:
                        return number_steps + 1
                    if temp in word_set:
                        q.append(temp)
                        word_set.remove(temp)
    return number_steps


def vertical_order(root):
    if not root:
        return []

    q = deque()
    col_index = 0

    q.append((col_index, root))
    hash_map = {}

    while q:
        curr_col_index, curr_node = q.popleft()

        if curr_col_index not in hash_map.keys():
            hash_map[curr_col_index] = [curr_node.data]
        else:
            hash_map[curr_col_index].append(curr_node.data)

        if curr_node.left:
            q.append((curr_col_index - 1, curr_node.left))
        if curr_node.right:
            q.append((curr_col_index + 1, curr_node.right))

    all_indices = list(hash_map.keys())
    all_indices.sort()

    result = []
    for idx in all_indices:
        result.append(hash_map[idx])
    return result


def is_symmetric(root: TreeNode) -> bool:
    if not root:
        return True

    q = deque()
    q.append(root.left)
    q.append(root.right)

    while q:
        left = q.popleft()
        right = q.popleft()

        if not left and not right:
            continue

        if (left and not right) or (right and not left) or left.data != right.data:
            return False

        q.append(left.left)
        q.append(right.right)
        q.append(left.right)
        q.append(right.left)

    return True


def serialize(root: TreeNode) -> List[int]:
    res = []
    _serialize(root, res)
    return res


def _serialize(root: TreeNode, res=[]):
    if not root:
        res.append(None)
    else:
        res.append(root.data)
        _serialize(root.left, res)
        _serialize(root.right, res)


def deserialize(stream: List[int]) -> TreeNode:
    global idx
    idx = 0
    return _deserialize(stream)


def _deserialize(stream: List[int]) -> TreeNode:
    global idx
    if not stream:
        return None
    next_element = stream[idx]
    idx += 1
    if not next_element:
        return None
    root = TreeNode(next_element)
    root.left = _deserialize(stream)
    root.right = _deserialize(stream)
    return root


def max_path_sum(root: TreeNode) -> int:
    global max_sum
    max_sum = -inf
    _max_contrib(root)
    return max_sum


def _max_contrib(root: TreeNode) -> int:
    global max_sum
    if not root:
        return 0
    left_subtree_max = _max_contrib(root.left)
    right_subtree_max = _max_contrib(root.right)

    left_subtree = 0
    right_subtree = 0

    if left_subtree_max > 0:
        left_subtree = left_subtree_max
    if right_subtree_max > 0:
        right_subtree = right_subtree_max

    path_max = root.data + left_subtree + right_subtree
    max_sum = max(max_sum, path_max)
    return root.data + max(left_subtree, right_subtree)


def build_tree(p_order, i_order):
    global p_idx
    p_idx = 0
    return _build_tree_rec(p_order, i_order)


def _build_tree_rec(p_order, i_order):
    global p_idx
    if not i_order:
        return None

    root_val = p_order[p_idx]
    p_idx += 1
    root = TreeNode(root_val)
    i_idx = i_order.index(root.data)

    root.left = _build_tree_rec(p_order, i_order[:i_idx])
    root.right = _build_tree_rec(p_order, i_order[i_idx + 1 :])
    return root


def mirror_binary_tree(root: TreeNode) -> TreeNode:
    if not root:
        return root
    return _mirror_binary_tree_rec(root)


def _mirror_binary_tree_rec(root: TreeNode) -> TreeNode:
    if not root:
        return None

    left_subtree = _mirror_binary_tree_rec(root.left)
    right_subtree = _mirror_binary_tree_rec(root.right)

    root.left = right_subtree
    root.right = left_subtree

    return root


def kth_smallest_element(root, k):
    # had to use a mutable datatype, not an int for K across recursive calls
    return _kth_smallest_element_rec(root, [k]).data


def _kth_smallest_element_rec(root, k):
    if not root:
        return None

    left = _kth_smallest_element_rec(root.left, k)

    if left:
        return left

    k[0] -= 1
    if k[0] == 0:
        return root

    return _kth_smallest_element_rec(root.right, k)


def lowest_common_ancestor(current_node, p, q):
    return _lowest_common_ancestor_rec(current_node, p, q)


def _lowest_common_ancestor_rec(current_node, p, q):
    if not current_node:
        return None

    mid = None
    left = None
    right = None

    mid = current_node.data in [p.data, q.data]
    left = _lowest_common_ancestor_rec(current_node.left, p, q)
    right = _lowest_common_ancestor_rec(current_node.right, p, q)

    if (mid and left) or (mid and right) or (left and right):
        return current_node

    if mid:
        return current_node
    if left:
        return left
    if right:
        return right
    return None


def find_max_depth(root) -> int:
    """CSE 143 problem: height()"""
    return _find_max_depth_rec(root)


def _find_max_depth_rec(root):
    if not root:
        return 0
    # AI complained that it wasn't DFS to call the recursive helper f(x)
    # within the max() comparator.
    left_max = _find_max_depth_rec(root.left)
    right_max = _find_max_depth_rec(root.right)

    return 1 + max(left_max, right_max)


def find_max_depth_EXAMPLE(root):
    if not root:
        return 0

    nodes_stack = deque([(root, 1)])

    max_depth = 0
    while nodes_stack:
        node, depth = nodes_stack.pop()
        if node.left:
            nodes_stack.append((node.left, depth + 1))
        if node.right:
            nodes_stack.append((node.right, depth + 1))
        if not node.left and not node.right:
            max_depth = max(max_depth, depth)
    return max_depth


def same_tree_dfs(p, q):
    return _same_tree_rec(p, q)


def _same_tree_rec(p, q):
    if not p or not q:
        return p is None and q is None
    left_nodes_same = _same_tree_rec(p.left, q.left)
    right_nodes_same = _same_tree_rec(p.right, q.right)

    return p.data == q.data and left_nodes_same and right_nodes_same


def same_tree_bfs(p, q):
    left_queue = deque()
    right_queue = deque()

    left_queue.append(p)
    right_queue.append(q)

    while left_queue or right_queue:
        left_tree = left_queue.popleft()
        right_tree = right_queue.popleft()

        if not left_tree and not right_tree:
            continue

        if (left_tree and not right_tree) or (right_tree and not left_tree):
            return False

        if left_tree.data != right_tree.data:
            return False

        left_queue.append(left_tree.left)
        left_queue.append(left_tree.right)
        right_queue.append(right_tree.left)
        right_queue.append(right_tree.right)
    return True


def is_subtree(root: TreeNode, sub_root: TreeNode) -> bool:
    if not root:
        return False
    if _is_subtree_rec(root, sub_root):
        return True
    return is_subtree(root.left, sub_root) or is_subtree(root.right, sub_root)


def _is_subtree_rec(root: TreeNode, sub_root: TreeNode) -> bool:
    if not root or not sub_root:
        return not root and not sub_root
    left_matches = _is_subtree_rec(root.left, sub_root.left)
    right_matches = _is_subtree_rec(root.right, sub_root.right)
    return root.data == sub_root.data and left_matches and right_matches


def validate_bst(root: TreeNode) -> bool:
    if not root:
        return True
    prev = [-inf]
    return _validate_bst_rec(root, prev)


def _validate_bst_rec(root: TreeNode, prev: List[int], curr_max=List[int]) -> bool:
    if not root:
        return True

    if not _validate_bst_rec(root.left, prev):
        return False

    if root.data <= prev[0]:
        return False
    prev[0] = root.data
    return _validate_bst_rec(root.right, prev)


def diameter_of_bst(root: TreeNode) -> int:
    curr_max = [-inf]
    _diameter_of_bst_rec(root, curr_max)
    return curr_max


def _diameter_of_bst_rec(root: TreeNode, curr_max: List[int]) -> int:
    if not root:
        return 0
    left_height = _diameter_of_bst_rec(root.left, curr_max)
    right_height = _diameter_of_bst_rec(root.right, curr_max)
    curr_diameter = left_height + right_height
    curr_max[0] = max(curr_diameter, curr_max[0])
    return 1 + max(left_height, right_height)


if __name__ == "__main__":
    values = [100]
    tree_nodes = [TreeNode(val) for val in values]
    tree = BinaryTree(tree_nodes)

    sub_tree_values = [100]
    sub_tree_nodes = [TreeNode(val) for val in sub_tree_values]
    sub_tree = BinaryTree(sub_tree_nodes)
    # print(level_order_traversal(tree.root))

    # print(serialize(tree.root))
    # print(max_path_sum(tree.root))
    # print(build_tree([3, 9, 20, 15, 7], [9, 3, 15, 20, 7]))
    print(is_subtree(tree.root, sub_tree.root))
