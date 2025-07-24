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


def _serialize(root: TreeNode, res=[]):
    if not root:
        res.append(None)
    else:
        res.append(root.data)
        _serialize(root.left, res)
        _serialize(root.right, res)


def serialize(root: TreeNode) -> List[int]:
    res = []
    _serialize(root, res)
    return res


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


def deserialize(stream: List[int]) -> TreeNode:
    global idx
    idx = 0
    return _deserialize(stream)


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


def max_path_sum(root: TreeNode) -> int:
    global max_sum
    max_sum = -inf
    _max_contrib(root)
    return max_sum


def _build_tree_rec(p_order, i_order):
    global p_idx
    if not i_order:
        return None
    
    root_val = p_order[p_idx]
    p_idx += 1
    root = TreeNode(root_val)
    i_idx = i_order.index(root.data)

    root.left = _build_tree_rec(p_order, i_order[:i_idx])
    root.right = _build_tree_rec(p_order, i_order[i_idx + 1:])
    return root


def build_tree(p_order, i_order):
    global p_idx
    p_idx = 0
    return _build_tree_rec(p_order, i_order)


if __name__ == "__main__":
    values = [-8, 2, 17, 1, 4, 19, 5]
    tree_nodes = [TreeNode(val) for val in values]
    tree = BinaryTree(tree_nodes)
    # print(level_order_traversal(tree.root))

    # print(serialize(tree.root))
    # print(max_path_sum(tree.root))
    print(build_tree([3,9,20,15,7], [9,3,15,20,7]))
