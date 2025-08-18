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


if __name__ == "__main__":
    values = [100]
    tree_nodes = [TreeNode(val) for val in values]
    tree = BinaryTree(tree_nodes)

    sub_tree_values = [100]
    sub_tree_nodes = [TreeNode(val) for val in sub_tree_values]
    sub_tree = BinaryTree(sub_tree_nodes)
