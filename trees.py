from typing import List
from collections import deque
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
        curr_word = q.popleft()
        seen_words = []
        for word in word_set:
            if word in seen_words:
                break
            # compare words
            offset = 0
            char_ptr = 0

            while char_ptr < len(curr_word):
                if curr_word[char_ptr] != word[char_ptr]:
                    offset += 1
                char_ptr += 1
            
            if offset == 0:
                return number_steps + 1
            elif offset == 1:
                q.append(word)
                seen_words.append(word)
        for word in seen_words:
            word_set.remove(word)
            number_steps += 1
    return number_steps


if __name__ == '__main__':
    values = [100, 50, 200, 25, 75, 300, 10, 350, 15]
    tree_nodes = [TreeNode(val) for val in values]
    tree = BinaryTree(tree_nodes)
    # print(level_order_traversal(tree.root))

    print(word_ladder("zz", "nx", ["nz", "nx"]))
