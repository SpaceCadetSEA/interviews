from typing import Tuple

from data_structures.nodes import TrieNode, TrieNodeV2
from data_structures.trie_v3 import Trie as TrieV3


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, string: str) -> None:
        p1 = 0
        curr = self.root
        while p1 < len(string) and string[p1] in curr.children:
            curr = curr.children[string[p1]]
            p1 += 1
        # Now we have traversed what is already inside the Trie
        # and we can add new values
        while p1 < len(string):
            curr_char = string[p1]
            curr.children[curr_char] = TrieNode()
            curr = curr.children[curr_char]
            p1 += 1
        # At the very end, when we've traversed the entire string,
        # we set the final node to is_word = True
        curr.is_word = True

    def _traverse_trie(self, string: str) -> Tuple[int, TrieNode]:
        p1 = 0
        curr = self.root
        while p1 < len(string) and string[p1] in curr.children:
            curr = curr.children[string[p1]]
            p1 += 1
        return p1, curr

    def search(self, string: str) -> bool:
        p1, curr_node = self._traverse_trie(string)
        return p1 == len(string) and curr_node.is_word

    def search_prefix(self, prefix: str) -> bool:
        p1, _ = self._traverse_trie(prefix)
        return p1 == len(prefix)

    def get_all_words(self):
        words = []
        for character in self.root.children.keys():
            self._get_all_words_helper(
                f"{character}", self.root.children[character], words
            )
        return words

    def _get_all_words_helper(self, sofar, trie_node, words):
        if trie_node.is_word:
            words.append(sofar)
        for character in trie_node.children.keys():
            self._get_all_words_helper(
                sofar + character, trie_node.children[character], words
            )


"""
Design a data structure called WordDictionary that supports the following
functionalities:

- Constructor: This function will initialize the object.

- Add Word(word): This function will store the provided word in the data
  structure.

- Search Word(word): This function will return TRUE if any string in the
  WordDictionary object matches the query word. Otherwise, it will return
  FALSE. If the query word contains dots, ., each dot is free to match any
  letter of the alphabet.
    - For example, the dot in the string “.ad” can have possible search results
      like “aad”, “bad”, “cad”, and so on.

- Get Words(): This function will return all the words in the WordDictionary
  class.
"""


class WordDictionary:
    ALPHA = "abcdefghijklmnopqrstuvwxyz"

    def __init__(self):
        self.root = TrieNodeV2()

    def _get_key(self, c):
        if c not in self.ALPHA:
            c = self._translate(c)
        return ord(c) - ord("a")

    def _translate(self, i):
        return self.ALPHA[i]

    def add_word(self, word):
        curr = self.root
        for c in word:
            key = self._get_key(c)
            if curr.children[key] is None:
                curr.children[key] = TrieNodeV2()
            curr = curr.children[key]
        curr.complete = True

    def search_word(self, word):
        return self._search_word_helper(word, self.root)

    def _search_word_helper(self, word, trie_node):
        if len(word) == 0:
            return trie_node.complete
        elif word[0] == ".":
            return any(
                [
                    self._search_word_helper(word[1:], trie_node.children[i])
                    for i, char in enumerate(trie_node.children)
                    if char is not None
                ]
            )
        else:
            key = self._get_key(word[0])
            if trie_node.children[key] is None:
                return False
            return self._search_word_helper(word[1:], trie_node.children[key])

    def get_words(self):
        words = []
        [
            self._get_words_helper(
                f"{self._translate(i)}", self.root.children[i], words
            )
            for i, char in enumerate(self.root.children)
            if char is not None
        ]
        return words

    def _get_words_helper(self, sofar, trie_node, words):
        if trie_node.complete:
            words.append(sofar)
        for i, c in enumerate(trie_node.children):
            if c is not None:
                self._get_words_helper(
                    sofar + self._translate(i), trie_node.children[i], words
                )


def find_strings(grid, words):
    trie = TrieV3()
    for w in words:
        trie.insert(w)

    found_words = set()
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            nodes_seen = set([(row, col)])
            character = grid[row][col]
            if character in trie.root.children:
                _find_strings_helper(
                    trie.root.children[character],
                    grid,
                    f"{grid[row][col]}",
                    row,
                    col,
                    found_words,
                    nodes_seen,
                )
    return list(found_words)


def _find_strings_helper(trie_node, grid, prefix, row, col, found_words, nodes_seen):
    if trie_node.is_string and prefix not in found_words:
        found_words.add(prefix)

    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    m = len(grid)
    n = len(grid[0])
    for dr, dc in directions:
        new_coords = new_row, new_col = (row + dr, col + dc)
        if (
            new_coords not in nodes_seen
            and 0 <= new_row < m
            and 0 <= new_col < n
        ):
            next_character = grid[new_row][new_col]
            if next_character in trie_node.children:
                nodes_seen.add(new_coords)
                _find_strings_helper(
                    trie_node.children[grid[new_row][new_col]],
                    grid,
                    prefix + grid[new_row][new_col],
                    new_row,
                    new_col,
                    found_words,
                    nodes_seen,
                )
                nodes_seen.remove(new_coords)


if __name__ == "__main__":
    print(find_strings([["H"], ["D"]], ["HD"]))
