from data_structures.nodes import TrieNode
from typing import Tuple


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

    def search_word(self, word):
        # if word has a wild card, we need to process that
        # and continue traversing down our trie
        curr = self.root
        words_found = []
        if "." in word:
            self._search_wildcard_helper(word, curr, words_found)
            return any(words_found)
        else:
            return self.search(word)

    def _search_wildcard_helper(self, word, trie_node, words_found):
        if trie_node.is_word:
            words_found.append(True)
        else:
            for c in word:
                if c == ".":
                    new_word = word[1:] if len(word) > 2 else ""
                    for child in trie_node.children.keys():
                        self._search_wildcard_helper(
                            new_word, trie_node.children[child], words_found
                        )
                elif c not in trie_node.children.keys():
                    return False
                else:
                    self._search_wildcard_helper(
                        word[1:], trie_node.children[c], words_found
                    )

    def search_prefix(self, prefix: str) -> bool:
        p1, _ = self._traverse_trie(prefix)
        return p1 == len(prefix)

    def get_all_words(self):
        dp = []
        for character in self.root.children.keys():
            self._get_all_words_helper(
                f"{character}", self.root.children[character], dp
            )
        return dp

    def _get_all_words_helper(self, sofar, trie_node, dp):
        if trie_node.is_word:
            dp.append(sofar)
        for character in trie_node.children.keys():
            self._get_all_words_helper(
                sofar + character, trie_node.children[character], dp
            )


if __name__ == "__main__":
    trie = Trie()
    trie.insert("hello")
    trie.insert("hi")
    trie.insert("hilt")
    trie.insert("help")
    print(trie.get_all_words())
    print(trie.search_word("h.."))
