from typing import List
from collections import Counter


def is_anagram(str1: str, str2: str) -> bool:
    """
    time complexity: O(N)
    space complexity: O(1) -- ONLY 26 English letters
    """
    if len(str1) != len(str2):
        return False

    char_count = {}
    for c in str1:
        if c not in char_count:
            char_count[c] = 1
        else:
            char_count[c] += 1

    for c in str2:
        if c not in char_count:
            return False
        else:
            char_count[c] -= 1
            if char_count[c] < 0:
                return False

    return not any(char_count.values())


def contains_duplicates(nums: List[int]) -> bool:
    num_set = set()
    for num in nums:
        if num in num_set:
            return True
        else:
            num_set.add(num)
    return False


def group_anagrams(strs: List[str]) -> List[List[str]]:
    anagram_dict = {}
    for word in strs:
        embedding = _embed_string(word)
        if embedding in anagram_dict:
            anagram_dict[embedding].add(word)
        else:
            anagram_dict[embedding] = [word]
    return sorted(list(anagram_dict.values()), key=lambda x: len(x), reverse=True)


def _embed_string(word):
    """
    given string, return embedding of len 26 where each index
    represents the character and each element represents frequency
    """
    alpha = "abcdefghijklmnopqrstuvwxyz"
    d = {b: a for a, b in enumerate(alpha)}
    list_embedding = [0 for _ in alpha]
    for c in word:
        index = d[c]
        list_embedding[index] += 1
    return "".join([str(i) for i in list_embedding])


def two_sum(arr, t):
    d = {}
    for i, num in enumerate(arr):
        if t - num not in d:
            d[t - num] = i
        if num in d and d[num] != i:
            return [d[num], i]


class TicTacToe:
    # Constructor will be used to initialize TicTacToe data members
    def __init__(self, n):
        self.rows = [0 for _ in range(n)]
        self.cols = [0 for _ in range(n)]
        self.diag = 0
        self.anti_diag = 0
        self.n = n

    # move will be used to play a move by a specific player and identify who
    # wins at each move
    def move(self, row, col, player):
        # player 1 == add
        # player 2 == subtract
        if player == 1:
            self.rows[row] += 1
            self.cols[col] += 1
            if row == col:
                self.diag += 1
            if row + col == self.n - 1:
                self.anti_diag += 1
        else:
            self.rows[row] -= 1
            self.cols[col] -= 1
            if row == col:
                self.diag -= 1
            if row + col == self.n - 1:
                self.anti_diag -= 1

        if (
            self.n in self.rows
            or self.n in self.cols
            or self.diag == self.n
            or self.anti_diag == self.n
        ):
            return 1
        if (
            -1 * self.n in self.rows
            or -1 * self.n in self.cols
            or self.diag == -1 * self.n
            or self.anti_diag == -1 * self.n
        ):
            return 2
        return 0


if __name__ == "__main__":
    print(group_anagrams())
