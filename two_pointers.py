from typing import List, Tuple
from math import inf


def is_palindrome(word: str) -> bool:
    p1, p2 = 0, len(word) - 1

    while p1 < p2:
        if word[p1] != word[p2]:
            return False
        p1 += 1
        p2 -= 1

    return True


def swap_string(word: list) -> list:
    p1, p2 = 0, len(word) - 1

    while p1 < p2:
        first, second = word[p1], word[p2]
        word[p1] = second
        word[p2] = first
        p1 += 1
        p2 -= 1

    return word


def find_sorted_array_sum(array: List[int], sum: int) -> Tuple[int, int]:
    p1, p2 = 0, len(array) - 1

    while p1 < p2:
        pair_sum = array[p1] + array[p2]
        if pair_sum == sum:
            return array[p1], array[p2]
        if pair_sum < sum:
            p1 += 1
        else:
            p2 -= 1

    return "not found"


def move_zeroes(array: List[int]) -> List[int]:
    p1, p2 = 0, len(array) - 1

    while p1 < p2:
        while array[p2] == 0:
            p2 -= 1
        while array[p1] != 0:
            p1 += 1
        if array[p1] == 0:
            first, second = array[p1], array[p2]
            array[p1], array[p2] = second, first
        p1 += 1
        p2 -= 1

    return array


def three_sum(array: List[int]) -> List[List[int]]:
    array.sort()
    result = []
    for i, num in enumerate(array):
        # If we have already seen this number in the proceeding loop, skip
        if i > 0 and num == array[i - 1]:
            continue
        # If our starting value is > 0, we cannot find a sum that equals 0, skip
        if num > 0:
            continue
        p1, p2 = i + 1, len(array) - 1
        while p1 < p2:
            first, second = array[p1], array[p2]
            current_sum = num + first + second
            if current_sum == 0:
                valid_tuple = [num, first, second]
                # The following checks ensure we don't duplicate results
                if len(result) > 0:
                    if valid_tuple == result[-1]:
                        p1 += 1
                        p2 -= 1
                        continue
                result.append(valid_tuple)
                p1 += 1
                p2 -= 1
            elif current_sum > 0:
                p2 -= 1
            else:
                p1 += 1
    return result


def container_with_most_water(heights: List[int]) -> int:
    p1, p2 = 0, len(heights) - 1
    # we have an initial height and a end height and we can calculate the amount
    # of water between then with the lower of the two heights * the distance.
    max_volume = -1
    while p1 < p2:
        # find distance between them
        distance = p2 - p1
        curr_max_volume = distance * min(heights[p1], heights[p2])
        max_volume = max(max_volume, curr_max_volume)
        if heights[p1] >= heights[p2]:
            p2 -= 1
        else:
            p1 += 1
    return max_volume


def product_except_self(arr: List[int]) -> List[int]:
    n = len(arr)
    res = [1] * n
    left_product, right_product = 1, 1
    l, r = 0, n - 1

    while l < n and r > -1:
        res[l] = res[l] * left_product
        res[r] = res[r] * right_product

        left_product *= arr[l]
        right_product *= arr[r]

        l += 1
        r -= 1

    return res


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def remove_nth_last_node(head: ListNode, n: int):
    left, right = head
    for _ in range(n):
        right = right.next
        if right is None:
            return head
    while right is not None:
        right = right.next
        left = left.next
    left.next = left.next.next
    return head



if __name__ == "__main__":
    # print(is_palindrome("madam"))
    # print(is_palindrome("racecar"))
    # print(is_palindrome("happy"))
    # print(is_palindrome("abccba"))

    # print(swap_string(["b", "a", "c", "k", "w", "a", "r", "d"]))
    # print(swap_string(["a", "b", "c"]))

    # print(find_sorted_array_sum([2, 3, 5, 7, 11, 13], 14))

    # print(move_zeroes([0, 0, 0, 1, 2, 3, 4, 0, 0]))

    # print(three_sum([0, 0, 0]))

    print(container_with_most_water([1, 8, 6, 2, 5, 4, 8, 3, 7]))
