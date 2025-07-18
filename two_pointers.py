from typing import List, Tuple
from math import inf
from data_structures.LinkedList import LinkedList, display


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
    """Best case: time: O(n) space: O(1)"""
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


def remove_nth_last_node(head: List[int], n: int):
    """
    0-> 1-> 2-> 3-> null

    [69,8,49,106,116,112] , 6
    """
    left = head
    right = head
    for _ in range(n):
        right = right.next
        if right is None:
            return head.next
    while right.next is not None:
        right = right.next
        left = left.next
    left.next = left.next.next
    return head


def middle_of_linked_list(head: ListNode) -> ListNode:
    fast = head
    slow = head

    while fast is not None and fast.next is not None:
        fast = fast.next.next
        slow = slow.next

    return slow


def detect_cycle_in_array(arr: List[int]) -> bool:
    if arr[0] < len(arr) - 1:
        return False
    
    slow = arr[0]
    fast = arr[arr[0]]

    while fast < len(arr) - 1:
        if fast == slow:
            return True
        slow = arr[slow]
        fast = arr[arr[fast]]

    return False


def detect_cycle_in_linked_list(head: ListNode) -> bool:
    if head is None:
        return False
    slow = head
    fast = head
    while fast is not None and fast.next is not None:
        # detect loop
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    # while loop has terminated, no loop found
    return False


def is_happy_number(n: int) -> bool:
    def sum_of_squares(n: int) -> int:
        running_total = 0
        while n != 0:
            # break off digit
            digit = n % 10
            # alter n
            n //= 10
            running_total += digit**2
            # continue
        return running_total
    slow = n
    fast = sum_of_squares(n)
    while fast != 1:
        slow = sum_of_squares(slow)
        fast = sum_of_squares(sum_of_squares(fast))
        if fast == slow:
            return False
    return True


def binary_search_rotated(nums: List[int], target: int) -> int:
    low, high = 0, len(nums) - 1
    while low <= high:
        mid = low + (high - low) // 2
        if nums[mid] == target:
            return mid
        # low to mid sorted
        if nums[low] <= nums[mid]:
            if nums[low] <= target and target < nums[mid]:
                high = mid - 1
            else:
                low = mid + 1
        # mid to high sorted
        else:
            if nums[mid] < target and target <= nums[high]:
                low = mid + 1
            else:
                high = mid - 1
    return -1


if __name__ == "__main__":
    # linked_list = LinkedList([69,8,49, 105,106,116,112])
    # display(middle_of_linked_list(linked_list.head))
    # print(detect_cycle_in_array([2, 3, 1, 4, 5, 9, 7]))
    print(is_happy_number(4))
