from math import inf
from typing import List, Tuple, Union

from data_structures.linked_list import LinkedList, display


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


def find_sorted_array_sum(array: List[int], sum: int) -> Union[Tuple[int, int], str]:
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


def three_sum(array):
    """
    This is a much better 3-sum implementation since we break out increment/
    decrement and duplication check into separate functions.
    """
    array.sort()
    result = set()
    i = 0
    while i < len(array) and array[i] <= 0:
        if 0 < i and array[i] == array[i - 1]:
            i += 1
            continue
        left, right = i + 1, len(array) - 1
        while left < right:
            curr_sum = sum([array[i], array[left], array[right]])
            if curr_sum == 0:
                result.add((array[i], array[left], array[right]))
                left = increment_left(array, left, right)
                right = decrement_right(array, left, right)
            elif curr_sum < 0:
                left = increment_left(array, left, right)
            else:
                right = decrement_right(array, left, right)
        i += 1
    return list(result)


def increment_left(nums, left, right):
    curr_left = left
    left += 1
    while left < right and nums[curr_left] == nums[left]:
        left += 1
    return left


def decrement_right(nums, left, right):
    curr_right = right
    right -= 1
    while left < right and nums[curr_right] == nums[right]:
        right -= 1
    return right


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


def remove_nth_last_node(head: ListNode, n: int):
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
    
    while right.next:
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


def find_min_in_rotated_array(arr: List[int]) -> int:
    if len(arr) == 1:
        return arr[0]

    left = 0
    right = len(arr) - 1

    if arr[right] > arr[0]:
        return arr[0]

    while right >= left:
        mid = left + (right - left) // 2
        if arr[mid] > arr[mid + 1]:
            return arr[mid + 1]

        if arr[mid] < arr[mid - 1]:
            return arr[mid]

        if arr[mid] > arr[left]:
            left = mid + 1
        else:
            right = mid - 1

    return -1


def threeSum(nums: List[int]) -> List[List[int]]:
    """
    Leetcode problem:
    https://leetcode.com/problems/3sum/description/

    Key here is runtime. Our original solution was very inefficient in both
    checking if the thruple already existed in our result and also iterating
    on combinations that would create the same thruple group.

    Using while loops properly set up to consume the index changes between
    first and last pointers, leds to a big speedup and reduces unnecessary
    visits to the inner while loop (first < last)
    """
    nums.sort()  # TimSort - O(n log n)
    i = 0
    res = []
    for i, num in enumerate(nums):
        if num > 0 or (i > 0 and nums[i - 1] == num):
            continue
        first, last = i + 1, len(nums) - 1
        while first < last:
            sum_of_values = num + nums[first] + nums[last]
            if sum_of_values == 0:
                thruple = [num, nums[first], nums[last]]
                res.append(thruple)
                # better way to consume similar answers from duplicated values in array
                while first < last and nums[first] == nums[first + 1]:
                    first += 1
                while first < last and nums[last] == nums[last - 1]:
                    last -= 1
            elif sum_of_values > 0:
                last -= 1
            else:
                first += 1
    return res


def pancake_sort(arr):
    """
    leetcode (medium):
    https://leetcode.com/problems/pancake-sorting/description/

    just need some time to figure out the pointer and max relationship :)
    """
    end = len(arr) - 1
    flips = []

    while end > 0:
        max_idx = arr.index(max(arr[: end + 1]))
        if max_idx == end:
            end -= 1
        elif max_idx == 0:
            flips.append(end + 1)
            arr = arr[: end + 1][::-1] + arr[end + 1 :]
            end -= 1
        else:
            flips.append(max_idx + 1)
            arr = arr[: max_idx + 1][::-1] + arr[max_idx + 1 :]

    return flips

def sort_colors(colors):
    left, right = 0, len(colors) - 1
    i = 0
    while i <= right:
        if colors[i] == 0:
            colors[i], colors[left] = colors[left], colors[i]
            left += 1
            i += 1
        elif colors[i] == 2:
            colors[i], colors[right] = colors[right], colors[i]
            right -= 1
        else:
            i += 1
    return colors


if __name__ == "__main__":
    # linked_list = LinkedList([69,8,49, 105,106,116,112])
    # display(middle_of_linked_list(linked_list.head))
    # print(detect_cycle_in_array([2, 3, 1, 4, 5, 9, 7]))
    # print(is_happy_number(4))
    print(sort_colors([0, 1, 0]))
