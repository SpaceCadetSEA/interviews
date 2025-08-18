from heapq import heappush, heappop, heappushpop, heapreplace
from collections import Counter
from math import sqrt
from typing import List

"""
FUNCTION findTopKElements(arr, k):
  # Initialize a min heap
  minHeap = new MinHeap()

  # Insert the first k elements into the heap
  FOR i FROM 0 TO k - 1:
    minHeap.insert(arr[i])

  # Process the remaining elements
  FOR i FROM k TO length of arr - 1:
    IF arr[i] > minHeap.peek():  # Compare with the smallest in heap
      minHeap.extractMin()  # Remove the smallest element
      minHeap.insert(arr[i])  # Insert the new larger element

  RETURN minHeap.toList()  # Convert heap to list and return
"""


def top_k_frequent(arr, k):
    """
    time complexity: O(n log k)
    space complexity: O(n + k)
    """
    h = []
    frequency = Counter(arr)

    for key, val in frequency.items():
        if len(h) < k:
            heappush(h, (val, key))
        elif h[0][0] < val:
            heapreplace(h, (val, key))
    return [key for key, _ in h]


def reorganize_string(string):
    frequency = Counter(string)
    max_heap = []

    for key, val in frequency.items():
        heappush(max_heap, (-1 * val, key))

    res = ""

    while max_heap:
        (next_val, next_char) = heappop(max_heap)
        if len(res) == 0 or res[-1] != next_char:
            res += next_char
            if next_val + 1 < 0:
                heappush(max_heap, (next_val + 1, next_char))
        elif res[-1] == next_char:
            if len(max_heap) > 0:
                (second_next_val, second_next_char) = heappop(max_heap)
                res += second_next_char
                if second_next_val + 1 < 0:
                    heappush(max_heap, (second_next_val + 1, second_next_char))
                heappush(max_heap, (next_val, next_char))
            else:
                return ""

    return res


class KthLargest:
    # Constructor to initialize heap and add values in it
    def __init__(self, k, nums):
        self.k = k
        self.nums = nums
        self.h = []
        for num in nums:
            self.add(num)

    # Adds element in the heap and return the Kth largest
    def add(self, val):
        if len(self.h) < self.k:
            heappush(self.h, val)
        else:
            curr_min = heappop(self.h)
            if val > curr_min:
                heappush(self.h, val)
            else:
                heappush(self.h, curr_min)
        return self.h[0]


def k_closest(points, k):
    h = []
    x0, y0 = (0, 0)
    for x, y in points:
        distance = sqrt(((x - x0) ** 2 + (y - y0) ** 2)) * -1
        if len(h) < k:
            heappush(h, (distance, (x, y)))
        else:
            curr_val, (curr_x, curr_y) = heappop(h)
            if distance > curr_val:
                heappush(h, (distance, (x, y)))
            else:
                heappush(h, (curr_val, (curr_x, curr_y)))
    return [[x, y] for _, (x, y) in h]


class MedianOfStream:
    def __init__(self):
        self.min_heap = []
        self.max_heap = []

    @property
    def min_heap_size(self):
        return len(self.min_heap)

    @property
    def max_heap_size(self):
        return len(self.max_heap)

    # This function should take a number and store it
    def insert_num(self, num):
        # if both empty, add to max
        if self.min_heap_size == 0 and self.max_heap_size == 0:
            heappush(self.max_heap, num * -1)
        # if smaller or equal to max heap
        elif num <= self.max_heap[0] * -1:
            heappush(self.max_heap, num * -1)
        else:
            heappush(self.min_heap, num)
        # balance heaps
        if self.max_heap_size > self.min_heap_size + 1:
            # max heap is overloaded... pop max and add to min
            max_val = heappop(self.max_heap)
            heappush(self.min_heap, max_val * -1)
        if self.min_heap_size > self.max_heap_size + 1:
            # min heap is overloaded... pop min and add to max
            min_val = heappop(self.min_heap)
            heappush(self.max_heap, min_val * -1)

    # This function should return the median of the stored numbers
    def find_median(self):
        # Replace this placeholder return statement with your code
        if self.max_heap_size == self.min_heap_size:
            return (self.max_heap[0] * -1 + self.min_heap[0]) / 2.0
        else:
            if self.max_heap_size > self.min_heap_size:
                return self.max_heap[0] * -1 / 1.0
            else:
                return self.min_heap[0] / 1.0


def minimum_machines(tasks):
    min_heap = []
    tasks.sort(key=lambda x: x[0])  # O(n log n) to sort

    for start, stop in tasks:  # O(n)
        if min_heap and start >= min_heap[0]:
            heappop(min_heap)  # O(log n)
        heappush(min_heap, stop)  # O(log n)
    return len(min_heap)


def maximum_capital(c: int, k: int, capitals: List[int], profits: List[int]) -> int:
    max_heap = []
    min_heap = []
    current_capital = c

    for index, capital in enumerate(capitals):
        heappush(min_heap, (capital, index))

    for _ in range(k):
        while min_heap and min_heap[0][0] <= current_capital:
            capital, index = heappop(min_heap)
            heappush(max_heap, -profits[index])
        if not max_heap:
            return current_capital
        else:
            profit = heappop(max_heap)
            current_capital += -profit

    return current_capital


def median_sliding_window(nums, k):
    """This problem stumped me"""
    medians = []
    outgoing_num = {}
    max_heap = []
    min_heap = []

    for i in range(k):
        heappush(max_heap, -1 * nums[i])

    for i in range(k // 2):
        element = heappop(max_heap)
        heappush(min_heap, -1 * element)

    balance = 0
    i = k
    while True:
        if k % 2 == 1:
            medians.append(-max_heap[0] / 1.0)
        else:
            medians.append((-max_heap[0] + min_heap[0]) / 2.0)

        if i >= len(nums):
            break

        out_num = nums[i - k]
        in_num = nums[i]
        i += 1

        if out_num <= (max_heap[0] * -1):
            balance -= 1
        else:
            balance += 1

        if out_num in outgoing_num:
            outgoing_num[out_num] = outgoing_num[out_num] + 1
        else:
            outgoing_num[out_num] = 1

        if max_heap and in_num <= (max_heap[0] * -1):
            balance += 1
            heappush(max_heap, in_num * -1)
        else:
            balance -= 1
            heappush(min_heap, in_num)

        if balance < 0:
            heappush(max_heap, (-1 * min_heap[0]))
            heappop(min_heap)
        elif balance > 0:
            heappush(min_heap, (-1 * max_heap[0]))
            heappop(max_heap)

        balance = 0

        while (
            (max_heap[0] * -1) in outgoing_num 
            and (outgoing_num[(max_heap[0] * -1)] > 0
        )):
            outgoing_num[max_heap[0] * -1] -= 1
            heappop(max_heap)

        while (
            min_heap
            and min_heap[0] in outgoing_num
            and (outgoing_num[min_heap[0]] > 0)
        ):
            outgoing_num[min_heap[0]] -= 1
            heappop(min_heap)

    return medians


if __name__ == "__main__":
    print(median_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3))
