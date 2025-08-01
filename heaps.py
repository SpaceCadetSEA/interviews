from heapq import heappush, heappop, heappushpop, heapreplace
from collections import Counter
from math import sqrt

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


if __name__ == "__main__":
    print(reorganize_string("aaabc"))
