from heapq import heappush, heappop, heappushpop, heapreplace
from collections import Counter

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
        pass

    return res


if __name__ == "__main__":
    print(
        top_k_frequent(
            [6, 0, 1, 4, 9, 7, -3, 1, -4, -8, 4, -7, -3, 3, 2, -3, 9, 5, -4, 0], 6
        )
    )
