from heapq import heappop, heappush


"""
FUNCTION kWayMerge(lists):
  # Initialize a min heap
  minHeap = new MinHeap()

  # Insert the first element of each list into the heap
  FOR each list in lists:
    IF list is not empty:
      minHeap.insert((list[0], listIndex, elementIndex))

  # Initialize an output list
  mergedList = []

  WHILE minHeap is not empty:
    # Extract the smallest element
    (value, listIndex, elementIndex) = minHeap.extractMin()
    
    # Add it to the merged list
    mergedList.append(value)

    # Insert the next element from the same list
    IF elementIndex + 1 < length of lists[listIndex]:
      minHeap.insert((lists[listIndex][elementIndex + 1], listIndex, elementIndex + 1))

  RETURN mergedList
"""