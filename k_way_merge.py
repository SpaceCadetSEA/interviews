from heapq import heappop, heappush


"""
Min Heap approach:

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

================================================================================
Pairwise Merge approach:

FUNCTION pairwiseMerge(lists):
  WHILE length of lists > 1:
    mergedLists = []

    # Merge lists in pairs
    FFOR i starting at 0, incrementing by 2, while i is less than the length of lists:
      IF i + 1 < length of lists:
        mergedLists.append(mergeTwoLists(lists[i], lists[i + 1]))
      ELSE:
        mergedLists.append(lists[i])  # Append last unpaired list

    # Update the list reference
    lists = mergedLists

  RETURN lists[0]
  
FUNCTION mergeTwoLists(list1, list2):
  mergedList = []
  index1, index2 = 0, 0

  # Standard two-way merge
  WHILE index1 < length of list1 AND index2 < length of list2:
    IF list1[index1] < list2[index2]:
      mergedList.append(list1[index1])
      index1 += 1
    ELSE:
      mergedList.append(list2[index2])
      index2 += 1

  # Append remaining elements
  mergedList.extend(list1[index1:])
  mergedList.extend(list2[index2:])

  RETURN mergedList
"""


def merge_sorted(nums1, m, nums2, n):
    # You have to modify nums1 in place
    
    return []