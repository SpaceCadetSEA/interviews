from heapq import heappop, heappush

from data_structures.linked_list import LinkedList, ListNode

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
    FOR i starting at 0, incrementing by 2, while i is less than the length of lists:
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
    # increment offest when value from nums2 is added to nums1
    p1 = m - 1
    p2 = n - 1
    p = len(nums1) - 1
    while p2 >= 0:
        if p1 < 0 or nums1[p1] <= nums2[p2]:
            nums1[p] = nums2[p2]
            p2 -= 1
            p -= 1
        else:
            nums1[p] = nums1[p1]
            p1 -= 1
            p -= 1
    return nums1


def merge_sorted(nums1, m, nums2, n):
    """
    Their solution is nice because it uses range to count down
    instead of up.
    """
    p1 = m - 1
    p2 = n - 1
    for p in range(n + m - 1, -1, -1):
        if p2 < 0:
            break
        if p1 >= 0 and nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
    return nums1


def merge_k_lists_pairwise(lists):
    if not lists:
        return None
    while len(lists) > 1:
        merged_lists = []
        for i in range(0, len(lists), 2):
            if i + 1 < len(lists):
                merged_lists.append(_merge_two_lists(lists[i], lists[i + 1]))
            else:
                merged_lists.append(lists[i])
        lists = merged_lists
    return lists[0]


def _merge_two_lists(list1, list2):
    merged_list = ListNode(0)
    curr_merged = merged_list
    while list1 and list2:
        if list1.val <= list2.val:
            curr_merged.next = list1
            list1 = list1.next
        else:
            curr_merged.next = list2
            list2 = list2.next
        curr_merged = curr_merged.next

    if not list2:
        curr_merged.next = list1
    else:
        curr_merged.next = list2
    return merged_list.next


def k_smallest_number(lists, k):
    # O((m + k) log m)
    # where m = len(lists)
    min_heap = []
    n = len(lists)
    i = 0
    for i in range(n):
        if len(lists[i]) > 0:
        # list val, list index, sublist index
            heappush(min_heap, (lists[i][0], i, 0))
    curr_val = 0
    for i in range(k):
        if not heap:
            break
        curr_val, list_index, sublist_index = heappop(min_heap)
        if sublist_index < len(lists[list_index]) - 1:
            heappush(
                min_heap,
                (lists[list_index][sublist_index + 1], list_index, sublist_index + 1),
            )
    return curr_val


def k_smallest_pairs(list1, list2, k):
    min_heap = []
    # node(sum of pairs, list1 index, list2 index)
    # knew midpoint values and could determine how many pairs would be needed

    for i, num in enumerate(list1):
        # O(N log N)
        heappush(min_heap, ((list2[0] + num), i, 0))

    result = []
    for _ in range(k):  # O (k log N)
        if not min_heap:
            continue
        (_, first_idx, second_idx) = heappop(min_heap)
        first_val, second_val = list1[first_idx], list2[second_idx]
        result.append([first_val, second_val])
        if second_idx + 1 < len(list2):
            heappush(min_heap, ((list2[second_idx + 1] + list1[first_idx]), first_idx, second_idx + 1))

    # Replace this placeholder return statement with your code
    return result


def k_smallest_pairs_forgetful(l1, l2, k):
    i1 = 0
    i2 = 0
    while i1 * i2 < k and (i1 < len(l1) or i2 < len(l2)):
        if i1 >= len(l1):
            i2 += 1
        elif i2 >= len(l2):
            i1 += 1
        elif l1[i1] <= l2[i2]:
            i1 += 1
        else:
            i2 += 1
    ret = []
    for x in l1[:i1]:  # O(M X N) > O((M X N) log (M + N))
        for y in l2[:i2]:
            ret.append([x, y])
 
    return ret[:k]


def merge_k_lists_heap(lists):
    n = len(lists)
    min_heap = []
    for i in range(len(lists)):
        if lists[i] is None:
            continue
        heappush(min_heap, (lists[i].val, i, lists[i]))

    dummy = ListNode(0)
    curr_node = dummy
    i = n
    while min_heap:
        _, _, node = heappop(min_heap)
        curr_node.next = node
        curr_node = curr_node.next
        if node.next:
            node = node.next
            heappush(min_heap, (node.val, i, node))
        i += 1  # its stupid and i'm mad.

    return dummy.next


if __name__ == "__main__":
    lists = [[2],[1,2,4],[25,56,66,72]]
    linked_lists = [LinkedList(x) for x in lists]
    print(merge_k_lists_heap(linked_lists))
