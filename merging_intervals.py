from typing import List
from heapq import heappop, heappush


def insert_interval(existing_intervals, new_interval):
    output = []

    i = 0
    # iterate over existing intervals and stop when if the next existing
    # interval's end value is greater than the new interval's start value.
    while i < len(existing_intervals) and existing_intervals[i][1] < new_interval[0]:
        output.append(existing_intervals[i])
        i += 1

    # the merge process... iterate over the existing intervals again and this
    # time include all existing intervals that start before or equal to the
    # merged end. Merge these by taking the min and max of start and stop values
    merged_start, merged_end = new_interval
    while i < len(existing_intervals) and existing_intervals[i][0] <= merged_end:
        merged_start = min(merged_start, existing_intervals[i][0])
        merged_end = max(merged_end, existing_intervals[i][1])
        i += 1
    # append the newly merged item into our output... this is either just the
    # new interval or the end result of merging all intervals together.
    output.append([merged_start, merged_end])

    # continue and finish iterating over the existing intervals to add the rest
    # of the existing values that have no overlap with the new interval or
    # merged values.
    while i < len(existing_intervals):
        output.append(existing_intervals[i])
        i += 1

    return output


def merge_intervals(intervals): 
    intervals.sort(key=lambda x: x[0])
    output = []

    output.append(intervals[0])
    for i in range(1, len(intervals)):
        start, stop = intervals[i]
        merged_start, merged_stop = output[-1]
        if start <= merged_stop:
            # we actually don't need this check since we already sorted values
            # merged_start = min(merged_start, start)
            merged_stop = max(merged_stop, stop)
            output[-1] = [merged_start, merged_stop]
        else:
            output.append(intervals[i])

    return output


def attend_all_meetings(intervals):
    if not intervals:
        return True

    intervals.sort(key=lambda x: x[0])
    _, curr_stop = intervals[0]

    for i in range(1, len(intervals)):
        start, stop = intervals[i]
        if start < curr_stop:
            return False
        curr_stop = stop
    
    return True


def find_sets(intervals):
    intervals.sort(key=lambda x: x[0])
    min_heap = []
    heappush(min_heap, intervals[0][1])

    for i in range(1, len(intervals)):
        start, stop = intervals[i]
        if min_heap and min_heap[0] <= start:
            heappop(min_heap)
        heappush(min_heap, stop)

    return len(min_heap)


def remove_min_intervals(intervals):
    intervals.sort(key=lambda x: x[1])
    remove = 0

    curr_stop = intervals[0][1]
    for i in range(1, len(intervals)):
        if intervals[i][0] < curr_stop:
            remove += 1
        else:
            curr_stop = intervals[i][1]

    return remove


def partition_labels(s: str) -> List[int]:
    """
    https://leetcode.com/problems/partition-labels/submissions/1722563883/
    """
    char_to_idx = {}
    for i, char in enumerate(s):  # O(N)
        if char not in char_to_idx:
            char_to_idx[char] = [i]
        else:
            char_to_idx[char].append(i)

    for key, vals in char_to_idx.items():  # O(k)
        char_to_idx[key] = [min(vals), max(vals)]

    intervals = sorted(list(char_to_idx.values()), key=lambda x: x[0])  # O(k log k)

    curr_start, curr_stop = intervals[0]
    res = []
    for i in range(1, len(intervals)):  # O(k)
        start, stop = intervals[i]
        if start < curr_stop:
            curr_stop = max(curr_stop, stop)
            curr_start = min(curr_start, start)
        else:
            res.append(curr_stop + 1 - curr_start)
            curr_stop = stop
            curr_start = start
    res.append(curr_stop + 1 - curr_start)
    return res


def partition_label_2(s):
    """
    This is a pythonic solution that runs in O(N) time, much better than my solution
    https://leetcode.com/problems/partition-labels/solutions/7041333/easy-python3-o-n-time-and-o-1-space-solution/
    """
    maxIndex = {}
    for i in range(len(s)):
        maxIndex[s[i]] = i

    index, ans = 0, []
    while index < len(s):
        length, maxi = 0, maxIndex[s[index]]
        while index <= maxi:
            if maxIndex[s[index]] > maxi:
                maxi = maxIndex[s[index]]
            index += 1
            length += 1
        ans.append(length)
    return ans


def intervals_intersection(interval_list_a, interval_list_b):
    
    return []


if __name__ == "__main__":
    print(partition_label_2("ababcbacadefegdehijhklij"))
