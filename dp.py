import math


def coin_change(coins, total):
    """
    time complexity: O(n * m), where n = total and m = num of coins
    space complexity:
    """
    if total < 1:
        return 0
    # memoization
    counter = [math.inf for _ in range(total)]  # O(n)
    return _coin_change_rec(
        coins, total, counter
    )  # also calling recursive function O(n+) times


def _coin_change_rec(coins, total, counter):
    if total < 0:
        return -1
    if total == 0:
        return 0
    # check for a memoized result
    if not math.isinf(counter[total - 1]):
        return counter[total - 1]
    # otherwise, we haven't seen the result yet
    minimum = math.inf
    for s in coins:  # looping over coins O(m)
        # Min(total) = Min(total − C) + 1
        result = _coin_change_rec(coins, total - s, counter)  # total decremented O(n)
        if result >= 0 and result < minimum:
            minimum = 1 + result

    counter[total - 1] = minimum if not math.isinf(minimum) else -1
    return counter[total - 1]


def longest_subsequence_bottom_up(nums):
    """
    This is the final version of the solution from educative.io. 
    It runs in O(n^2)
    """
    n = len(nums)
    cache = [1 for _ in range(n)]

    for curr in range(1, n):
        for prev in range(curr):
            if nums[prev] < nums[curr]:
                cache[curr] = max(cache[curr], cache[prev] + 1)

    return max(cache)


def longest_subsequence(nums):
    """
    YOU DO NOT USE BS ON THE NUMS SEQUENCE...

    You use BS in order to find the index that the value WOULD have
    if it were in a sorted list.

    O(n log n)
    """
    tails = []
    for num in nums:
        pos = _longest_subsequence_bst(num, tails)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)


def _longest_subsequence_bst(target, tails):
    left, right = 0, len(tails)
    while left < right:
        mid = (left + right) // 2
        if tails[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left


def climb_stairs(n):
    """
    space/time: O(n)
    """
    lookup = [0] * n
    for i in range(n - 1, -1, -1):
        if i == n:
            lookup[i] = 1
        if i == n - 1:
            lookup[i] = 1
        else:
            lookup[i] = lookup[i + 1] + lookup[i + 2]
    return lookup[0]


def climb_stairs(n):
    cache = [-1] * (n + 1)
    return _climb_stairs_rec(n, cache)


def _climb_stairs_rec(n, cache):
    # base cases
    if n == 0:
        return 1
    elif n == 1:
        return 1
    # check cache
    if cache[n] > 0:
        return cache[n]
    # calculate new values
    cache[n] = _climb_stairs_rec(n - 1) + _climb_stairs_rec(n - 2)
    return cache[n]


if __name__ == "__main__":
    print(climb_stairs(5))
