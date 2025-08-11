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


def climb_stairs_top_down(n):
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


def climb_stairs_bottom_up(n):
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
    cache[n] = _climb_stairs_rec(n - 1, cache) + _climb_stairs_rec(n - 2, cache)
    return cache[n]


def combination_sum(nums, target):
    """
    I was just missing the sort before adding to our cache array.
    Sorting before adding allows us to do the check quickly for any duplicate
    entries already.

    Overall: B+
    """
    arr = [[] for _ in range(target + 1)]

    for i in range(1, target + 1):
        for num in nums:
            if num == i:
                arr[i].append([num])
            if num <= i:
                for prev in arr[i - num]:
                    to_add = [num] + prev
                    # This is what i was missing... we were so close...
                    to_add.sort()
                    if to_add not in arr[i]:
                        arr[i].append(to_add)
    return arr[target]


def counting_bits(n):
    """
    There are two patterns to the bits that are key to solving this:
    1. if i is equal, the number of 1 bits at i is equal to the bits at i // 2
    2. if i is odd, we use right bit shift to read the already computed value at
       the shifted bit plus 1.

    EXAMPLE:
    5  => 00000101 >> 00000010 (2)  + 1  =  2
    7  => 00000111 >> 00000011 (3)  + 1  =  3
    9  => 00001001 >> 00000100 (4)  + 1  =  2
    11 => 00001011 >> 00000101 (5)  + 1  =  3
    13 => 00001101 >> 00000110 (6)  + 1  =  3

    This is true, but this result came from the educative.io AI...

    A more intuitive solution would be to continue to use i // 2 for odd
    numbers, then add 1 becuase i // 2 will always take the floor value like
    our even values and the odd number will always have one additional bit
    since it is one larger
    """
    result = [0 for _ in range(n + 1)]
    if n == 0:
        return result
    result[1] = 1
    for i in range(2, n + 1):
        if i % 2 == 0:
            result[i] = result[i // 2]
        else:
            result[i] = result[i // 2] + 1
    return result


def rob_houses(nums):
    """
    So this worked, but it doesn't really encapsulate the dynamic programming
    pattern they want...
    """
    n = len(nums)
    if n == 0:
        return -1  # this should be zero
    if n <= 2:
        return max(nums)

    result = [0] * n
    result[0] = nums[0]
    result[1] = nums[1]  # they want us to track the max of i = 0..1 in this space

    for i in range(3, n):
        result[i] = nums[i] + max(result[i - 2], result[i - 3])

    return max(result)


def rob_houses_dp(nums):
    n = len(nums)
    if n == 0:
        return 0
    if n <= 2:
        return max(nums)

    result = [0] * n
    result[0] = nums[0]
    result[1] = max(nums[1], result[0])

    for i in range(2, n):
        result[i] = max(result[i - 1], nums[i] + result[i - 2])

    return result[-1]


def house_robber(money):
    """
    I tried to do this in a single for loop, and it works, but the use of all
    the +1 and -1 instances does leave this open to off-by-one errors.
    """
    n = len(money)
    if n == 0:
        return 0
    if n <= 2:
        return max(money)

    result_front = [0] * n
    result_back = [0] * n

    result_front[1] = money[0]
    result_back[1] = money[1]

    for i in range(1, n - 1):
        result_front[i + 1] = max(result_front[i], result_front[i - 1] + money[i])
        result_back[i + 1] = max(result_back[i], result_back[i - 1] + money[i + 1])

    max_front = max(result_front)
    max_back = max(result_back)

    return max(max_front, max_back)


def house_robber_solution(money):
    if len(money) == 0 or money is None:
        return 0
    if len(money) == 1:
        return money[0]
    # This is by far cleaner... defining the function to process any list and
    # then calling it on the two sections we split the input data into.
    return max(house_robber_helper(money[:-1]), house_robber_helper(money[1:]))


def house_robber_helper(money):
    # Creates our lookup array
    lookup_array = [0 for x in range(len(money) + 1)]
    # Set our two base cases for 0 and 1
    lookup_array[0] = 0
    lookup_array[1] = money[0]
    # Iterate over the remaining indices through n + 1
    for i in range(2, len(money) + 1):
        # For each index, set the lookup array to the max of either the current
        # house money value + the running max seen at i - 2 or the previous
        # lookup element at i - 1, whichever is greater
        lookup_array[i] = max(money[i - 1] + lookup_array[i - 2], lookup_array[i - 1])

    return lookup_array[len(money)]


def longest_common_subsequence_bottom_up(str1: str, str2: str):
    """
    O(M + N) seems unavoidable since we need to check all characters
    """
    m = len(str1)
    n = len(str2)
    cache = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(
            n - 1, -1, -1
        ):  # is there a way to update this so we don't visit every one?
            if str1[i] == str2[j]:
                cache[i][j] = 1 + cache[i + 1][j + 1]
                continue
            else:
                cache[i][j] = max(cache[i + 1][j], cache[i][j + 1])
    return cache[0][0]


def longest_common_subsequence_top_down(str1, str2):
    """
    I think the recursive solution is just confusing and the call stack
    traversal is a lot less intuitive than the iterative approach above.
    """
    n = len(str1)
    m = len(str2)
    dp = [[-1 for x in range(m)] for y in range(n)]
    return longest_common_subsequence_helper(str1, str2, 0, 0, dp)


# Helper function with updated signature: i is current index in str1, j is current index in str2
def longest_common_subsequence_helper(str1, str2, i, j, dp):
    # Our base case when we've consumed all of one string
    if i == len(str1) or j == len(str2):
        return 0
    # Here is where we only compute if we have to
    elif dp[i][j] == -1:
        if str1[i] == str2[j]:
            # If the strings equal at this point, update cache by adding 1 and
            # then increment both pointers.
            dp[i][j] = 1 + longest_common_subsequence_helper(
                str1, str2, i + 1, j + 1, dp
            )
        else:
            # Otherwise, update cache by checking both possible subsequences
            # incrementing one of i or j
            dp[i][j] = max(
                longest_common_subsequence_helper(str1, str2, i + 1, j, dp),
                longest_common_subsequence_helper(str1, str2, i, j + 1, dp),
            )
    return dp[i][j]


def longest_palindromic_substring(s):
    if len(s) == 0:
        return ""
    n = len(s)
    results = [0, 0]
    cache = [[False for _ in range(n)] for _ in range(n)]
    # Basecase 1: single letters in the string
    for i in range(n):
        cache[i][i] = True
    # Basecase 2: pairs of letters in string
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            cache[i][i + 1] = True
            results = [i, i + 1]
    # iterate over all windows of strings starting from 3 and going up to the
    # full length of the input string.
    for length in range(3, n + 1):
        i = 0
        for j in range(length - 1, n):
            cache[i][j] = cache[i + 1][j - 1] and s[i] == s[j]
            # if we find a palindrome in this process, we overwrite the
            # current results variable which is tracking our index range
            if cache[i][j]:
                results = [i, j]
            i += 1
    return s[results[0] : results[1] + 1]


def max_product(nums):
    if len(nums) == 0:
        return 0
    pass


if __name__ == "__main__":
    print(longest_palindromic_substring("aabbccddccbbae"))
