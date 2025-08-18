from functools import cache
import math
from typing import List


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


def climb_stairs_bottom_up(n):
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


def climb_stairs_top_down(n):
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
        for j in range(n - 1, -1, -1):
            # is there a way to update this so we don't visit every one?
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
    """
    This is a beast of a solution... Claude recommends the "expand-from-center"
    approach as it uses constant space and is "more intuitive".
    """
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


def longest_palindromic_subsequence_center_expansion(s):
    if not s:
        return ""

    start, max_len = 0, 1
    for i in range(len(s)):
        # base case 1 - odd palindrome
        len1 = expand_around_center(s, i, i)
        # base case 2 - even palindrome
        len2 = expand_around_center(s, i, i + 1)

        current_max = max(len1, len2)
        if current_max > max_len:
            # calculate our start value
            start = i - (current_max - 1) // 2

    return s[start : start + max_len]


def expand_around_center(s, left, right):
    while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
    return right - left - 1  # length of palindrome


def max_product(nums):
    # O(N) time, O(1) space
    if len(nums) == 0:
        return 0
    max_so_far = min_so_far = nums[0]
    result = max_so_far

    for i in range(1, len(nums)):
        # second variable for curr_max_so_far is needed to prevent the
        # computation of max_so_far from overwriting the previous value before
        # we use it in the min_so_far calculation.
        curr_max_so_far = max_so_far
        max_so_far = max(nums[i], nums[i] * max_so_far, nums[i] * min_so_far)
        min_so_far = min(nums[i], nums[i] * min_so_far, nums[i] * curr_max_so_far)
        result = max(result, max_so_far)

    return result


def count_palindromic_substrings_middle_out(s):
    """
    This O(N^2) time and O(1) space
    """
    if not s:
        return 0

    count = 0
    # expand out pattern with odd-sized palindromes
    for i in range(len(s)):
        count += expand_from_center(s, i, i)
    # expand out pattern with even-sized palindromes
    for i in range(len(s) - 1):
        count += expand_from_center(s, i, i + 1)

    return count


def expand_from_center(s, left, right):
    curr_count = 0
    while left >= 0 and right < len(s) and s[left] == s[right]:
        curr_count += 1
        left -= 1
        right += 1
    return curr_count


def count_palindromic_substrings(s):
    """
    This solution is both O(N^2) time and space complexity
    """
    if not s:
        return 0
    count = 0

    dp = [[False for _ in range(len(s))] for _ in range(len(s))]
    # Base case 1: single letters
    for i in range(len(s)):
        dp[i][i] = True
        count += 1
    # Base case 2: pairs of letters
    for i in range(len(s) - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            count += 1
    # Iterate over a gradually increasing window of characters
    for length in range(3, len(s) + 1):
        i = 0  # i represents our starting index, while length is our window
        for j in range(length - 1, len(s)):
            dp[i][j] = s[i] == s[j] and dp[i + 1][j - 1]
            if dp[i][j]:
                count += 1
            i += 1
    return count


def unique_paths(m, n):
    """
    O(M x N) time and space complexity
    """
    if m == 0 or n == 0:
        return 0

    # This creation of the dp cache is much simplier, it precludes our need to
    # do initial passes that set row and columns as 1.
    cache = [[1 for _ in range(n)] for _ in range(m)]

    # It's easier to iterate from the front and work through the entire matrix
    for row in range(1, m):
        for col in range(1, n):
            cache[row][col] = cache[row][col - 1] + cache[row - 1][col]

    return cache[m - 1][n - 1]


def unique_paths_rolling_array(m, n):
    """
    O(M x N) time, O(N) space solution

    Very clever approach that reuses a single list of N elements to keep a
    rolling total and computes the next value in the sequence using the
    previous column value and the updated current column.
    """
    rolling_array = [1] * n

    for _ in range(1, m):
        for col in range(1, n):
            rolling_array[col] += rolling_array[col - 1]

    return rolling_array[-1]


def word_break(s, word_dict):
    """
    O(N^2) time, O(N) space

    Take another look at how it is reusing precomputed values...
    WE NEED TO LEARN THE PATTERN
    """
    if not s:
        return True
    dp = [False for _ in range(len(s) + 1)]
    dp[0] = True
    # These loops find all possible prefixes of the input string
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_dict:
                dp[i] = True
                break
    return dp[len(s)]


def num_of_decodings(decode_str):
    """
    time: O(N)
    space: O(N)
    """
    if not decode_str or decode_str[0] == "0":
        return 0
    n = len(decode_str)
    res = [0] * (n + 1)
    # be explicit about the base cases...
    # res[0] represents the empty string
    res[0] = 1
    # res[1] represents the single digit string which must be valid if the
    # first element in decode_str is not 0.
    if decode_str[0] > "0":
        res[1] = 1
    # with our two base casses, we can start iterating
    for i in range(2, n + 1):
        # like climbing stairs, we look back one and two places in our cache.
        # first we eval the single character
        first = decode_str[i - 1 : i]
        # then we evaluate two character strings
        second = decode_str[i - 2 : i]
        # separate rules for the single and double digit strings
        if first > "0":
            res[i] = res[i - 1]
        if "10" <= second <= "26":
            res[i] += res[i - 2]
    return res[n]


def max_sub_array(nums):
    if not nums:
        return 0
    n = len(nums)
    res = [-math.inf] * (n + 1)
    res[0] = nums[0]
    for i in range(1, n):
        res[i] = max(nums[i], nums[i] + res[i - 1])
    return max(res)


def max_sub_array_constant(nums):
    """
    Constant time converts our list of stored results for the last subarray
    max to a
    """
    if not nums:
        return 0
    n = len(nums)
    curr_max = nums[0]
    curr_subarray_max = nums[0]
    for i in range(1, n):
        curr_subarray_max = max(nums[i], nums[i] + curr_subarray_max)
        curr_max = max(curr_max, curr_subarray_max)
    return curr_max


def calculate_minimum_hp(dungeon):
    """
    Leetcode problem:
    https://leetcode.com/problems/dungeon-game/

    This was a hard dynamic programming question
    """
    if not dungeon or len(dungeon[0]) == 0:
        return 0
    if len(dungeon[0]) == 1:
        return abs(dungeon[0][0]) + 1

    m = len(dungeon)
    n = len(dungeon[0])
    # too much pattern matching without understanding why
    # when we instantiated our table (dp) to m + 1 and n + 1
    dp = [[0 for _ in range(n)] for _ in range(m)]
    dp[m - 1][n - 1] = max(1, 1 - dungeon[m - 1][n - 1])

    # two base cases to consider, when we can only move right or down
    # aka - the edges of the matrix
    for i in range(m - 2, -1, -1):
        dp[i][n - 1] = max(1, dp[i + 1][n - 1] - dungeon[i][n - 1])
    for j in range(n - 2, -1, -1):
        dp[m - 1][j] = max(1, dp[m - 1][j + 1] - dungeon[m - 1][j])

    # Looping from m - 2 and n - 2
    for i in range(m - 2, -1, -1):
        for j in range(n - 2, -1, -1):
            dp[i][j] = max(
                1, min(dp[i + 1][j] - dungeon[i][j], dp[i][j + 1] - dungeon[i][j])
            )

    return dp[0][0]


def calculate_min_hp_top_down(dungeon):
    m, n = len(dungeon), len(dungeon[0])
    dp = [[-1] * n for _ in range(m)]
    return _calc_min_hp_rec(0, 0, dungeon, dp)


def _calc_min_hp_rec(i, j, dungeon, dp):
    m, n = len(dungeon), len(dungeon[0])
    if i == m - 1 and j == n - 1:
        return max(1, 1 - dungeon[i][j])
    if i >= m or j >= n:
        return float("inf")
    if dp[i][j] != -1:
        return dp[i][j]

    right = _calc_min_hp_rec(i, j + 1, dungeon, dp)
    down = _calc_min_hp_rec(i + 1, j, dungeon, dp)

    min_health = min(right, down) - dungeon[i][j]
    dp[i][j] = max(1, min_health)

    return dp[i][j]


def find_max_knapsack_profit(capacity, weights, values):
    if not weights or capacity < min(weights):
        return 0
    w = len(weights)
    dp = [0] * (capacity + 1)

    for i in range(1, w + 1):
        lower_limit = weights[i - 1]
        for j in range(capacity, lower_limit - 1, -1):
            dp[j] = max(dp[j], dp[j - lower_limit] + values[i - 1])

    return dp[capacity]


def find_tribonacci_naive(n):
    """
    Naive recursive solution
    """
    if n == 0:
        return 0
    elif n == 1:
        return 1
    elif n == 2:
        return 1
    else:
        return (
            find_tribonacci_naive(n - 3)
            + find_tribonacci_naive(n - 2)
            + find_tribonacci_naive(n - 1)
        )


def find_tribonacci_top_down(n):
    """
    Linear DP
    """
    dp = {}
    return find_tribonacci_helper(n + 1, dp)


def find_tribonacci_helper(n, dp):
    if n <= 1:
        return 0
    elif n == 2:
        return 1
    elif n == 3:
        return 1

    if n in dp:
        return dp[n]

    else:
        dp[n] = (
            find_tribonacci_helper(n - 3, dp)
            + find_tribonacci_helper(n - 2, dp)
            + find_tribonacci_helper(n - 1, dp)
        )
        return dp[n]


def find_tribonacci_bottom_up(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = 1
    dp[2] = 1

    for i in range(3, n + 1):
        dp[i] = dp[i - 3] + dp[i - 2] + dp[i - 1]
        
    return dp[n]


def find_tribonacci_constant(n):
    if n < 3:
        return 1 if n > 0 else 0
    
    first, second, third = 0, 1, 1
    for _ in range(2, n - 2):
        first, second, third = second, third, first + second + third
    return third


def find_fibonacci_constant(n):
    if n < 2:
        return 1 if n > 0 else 0
    first, second = 0, 1
    for _ in range(1, n - 1):
        first, second = second, first + second
    return second


def pascals_triangle(n):
    """
    Where n is the number of rows requested.
    
    Output: List[List[int]]
    """
    triangle = []
    for i in range(n):
        row = [1]
        if i > 0:
            prev_row = triangle[i - 1]
            for j in range(1, i):
                row.append(prev_row[j - 1] + prev_row[j])
        if i > 0:
            row.append(1)
        triangle.append(row)
    return triangle


def beautifulNumbers(l: int, r: int) -> int:
    """
    leetcode (hard): https://leetcode.com/problems/count-beautiful-numbers/description/
    """
    return count_beautiful(r) - count_beautiful(l - 1)

def count_beautiful(n):
    digits = list(map(int, str(n)))

    @cache
    def dp(pos, sum_, product, tight, leading_zero):
        if pos == len(digits):
            return int(not leading_zero and sum_ > 0 and product % sum_ == 0)
        limit = digits[pos] if tight else 9
        count = 0
        # investigate the use of "tight" to mark whether we are in the upper bound or not
        for d in range(limit + 1):
            new_tight = tight and (d == limit)
            new_leading_zero = leading_zero and d == 0
            new_sum = sum_ + d
            new_product = product if new_leading_zero else (product * d if d != 0 else 0)
            count += dp(pos + 1, new_sum, new_product, new_tight, new_leading_zero)

        return count

    return dp(0, 0, 1, True, True)


def len_of_diagonal(grid: List[List[int]]) -> int:
    """
    leetcode (hard):
    https://leetcode.com/problems/length-of-longest-v-shaped-diagonal-segment/description/
    
    Pretty straightforward DP problem. memoization occurs on the recursive function using @cache
    """
    m, n = len(grid), len(grid[0])

    @cache
    def len_of_v_helper(row, col, can_turn, direction, target):
        # make sure we are in bounds...
        # return 0 if not
        if row < 0 or row > m - 1 or col < 0 or col > n - 1:
            return 0
        
        if grid[row][col] != target:
            return 0

        # if we are in bounds, flip the target and check the 4 possible directions
        #   and rotating or not.
        directions = {0: (-1, -1), 1: (-1, 1), 2: (1, 1), 3: (1, -1)}
        curr_direction = directions[direction]
        new_direction = direction + 1 if direction < 3 else 0
        turn_direction = directions[new_direction]
        new_target = 2 if target == 0 else 0
        if can_turn:
            return 1 + max(
                len_of_v_helper(
                    row + curr_direction[0],
                    col + curr_direction[1],
                    can_turn,
                    direction,
                    new_target,
                ),
                len_of_v_helper(
                    row + turn_direction[0],
                    col + turn_direction[1],
                    False,
                    new_direction,
                    new_target,
                ),
            )
        return 1 + len_of_v_helper(
            row + curr_direction[0],
            col + curr_direction[1],
            can_turn,
            direction,
            new_target,
        )

    longest = 0
    for row in range(m):
        for col in range(n):
            if grid[row][col] == 1:
                longest = max(
                    longest,
                    1
                    + max(
                        len_of_v_helper(row - 1, col - 1, True, 0, 2),
                        len_of_v_helper(row - 1, col + 1, True, 1, 2),
                        len_of_v_helper(row + 1, col + 1, True, 2, 2),
                        len_of_v_helper(row + 1, col - 1, True, 3, 2),
                    ),
                )
    return longest


if __name__ == "__main__":
    print(
        len_of_diagonal(
            [
                [2, 2, 2, 2, 2],
                [2, 0, 2, 2, 0],
                [2, 0, 1, 1, 0],
                [1, 0, 2, 2, 2],
                [2, 0, 0, 2, 2],
            ]
        )
    )
