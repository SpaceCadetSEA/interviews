from collections import defaultdict, Counter
import math

"""
FIXED WINDOW
FUNCTION slidingWindow(arr, k, processWindow):
  # Initialize the window result (sum, product, count, etc.)
  windowResult = INITIAL_VALUE

  # Compute the initial window's result
  FOR i FROM 0 TO k - 1:
    UPDATE windowResult WITH arr[i]

  # Process the first window
  processWindow(windowResult)

  # Slide the window across the array
  FOR i FROM k TO length of arr - 1:
    UPDATE windowResult BY ADDING arr[i]  # Add a new element to the window
    UPDATE windowResult BY REMOVING arr[i - k]  # Remove outgoing element
    processWindow(windowResult)  # Operation on the updated window


DYNAMIC WINDOW
FUNCTION slidingWindow(arr, condition, processWindow):
  left = 0
  windowState = INITIAL_VALUE

  FOR right FROM 0 TO length of arr - 1:
    UPDATE windowState WITH arr[right]  # expand window

    WHILE NOT condition(windowState):   # shrink window if needed
      UPDATE windowState BY REMOVING arr[left]
      left = left + 1

    processWindow(windowState, left, right)
"""


def max_profit(prices):
    # dynamic window
    # start with a current when the first value in the window is less than the
    # next value in the window.
    maxx = 0
    buy = 0
    sell = 1
    while sell < len(prices):
        if prices[buy] <= prices[sell]:
            curr_max = prices[sell] - prices[buy]
            maxx = max(curr_max, maxx)
        else:
            buy = sell

        sell += 1
    return maxx


def find_longest_substring(input_str):
    left = 0
    right = 1
    chars = set()
    chars.add(input_str[left])
    maxx = 1
    while right < len(input_str):
        if input_str[right] in chars:
            chars.remove(input_str[left])
            left += 1
        else:
            chars.add(input_str[right])
            maxx = max(maxx, len(chars))
            right += 1
    return maxx


def longest_repeating_character_replacement(s, k):
    start = 0
    # utilizes a defaultdict of int to maintain our character count of the
    # current window
    char_freq = defaultdict(int)
    most_freq_char = 1
    len_of_max_substring = 1

    # we can dynamically move through our window, but increment right pointer
    # at a fixed rate.
    for end in range(len(s)):
        # add the current edge of window to the character count
        char_freq[s[end]] += 1
        # update our most frequent character with the rightward edge of the
        # window
        most_freq_char = max(most_freq_char, char_freq[s[end]])
        # check the window for validity...
        # length of the window (+1, due to 0 indexing) minus the most
        # frequent character...
        if end - start + 1 - most_freq_char > k:
            # if its in this invalid state, we have to shrink the window
            char_freq[s[start]] -= 1
            start += 1
        # after shrinking the window, we update the length of the max substring
        len_of_max_substring = max(end - start + 1, len_of_max_substring)

    return len_of_max_substring


def min_window(s, t):
    """
    Seems like we just want to maintain a dict where the items in t appear at
    most once...

    while we move our window from left to right, consuming the right value
    and then shrinking on the left side, we keep the
    """
    min_window_size = math.inf
    start = end = 0

    min_window_start = min_window_end = 0

    target_char_count = Counter(t)
    window_char_count = defaultdict(int)

    while end < len(s):
        window_char_count[s[end]] += 1

        # Check min window size if we have what we need
        # TODO: By not keeping the entire window in a dict and only relevant
        # keys, we can greatly simplify our check for inclusion.
        while target_char_count.items() <= window_char_count.items():
            if end - start + 1 < min_window_size:
                min_window_start = start
                min_window_end = end
                min_window_size = end - start + 1
            window_char_count[s[start]] -= 1
            start += 1

        end += 1

    return (
        s[min_window_start : min_window_end + 1] if min_window_size != math.inf else ""
    )


def subset_of_dict(primary, secondary):
    """
    Check if the secondary dictionary is a subset of the primary dictionary
    """
    subset = {}
    for k, v in secondary.items():
        subset[k] = primary[k] <= v

    return all([v for v in subset.values()])


def min_window_official(s, t):
    if not t:
        return ""

    req_count = {}
    window = {}

    for char in t:
        req_count[char] = req_count.get(char, 0) + 1

    current = 0
    required = len(req_count)

    res = [-1, -1]
    res_len = float("inf")

    left = 0
    # Expand window to the right on each iteration
    for right in range(len(s)):
        char = s[right]
        # If the character is needed... add to window hashmap
        if char in req_count:
            window[char] = window.get(char, 0) + 1
            # Manually increment current counter
            if window[char] == req_count[char]:
                current += 1

        # Current counter matched with required, length of req_count
        # This tells us whether we can shrink our window from the left.
        while current == required:
            # If the smaller window is smaller than our previous window seen...
            if right - left + 1 < res_len:
                res = [left, right]
                res_len = right - left + 1
            # Before we shrink the window on the left, we need to process
            # the character we are going to be excluding.
            left_char = s[left]
            if left_char in req_count:
                window[left_char] -= 1
                # If removing the left char from the window means we no longer
                # meet the required count of characters, we need to decrement
                # our current variable and invalidate the while loop and stop
                # shrinking the left side.
                if window[left_char] < req_count[left_char]:
                    current -= 1
            left += 1

    return s[res[0] : res[1] + 1] if res_len != float("inf") else ""


if __name__ == "__main__":
    # longest_repeating_character_replacement("abbcab", 2)
    min_window("ABDFGDCKAB", "ABCD")
