"""
Valid palindrome:
https://www.educative.io/courses/grokking-coding-interview-in-python/valid-palindrome
"""


def is_palindrome(s):
    left, right = 0, len(s) - 1

    while left <= right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if s[left].lower() != s[right].lower():
            return False
        left += 1
        right -= 1
    return True


"""
Three sum:
https://www.educative.io/courses/grokking-coding-interview-in-python/3sum
"""


def three_sum(array):
    array.sort()
    result = set()
    i = 0
    # i < length of array to avoid index error and array[i] <= 0 because we
    # cannot get a sum of 0 if the value is greater than 0 in our sorted list.
    while i < len(array) and array[i] <= 0:
        if 0 < i and array[i] == array[i - 1]:
            i += 1
            continue
        left, right = i + 1, len(array) - 1
        while left < right:
            curr_sum = sum([array[i], array[left], array[right]])
            if curr_sum == 0:
                result.add((array[i], array[left], array[right]))
                left = increment_left(array, left, right)
                right = decrement_right(array, left, right)
            elif curr_sum < 0:
                left = increment_left(array, left, right)
            else:
                right = decrement_right(array, left, right)
        i += 1
    return list(result)


def increment_left(nums, left, right):
    curr_left = left
    left += 1
    while left < right and nums[curr_left] == nums[left]:
        left += 1
    return left


def decrement_right(nums, left, right):
    curr_right = right
    right -= 1
    while left < right and nums[curr_right] == nums[right]:
        right -= 1
    return right


"""
Remove nth node from end of list:
https://www.educative.io/courses/grokking-coding-interview-in-python/remove-nth-node-from-end-of-list
"""


def remove_nth_last_node(head, n):
    right = head
    for i in range(n):
        right = right.next

    if not right:
        return head.next

    left = head
    while right.next:
        right = right.next
        left = left.next

    left.next = left.next.next
    return head


"""
Sort colors:
https://www.educative.io/courses/grokking-coding-interview-in-python/sort-colors
"""


def sort_colors(colors):
    left, right = 0, len(colors) - 1
    i = 0
    # Three pointers... left, right, and i
    # We loop until i is greater than the right pointer...
    # Values in colors are either 0, 1, or 2.
    while i <= right:
        # If the color value we have is 0, we can safely swap with the left
        # pointer which starts at the 0th index. This way, a zero is guaranteed
        # to be at the front of the list. We then increment both i and left to
        # move on to the next value AND move the 0-color wall one forward.
        if colors[i] == 0:
            colors[i], colors[left] = colors[left], colors[i]
            left += 1
            i += 1
        # If the color value at i is 2, we swap with the right pointer which
        # starts at the end of the color list. This way, a 2 is guaranteed to
        # be at the end of the list. Then we decrement the right value, but
        # leave the ith pointer alone because we don't know what color value
        # we have now swapped with, so we need to check i at its current value
        # again.
        elif colors[i] == 2:
            colors[i], colors[right] = colors[right], colors[i]
            right -= 1
        # Since we're moving 0's to the front and 2's to the back, coming
        # across a 1 should mean we just increment i until we get to a value
        # we can move.
        else:
            i += 1
    return colors


"""
Reverse words in a string:
https://www.educative.io/courses/grokking-coding-interview-in-python/reverse-words-in-a-string
"""


def reverse_words(sentence):
    # Naive approach would be to convert string into a list of words with split.
    # Then we would reverse the string and concatenate. But this is 2 pointers.

    # I spent too long trying to mutate the string inplace.
    # They want you to use a list and then reverse the elements in the list
    # with left and right pointers that increment by one element each time.
    # This way you don't have to do the complicated math of understanding where
    # to add the start and end that you've already processed.

    # Using lists makes this really trivial.
    res = sentence.split()
    left, right = 0, len(res)

    while left <= right:
        res[left], res[right] = res[right], res[left]

        left += 1
        right -= 1

    return " ".join(res)


"""
Count pairs whose sum is less than target
https://www.educative.io/courses/grokking-coding-interview-in-python/count-pairs-whose-sum-is-less-than-target
"""


def count_pairs(nums, target):
    # sort
    nums.sort()
    left, right = 0, len(nums) - 1
    count = 0

    while left < right:
        if nums[left] + nums[right] < target:
            # This is the key that makes it not O(N^2)...
            # We can count all pairs between a valid pair without having to
            # include them in an iteration. This allows us to find all pairs
            # between the left..right values and then just continue moving
            # forward with left + 1 and the current right value.
            count += right - left
            left += 1
        else:
            right -= 1

    return count


if __name__ == "__main__":
    print(reverse_words("   Greeting123   "))
