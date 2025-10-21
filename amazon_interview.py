from typing import List
from heapq import heappop, heappush
from collections import defaultdict
import math


class Node:
    def __init__(self, value):
        self.data = value
        self.next = None


"""
k-largest elements in array

heap!

Use a min heap and compare new values to minimum value in heap.
If new value is greater than minimum value in heap, remove min value from heap
and replace it with the larger value.
In the end, you will have the three largest values in reverse order from how
we want to return it.
"""


def k_largest(nums: List[int], k: int) -> List[int]:
    heap = []

    for i in range(k):
        heappush(heap, nums[i])

    for i in range(k, len(nums)):
        if heap[0] < nums[i]:
            heappop(heap)
            heappush(heap, nums[i])

    return sorted(heap, reverse=True)


"""
pythagorian triplet in array

Best case is N^2 since we have to check all the values. We do some precomputing
to get all the squares and then convert them into a set.
We can then add squares together as we iterate in nested for loops and
terminate as soon as we find a value in our squares set.
"""


def pythagorean_triplet(arr: List[int]) -> bool:
    # square the entire list
    squares = [num**2 for num in arr]  # O(N)
    squaresset = set(squares)  # O(N)

    for i in range(len(squares)):
        for j in range(i + 1, len(squares)):  # O(N^2)
            if squares[i] + squares[j] in squaresset:  # O(1)
                return True

    return False


"""
convert binary tree to doubly-linked-list

Key insight is using two class-variables as pointers to track the new head node
and keep a cache of the previous node as we recurse down the tree.

self.prev will be None until we get to the left-most node of the tree. Once
there, we set self.head to this node. We then cache this node as self.prev for
the first time and recurse down the right subtree.

Now that self.prev is no longer null, we will set self.prev.right as current
node and node.left as self.prev to create the double link.
"""


class bToDLL:
    def solution(self, root):
        self.head = None
        self.prev = None
        self.bToDLL_rec(root)
        return self.head

    def bToDLL_rec(self, node):
        if not node:
            return node

        self.bToDLL_rec(node.left)
        if not self.prev:
            self.head = node
        else:
            self.prev.right = node
            node.left = self.prev
        self.prev = node
        self.bToDLL_rec(node.right)


"""
lowest common ancestor in a tree

Classic problem we have seen before. Keep three pointers at each node to denote
whether the values we are seeking are found in the current node or when
recursing down the left and right subtrees. The first time two of these three
values are True, we've found the lowest common ancestor.
"""


def lca(root, n1, n2):
    return lca_rec(root, n1, n2)


def lca_rec(node, n1, n2):
    if not node:
        return node

    mid = node.data in [n1, n2]
    left = lca_rec(node.left, n1, n2)
    right = lca_rec(node.right, n1, n2)

    if (mid and left) or (mid and right) or (left and right):
        return node

    if mid:
        return mid
    if left:
        return left
    if right:
        return right

    return None


"""
special stack data structure
"""


class SpecialStack:
    def __init__(self):
        self.st = []

    def push(self, element):
        if len(self.st) == 0:
            self.st.append((element, element))
        else:
            self.st.append((element, min(element, self.get_min())))

    def pop(self):
        if self.is_empty():
            return -1
        return self.st.pop()

    def get_min(self):
        if self.is_empty():
            return -1
        return self.st[-1][1]

    def is_empty(self):
        return len(self.st) == 0

    def is_full(self, n):
        return len(self.st) == n


"""
reverse a linked list in groups of k

We've seen this before too. The key is to use a dummy node as the new head and
then a ptr node to traverse the groups of length k. Then its the classic 
reverse a linked list algorithm on those groups, managing the pointers.

**CAUTION** The version we have seen did not reverse a group that was not full
at the end. The question found on the geeks for geeks site explicitly told us
to reverse even a partial group.
"""


def reverse_linked_list_groups(head: Node, k):
    if not head:
        return head

    curr = head
    new_head = None
    tail = None

    while curr:
        group_head = curr
        prev = None
        next = None
        count = 0

        while curr and count < k:
            next = curr.next
            curr.next = prev
            prev = curr
            curr = next
            count += 1

        if not new_head:
            new_head = prev

        if tail:
            tail.next = prev

        tail = group_head

    return new_head


"""
add two linked lists

With potential leading zeroes and lists of unequal length.
Recursive solution is the best to avoid reversing the list...

Recursive solution relies on maintaining two variables, carry and diff. Diff 
allows us to traverse the longer of the two lists until we are on the same 
level, whereas carry denotes whether the previous sum in a lower digit was 
greater than 10 and we need to carry the one.

The suggested strategy is to reverse both lists and add from the back.
Then you reverse the final list after generating it.
"""


def add_two_lists(head1, head2):

    def get_length(llist):
        length = 0
        while llist:
            length += 1
            llist = llist.next
        return length

    def add_two_lists_rec(longer, shorter, carry, diff):
        if not longer:
            return (None, 0)
        if diff > 0:
            node, carry = add_two_lists_rec(
                longer.next, shorter, carry, diff - 1)
            value = longer.data + carry
        else:
            node, carry = add_two_lists_rec(
                longer.next, shorter.next, carry, diff)
            value = longer.data + shorter.data + carry

        digit = value % 10
        carry = value // 10

        new_node = Node(digit)
        new_node.next = node
        return (new_node, carry)

    left_len = get_length(head1)
    right_len = get_length(head2)
    if left_len > right_len:
        longer = head1
        shorter = head2
        diff = left_len - right_len
    else:
        longer = head2
        shorter = head1
        diff = right_len - left_len

    node, carry = add_two_lists_rec(longer, shorter, 0, diff)

    if carry:
        head = Node(1)
        head.next = node
    else:
        head = node

    while head.data == 0:
        head = head.next

    return head


"""
rotate matrix 90 degrees counter-clockwise

** CAUTION ** make sure you set up the top left/right bottom left/right
variables correctly! The rows and cols usage for the actual indexing is not
straight forward.
"""
def rotate_matrix(matrix):
    n = len(matrix)
    if n < 2:
        return matrix

    for row in range(n - 1 // 2):
        for col in range(row, n - 1 - row):
            top_left = (row, col)
            top_right = (col, n - 1 - row)
            bottom_left = (n - 1 - col, row)
            bottom_right = (n - 1 - row, n - 1 - col)

            # swap top left / top right
            matrix[top_left[0]][top_left[1]], matrix[top_right[0]][top_right[1]] = (
                matrix[top_right[0]][top_right[1]
                                     ], matrix[top_left[0]][top_left[1]]
            )
            # swap top right / bottom right
            matrix[top_right[0]][top_right[1]], matrix[bottom_right[0]][bottom_right[1]] = (
                matrix[bottom_right[0]][bottom_right[1]
                                        ], matrix[top_right[0]][top_right[1]]
            )
            # swap bottom right / bottom left
            matrix[bottom_right[0]][bottom_right[1]], matrix[bottom_left[0]][bottom_left[1]] = (
                matrix[bottom_left[0]][bottom_left[1]
                                       ], matrix[bottom_right[0]][bottom_right[1]]
            )
    return matrix


"""
Given an array of integers of at least length > 15, allow the first 15 integers 
to pass without inspection. Starting at the 16th integer: Return the first 
integer in the list that is not a sum of any two integers found in the PREVIOUS 
15 integers in the list. 

If all integers in the array after the first 15 pass 
this test, return 0. 

The array only includes positive integers, but is not 
guaranteed to be strictly increasing or decreasing. The array can include 
duplicates, but will not include 0.

Using a set/hashmap to contain values and their count.
"""
def sum_of_two(arr):
    hashmap = defaultdict(int)
    for i in range(len(arr[:15])):
        hashmap[arr[i]] += 1
    
    for i in range(15, len(arr)):
        curr_val = arr[i]
        found = False
        
        for val in hashmap.keys():
            diff = abs(curr_val - val)
            if val == diff:
                found = hashmap[val] >= 2
                break
            else:
                if diff in hashmap:
                    found = True
                    break
        if not found:
            return curr_val
        
        # slide the window
        old_num = arr[i - 15]
        hashmap[old_num] -= 1
        if hashmap[old_num] == 0:
            del hashmap[old_num]
        hashmap[curr_val] += 1
    return 0        


"""
house robber

Classic DP problem that is easiest to understand from the bottom-up approach.
"""
def house_robber(arr: List[int]):
    n = len(arr)
    if n < 3:
        return max(arr)
    res = [0] * n
    res[0] = arr[0]
    res[1] = max(res[0], arr[1])
    
    for i in range(2, n):
        res[i] = max(arr[i] + res[i - 2], res[i - 1])
        
    return res[-1]


"""
minimum edit distance

This is a hard one. I need to watch a video on this one and other similar
dynamic programming questions that are looking for common substrings, etc.
"""
def edit_distance(s1: str, s2: str) -> int:
    if s1 == s2:
        return 0
    m = len(s1)
    n = len(s2)
    
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Set our default values for each string alone
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                dp[i + 1][j + 1] = dp[i][j]
            else:
                dp[i + 1][j + 1] = 1 + min(dp[i + 1][j], dp[i][j + 1], dp[i][j])

    return dp[m][n]


"""
assembly line

Very confusing prompt. But ultimately a shortest path through two assembly
lines where we need to explore both paths.
"""
def assembly_line(n: int, a: List[List[int]], T: List[List[int]], e: List[int], x: List[int]):
    dp = [[0] * n for _ in range(2)]
    
    dp[0][0] = e[0] + a[0][0]
    dp[1][0] = e[1] + a[1][0]
    
    for i in range(1, n):
        dp[0][i] = min(
            dp[0][i - 1] + a[0][i],
            dp[1][i - 1] + T[1][i] + a[0][i]
        )
        dp[1][i] = min(
            dp[1][i - 1] + a[1][i],
            dp[0][i - 1] + T[0][i] + a[1][i]
        )
        
    return min(dp[0][n - 1] + x[0], dp[1][n - 1] + x[1])


"""
ACTUAL INTERVIEW QUESTION


Given an ip cidr range, we have to write api to allocate IP address from it.
For example, given a cidr 10.10.0.0/16, a call to the allocation method will 
yield, say, 10.10.255.1, next call can yield 10.10.240.1.

To make things easier to code, lets use an integer range instead, like(9, 80)

Now lets consider that the user of the IP will release them and we can reuse 
them for the next allocation. Doesnt mean we have to, but we can.

Always return the lowest available IP


def main():
    cidr = MyCIDR(9, 80)
    ip = cidr.allocate_ip()  # Return 9
    ip2 = cidr.allocate_ip()  # Return 10
    //I do my work
    cidr.release_ip(ip2)
    ip2 = cidr.allocate_ip()  # Potentially return 10
"""


class MyCIDR:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop
        self.last_used_resource = start
        self.released_ip_addresses = []
        
    def create_from_ip_addresses(self, list_of_ips, start, stop):
        pass
    
    def allocate_ip(self) -> int:
        # get an ip value
        # TODO (atheis4): investigate randomness
        if self.released_ip_addresses:  # O(1)
            ip_address = heappop(self.released_ip_addresses)  # O(log N)
            return ip_address
        if self.last_used_resource == self.stop:
            raise Exception("No more IP Addresses to allocate")
        # invariant where last_used_resource >= value in released ip address heap
        ip_address = self.last_used_resource
        self.last_used_resource += 1
        return ip_address
        
    def release_ip(self, ip_address):
        # cache the removed ip address
        heappush(self.released_ip_addresses, ip_address)  # O(log N)
        # update allocate_ip address to first check the release cache


if __name__ == "__main__":
    print(edit_distance("geek", "gesek"))
