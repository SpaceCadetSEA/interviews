from data_structures.LinkedList import LinkedList, display, ListNode


def fold_linked_list(head: ListNode) -> ListNode:
    if not head:
        return head
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    prev, curr = None, slow

    while curr:
        next = curr.next
        curr.next = prev
        prev = curr
        curr = next

    first = head
    second = prev

    while second.next:
        # save the next nodes
        first_next = first.next
        second_next = second.next

        # insert the current node from 2 after the current node from 1
        second.next = first.next
        first.next = second

        # update pointers for next iteration
        first = first_next
        second = second_next
    return head


def reverse_k_groups(head, k):
    dummy = ListNode(0)
    dummy.next = head
    ptr = dummy

    while ptr:
        # Keep track of the current position
        tracker = ptr
        # Traverse k nodes to check if there are enough nodes to reverse
        for i in range(k):
            # If there are not enough nodes to reverse, break out of the loop
            if tracker == None:
                break
            tracker = tracker.next

        if tracker == None:
            break

        # Reverse the current group of k nodes
        previous = None
        current = ptr.next
        next = None
        for _ in range(k):
            # temporarily store the next node
            next = current.next
            # reverse the current node
            current.next = previous
            # before we move to the next node, point previous to the
            # current node
            previous = current
            # move to the next node
            current = next

        # Connect the reversed group to the rest of the linked list
        last_node_of_reversed_group = ptr.next
        last_node_of_reversed_group.next = current
        ptr.next = previous
        ptr = last_node_of_reversed_group

    return dummy.next


def valid_parens(string: str) -> bool:
    stack = []
    parens = {"}": "{", ")": "(", "]": "["}

    for s in string:
        if s not in parens.keys():
            stack.append(s)
        else:
            if len(stack) == 0:
                return False
            curr_parens = stack.pop()
            if curr_parens != parens[s]:
                return False

    if len(stack) > 0:
        return False
    return True


def remove_duplicates(string: str) -> str:
    stack = []
    for s in string:
        if not stack:
            stack.append(s)
        else:
            last = stack[-1]
            if s == last:
                stack.pop()
            else:
                stack.append(s)
    return ''.join(stack)


def calculator(expression: str) -> int:
    number = 0
    sign_value = 1
    result = 0
    operations_stack = []

    for c in expression:
        if c.isdigit():
            number = number * 10 + int(c)
        if c in "+-":
            result += number * sign_value
            sign_value = -1 if c == '-' else 1
            number = 0
        elif c == '(':
            operations_stack.append(result)
            operations_stack.append(sign_value)
            result = 0
            sign_value = 1

        elif c == ')':
            result += sign_value * number
            pop_sign_value = operations_stack.pop()
            result *= pop_sign_value

            second_value = operations_stack.pop()
            result += second_value
            number = 0
    
    return result + number * sign_value
            
        







if __name__ == "__main__":
    linked_list = LinkedList([1, 2, 3, 4, 5])
    # display(reverse_k_groups(linked_list.head, 2))
    print(calculator("12 - (6 + 2) + 5"))
