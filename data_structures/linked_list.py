
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class LinkedList:
    def __init__(self, values=None):
        self.head = None
        if values:
            self._create_linked_list(values)

    def _create_linked_list(self, values):
        if not values:
            self.head = None
            return

        self.head = ListNode(values[0])
        current = self.head
        for value in values[1:]:
            current.next = ListNode(value)
            current = current.next

    
def display(head):
    current = head
    while current:
        print(current.val, end=" -> ")
        current = current.next
    print("None")