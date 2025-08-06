
class Interval:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.closed = True  # by default, the interval is closed

    # set the flag for closed/open

    def set_closed(self, closed):
        self.closed = closed

    def __str__(self):
        return (
            "[" + str(self.start) + ", " + str(self.end) + "]"
            if self.closed
            else "(" + str(self.start) + ", " + str(self.end) + ")"
        )
    
    def __eq__(self, other):
        return self.start == other.start
    
    def __ne__(self, other):
        return self.start != other.start
    
    def __lt__(self, other):
        return self.start < other.start

    def __gt__(self, other):
        return self.start > other.start

    def __le__(self, other):
        return self.start <= other.start

    def __ge__(self, other):
        return self.start >= other.start
