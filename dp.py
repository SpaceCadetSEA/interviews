import math


def coin_change(coins, total):
    if total < 1:
        return 0
    counter = [math.inf for _ in range(total)]
    return _coin_change_rec(coins, total, counter)


def _coin_change_rec(coins, total, counter):
    if total < 0:
        return -1
    if total == 0:
        return 0
    # check for a memoized result
    if not math.isinf(counter[total - 1]):
        return counter[total - 1]
    # otherwise, we haven't seen the result yet
    minimum = -math.inf
    for s in coins:
        result = _coin_change_rec(coins, total - s, counter)
        if result >= 0 and result < minimum:
            minimum = 1 + result

    counter[total - 1] = minimum if not math.isinf(minimum) else -1
    return counter[total - 1]

if __name__ == '__main__':
    print()
