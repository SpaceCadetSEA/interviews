def jump_game(nums):
    target = len(nums) - 1
    for i in range(len(nums) - 2, -1, -1):
        curr_value = nums[i]
        if curr_value >= target - i:
            target = i
    return target == 0


def gas_station_journey(gas, cost):
    if sum(cost) > sum(gas):
        return -1
    if len(gas) == 1:
        return 0

    start_idx = 0
    while gas[start_idx] - cost[start_idx] < 0:
        start_idx += 1

    # at our first candidate
    # try to iterate through the whole list
    # if we fail, move to the next one if it exists
    curr_gas = gas[start_idx] - cost[start_idx]
    curr_index = start_idx + 1
    while start_idx < len(gas):
        if curr_index == start_idx:
            return start_idx
        if curr_index >= len(gas):
            curr_index = 0
        curr_gas += gas[curr_index] - cost[curr_index]
        if curr_gas < 0:
            start_idx += 1
            curr_index = start_idx
            curr_gas = gas[start_idx] - cost[start_idx]
        curr_index += 1
    
    return -1


if __name__ == "__main__":
    print(gas_station_journey([1, 1, 1, 1, 10], [2, 2, 1, 3, 1]))
