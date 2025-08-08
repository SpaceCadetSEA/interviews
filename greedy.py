def jump_game(nums):
    target = len(nums) - 1
    for i in range(len(nums) - 2, -1, -1):
        curr_value = nums[i]
        if curr_value >= target - i:
            target = i
    return target == 0


def gas_station_journey(gas, cost):
    """
    Deceptively simple solution. 
    
    We iterate over the list a single time and just check the gas - cost at
    each index, incrementing starting index when the current is less than 0.
    """
    if sum(cost) > sum(gas):
        return -1

    current_gas = starting_idx = 0

    for i in range(len(gas)):
        current_gas += gas[i] - cost[i]
        if current_gas < 0:
            current_gas = 0
            starting_idx = i + 1

    return starting_idx


def min_refuel_stops(target, start_fuel, stations):
    return 0


if __name__ == "__main__":
    print(gas_station_journey([1, 1, 1, 1, 10], [2, 2, 1, 3, 1]))
