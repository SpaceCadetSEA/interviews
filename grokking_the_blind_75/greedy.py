from heapq import heappop, heappush
import math


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
    """
    Much more simplified solution does not simulate the fuel usage whatsoever.
    """
    if start_fuel >= target:
        return 0

    max_heap = []
    max_distance = start_fuel

    station = 0
    stops = 0
    while target > max_distance:
        if station < len(stations) and stations[station][0] <= max_distance:
            _, gas = stations[station]
            heappush(max_heap, -gas)
            station += 1
        elif not max_heap:
            return -1
        else:
            max_distance += -heappop(max_heap)
            stops += 1

    return stops


def rescue_boats(people, limit):
    people.sort()
    start, end = 0, len(people) - 1

    boats = 0

    while start <= end:
        first_passenger = people[start]
        last_passenger = people[end]

        if first_passenger + last_passenger > limit:
            boats += 1
            last_passenger -= 1
        else:
            boats += 1
            first_passenger += 1
            last_passenger -= 1

    return boats


if __name__ == "__main__":
    print(min_refuel_stops(120, 10, [[10, 60], [20, 25], [30, 30], [60, 40]]))
