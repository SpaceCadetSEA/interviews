

def jump_game(nums):
    target = len(nums) - 1
    for i in range(len(nums) - 2, -1, -1):
        curr_value = nums[i]
        if curr_value >= target - i:
            target = i
    return target == 0


def gas_station_journey(gas, cost):
    return -1


if __name__ == "__main__":
    print(jump_game([2,3,1,1,9]))