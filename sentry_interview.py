"""
ParkingGarage
- spots: vector (list)
    spot number as index : lic. no
    inital condition -- empty 0, null
    
- available_spots
- is_full
- add_vehicle(lic_no) -> bool
    not is_full (if full -> false)
    available_spots
        linear time
        insert lic_no to the spots list
        are able to park (-> true)
- remove_vehicle(lic_no)
    linear scan here again
    
    
ParkingSpace
    - 

Lic. no. <> spot number
Time component

Vehicle
- lic. no.
"""


class ParkingGarage:
    def __init__(self, size=100):
        self.car_capacity = 100
        self.motorcycle_capacity = 20
        # Why do I do this? Why not keep a hashmap of lic_no to spot number?
        # spot number will go up and down as cars enter and leave.
        self.car_spots = [None] * self.car_capacity
        self.motorcycle_spots = [None] * self.motorcycle_capacity
        self.size = self.car_capacity + self.motorcycle_capacity
        self.available_car_capacity = 0
        self.available_motorcycle_capacity = 0
    
    def is_car_capacity_full(self):
        return self.available_car_capacity == len(self.car_spots)
    
    def is_full(self, vehicle_type):
        # invariant car or motorcycle vehicle_type
        if vehicle_type == 'car':
            return self.is_car_capacity_full()
        else:
            return (self.available_motorcycle_capacity == len(self.motorcycle_spots)) or self.is_car_capacity_full()
    
    def add_vehicle(self, lic_no : str, vehicle_type : str) -> bool:
        # TODO check if lic_no already in garage
        if self.is_full(vehicle_type):
            return False
        if vehicle_type == 'car':
            current_spot = None
            for i, spot in enumerate(self.car_spots):
                if not spot:
                    current_spot = i
                    break;
            self.car_spots[current_spot] = lic_no
            self.available_car_capacity += 1
            return True
        else:
            # want to check motorcycle spots first
            # if none avaiable, use vehicle spot
            current_spot = None
            for i, spot in enumerate(self.motorcycle_spots):
                if not spot:
                    current_spot = i
                    break;
            if not current_spot:
                for i, spot in enumerate(self.car_spots):
                    if not spot:
                        current_spot = i
                        break;
                    self.car_spots[current_spot] = lic_no
                    self.available_car_capacity += 1
                    return True
            else:
                self.motorcycle_spots[current_spot] = lic_no
                self.available_motorcycle_capacity += 1
                return True
        
    def vehicle_in_garage(self, lic_no : str, vehicle_type : str) -> bool:
        vehicle_exists = False
        for lic in self.spots:
            if lic_no == lic:
                vehicle_exists = True
        return vehicle_exists
    
    def remove_vehicle(self, lic_no : str, vehicle_type : str) -> bool:
        if vehicle_type == 'car':
            try:
                spot = self.car_spots.index(lic_no)
                # TODO remove 
                print(spot)
                self.spots[spot] = None
                self.current_capacity -= 1
            except Exception as e:
                # ValueError
                # logging 
                raise e
            return True
        else:
            try:
                spot = self.motorcycle_spots.index()
    
        
        
garage = ParkingGarage(100)
print(garage.is_full())
print(garage.add_vehicle('1234'))
# for i in range(101):
#     lic_no = f'{i}'
#     garage.add_vehicle(lic_no)
# print(garage.is_full())  # true
# lookup whether a specific lic. is in garage
# adding the same license plate
garage.add_vehicle('2344')
print(garage.current_capacity == 2)
print(garage.remove_vehicle('1234'))
print(garage.current_capacity == 1)


