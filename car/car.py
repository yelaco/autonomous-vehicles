from AlphaBot import AlphaBot

class Car:
    FORWARD = 0
    LEFT = 1
    RIGHT = 2
    
    def __init__(self):
        self.Ab = AlphaBot()
        self.stopped = False
    
    def make_decision(self, mod_func, *args):
        action = mod_func(*args)
        if action == Car.FORWARD:
            self.Ab.forward()
            print("Forward")
        elif action == Car.LEFT:
            self.Ab.left()
            print("Left")
        elif action == Car.RIGHT:
            self.Ab.right()
            print("Right")
        else:
            self.Ab.stop()
            print("Stop")
    
    def stop(self):
        print("Stop")
        self.Ab.stop()
        self.stopped = True
    
    def ultrasonic_sensors(self): 
        return [(int(dist) if dist < 100 else 100) if dist >= 0 else 0 for dist in self.Ab.SR04()]