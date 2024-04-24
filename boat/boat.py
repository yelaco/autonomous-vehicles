import math
import subprocess
import sys
import time
from AlphaBoat import AlphaBoat
from ultrasonic import UltrasonicThread
from pid import PIDController

def measure(): 
    return -1

def manual():
    pass

def auto(): 
    measured_value = measure()
    if measured_value > 0:
        Ab.move(pid.update(measured_value))
    else:
        Ab.stop()

Ab = AlphaBoat()
ultrasonic = UltrasonicThread(Ab)
pid = PIDController(kp=0.5, ki=0.1, kd=0.2)

# For calculating state
# BOAT_WIDTH = 30
# DELTA_C = 20
# UD = int((DELTA_C + BOAT_WIDTH/2) / math.cos(math.pi * 65 / 90))
# LD = int((DELTA_C + BOAT_WIDTH/2) / math.cos(math.pi * 40 / 90))

# Mode for controlling the boat
CONTROL_MODE = 2 # 0: shutdown, 1: manual, 2: auto

while True:
    if CONTROL_MODE == 1:
        manual()
    elif CONTROL_MODE == 2: 
        auto()
    else: break
        
Ab.shutdown()


