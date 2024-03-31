import RPi.GPIO as GPIO

import time
 
from AlphaBot import AlphaBot
#import smbus
import curses
import random 
import pickle
import numpy as np
#from mpu6050 import mpu6050


#mpu = mpu6050(0x68)
Ab = AlphaBot()

#screen = curses.initscr()
 
#curses.noecho() 
 
#curses.cbreak()
 
#curses.halfdelay(3)
 
#screen.keypad(True)
 
#PWM = 10
 
def greedy_policy(Qtable, state):
    action = np.argmax(Qtable[tuple(state)])
    return action

def get_sensor_values(distances):
    k1 = 2
    k2 = 2
    k3 = 3
    k4 = 3

    # check the left side sensor
    dist = min(distances[0:3])
    if (dist > 40 and dist < 70):
        k1 = 1  #zone 1
    elif (dist <= 40):
        k1 = 0  #zone 0

    # check the right side sensor
    dist = min(distances[2:])
    if (dist > 40 and dist < 70):
        k2 = 1  #zone 1
    elif (dist <= 40):
        k2 = 0  #zone 0

    detected = [distance < 100 for distance in distances]
    # the left sector of the vehicle
    if detected[0] and detected[2]:
        k3 = 0 # both subsectors have obstacles
    elif (detected[1] or detected[2]) and not detected[0]:
        k3 = 1 # inner left subsector
    elif (detected[0] or detected[1]) and not detected[2]:
        k3 = 2 # outter left subsector

    # the right sector of the vehicle
    if detected[2] and detected[4]:
        k4 = 0 # both subsectors have obstacles
    elif (detected[2] or detected[3]) and not detected[4]:
        k4 = 1 # inner right subsector
    elif (detected[3] or detected[4]) and not detected[2]:
        k4 = 2 # outter right subsector 

    return [k1, k2, k3, k4]

with open('q_table.pkl', 'rb') as f:
    Qtable_rlcar = pickle.load(f)

try:
 
    while True:
        distances = [(int(dist) if dist < 100 else 100) if dist >= 0 else 0 for dist in Ab.SR04()]
        print(distances)
        state = get_sensor_values(distances)

        print(state)
        action = greedy_policy(Qtable_rlcar, state)
        if action == 0:
            print("Forward")
            Ab.forward()
        elif action == 1:
            print("Left")
            Ab.left()
        elif action == 2:
            print("Right")
            Ab.right()
        else:
            Ab.stop()

             
            
        #Ab.forward()
        #accel_data = mpu.get_accel_data()
        #gyro_data = mpu.get_gyro_data()
        #time.sleep(0.5)
        #print(accel_data)
        #print(gyro_data)
       
        # elif char == curses.KEY_UP:
 
        #     Ab.forward()
 
        # elif char == curses.KEY_DOWN:
 
        #     Ab.backward()
 
        # elif char == curses.KEY_RIGHT:
 
        #     Ab.right()
 
        # elif char == curses.KEY_LEFT:
 
        #     Ab.left()
 
        # elif char == 10:
 
        #     Ab.stop()
 
finally:
 
    #Close down curses properly, inc turn echo back on!
    screen=curses.initscr()
    curses.nocbreak();
    screen.keypad(0);
    curses.echo()
    curses.endwin()
    GPIO.cleanup()
