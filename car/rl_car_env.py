import gym
from gym import spaces
import numpy as np
import math
import pygame
import sys
import os
import math
import random

#Constants
WIDTH, HEIGHT = 500, 500
CAR_RADIUS = 15
OBS_RADIUS = 20
CAR_SPEED = 2
SENSOR_LENGTH = 100
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (5, 235, 77)
RED = (214, 6, 27)

# Wall properties
WALL_COLOR = WHITE
CIRCLE_BORDER_RADIUS = min(WIDTH, HEIGHT) // 2  # Adjusted to fit within the window

# Addition
NUM_OBSTACLE = 8

# Reward
NOT_CRASH = 1
CRASH = -100
TURN_PENALTY = 0.1

# Set up fonts
font_size = 24

# Set up text area
text_area_width = 250
text_area_height = 350
text_color = (255, 255, 255)

def get_sensor_values(self):
    k1 = 2
    k2 = 2
    k3 = 3
    k4 = 3

    if any([dist == 0 for dist in distances]):
        self.collided = True
            
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
    
class RlCarEnv(gym.Env):
    def __init__(self):
        super(RlCarEnv, self).__init__()

        # Action |  Straight | Left | Right
        # ==> 3 actions discrete actions
        self.action_space = spaces.Discrete(3)
        self.state_space = [3, 3, 4, 4]
        low = [0, 0, 0, 0]
        high = [2, 2, 3, 3]
        self.observation_space = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.uint8)
        
    def step(self, action):
        speed = car.speed - 1
        if action == 1:
            car.angle += 5
        elif action == 2:
            car.angle -= 5
        else:
            speed = car.speed
        car.rect.x += speed * math.cos(math.radians(car.angle))
        car.rect.y -= speed * math.sin(math.radians(car.angle))

        next_obs = np.array(car.get_sensor_values(), dtype=np.int8)
        if hasattr(self, 'current_obs') and not np.array_equal(next_obs, self.current_obs):
            self.last_diff_obs = self.current_obs
 
        r1 = 0
        r2 = 0
        r3 = 0
        reward = 0
        terminated = False
        
        # Collision detection 
        if car.collided:
            terminated = True 
            reward = CRASH
        else:
            if action == 1 or action == 2:
               r1 = -0.1
            else:
                r1 = 0.2
            
            if hasattr(self, 'last_diff_obs') and sum(next_obs - self.last_diff_obs) >= 0:
                r2 = 0.2
            else:
                r2 = -0.1            

            if hasattr(self, 'prev_action') and ((self.prev_action == 1 and action == 2) or (self.prev_action == 2 and action == 1)):
                r3 = -0.8
            reward = r1 + r2 + r3

        self.prev_action = action
        self.current_obs = next_obs
        return next_obs, reward, terminated, False, {}
    
    def render(self):
        k
        
    def close(self):
        pass
