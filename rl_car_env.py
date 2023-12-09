import gym
from gym import spaces
import numpy as np
import pygame
import sys
import os
import math
import random

#Constants
WIDTH, HEIGHT = 800, 600
CAR_RADIUS = 12
CAR_SPEED = 1
MIN_CAR_SPEED = 1
MAX_CAR_SPEED = 8
OBSTACLE_SIZE = 45
SENSOR_LENGTH = 400
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)

# Wall properties
WALL_THICKNESS = 1
WALL_COLOR = BLACK

# Addition
NUM_OBSTACLE = 24

# Wall positions (left, top, width, height)
WALLS = [
    pygame.Rect(0, 0, WIDTH, WALL_THICKNESS),  # Top wall
    pygame.Rect(0, 0, WALL_THICKNESS, HEIGHT),  # Left wall
    pygame.Rect(WIDTH - WALL_THICKNESS, 0, WALL_THICKNESS, HEIGHT),  # Right wall
    pygame.Rect(0, HEIGHT - WALL_THICKNESS, WIDTH, WALL_THICKNESS),  # Bottom wall
]

# Reward
NOT_CRASH = 1
CRASH = -100
SLOW = -1
TURN_PENALTY = -0.5

# Car class
class Car(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        
        self.image = pygame.Surface((2 * CAR_RADIUS, 2 * CAR_RADIUS), pygame.SRCALPHA)
        pygame.draw.circle(self.image, WHITE, (CAR_RADIUS, CAR_RADIUS), CAR_RADIUS)
        self.rect = self.image.get_rect(center=(x, y))
        self.speed = CAR_SPEED
        self.angle = 0
        self.collided = False

        # Sensor directions (relative angles)
        # represent left, front, right sensor
        self.sensor_angles = [-45, 0, 45]
        self.sensors = [Sensor(self.rect.center, angle) for angle in self.sensor_angles]
        
    def update(self):
        # Check for collisions with obstacles
        self.collided = False
        collisions = pygame.sprite.spritecollide(self, obstacles, False)

        if collisions:
            self.reset_car_position()

        # Update sensor positions and distances
        for sensor in self.sensors:
            sensor.update(self.rect.center, self.angle)

    def get_sensor_values(self):
        return [sensor.distance // 4 for sensor in self.sensors] + [self.speed-1]

    def reset_car_position(self):
        # If collision with an obstacle, respawn the car at the center
        self.rect.center = (20, HEIGHT // 2)
        self.angle = 0
        self.collided = True
    
# Sensor class
class Sensor(pygame.sprite.Sprite):
    def __init__(self, start_pos, angle_offset):
        super().__init__()
        self.image = pygame.Surface((0, 0), pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        self.start_pos = start_pos
        self.angle_offset = angle_offset
        self.end_pos = start_pos
        self.distance = SENSOR_LENGTH

    def update(self, car_center, car_angle):
        self.start_pos = car_center
        # Calculate sensor position based on car's center and angle
        angle = math.radians(car_angle + self.angle_offset)
        self.end_pos = (
            int(car_center[0] + SENSOR_LENGTH * math.cos(angle)),
            int(car_center[1] - SENSOR_LENGTH * math.sin(angle))
        )
        self.rect.topleft = car_center

        # Calculate the distance to the closest obstacle
        closest_obstacle = None
        closest_distance = SENSOR_LENGTH

        for obstacle in obstacles:
            # Calculate the intersection point of the line segment and the obstacle's rect
            intersection_point = self.get_line_rect_intersection(self.start_pos, self.end_pos, obstacle.rect)
            if intersection_point:
                distance = math.sqrt((car_center[0] - intersection_point[0]) ** 2 +
                                     (car_center[1] - intersection_point[1]) ** 2)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_obstacle = obstacle

        for wall in WALLS:
            # Calculate the intersection point of the line segment and the wall
            intersection_point = self.get_line_rect_intersection(self.start_pos, self.end_pos, pygame.Rect(wall))
            if intersection_point:
                distance = math.sqrt((car_center[0] - intersection_point[0]) ** 2 +
                                     (car_center[1] - intersection_point[1]) ** 2)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_obstacle = None  # Walls override other obstacles

        if closest_obstacle:
            self.distance = abs(int(closest_distance - CAR_RADIUS))
        else:
            self.distance = SENSOR_LENGTH

    @staticmethod
    def get_line_rect_intersection(start_pos, end_pos, rect):
        # Get the four corners of the rectangle
        rect_corners = [(rect.left, rect.top), (rect.right, rect.top), (rect.right, rect.bottom), (rect.left, rect.bottom)]

        for i in range(4):
            # Check for intersection with each line segment of the rectangle
            line_start = rect_corners[i]
            line_end = rect_corners[(i + 1) % 4]

            intersection = line_intersection(start_pos, end_pos, line_start, line_end)
            if intersection:
                return intersection

        return None

# Function to find the intersection point of two lines
def line_intersection(start1, end1, start2, end2):
    x1, y1 = start1
    x2, y2 = end1
    x3, y3 = start2
    x4, y4 = end2

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if denominator == 0:
        return None

    x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator

    # Check if the intersection point is within the line segments
    if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2) and \
       min(x3, x4) <= x <= max(x3, x4) and min(y3, y4) <= y <= max(y3, y4):
        return int(x), int(y)
    else:
        return None

# Obstacle class
class Obstacle(pygame.sprite.Sprite):
    def __init__(self, x, y, color):
        super().__init__()
        self.image = pygame.Surface((OBSTACLE_SIZE, OBSTACLE_SIZE))
        self.image.fill(color)
        self.rect = self.image.get_rect(topleft=(x, y))

# Function to create obstacles with random positions
def create_random_obstacles(num_obstacles, all_sprites, car):
    obstacles = pygame.sprite.Group()
    for _ in range(num_obstacles):
        while (True):
            x = random.randint(120, WIDTH - OBSTACLE_SIZE)
            y = random.randint(0, HEIGHT - OBSTACLE_SIZE)
            # for better randomness ? not sure
            x = random.randint(120, WIDTH - OBSTACLE_SIZE)
            y = random.randint(0, HEIGHT - OBSTACLE_SIZE)
            
            color = WHITE
            obstacle = Obstacle(x, y, color)
            if not pygame.sprite.collide_rect(car, obstacle):
                break
        obstacles.add(obstacle)
    
    for wall in WALLS:
        wall_obstacle = Obstacle(wall[0], wall[1], WALL_COLOR)
        wall_obstacle.rect.size = (wall[2], wall[3])
        obstacles.add(wall_obstacle)
        all_sprites.add(wall_obstacle)
    
    return obstacles

# Create sprites
all_sprites = pygame.sprite.Group()
car = Car(20, HEIGHT // 2)

# Initialize sprites outside the game loop
obstacles = create_random_obstacles(NUM_OBSTACLE, all_sprites, car)  # Initial number of obstacles
all_sprites.add(car, *obstacles) 

class RlCarEnv(gym.Env):
    def __init__(self):
        super(RlCarEnv, self).__init__()

        # Direction |  Left | Left | Left | Forward | Forward | Forward | Right | Right | Right |
        # Speed     |   Up  | Down | Keep |   Up    |   Down  |   keep  |  Up   | Down  | Keep  |
        # ==> 6 actions discrete actions
        self.action_space = spaces.Discrete(9)
        low = [0] * 3 + [0] # 3 distance sensors and 1 velocity sensor
        high = [100] * 3 + [7]
        self.observation_space = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.uint8)
        
        # Initialize Pygame
        pygame.init()

        # Initialize screen
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Autonomous Car Simulation")
            
        # Game loop
        self.clock = pygame.time.Clock()

    def reset(self):
        # Reset the environment to its initial state
        car.reset_car_position()
        
        return np.array(car.get_sensor_values(), dtype=np.uint8)

    def step(self, action):
        # Take a step in the environment given the action
        # Return the next observation, reward, done flag, and additional information
        if action == 1 or action == 4 or action == 7:
            if car.speed - 1 >= MIN_CAR_SPEED:
                car.speed -= 1
        elif action == 0 or action == 3 or action == 6:
            if car.speed + 1 <= MAX_CAR_SPEED:
                car.speed += 1

        direction = action // 3
        car.angle += (direction - 1) * 5
        car.rect.x += car.speed * math.cos(math.radians(car.angle))
        car.rect.y -= car.speed * math.sin(math.radians(car.angle))
        
        # Collision detection 
        if car.collided:
            reward = CRASH
        elif car.speed <= 2:
            reward = SLOW
        else:
            reward = NOT_CRASH
            # Penalize for turning
            if direction != 1:  # Direction 1 correspond to forward
                reward += TURN_PENALTY

        next_obs = np.array(car.get_sensor_values(), dtype=np.uint8)
        return next_obs, reward, False, False, {}

    def render(self, mode='human'):
        # Update
        all_sprites.update()

        # Draw and display
        self.screen.fill(BLACK)
        for sensor in car.sensors:
            pygame.draw.line(self.screen, YELLOW, sensor.start_pos, sensor.end_pos, 2)
        all_sprites.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(FPS)
        
    def close(self):
        # Perform cleanup operations, if needed
        pygame.quit()
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]