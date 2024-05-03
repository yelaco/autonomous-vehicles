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
WIDTH, HEIGHT = 600, 600
CAR_RADIUS = 15
OBSTACLE_SIZE = 40
CAR_SPEED = 4
SENSOR_LENGTH = 100
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (5, 235, 77)
RED = (214, 6, 27)

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
TURN_PENALTY = 0.1

# Set up fonts
font_size = 24

# Set up text area
text_area_width = 250
text_area_height = 350
text_color = (255, 255, 255)

class Car(pygame.sprite.Sprite):
    def __init__(self, x, y, angle):
        super().__init__()
        self.image = pygame.Surface((2 * CAR_RADIUS, 2 * CAR_RADIUS), pygame.SRCALPHA)
        pygame.draw.circle(self.image, WHITE, (CAR_RADIUS, CAR_RADIUS), CAR_RADIUS)
        self.rect = self.image.get_rect(center=(x, y))
        self.speed = CAR_SPEED
        self.angle = angle
        self.collided = False
        self.initial_angle = angle
        self.initial_pos = (x, y)

        # Sensor directions (relative angles)
        # represent left, front, right sensor
        self.sensor_angles = [50, 25, 0, -25, -50]
        self.sensors = [Sensor(self.rect.center, angle) for angle in self.sensor_angles]

    def update(self):
        # Check for collisions with obstacles and walls
        self.collided = False
        
        for obstacle in obstacles:
            intersected = self.circle_rect_intersection(self.rect.center, CAR_RADIUS, obstacle.rect)
            if intersected:
                self.collided = True
                break

        # Update sensor positions and distances
        for sensor in self.sensors:
            sensor.update(self.rect.center, self.angle)

    def get_sensor_values(self):
        distances = [sensor.distance for sensor in self.sensors]
        return distances

    def reset_car_position(self):
        # If collision with an obstacle, respawn the car at the center
        self.initial_angle = (self.initial_angle + 45) % 360
        self.rect.center = self.initial_pos
        self.angle = self.initial_angle
        self.update()
    
    @staticmethod
    def circle_rect_intersection(circle_center, radius, rect):
        closest_x = max(rect.left, min(circle_center[0], rect.right))
        closest_y = max(rect.top, min(circle_center[1], rect.bottom))

        distance = math.sqrt((closest_x - circle_center[0]) ** 2 + (closest_y - circle_center[1]) ** 2)

        if distance < radius:
            return True
        else:
            return False
    
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

class MovingObstacle(Obstacle):
    def __init__(self, radius, moving_radius, angular_speed, clockwise=True):
        super().__init__(WIDTH // 2 + int(moving_radius * math.cos(0)), HEIGHT // 2 + int(moving_radius * math.sin(0)), radius)
        self.image = pygame.Surface((2 * radius, 2 * radius), pygame.SRCALPHA)
        pygame.draw.circle(self.image, (214, 6, 27), (radius, radius), radius)
        self.moving_radius = moving_radius 
        self.angular_speed = angular_speed
        self.angle = 0
        self.clockwise = clockwise

    def update(self):
        if self.clockwise: 
            self.angle -= math.radians(self.angular_speed)
        else:
            self.angle += math.radians(self.angular_speed)
        self.rect.x = WIDTH // 2 + int(self.moving_radius * math.cos(self.angle))
        self.rect.y = HEIGHT // 2 + int(self.moving_radius * math.sin(self.angle))
         
    
# Function to create obstacles with random positions
def create_obstacles(num_obstacles):
    obstacles = pygame.sprite.Group()
    for _ in range(num_obstacles):
        while (True):
            x1 = random.randint(0, 150)
            x2 = random.randint(250, WIDTH - OBSTACLE_SIZE)
            x = random.choice([x1, x2])
            y1 = random.randint(0, HEIGHT // 2 - 50)
            y2 = random.randint(HEIGHT // 2 + 50, HEIGHT - OBSTACLE_SIZE)
            y = random.choice([y1, y2])
            
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
car = Car(x=200, y=HEIGHT // 2, angle=0)

# Initialize sprites outside the game loop
obstacles = create_obstacles(NUM_OBSTACLE)  # Initial number of obstacles
all_sprites.add(car, *obstacles)

class RlCarEnv(gym.Env):
    def __init__(self):
        super(RlCarEnv, self).__init__()

        # Action |  Straight | Left | Right
        # ==> 3 actions discrete actions
        self.action_space = spaces.Discrete(3)
        self.state_space = [101, 101, 101, 101, 101]
        low = [0, 0, 0, 0, 0]
        high = [100, 100, 100, 100, 100]
        self.observation_space = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.uint8)
        
        # Initialize Pygame
        pygame.init()


        # State
        self.state_sensors = [
            [(WIDTH // 2 - 11, HEIGHT + 200), (WIDTH // 2 - 28, HEIGHT + 158), (WIDTH // 2 - 41, HEIGHT + 168) ],
            [(WIDTH // 2 - 4, HEIGHT + 200), (WIDTH // 2 - 4, HEIGHT + 149), (WIDTH // 2 - 21, HEIGHT + 155) ],
            [(WIDTH // 2 - 43, HEIGHT + 115), (WIDTH // 2 - 30, HEIGHT + 150), (WIDTH // 2 - 46, HEIGHT + 162), (WIDTH // 2 - 70, HEIGHT + 135) ],
            [(WIDTH // 2 - 4, HEIGHT + 100), (WIDTH // 2 - 4, HEIGHT + 142), (WIDTH // 2 - 22, HEIGHT + 148), (WIDTH // 2 - 35, HEIGHT + 110) ],
            [(WIDTH // 2 + 4, HEIGHT + 200), (WIDTH // 2 + 4, HEIGHT + 149), (WIDTH // 2 + 21, HEIGHT + 155) ],
            [(WIDTH // 2 + 11, HEIGHT + 200), (WIDTH // 2 + 28, HEIGHT + 158), (WIDTH // 2 + 41, HEIGHT + 168) ],
            [(WIDTH // 2 + 4, HEIGHT + 100), (WIDTH // 2 + 4, HEIGHT + 142), (WIDTH // 2 + 22, HEIGHT + 148), (WIDTH // 2 + 35, HEIGHT + 110) ],
            [(WIDTH // 2 + 43, HEIGHT + 115), (WIDTH // 2 + 30, HEIGHT + 150), (WIDTH // 2 + 46, HEIGHT + 162), (WIDTH // 2 + 70, HEIGHT + 135) ],
        ]

        # Initialize screen
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("RL Car Simulation")
            
        # Game loop
        self.clock = pygame.time.Clock()
    
    def change_map(self):
        global all_sprites
        global obstacles
        all_sprites.remove(*obstacles)
        obstacles = create_obstacles(NUM_OBSTACLE)
        all_sprites.add(*obstacles) 

    def reset(self):
        # Reset the environment to its initial state
        car.reset_car_position()
        return np.array(car.get_sensor_values(), dtype=np.uint8)

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
 
        r1 = 0
        r2 = 0
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
                r1 = 1       
                
            if hasattr(self, 'prev_action') and ((self.prev_action == 1 and action == 2) or (self.prev_action == 2 and action == 1)):
                r2 = -0.8
                
            reward = r1 + r2

        self.prev_action = action
        self.current_obs = next_obs
        return next_obs, reward, terminated, False, {}

    def render(self, info=""):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return True
            
        # Update
        all_sprites.update()

        # Draw and display
        self.screen.fill(BLACK)
         # Render circular walls
        for wall in WALLS:
            pygame.draw.circle(self.screen, WALL_COLOR, wall[:2], wall[2], 1)
        for sensor in car.sensors:
            pygame.draw.line(self.screen, YELLOW, sensor.start_pos, sensor.end_pos, 2)
        pygame.draw.line(self.screen, WHITE, (WIDTH + 50, 0), (WIDTH + 50, HEIGHT + 300), 2) 

        all_sprites.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(FPS)
        return False
        
    def close(self):
        # Perform cleanup operations, if needed
        pygame.quit()
