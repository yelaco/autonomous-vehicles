import gym
from gym import spaces
import numpy as np
import pygame
import sys
import os
import math
import random

#Constants
WIDTH, HEIGHT = 400, 400
CAR_RADIUS = 10
OBS_RADIUS = 20
CAR_SPEED = 3
SENSOR_LENGTH = 100
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)

# Wall properties
WALL_COLOR = WHITE
CIRCLE_BORDER_RADIUS = min(WIDTH, HEIGHT) // 2  # Adjusted to fit within the window

# Addition
NUM_OBSTACLE = 2

# Reward
NOT_CRASH = 1
CRASH = -20
TURN_PENALTY = 0.1

class Car(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((2 * CAR_RADIUS, 2 * CAR_RADIUS), pygame.SRCALPHA)
        pygame.draw.circle(self.image, WHITE, (CAR_RADIUS, CAR_RADIUS), CAR_RADIUS)
        self.rect = self.image.get_rect(center=(x, y))
        self.speed = CAR_SPEED
        self.angle = 0
        self.collided = False
        self.initial_pos = (x, y)

        # Sensor directions (relative angles)
        # represent left, front, right sensor
        self.sensor_angles = [-45, 0, 45]
        self.sensors = [Sensor(self.rect.center, angle) for angle in self.sensor_angles]

    def update(self):
        # Check for collisions with obstacles and walls
        self.collided = False

        for obstacle in obstacles:
            intersection = self.get_circle_circle_intersection(self.rect.center, CAR_RADIUS, obstacle.rect.center, obstacle.radius)
            if intersection:
                self.collided = True

        for wall in CIRCULAR_WALLS:
            intersection = self.get_circle_circle_intersection(self.rect.center, CAR_RADIUS, wall[:2], wall[2])
            if intersection:
                self.collided = True

        # Update sensor positions and distances
        for sensor in self.sensors:
            sensor.update(self.rect.center, self.angle)

    def get_circle_circle_intersection(self, center1, radius1, center2, radius2):
        distance = math.dist(center1, center2)

        # Check if circles have the same center
        if distance == 0:
            return None
        
        # Check if circles do not intersect
        if distance > radius1 + radius2:
            return None

        # Calculate intersection points using circle intersection formula
        a = (radius1**2 - radius2**2 + distance**2) / (2 * distance)
        discriminant = radius1**2 - a**2

        # Check if the discriminant is non-negative
        if discriminant < 0:
            return None

        h = math.sqrt(discriminant)

        # Calculate coordinates of intersection points
        x0 = center1[0] + a * (center2[0] - center1[0]) / distance
        y0 = center1[1] + a * (center2[1] - center1[1]) / distance

        intersection1 = (
            int(x0 + h * (center2[1] - center1[1]) / distance),
            int(y0 - h * (center2[0] - center1[0]) / distance)
        )

        intersection2 = (
            int(x0 - h * (center2[1] - center1[1]) / distance),
            int(y0 + h * (center2[0] - center1[0]) / distance)
        )

        return intersection1, intersection2

    def get_sensor_values(self):
        # Initial values
        # Which mean that there are no obstacle in corresponding zones and sectors
        k1 = 2 
        k2  = 2
        k3 = 3
        k4 = 3
        
        distances = [sensor.distance for sensor in self.sensors]
        # check the right side sensor
        dist = min(distances[0:2])
        if (dist > 40 and dist < 70):
            k1 = 1  #zone 1
        elif (dist <= 40):
            k1 = 0  #zone 0

        # check the left side sensor
        dist = min(distances[1:])
        if (dist > 40 and dist < 70):
            k2 = 1  #zone 1
        elif (dist <= 40):
            k2 = 0  #zone 0
        
        detected = [distance < 100 for distance in distances]
        # the right sector of the vehicle
        if detected[1] and detected[0]:
            k3 = 0
        elif detected[0] and not detected[1]:
            k3 = 1
        elif detected[1] and not detected[0]:
            k3 = 2
            
        # the left sector of the vehicle
        if detected[1] and detected[2]:
            k4 = 0
        elif detected[2] and not detected[1]:
            k4 = 1
        elif detected[1] and not detected[2]:
            k4 = 2
        # check the left side sendor
        return [k1, k2, k3, k4]

    def reset_car_position(self):
        # If collision with an obstacle, respawn the car at the center
        self.rect.center = self.initial_pos
        self.angle = 0
        self.collided = False
    
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

        # Calculate the distance to the closest obstacle or circular wall
        closest_obstacle = None
        closest_distance = SENSOR_LENGTH

        for obstacle in obstacles:
            intersection = self.get_line_circle_intersection(self.start_pos, self.end_pos, obstacle.rect.center, obstacle.radius)
            if intersection:
                # Choose the intersection point closer to the sensor
                distance1 = math.dist(self.start_pos, intersection[0])
                distance2 = math.dist(self.start_pos, intersection[1])
                distance = min(distance1, distance2)

                if distance < closest_distance:
                    closest_distance = min(distance, SENSOR_LENGTH)  # Cap distance at sensor length
                    closest_obstacle = obstacle

        for wall in CIRCULAR_WALLS:
            intersection = self.get_line_circle_intersection(self.start_pos, self.end_pos, wall[:2], wall[2])
            if intersection:
                # Choose the intersection point closer to the sensor
                distance1 = math.dist(self.start_pos, intersection[0])
                distance2 = math.dist(self.start_pos, intersection[1])
                distance = min(distance1, distance2)

                if distance < closest_distance:
                    closest_distance = min(distance, SENSOR_LENGTH)  # Cap distance at sensor length
                    closest_obstacle = wall
                    
        if closest_obstacle:
            self.distance = abs(int(closest_distance - CAR_RADIUS))
        else:
            self.distance = SENSOR_LENGTH

    @staticmethod
    def get_line_circle_intersection(start_pos, end_pos, circle_center, circle_radius):
        x1, y1 = start_pos
        x2, y2 = end_pos
        cx, cy = circle_center

        # Translate the points so that the circle is centered at the origin
        x1 -= cx
        y1 -= cy
        x2 -= cx
        y2 -= cy

        # Parameterized line equation
        dx = x2 - x1
        dy = y2 - y1
        a = dx ** 2 + dy ** 2
        b = 2 * (x1 * dx + y1 * dy)
        c = x1 ** 2 + y1 ** 2 - circle_radius ** 2

        # Discriminant
        disc = b ** 2 - 4 * a * c

        if disc < 0:
            return None  # No intersection

        t1 = (-b + math.sqrt(disc)) / (2 * a)
        t2 = (-b - math.sqrt(disc)) / (2 * a)

        # Check if the intersection points are within the line segment
        if 0 <= t1 <= 1 or 0 <= t2 <= 1:
            # At least one intersection point is within the line segment
            intersection_x1 = x1 + t1 * dx
            intersection_y1 = y1 + t1 * dy
            intersection_x2 = x1 + t2 * dx
            intersection_y2 = y1 + t2 * dy

            # Translate the intersection points back to the original coordinate system
            intersection_x1 += cx
            intersection_y1 += cy
            intersection_x2 += cx
            intersection_y2 += cy

            return (int(intersection_x1), int(intersection_y1)), (int(intersection_x2), int(intersection_y2))

        return None

    @staticmethod
    def is_within_line_segment(intersection):
        x1, y1, x2, y2 = intersection
        return 0 <= x1 <= x2 or 0 <= x2 <= x1


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
    def __init__(self, x, y, radius):
        super().__init__()
        self.radius = radius
        self.image = pygame.Surface((2 * radius, 2 * radius), pygame.SRCALPHA)
        pygame.draw.circle(self.image, WHITE, (radius, radius), radius)
        self.rect = self.image.get_rect(center=(x, y))

# Function to create obstacles with random positions
def create_obstacles(num_obstacles, car, fixed=False, no_obs=False):
    obstacles = pygame.sprite.Group()
    if (fixed):
        obstacles.add(Obstacle(200, 0, OBS_RADIUS + 10))
        obstacles.add(Obstacle(250, 400, OBS_RADIUS + 10))
        obstacles.add(Obstacle(0, 200, OBS_RADIUS + 10))
        obstacles.add(Obstacle(400, 200, OBS_RADIUS + 10))
        obstacles.add(Obstacle(125, 125, OBS_RADIUS + 5))
        obstacles.add(Obstacle(275, 275, OBS_RADIUS + 5))
        obstacles.add(Obstacle(125, 275, OBS_RADIUS + 5))
        obstacles.add(Obstacle(275, 125, OBS_RADIUS + 5))
        obstacles.add(Obstacle(WIDTH // 2, HEIGHT // 2, OBS_RADIUS))
    elif no_obs:
        pass
    else:
        for _ in range(num_obstacles):
            while True:
                x = random.randint(0, WIDTH)
                y = random.randint(0, HEIGHT)

                obstacle = Obstacle(x, y)
                if not pygame.sprite.collide_rect(car, obstacle):
                    break
            obstacles.add(obstacle)
    return obstacles

# Create sprites
all_sprites = pygame.sprite.Group()
car = Car(WIDTH // 2, HEIGHT // 2)

# Initialize circular walls
CIRCULAR_WALLS = [(WIDTH // 2, HEIGHT // 2, CIRCLE_BORDER_RADIUS)]
WALLS = CIRCULAR_WALLS  # Use this for collision detection

# Initialize sprites outside the game loop
obstacles = create_obstacles(NUM_OBSTACLE, car, fixed=False, no_obs=True)  # Initial number of obstacles
all_sprites.add(car, *obstacles) 

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
        speed = car.speed - 1
        if action == 1:
            car.angle += 5
        elif action == 2:
            car.angle -= 5
        else:
            speed = car.speed
        car.rect.x += speed * math.cos(math.radians(car.angle))
        car.rect.y -= speed * math.sin(math.radians(car.angle))

        next_obs = np.array(car.get_sensor_values(), dtype=np.uint8)
        print(car.get_sensor_values())
        
        r1 = 0
        r2 = 0
        r3 = 0
        reward = 0
        terminated = False
        
        # Collision detection 
        if car.collided:
            reward = CRASH
            terminated = True
        else:
            if action == 1 or action == 2:
               r1 = -0.1
            else:
                r1 = 0.2 
            
            if hasattr(self, 'current_obs') and sum(next_obs - self.current_obs) >= 0:
                r2 = 0.2
            else:
                r2 = -0.2            

            if hasattr(self, 'prev_action') and ((self.prev_action == 1 and action == 2) or (self.prev_action == 2 and action == 1)):
                r3 = -0.8
            reward = r1 + r2 + r3

        self.prev_action = action
        self.current_obs = next_obs
        return next_obs, reward, terminated, False, {}

    def render(self, mode='human'):
        # Update
        all_sprites.update()

        # Draw and display
        self.screen.fill(BLACK)
         # Render circular walls
        for wall in CIRCULAR_WALLS:
            pygame.draw.circle(self.screen, WALL_COLOR, wall[:2], wall[2], 1)
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
