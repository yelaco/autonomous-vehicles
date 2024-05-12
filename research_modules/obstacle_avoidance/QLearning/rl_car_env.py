import gym
from gym import spaces
import numpy as np
import math
import pygame
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
            intersection = self.get_circle_circle_intersection(self.rect.center, CAR_RADIUS, obstacle.rect.center, obstacle.radius)
            if intersection:
                self.collided = True

        for mov in moving_obstacles:
            intersection = self.get_circle_circle_intersection(self.rect.center, CAR_RADIUS, mov.rect.center, mov.radius)
            if intersection and self.check_other_moving_cars([mov]):
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

    def check_other_moving_cars(self, other_cars):
        """
        Check if the current car can "see" another car within its line of sight.
        
        Parameters:
        other_car (Car): The other Car object to check against.

        Returns:
        bool: True if the other car is visible within the current car's line of sight, False otherwise.
        """
        # Calculate vector from current car to other car
        for other_car in other_cars:
            dx = other_car.rect.centerx - self.rect.centerx
            dy = other_car.rect.centery - self.rect.centery

            # Calculate angle between current car's heading direction and vector to other car
            angle_to_other_car = math.degrees(math.atan2(-dy, dx))  # Angle in degrees (0 to 360)

            # Normalize angles to be within 0 to 360 degrees
            angle_diff = (angle_to_other_car - self.angle) % 360
            if angle_diff > 180:
                angle_diff -= 360

            # Define line of sight threshold (adjust as needed)
            line_of_sight_angle = 100  # Angle in degrees within which another car is considered visible

            # Check if the other car is within the line of sight angle
            if abs(angle_diff) <= line_of_sight_angle / 2:
                # Now check if the other car is within the distance threshold (adjust as needed)
                distance_threshold = SENSOR_LENGTH  # Use your sensor length or visibility range
                distance_squared = dx**2 + dy**2

                if distance_squared <= distance_threshold**2:
                    return True

        return False

    def get_sensor_values(self):
        k1 = 2
        k2 = 2
        k3 = 3
        k4 = 3
        k5 = 1

        distances = [sensor.distance for sensor in self.sensors]

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

        if self.check_other_moving_cars(moving_obstacles):
            k5 = 0

        return [k1, k2, k3, k4, k5] 

    def reset_car_position(self):
        # If collision with an obstacle, respawn the car at the center
        self.initial_angle = (self.initial_angle + 45) % 360
        self.rect.center = self.initial_pos
        self.angle = self.initial_angle
        self.update()
    
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
            int(car_center[0] + (SENSOR_LENGTH + CAR_RADIUS) * math.cos(angle)),
            int(car_center[1] - (SENSOR_LENGTH + CAR_RADIUS) * math.sin(angle))
        )
        self.rect.topleft = car_center

        # Calculate the distance to the closest obstacle or circular wall
        closest_obstacle = None
        closest_distance = SENSOR_LENGTH

        for obstacle in obstacles:
            intersection = self.get_line_circle_intersection(self.start_pos, self.end_pos, obstacle.rect.center, obstacle.radius)
            if intersection:
                # Choose the intersection point closer to the sensor
                distance1 = math.dist(self.start_pos, intersection[0]) - CAR_RADIUS
                distance2 = math.dist(self.start_pos, intersection[1]) - CAR_RADIUS
                distance = min(distance1, distance2)

                if distance < closest_distance:
                    closest_distance = min(distance, SENSOR_LENGTH)  # Cap distance at sensor length
                    closest_obstacle = obstacle
        
        for obstacle in moving_obstacles:
            intersection = self.get_line_circle_intersection(self.start_pos, self.end_pos, obstacle.rect.center, obstacle.radius)
            if intersection:
                # Choose the intersection point closer to the sensor
                distance1 = math.dist(self.start_pos, intersection[0]) - CAR_RADIUS
                distance2 = math.dist(self.start_pos, intersection[1]) - CAR_RADIUS
                distance = min(distance1, distance2)

                if distance < closest_distance:
                    closest_distance = min(distance, SENSOR_LENGTH)  # Cap distance at sensor length
                    closest_obstacle = obstacle

        for wall in CIRCULAR_WALLS:
            intersection = self.get_line_circle_intersection(self.start_pos, self.end_pos, wall[:2], wall[2])
            if intersection:
                # Choose the intersection point closer to the sensor
                distance1 = math.dist(self.start_pos, intersection[0]) - CAR_RADIUS
                distance2 = math.dist(self.start_pos, intersection[1]) - CAR_RADIUS
                distance = min(distance1, distance2)

                if distance < closest_distance:
                    closest_distance = min(distance, SENSOR_LENGTH)  # Cap distance at sensor length
                    closest_obstacle = wall
                    
        if closest_obstacle:
            self.distance = int(closest_distance)
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
def create_obstacles(num_obstacles, map='none'):
    obstacles = pygame.sprite.Group()
    if map == 'static_fixed':
        x = 160
        obstacles.add(Obstacle(WIDTH // 2, 0, OBS_RADIUS + 10))
        obstacles.add(Obstacle(WIDTH // 2, HEIGHT, OBS_RADIUS + 10))
        obstacles.add(Obstacle(0, HEIGHT // 2, OBS_RADIUS + 10))
        obstacles.add(Obstacle(WIDTH, HEIGHT // 2, OBS_RADIUS + 10))
        obstacles.add(Obstacle(x, x, OBS_RADIUS + 5))
        obstacles.add(Obstacle(HEIGHT - x, HEIGHT - x, OBS_RADIUS + 5))
        obstacles.add(Obstacle(x, HEIGHT - x, OBS_RADIUS + 5))
        obstacles.add(Obstacle(HEIGHT - x, x, OBS_RADIUS + 5))
        obstacles.add(Obstacle(WIDTH // 2, HEIGHT // 2, OBS_RADIUS))
    elif map == 'static_random':
        for _ in range(num_obstacles):
            while True:
                x_1 = random.randint(WIDTH // 2 - 100, WIDTH)
                x_2 = random.randint(WIDTH // 2 - 200, WIDTH)
                x = random.choice([x_1, x_2])
                y_offset = int(math.sqrt(CIRCLE_BORDER_RADIUS**2 - abs(WIDTH // 2 - x)**2)) 
                y = random.randint(HEIGHT // 2 - y_offset, HEIGHT // 2 + int(y_offset))

                obstacle = Obstacle(x, y, OBS_RADIUS + 10)
                if not pygame.sprite.collide_rect(car, obstacle):
                    break
            obstacles.add(obstacle)
    elif map == 'dynamic':
        obstacles.add(MovingObstacle(OBS_RADIUS - 5, 50, 0.5, clockwise=False))
        obstacles.add(MovingObstacle(OBS_RADIUS - 5, 150, 0.5))
    elif map == 'static_dynamic':
        x = 160
        moving_obstacles = pygame.sprite.Group()
        obstacles.add(Obstacle(WIDTH // 2, 0, OBS_RADIUS + 10))
        obstacles.add(Obstacle(WIDTH // 2, HEIGHT, OBS_RADIUS + 10))
        obstacles.add(Obstacle(0, HEIGHT // 2, OBS_RADIUS + 10))
        obstacles.add(Obstacle(WIDTH, HEIGHT // 2, OBS_RADIUS + 10))
        obstacles.add(Obstacle(x, x, OBS_RADIUS + 5))
        obstacles.add(Obstacle(HEIGHT - x, HEIGHT - x, OBS_RADIUS + 5))
        obstacles.add(Obstacle(x, HEIGHT - x, OBS_RADIUS + 5))
        obstacles.add(Obstacle(HEIGHT - x, x, OBS_RADIUS + 5))
        obstacles.add(Obstacle(WIDTH // 2, HEIGHT // 2, OBS_RADIUS))
        mov_1 = MovingObstacle(OBS_RADIUS - 5, 75, 0.5, clockwise=False)
        mov_2 = MovingObstacle(OBS_RADIUS - 5, 175, 0.5)
        moving_obstacles.add(mov_1)
        moving_obstacles.add(mov_2)
        return obstacles, moving_obstacles
    else:
        pass
    return obstacles

# Create sprites
all_sprites = pygame.sprite.Group()
car = Car(x=WIDTH // 2 - 120, y=HEIGHT // 2, angle=0)

# Initialize circular walls
CIRCULAR_WALLS = [(WIDTH // 2, HEIGHT // 2, CIRCLE_BORDER_RADIUS)]
WALLS = CIRCULAR_WALLS  # Use this for collision detection

# Initialize sprites outside the game loop
obstacles, moving_obstacles = create_obstacles(NUM_OBSTACLE, map='static_dynamic')  # Initial number of obstacles
all_sprites.add(car, *obstacles, *moving_obstacles)

class RlCarEnv(gym.Env):
    def __init__(self):
        super(RlCarEnv, self).__init__()

        # Action |  Straight | Left | Right
        # ==> 3 actions discrete actions
        self.action_space = spaces.Discrete(3)
        self.state_space = [3, 3, 4, 4, 2]
        low = [0, 0, 0, 0, 0]
        high = [2, 2, 3, 3, 1]
        self.observation_space = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.uint8)
        
        # Initialize Pygame
        pygame.init()

        # Load eval graph image
        self.image_path = 'evaluate.png'
        self.update_eval_graph()

        # Text
        self.text_area_rect = pygame.Rect(WIDTH + 70, self.image.get_height() + 15, text_area_width, text_area_height)
        self.font = pygame.font.Font(None, font_size)

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
        self.screen = pygame.display.set_mode((WIDTH + self.image.get_width() + 50, self.image.get_height() + 15 + text_area_height + 15))
        pygame.display.set_caption("RL Car Simulation")
            
        # Game loop
        self.clock = pygame.time.Clock()
    
    def change_map(self, map='static_fixed'):
        global all_sprites
        global obstacles
        all_sprites.remove(*obstacles)
        obstacles = create_obstacles(NUM_OBSTACLE, map)
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
    
    def update_eval_graph(self):
        self.image = pygame.image.load(self.image_path)
        self.image_rect = self.image.get_rect(topleft=(WIDTH + 50, 0))

    def state_translate(self, state):
        colors = [WHITE] * 8
        
        if state[0] == 0:
            left_offset = 0
        elif state[0] == 1: 
            left_offset = 2
        if state[1] == 0:
            right_offset = 0
        elif state[1] == 1:
            right_offset = 2
        
        if state[0] != 2:
            if state[2] == 0:
                colors[0 + left_offset] = RED 
                colors[1 + left_offset] = RED 
            elif state[2] == 1:
                colors[1 + left_offset] = RED
            elif state[2] == 2:
                colors[0 + left_offset] = RED
        
        if state[1] != 2:
            if state[3] == 0:
                colors[4 + right_offset] = RED
                colors[5 + right_offset] = RED 
            elif state[3] == 1:
                colors[4 + right_offset] = RED
            elif state[3] == 2:
                colors[5 + right_offset] = RED 
        
        return colors

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
        for wall in CIRCULAR_WALLS:
            pygame.draw.circle(self.screen, WALL_COLOR, wall[:2], wall[2], 1)
        for sensor in car.sensors:
            pygame.draw.line(self.screen, YELLOW, sensor.start_pos, sensor.end_pos, 2)
        pygame.draw.line(self.screen, WHITE, (WIDTH + 50, 0), (WIDTH + 50, HEIGHT + 300), 2) 
        
        pygame.draw.rect(self.screen, BLACK, self.text_area_rect, 2)

        pol_colors = self.state_translate(self.current_obs)
        for i, polygon in enumerate(self.state_sensors):
            pygame.draw.polygon(self.screen, pol_colors[i], polygon)

        # Render and display the input text
        lines = info.split('\n')
        y_offset = 15  # Initial y-offset
        for line in lines:
            text_surface = self.font.render(line, True, text_color)
            text_rect = text_surface.get_rect(topleft=(self.text_area_rect.left + 10, self.text_area_rect.top + y_offset))
            self.screen.blit(text_surface, text_rect)
            y_offset += font_size + 2
        
        # Render image
        self.screen.blit(self.image, self.image_rect)
    
        all_sprites.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(FPS)
        return False
        
    def close(self):
        # Perform cleanup operations, if needed
        pygame.quit()
