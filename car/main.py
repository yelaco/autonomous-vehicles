from car import Car
from obstacle_avoiding import ObstacleAvoidingThread
from object_detecting import ObjectDetectingThread
import time

car = Car()
object_avoider = ObstacleAvoidingThread(car, avoiding=False)
object_detector = ObjectDetectingThread(car)

while True:
    if object_detector.tracking:
        object_avoider.avoiding = False
    else:
        if car.stopped:
            time.sleep(5)
            car.stopped = True
        object_avoider.avoiding = True
