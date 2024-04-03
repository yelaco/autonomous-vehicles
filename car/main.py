from car import Car
from obstacle_avoiding import ObstacleAvoidingThread
from object_detecting import ObjectDetectingThread

car = Car()
object_avoider = ObstacleAvoidingThread(car)
object_detector = ObjectDetectingThread(car, detecting=False)

while True:
    if object_detector.tracking:
        object_avoider.avoiding = False
    else:
        object_avoider.avoiding = True