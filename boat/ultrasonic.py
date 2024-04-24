import numpy as np
import threading

class UltrasonicThread(threading.Thread):
    def __init__(self, Ab, name='ultrasonic-thread'):
        self.running = True
        self.latest_measure = [100, 100, 100, 100, 100]
        self.Ab = Ab
        super(UltrasonicThread, self).__init__(name=name)
        self.start()

    def run(self):
        while self.running:
            self.latest_measure = [(int(dist) if dist < 100 else 100) if dist >= 0 else 0 for dist in self.Ab.SR04()]
            
    def measure_distances(self):
        return self.latest_measure 
