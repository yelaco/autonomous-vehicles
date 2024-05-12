class Ultrasonic:
    def __init__(self, Ab):
        self.latest_measure = [100, 100, 100, 100, 100]
        self.Ab = Ab
            
    def measure_distances(self):
        self.latest_measure = [(int(dist) if dist < 100 else 100) if dist >= 0 else 0 for dist in self.Ab.SR04()]
