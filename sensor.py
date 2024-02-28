import numpy as np
import threading

c = 1500

class Sensor:
    def __init__(self, frame):
        self.frame = frame
        self.distance = 0


    def random_test(self, distance, sigma):
        while(True):
            self.distance = distance + np.random.normal(0, sigma)


    def run_random(self, distance, sigma):
        thread = threading.Thread(target=self.random_test)


    def get_distance(self):
        return self.distance