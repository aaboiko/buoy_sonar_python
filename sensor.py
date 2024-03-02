import numpy as np
import threading

c = 1500

class Sensor:
    def __init__(self, frame, buffer_size):
        self.frame = frame
        self.distance = 0
        self.distribution = []
        self.buffer_size = buffer_size
        self.meas_buffer = np.zeros(buffer_size)


    def random_test(self, distance, sigma):
        while(True):
            self.distance = distance + np.random.normal(0, sigma)
            self.meas_buffer = np.roll(self.meas_buffer, 1)
            self.meas_buffer[0] = self.distance
            print(self.meas_buffer)


    def run_random(self, distance, sigma):
        thread = threading.Thread(target=self.random_test, args=[distance, sigma])
        thread.start()


    def get_distance(self):
        return self.distance


    def get_distribution(self):
        return self.distribution
    

    def get_buffer(self):
        return self.meas_buffer
    

sensor = Sensor([0, 0, 0, 0, 0, 0], 10)
sensor.run_random(10, 1)
