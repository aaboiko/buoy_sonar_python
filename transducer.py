import numpy as np
import threading

class Transducer:
    def __init__(self, angle, pan=0, tilt=0, sigma_r=0.5, range_max=300, sigma_a=0):
        self.angle = angle
        self.sigma_r = sigma_r
        self.sigma_a = sigma_a
        self.sigma_angle = self.angle / (3 * np.sqrt(2))

        self.sigma = np.array([[self.sigma_r**2, 0, 0, 0],
                               [0, self.sigma_angle**2, 0, 0],
                               [0, 0, self.sigma_angle**2, 0],
                               [0, 0, 0, self.sigma_a**2]])
        
        self.r = 0
        self.amplitude = 0
        self.pan = pan
        self.tilt = tilt
        self.range_max = range_max


    def diagram(self, angle):
        return np.exp((angle**2) / (2 * self.sigma_fi**2))
    

    def set_value(self, r, a):
        self.r = r + np.random.normal(0, self.sigma_r)
        self.amplitude = a + np.random.normal(0, self.sigma_a)


    def get_pan_tilt(self):
        return self.pan, self.tilt
    
    
    def get_sigma(self):
        return self.sigma
    

    def get_value(self):
        return np.array([self.r, self.amplitude])

    