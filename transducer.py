import numpy as np
import threading

class Transducer:
    def __init__(self, angle, pan=0, tilt=0, sigma_r=0.5, range_max=300):
        self.angle = angle
        self.sigma_r = sigma_r
        self.sigma_fi = self.angle / 3
        self.sigma = np.array([[self.sigma_r**2, 0],
                               [0, self.sigma_fi**2]])
        
        self.r = 0
        self.amplitude = 0
        self.pan = pan
        self.tilt = tilt
        self.range_max = range_max


    def deg_to_rad(self, angle):
        return angle / (180 / np.pi)


    def diagram(self, angle):
        return np.exp((angle**2) / (2 * self.sigma_fi**2))
    

    def set_value(self, r, a):
        self.r = r + np.random.normal(0, self.sigma_r)
        self.amplitude = a


    def get_pan_tilt(self):
        return self.pan, self.tilt
    
    
    def run(self):
        pass

    