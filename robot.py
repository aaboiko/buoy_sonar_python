import numpy as np
from transducer import Transducer
from imu import IMU

class Robot:
    def __init__(self, angle, pan_min=0, pan_max=360, tilt_min=-5, tilt_max=5):
        self.pose = {
            "x": 0,
            "y": 0,
            "z": 0,
            "roll": 0,
            "pitch": 0,
            "yaw": 0
        }

        self.transducers = self.create_transducers(pan_min, pan_max, tilt_min, tilt_max, angle)
        self.imu = IMU()


    def create_transducers(self, pan_min, pan_max, tilt_min, tilt_max, angle):
        n_pan = (pan_max - pan_min) // angle
        n_tilt = (tilt_max - tilt_min) // angle
        n = n_pan * n_tilt
        transducers = []

        for i in range(n_pan):
            for j in range(n_tilt):
                tranducer = Transducer(angle, pan=angle*i, tilt=angle*j)
                transducers.append(tranducer)

        return transducers
    

    def get_pose(self):
        self.pose = self.imu.get_pose()
        x = self.pose["x"]
        y = self.pose["y"]
        z = self.pose["z"]
        roll = self.pose["roll"]
        pitch = self.pose["pitch"]
        yaw = self.pose["yaw"]
        
        return np.array([x, y, z, roll, pitch, yaw])
    

    def set_transducer_value(self, index, r, a):
        self.transducers[index].set_value(r, a)


    def get_transducers(self):
        return self.transducers
    

    def set_transducers(self, transducers):
        self.transducers = transducers