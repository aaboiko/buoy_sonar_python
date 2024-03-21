import numpy as np
from transducer import Transducer
from imu import IMU
from map import Map
import threading
import time

class Robot:
    def __init__(self):
        self.pose = {
            "x": 6,
            "y": 6,
            "z": 0,
            "roll": 0,
            "pitch": 0,
            "yaw": 0
        }

        self.imu = IMU()


    def create_transducers(self, pan_min, pan_max, tilt_min, tilt_max, angle):
        transducers = []

        for tilt in np.arange(tilt_min, tilt_max, angle):
            sensor_row = []

            for pan in np.arange(pan_min, pan_max, angle):
                tranducer = Transducer(angle, pan, tilt)
                sensor_row.append(tranducer)

            transducers.append(sensor_row)

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