import numpy as np
from transducer import Transducer
from imu import IMU
from map import Map
import threading
import time

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

        self.thread_mapping = threading.Thread(target=self.run_mapping)
        self.thread_mapping.start()


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


    def run_mapping(self):
        pass