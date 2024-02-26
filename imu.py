import numpy as np
import time

class IMU:
    def __init__(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0):
        self.state = {
            "x" : x,
            "y": y,
            "z": z,
            "roll" : roll,
            "pitch": pitch,
            "yaw": yaw
        }

        self.disturbances = {
            "x": {"amps:": [], "phases": []},
            "y":{"amps:": [], "phases": []},
            "z": {"amps:": [], "phases": []},
            "roll": {"amps:": [], "phases": []},
            "pitch": {"amps:": [], "phases": []},
            "yaw": {"amps:": [], "phases": []}
        }

    def get_state(self):
        return self.state
    
    def set_state(self, state):
        self.state = state

    def set_disturbances(self, disturbances):
        self.disturbances = disturbances

    def harmonic(self, k, amp, phase, t):
        w = 2 * np.pi * k
        return amp * np.sin(w * t + phase)
    
    def sway_func(self, amps, phases, t):
        res = 0
        for k in range(len(amps)):
            res += self.harmonic(k, amps[k], phases[k], t)

        return res
    

    def run_sway(self):
        t = time.time_ns() / 1000000
        
        x = self.sway_func(self.disturbances["x"]["amps"], self.disturbances["x"]["phases"], t)
        y = self.sway_func(self.disturbances["y"]["amps"], self.disturbances["y"]["phases"], t)
        z = self.sway_func(self.disturbances["z"]["amps"], self.disturbances["z"]["phases"], t)
        roll = self.sway_func(self.disturbances["roll"]["amps"], self.disturbances["roll"]["phases"], t)
        pitch = self.sway_func(self.disturbances["pitch"]["amps"], self.disturbances["pitch"]["phases"], t)
        yaw = self.sway_func(self.disturbances["yaw"]["amps"], self.disturbances["yaw"]["phases"], t)

        self.state = {
            "x" : x,
            "y": y,
            "z": z,
            "roll" : roll,
            "pitch": pitch,
            "yaw": yaw
        }