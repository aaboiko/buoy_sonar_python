import numpy as np
import time
import threading
import json

class IMU:
    def __init__(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0):
        self.state = {
            "pose": {
                "x": x,
                "y": y,
                "z": z,
                "roll" : roll,
                "pitch": pitch,
                "yaw": yaw
            },
            "speed": {
                "x": 0,
                "y": 0,
                "z": 0,
                "roll" : 0,
                "pitch": 0,
                "yaw": 0
            }
        }

        file = open('config/imu_disturbances.json')
        self.disturbances = json.load(file)
        file.close()

        thread_imu = threading.Thread(target=self.run_sway)
        #thread_imu.start()


    def get_state(self):
        return self.state
    

    def get_pose(self):
        return self.state["pose"]


    def harmonic(self, amp, freq, phase, t):
        w = 2 * np.pi * freq
        return amp * np.sin(w * t + phase)
    

    def sway_func(self, disturbance, t):
        res = 0
        
        for harm in disturbance:
            amp = harm["amp"]
            freq = harm["freq"]
            phase = harm["phase"]
            res += self.harmonic(amp, freq, phase, t)

        return res
    

    def run_sway(self):
        time_start = time.time()

        while(True):
            t = time.time() - time_start
            
            x = self.sway_func(self.disturbances["x"], t)
            y = self.sway_func(self.disturbances["y"], t)
            z = self.sway_func(self.disturbances["z"], t)
            roll = self.sway_func(self.disturbances["roll"], t)
            pitch = self.sway_func(self.disturbances["pitch"], t)
            yaw = self.sway_func(self.disturbances["yaw"], t)

            self.state["pose"] = {
                "x": x,
                "y": y,
                "z": z,
                "roll" : roll,
                "pitch": pitch,
                "yaw": yaw
            }
