from robot import Robot
from environment import Environment, Sphere, Cube
from transform import Transform as tf
from transducer import Transducer
import time
import numpy as np

class Supervisor:
    def __init__(self, sonar_angle) -> None:
        self.environment = Environment()
        self.robot = Robot(sonar_angle)


    def spin(self):
        transducers = self.robot.get_transducers()
        objects = self.environment.get_objects()

        for obj in objects:
            for transducer in transducers:
                pan, tilt = transducer.get_pan_tilt()


    def run(self):
        while(True):
            pass


def main():
    supervisor = Supervisor(10)

if __name__ == '__main__':
    main()