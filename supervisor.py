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


    def amp_normalizer(self, A):
        return A


    def spin(self):
        transducers = self.robot.get_transducers()
        objects = self.environment.get_objects()
        robot_pose = self.robot.get_pose()
        transducers_new = []

        for obj in objects:
            vertices = obj.get_vertices()

            for transducer in transducers:
                a_sum = 0
                r_sum = 0
                pan, tilt = transducer.get_pan_tilt()

                for p in vertices:
                    p_robot = tf.world_to_local(p, robot_pose)
                    p_sph = tf.cart_to_spherical(p_robot)
                    p_r, p_phi, p_theta = p_sph
                    delta_angle = tf.angle_diff(p_theta, p_phi, pan, tilt)
                    A = transducer.diagram(delta_angle)
                    a_sum += A
                    r_sum += self.amp_normalizer(A) * p_r

                r = r_sum / a_sum
                transducer.set_value(r, a_sum)
                transducers_new.append(transducer)

        self.robot.set_transducers(transducers_new)


    def run(self):
        while(True):
            pass


def main():
    supervisor = Supervisor(10)

if __name__ == '__main__':
    main()