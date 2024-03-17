import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
import time

from robot import Robot
from environment import Environment, Sphere, Ellipsoid
from transform import Transform as tf
from transducer import Transducer

def create_transducers(pan_min, pan_max, tilt_min, tilt_max, angle):
        transducers = []

        for tilt in np.arange(tilt_min, tilt_max, angle):
            sensor_row = []

            for pan in np.arange(pan_min, pan_max, angle):
                tranducer = Transducer(angle, pan, tilt)
                sensor_row.append(tranducer)

            transducers.append(sensor_row)

        return transducers


def measure_sphere(sonar, center, radius):
    theta, phi = sonar.get_pan_tilt() 
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi) 
    ax, ay, az = tf.spherical_to_cart(np.array([1, phi, theta]))
    xc, yc, zc = center

    a_sq = radius / np.sqrt(ax**2 + ay**2 + az**2)
    b_sq = -2 * (ax*xc + ay*yc + az*zc)
    c_sq = xc**2 + yc**2 + zc**2 - radius**2
    D = b_sq**2 - 4 * a_sq * c_sq

    if D >= 0:
        t = (-b_sq - np.sqrt(D)) / (2 * a_sq)
        r = np.array([ax*t, ay*t, az*t])
        sigma = sonar.get_sigma()
        return np.linalg.norm(r) + np.random.normal(0, sigma[0, 0])
    
    return np.inf


def measure_ellipse(sonar, center, a, b, c):
    theta, phi = sonar.get_pan_tilt()
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi) 
    ax, ay, az = tf.spherical_to_cart(np.array([1, phi, theta]))
    xc, yc, zc = center

    a_sq = 1 / np.sqrt((ax/a)**2 + (ay/b)**2 + (az/c)**2)
    b_sq = -2 * ((ax/a)*xc + (ay/b)*yc + (az/c)*zc)
    c_sq = (xc*a)**2 + (yc/b)**2 + (zc/c)**2 - 1
    D = b_sq**2 - 4 * a_sq * c_sq

    if D >= 0:
        t = (-b_sq - np.sqrt(D)) / (2 * a_sq)
        r = np.array([ax*t, ay*t, az*t])
        sigma = sonar.get_sigma()
        return np.linalg.norm(r) + np.random.normal(0, sigma[0, 0])
    
    return np.inf
     

sonars = create_transducers(0, 90, -20, 20, 1)
print('sonars massive created: ' + str(len(sonars)) + 'x' + str(len(sonars[0])) + ' = ' + str(len(sonars)*len(sonars[0])) + ' sonars')

env = Environment()
obj = Sphere(1, np.array([2, 2, 0]))
env.add_object(obj)

robot = Robot()
robot.set_transducers(sonars)
robot_pose = robot.get_pose()
robot_xyz = robot_pose[0:2]
robot_rpy = robot_pose[3:5]

measurements = []
for sonar_row in sonars:
    meas_row = []
    for sonar in sonar_row:
        if obj.type == "sphere":
            center = obj.get_pose()
            r = measure_sphere(sonar, center, obj.radius)
            meas_row.append(r)

    measurements.append(meas_row)
        
        
#plt.imshow(measurements)
#plt.show()
    
