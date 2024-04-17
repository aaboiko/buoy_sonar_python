import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse, Circle
import time
import pickle
import os

from robot import Robot
from environment import Environment, Sphere, Ellipsoid
from transform import Transform as tf
from transducer import Transducer, SingleSonar
from map import Map
from pointcloud_processor import PointcloudProcessor as pp
from filters import CTKalmanFilter, CVKalmanFilter, CVKalmanFilter_7D, CTKalmanFilter_7D, DubinsKalmanFilter

from sklearn import model_selection
from sklearn import linear_model, svm

SONAR_MAX_RANGE = 150
T = 0.1

def create_transducers(n_sonars, n_rays):
    transducers = []
    step = 360 // n_sonars

    for i in range(n_sonars):
        transducer = SingleSonar(i * step, 0, step, n_rays)
        transducers.append(transducer)

    return transducers


def draw_scene(objs, angle):
    angle = np.deg2rad(angle)

    for obj in objs:
        pose = obj.get_pose()
        ellipse = Ellipse(xy=[pose[0], pose[1]], width=2*obj.a, height=2*obj.b, angle=np.rad2deg(pose[3]))
        ellipse.set_alpha(0.25)
        plt.gca().add_artist(ellipse)

    x_end = SONAR_MAX_RANGE * np.cos(angle/2)
    y_end = SONAR_MAX_RANGE * np.sin(angle/2)
    plt.plot([0, x_end], [0, y_end], 'r-', color='blue')
    plt.plot([0, x_end], [0, -y_end], 'r-', color='blue')

    plt.gca().set_aspect('equal')
    plt.show()


def visualize_measures(meas):
    aa = []
    rr = []

    for i in np.arange(0, SONAR_MAX_RANGE, sonar.r_step):
        rr.append(i)
        aa.append(meas[i])
        
    plt.plot(rr, aa)
    plt.show()


def measure_ellipsoid(axis, center, bias, A):
    ax, ay, az = axis
    xb, yb, zb = bias - center
    
    a11, a12, a13 = A[0,:]
    a21, a22, a23 = A[1,:]
    a31, a32, a33 = A[2,:]

    a_12 = a12 + a21
    a_13 = a13 + a31
    a_23 = a23 + a32

    a_sq = a11*ax**2 + a22*ay**2 + a33*az**2 + a_12*ax*ay + a_13*ax*az + a_23*ay*az
    b_sq = 2*(a11*ax*xb + a22*ay*yb + a33*az*zb) + a_12*(ax*yb + ay*xb) + a_13*(ax*zb + az*xb) + a_23*(ay*zb + az*yb)
    c_sq = a11*xb**2 + a22*yb**2 + a33*zb**2 + a_12*xb*yb + a_13*xb*zb + a_23*yb*zb - 1
    D = b_sq**2 - 4 * a_sq * c_sq

    if D >= 0:
        t = (-b_sq - np.sqrt(D)) / (2 * a_sq)
        if t >= 0:
            r = np.array([ax*t, ay*t, az*t])
            return np.linalg.norm(r)

    return SONAR_MAX_RANGE


def single_sonar_measure(sonar, obj, robot_xyz, robot_rpy, visualize=False):
    rays = sonar.get_rays()
    n = len(rays)
    a = 0
    r_res = SONAR_MAX_RANGE
    
    meas = dict()
    for i in np.arange(0, SONAR_MAX_RANGE + 1, sonar.r_step):
        meas[i] = 0

    progress = 0
    prev = 0
    iter = 0

    for ray in rays:
        progress = int(100 * iter / n)

        if progress > prev:
            print('single sonar measure in progress: ' + str(progress) + '%')
            prev = progress
        iter += 1

        tilt = ray["phi"]
        pan = ray["theta"]
        k = ray["k"]

        phi = np.deg2rad(tilt)
        theta = np.deg2rad(pan)
        a_loc = tf.spherical_to_cart(np.array([1, phi, theta]))
        a_world = tf.rotate_vector(a_loc, robot_rpy)

        center = obj.center
        r = measure_ellipsoid(a_world, center, robot_xyz, obj.A)

        a += k
        r_res = min(r_res, r)
        if r <= SONAR_MAX_RANGE:
            meas[int(r // sonar.r_step) * sonar.r_step] += k

        if visualize and r < SONAR_MAX_RANGE:
            plt.scatter(theta, phi, s=1, color='blue')

    if visualize:
        plt.show()

    return r, a, meas


def measure(sonars, objs, robot_pose):
    measurements = []
    robot_xyz = robot_pose[0:3]
    robot_rpy = robot_pose[3:6]

    for sonar in sonars:
        sonar.reset()
        angle = np.deg2rad(sonar.angle)

        for obj in objs:
            r, a, meas = single_sonar_measure(sonar, obj, robot_xyz, robot_rpy)
            sonar.set_value(r, a, meas)


#Main code in launched here

env = Environment()
robot = Robot()
obj = Ellipsoid(20, 30, 20, np.array([100, 0, 0, np.pi/4, 0, 0]), name='sphere')
sonar  = SingleSonar(0, 0, 45, 500)

r, a, meas = single_sonar_measure(sonar, obj, robot.get_pose()[0:3], robot.get_pose()[3:6])
print(meas[4.5])
aa = []
rr = []

for i in np.arange(0, SONAR_MAX_RANGE, sonar.r_step):
    rr.append(i)
    aa.append(meas[i])

plt.plot(rr, aa)
plt.show()

draw_scene([obj], 45)