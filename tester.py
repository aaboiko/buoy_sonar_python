import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
import time
import pickle
import os

from robot import Robot
from environment import Environment, Sphere, Ellipsoid
from transform import Transform as tf
from transducer import Transducer
from map import Map

SONAR_MAX_RANGE = 150

def create_transducers(pan_min, pan_max, tilt_min, tilt_max, angle):
        transducers = []

        for tilt in np.arange(tilt_min, tilt_max, angle):
            sensor_row = []

            for pan in np.arange(pan_min, pan_max, angle):
                tranducer = Transducer(angle, pan, tilt, sigma_r=0.1)
                sensor_row.append(tranducer)

            transducers.append(sensor_row)

        return transducers


def measure_sphere(a, center, bias, radius):
    ax, ay, az = a
    xc, yc, zc = center - bias

    a_sq = ax**2 + ay**2 + az**2
    b_sq = -2 * (ax*xc + ay*yc + az*zc)
    c_sq = xc**2 + yc**2 + zc**2 - radius**2
    D = b_sq**2 - 4 * a_sq * c_sq

    if D >= 0:
        t = (-b_sq - np.sqrt(D)) / (2 * a_sq)
        r = np.array([ax*t, ay*t, az*t]) - bias
        return np.linalg.norm(r) 
    
    return SONAR_MAX_RANGE


def measure_ellipse(axis, center, bias, a, b, c):
    ax, ay, az = axis
    xc, yc, zc = center - bias

    a_sq = 1 / np.sqrt((ax/a)**2 + (ay/b)**2 + (az/c)**2)
    b_sq = -2 * ((ax/(a**2))*xc + (ay/(b**2))*yc + (az/(c**2))*zc)
    c_sq = (xc*a)**2 + (yc/b)**2 + (zc/c)**2 - 1
    D = b_sq**2 - 4 * a_sq * c_sq

    if D >= 0:
        t = (-b_sq - np.sqrt(D)) / (2 * a_sq)
        r = np.array([ax*t, ay*t, az*t]) - bias
        return np.linalg.norm(r)
    
    return SONAR_MAX_RANGE
     

def measure_distances(sonars, obj, robot_pose):
    measurements = []
    robot_xyz = robot_pose[0:3]
    robot_rpy = robot_pose[3:6]

    for sonar_row in sonars:
        meas_row = []
        for sonar in sonar_row:
            pan, tilt = sonar.get_pan_tilt()
            phi = np.deg2rad(tilt)
            theta = np.deg2rad(pan)
            a_loc = tf.spherical_to_cart(np.array([1, phi, theta]))
            a_world = tf.rotate_vector(a_loc, robot_rpy)

            if obj.type == "sphere":
                center = obj.get_pose()
                r = measure_sphere(a_world, center, robot_xyz, obj.radius)
                meas_row.append(r)
                sonar.set_value(r, 1)

        measurements.append(meas_row)

    return measurements


def get_measurements_xyz(sonars, robot_pose):
    n_rows = len(sonars)
    n_cols = len(sonars[0])

    measurements_xyz = []

    for i in range(n_rows):
        xyz_row = []
        for j in range(n_cols):
            sonar = sonars[i][j]
            r = sonar.get_value()[0]

            if r < SONAR_MAX_RANGE and r != -np.inf:
                pan, tilt = sonar.get_pan_tilt()
                sigma = sonar.sigma_angle
        
                theta = np.deg2rad(pan) + np.deg2rad(np.random.normal(0, sigma))
                phi = np.deg2rad(tilt) + np.deg2rad(np.random.normal(0, sigma))
                p_cart_local = tf.spherical_to_cart(np.array([r, phi, theta]))
                p_cart_global = tf.local_to_world(p_cart_local, robot_pose)
                xyz_row.append(p_cart_global)

        measurements_xyz.append(xyz_row)

    return measurements_xyz


def move_traj_and_save_meas(traj, sonars, obj, robot_pose):
    map = []

    for point in traj:
        print('New point processing: ' + str(point))
        obj.set_pose(point)
        robot_pose = robot.get_pose()
        measure_distances(sonars, obj, robot_pose)
        measurements = get_measurements_xyz(sonars, robot_pose)
        map.append(measurements)

    pickle.dump(map, open('datasets/synthetic/sphere_linear_right_down.bin', 'wb'))
    print('done')


def update(t):
    x = t
    y = t
    point.set_data([x], [y])
    return point,


sonars = create_transducers(0, 90, -20, 20, 1)
print('sonars massive created: ' + str(len(sonars)) + 'x' + str(len(sonars[0])) + ' = ' + str(len(sonars)*len(sonars[0])) + ' sonars')

env = Environment()
obj = Sphere(1, np.array([2, 2, 0]))
print('Object created: type = ' + obj.type + ', pose = ' + str(obj.pose))
env.add_object(obj)

robot = Robot()
robot.set_transducers(sonars)

#robot_pose = robot.get_pose()
#measure_distances(sonars, obj, robot_pose)
#measurements = get_measurements_xyz(sonars, robot_pose)
        
traj = np.loadtxt('trajectories/traj_linear_right_down_diag.txt', delimiter=' ')

fig, ax = plt.subplots()
ax.axis([-10, 10, -10, 10])
ax.set_aspect('equal')

'''for meas_row in measurements:
    for meas in meas_row:
        x, y, z = meas
        ax.scatter(x, y, s=2, color='blue')

plt.show()'''

'''ani = FuncAnimation(fig, update, interval=100, blit=True, repeat=True,
                    frames=np.linspace(0, 100, 1000, endpoint=False))

plt.show()'''

#move_traj_and_save_meas(traj, sonars, obj, robot_pose)