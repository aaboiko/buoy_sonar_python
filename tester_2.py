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
    step = 360.0 / n_sonars

    for i in range(n_sonars):
        transducer = SingleSonar(i * step, 0, step, n_rays)
        transducers.append(transducer)

    return transducers


def create_sonars_by_pan(pans, angle, n_rays):
    sonars = []

    for pan in pans:
        sonar = SingleSonar(pan, 0, angle, n_rays)
        sonars.append(sonar)

    return sonars


def draw_scene(objs, angle, ray_len):
    angle = np.deg2rad(angle)

    for obj in objs:
        pose = obj.get_pose()
        ellipse = Ellipse(xy=[pose[0], pose[1]], width=2*obj.a, height=2*obj.b, angle=np.rad2deg(pose[3]))
        ellipse.set_alpha(0.25)
        plt.gca().add_artist(ellipse)

    x_end = ray_len * np.cos(angle/2)
    y_end = ray_len * np.sin(angle/2)
    plt.plot([0, x_end], [0, y_end], 'r-', color='blue')
    plt.plot([0, x_end], [0, -y_end], 'r-', color='blue')

    plt.gca().set_aspect('equal')
    plt.show()


def visualize_measures(meas):
    aa = []
    rr = []

    for i in np.arange(0, SONAR_MAX_RANGE, 0.5):
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

    return r_res, a, meas


def measure(sonars, obj, robot_pose):
    cloud = []
    meas_arrays = []
    robot_xyz = robot_pose[0:3]
    robot_rpy = robot_pose[3:6]

    for sonar in sonars:
        sonar.reset()
        pan, tilt = sonar.get_pan_tilt()
        sigma = sonar.sigma_angle
        sigma_r = sonar.sigma_r
        theta = np.deg2rad(pan) + np.deg2rad(np.random.normal(0, sigma))
        phi = np.deg2rad(tilt) + np.deg2rad(np.random.normal(0, sigma))

        r, a, meas = single_sonar_measure(sonar, obj, robot_xyz, robot_rpy)
        r += np.random.normal(0, sigma_r)
        p_cart_local = tf.spherical_to_cart(np.array([r, phi, theta]))
        p_cart_global = tf.local_to_world(p_cart_local, robot_pose)

        sonar.set_value(r, a, meas)
        cloud.append(p_cart_global)
        meas_arrays.append(meas)

    return cloud, meas_arrays


def get_arl_from_meas(meas):
    a = 0
    r = 0
    amax = 0
    l = 0

    for i in np.arange(0, 150, 0.5):
        a += meas[i]
        if meas[i] > amax:
            amax = meas[i]
            r = i

    for i in np.arange(r, 150, 0.5):
        if meas[i] > 0:
            l += 0.5
        else:
            break

    return a, r, l

            
def generate_dataset(angle, env, objs, robot, traj_len, num_trajs, num_rays, pans):
    dataset = []
    n = len(objs) * num_trajs

    for obj in objs:
        if obj.name == 'sphere':
            v_mean = 1
            sigma_v = 0.1
            sigma_angle = 0
        if obj.name == 'human':
            v_mean = 1
            sigma_v = 0.3
            sigma_angle = np.pi / 10
        if obj.name == 'dolphin':
            v_mean = 5
            sigma_v = 1
            sigma_angle = np.pi / 8
        if obj.name == 'drone':
            v_mean = 10
            sigma_v = 1
            sigma_angle = 0

        prev = -1

        for i in range(num_trajs):
            progress = int(100 * i / num_trajs)
            if progress > prev:
                print('in progress for: ' + obj.name + ' ' + str(progress) + '%')
                prev = progress

            x_start = np.random.uniform(20, 30)
            y_start = np.random.uniform(20, 30)
            start = np.array([x_start, y_start, 0])
            angle_start = np.random.uniform(0, 2 * np.pi)
            traj = env.generate_random_trajectory(start, angle_start, v_mean, sigma_v, sigma_angle, traj_len)

            clouds = []
            graph_meas = []
            iter = 0

            for point in traj:
                print('processing point #' + str(iter) + ' ' + str(point))
                iter += 1

                env.clear()
                env.add_object(obj)
                obj.set_pose(point)

                sonars = create_sonars_by_pan(pans, angle, num_rays)
                robot.set_transducers(sonars)

                robot_pose = robot.get_pose()
                cloud, meas_arrays = measure(sonars, obj, robot_pose)

                clouds.append(cloud)
                graph_meas.append(meas_arrays)

            data_obj = {
                "name": obj.name,
                "clouds": clouds,
                "graph_meas": graph_meas
            }

            dataset.append(data_obj)

    print('generating completed. Saving...')
    pickle.dump(dataset, open('datasets/synthetic/single/single_1.bin', 'wb'))
    print('saved succesfully')


def extract_features(dataset, sigma):
    res_arr = []
    n = len(dataset)
    prev = -1
    step = 0

    for data_obj in dataset:
        progress = int(100 * step / n)
        if progress > prev:
            print('feature extractor is in progress: ' + str(progress) + '%')
            prev = progress

        name = data_obj["name"]
        clouds = data_obj["clouds"]
        graph_meas = data_obj["graph_meas"]

        #extracting features from clouds
        x_size, y_size, z_size = pp.get_mean_size(pp, clouds)

        iter = -2
        step += 1

        ftr = CVKalmanFilter(sigma)
        x_cur = np.array([0, 0, 0, 1, 1, 0])

        x_f = []
        traj = []
        flag = 2
        clouds_len = len(clouds)
        cloud_step = 0

        for cloud in clouds:
            cloud_step += 1
            com = pp.center_of_mass(cloud)
            print('cloud is processing: ' + str(cloud_step) + '/' + str(clouds_len))
            y_k = com

            if flag == 2:
                x_cur[0:3] = com
                flag -= 1
            elif flag == 1:
                x_cur[3:6] = com - x_cur[0:3]
                x_f.append(x_cur)
                traj.append(x_cur[0:3])
                flag -= 1
            else:
                x_new = ftr.EKF(x_f[iter], y_k)
                x_f.append(x_new)
                traj.append(x_new[0:3])

            iter += 1

        v_mean, v_sigma, curvature = pp.get_traj_params(traj)

        #extracting features from single sonar data
        for meas_arrays in graph_meas:
            for meas in meas_arrays:
                a, r, l = get_arl_from_meas(meas)
            


#Main code in launched here

env = Environment()
robot = Robot()
obj = Ellipsoid(100, 10, 10, np.array([100, 0, 0, np.pi/2, 0, 0]), name='sphere')
#sonar  = SingleSonar(0, 0, 15, 500)

#r, a, meas = single_sonar_measure(sonar, obj, robot.get_pose()[0:3], robot.get_pose()[3:6])

#visualize_measures(meas)
#draw_scene([obj], 15, 20)
pans = [0]
#pans = [-15, 0, 15, 30, 45, 60, 75]
sonars = create_sonars_by_pan(pans, 15, 300)


