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
from transducer import Transducer
from map import Map
from pointcloud_processor import PointcloudProcessor as pp
from filters import CTKalmanFilter, CVKalmanFilter, CVKalmanFilter_7D, CTKalmanFilter_7D, DubinsKalmanFilter

SONAR_MAX_RANGE = 150
T = 0.1

def single_measure_test(env, sonars, objs, robot, frame_size):
    env.add_objects(objs)
    robot_pose = robot.get_pose()
    measure_distances(sonars, objs, robot_pose)
    measurements = get_measurements_xyz(sonars, robot_pose)

    for meas_row in measurements:
        for meas in meas_row:
            x, y, z = meas
            plt.scatter(x, y, s=2, color='blue')
            #print('x = ' + str(x) + ', y = ' + str(y))
            
    plt.axis(frame_size)
    plt.gca().set_aspect('equal')
    plt.show()


def create_transducers(pan_min, pan_max, tilt_min, tilt_max, angle, sigma):
        transducers = []

        for tilt in np.arange(tilt_min, tilt_max, angle):
            sensor_row = []

            for pan in np.arange(pan_min, pan_max, angle):
                tranducer = Transducer(angle, pan, tilt, sigma_r=sigma)
                sensor_row.append(tranducer)

            transducers.append(sensor_row)

        return transducers


def measure_sphere(a, center, bias, radius, angle):
    ax, ay, az = a
    xc, yc, zc = center - bias

    a_sq = ax**2 + ay**2 + az**2
    b_sq = -2 * (ax*xc + ay*yc + az*zc)
    c_sq = xc**2 + yc**2 + zc**2 - radius**2
    D = b_sq**2 - 4 * a_sq * c_sq

    if D >= 0:
        t = (-b_sq - np.sqrt(D)) / (2 * a_sq)
        if t >= 0:
            r = np.array([ax*t, ay*t, az*t]) 
            return np.linalg.norm(r) 
    else:
        v_c = center - bias
        delta_angle = tf.angle_between_vectors(v_c, a)

        if delta_angle <= angle:
            return np.linalg.norm(v_c) - radius
    
    return SONAR_MAX_RANGE


def measure_ellipsoid(axis, center, bias, A, angle):
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
    else:
        U, J, V = np.linalg.svd(A)
        semiax = np.min(J)
        v_c = center - bias
        delta_angle = tf.angle_between_vectors(v_c, axis)
        #print('delta_angle = ' + str(np.rad2deg(angle)))

        if delta_angle <= angle:
            #print('Object in sector! angle: ' + str(np.rad2deg(delta_angle)))
            return np.linalg.norm(v_c) - semiax

    return SONAR_MAX_RANGE


def reset_values(sonars):
    for sonar_row in sonars:
        for sonar in sonar_row:
            sonar.set_value(SONAR_MAX_RANGE, 0)


def measure_distances(sonars, objs, robot_pose):
    measurements = []
    robot_xyz = robot_pose[0:3]
    robot_rpy = robot_pose[3:6]

    for sonar_row in sonars:
        meas_row = []
        for sonar in sonar_row:
            sonar.reset()
            angle = np.deg2rad(sonar.angle)

            for obj in objs:
                pan, tilt = sonar.get_pan_tilt()
                phi = np.deg2rad(tilt)
                theta = np.deg2rad(pan)
                a_loc = tf.spherical_to_cart(np.array([1, phi, theta]))
                a_world = tf.rotate_vector(a_loc, robot_rpy)

                if obj.type == "sphere":
                    center = obj.get_pose()
                    r = measure_sphere(a_world, center, robot_xyz, obj.radius, angle)
                    if sonar.get_value()[0] > r:
                        meas_row.append(r)
                        sonar.set_value(r, 1)

                if obj.type == "ellipsoid":
                    center = obj.center
                    r = measure_ellipsoid(a_world, center, robot_xyz, obj.A, angle)
                    if sonar.get_value()[0] > r:
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


def get_cloud_from_measurements(measurements):
    cloud = []

    for meas_row in measurements:
        for meas in meas_row:
            cloud.append(meas)

    return cloud


def move_traj_animation(traj, sonars, obj, robot, clear=True, ftr_type='cv7d'):
    fig, ax = plt.subplots()
    #plt.set_aspect('equal')

    #n = len(trajs)
    #assert len(trajs) == len(objs)
    iter = -2
    sigma = sonars[0][0].get_sigma()
    bounds = env.get_trajectory_bounds(traj)

    if ftr_type == 'cv7d':
        ftr = CVKalmanFilter_7D(sigma)
        x_cur = np.array([0, 0, 0, 1, 1, 0])  
    if ftr_type == 'ct7d':
        ftr = CTKalmanFilter_7D(sigma)
        x_cur = np.array([0, 0, 0, 1, 1])
    if ftr_type == 'cv':
        ftr = CVKalmanFilter(sigma)
        x_cur = np.array([0, 0, 0, 1, 1, 0])
    if ftr_type == 'ct':
        ftr = CTKalmanFilter(sigma)
        x_cur = np.array([0, 0, 0, 1, 1])
    if ftr_type == 'dub':
        ftr = DubinsKalmanFilter(sigma)
        x_cur = np.array([0, 0, 0, 1, 0, 0])

    x_f = []
    flag = 2

    for point in traj:
        print(point)
        obj.set_pose(point)
        robot_pose = robot.get_pose()
        measure_distances(sonars, [obj], robot_pose)
        measurements = get_measurements_xyz(sonars, robot_pose)
        cloud = get_cloud_from_measurements(measurements)

        '''for p in cloud:
            x, y, z = p
            plt.scatter(x, y, s=2, color='blue')'''

        com = pp.center_of_mass(cloud)
        p_xmin, p_xmax, p_ymin, p_ymax, p_zmin, p_zmax = pp.get_margin_points(cloud)
        #y_k = com[0:3]

        if ftr_type == 'cv7d':
            y_k = np.block([com, p_xmin, p_xmax, p_ymin, p_ymax, p_zmin, p_zmax])
        if ftr_type == 'ct7d':
            y_k = np.block([com[0:2], p_xmin[0:2], p_xmax[0:2], p_ymin[0:2], p_ymax[0:2], p_zmin[0:2], p_zmax[0:2]])
        if ftr_type == 'cv' or ftr_type == 'dub':
            y_k = com
        if ftr_type == 'ct':
            y_k = com[0:2]

        plt.scatter(com[0], com[1], s=2, color='red')

        if flag == 2:
            if ftr_type == 'cv7d' or ftr_type == 'cv' or ftr_type == 'dub':
                x_cur[0:3] = com
            if ftr_type == 'ct7d' or ftr_type == 'ct':
                x_cur[0:2] = com[0:2]

            flag -= 1
        elif flag == 1:
            if ftr_type == 'cv7d' or ftr_type == 'cv':
                x_cur[3:6] = com - x_cur[0:3]
            if ftr_type == 'ct7d' or ftr_type == 'ct':
                x_cur[2:4] = np.zeros(2)
            if ftr_type == 'dub':
                x_cur[3:6] = np.zeros(3)

            x_f.append(x_cur)
            flag -= 1
        else:
            x_new = ftr.EKF(x_f[iter], y_k)
            x_f.append(x_new)
            com_filtered = x_new[0:3]
            #print('com_filtered: ' + str(com_filtered))
            plt.scatter(com_filtered[0], com_filtered[1], s=3, color='blue')
        
        ellipse = Ellipse(xy=[point[0], point[1]], width=2*obj.a, height=2*obj.b, angle=np.rad2deg(point[3]))
        ellipse.set_alpha(0.25)
        plt.gca().add_artist(ellipse)

        plt.axis(bounds)           
        plt.gca().set_aspect('equal')

        plt.plot(traj[0:iter, 0], traj[0:iter, 1], 'r-', color='green')

        plt.pause(0.01)

        ellipse.remove()
        if clear:
            plt.clf()
        #print(measurements)
        iter += 1

    plt.show()


def move_traj_and_save_meas(traj, sonars, obj, robot):
    map = []

    for point in traj:
        print('New point processing: ' + str(point))
        obj.set_pose(point)
        robot_pose = robot.get_pose()
        measure_distances(sonars, [obj], robot_pose)
        measurements = get_measurements_xyz(sonars, robot_pose)
        map.append(measurements)

    pickle.dump(map, open('datasets/synthetic/sphere_linear_right_down.bin', 'wb'))
    print('done')


def move_traj_and_concat_cloud(traj, sonars, obj, robot):
    cloud_sum = []
    n = len(traj)
    iter = 0
    progress_prev = -1

    for point in traj:
        progress = int(iter / n) * 100
        if progress > progress_prev:
            print('in progress: ' + str(progress) + str('%'))
            progress_prev = progress

        obj.set_pose(point)
        robot_pose = robot.get_pose()
        measure_distances(sonars, [obj], robot_pose)
        measurements = get_measurements_xyz(sonars, robot_pose)
        cloud = get_cloud_from_measurements(measurements)
        cloud_c = pp.centrate(cloud)
        #print('iter: '+ str(iter))
        iter += 1

        

        for point in cloud_c:
            cloud_sum.append(point)
            x, y, z = point 
            plt.scatter(x, y, s=2, color='blue')

    plt.show()


def generate_dataset_morphologic(angle, env, objs, robot):
    dataset = []

    for obj in objs:
        cloud_obj = []

        for x in np.arange(10, 150, 10):
            print('processing: ' + str(int(100 * x / 140)) + '%')
            for roll in np.linspace(0, 2*np.pi, 10):
                for pitch in np.linspace(0, np.pi/2, 10):
                    env.clear()
                    env.add_object(obj)
                    obj_pose = np.array([x, 0, 0, roll, pitch, 0])
                    obj.set_pose(obj_pose)

                    sonars = create_transducers(-45, 45, -10, 10, angle, 0.1)
                    #print('sonars massive created: ' + str(len(sonars)) + 'x' + str(len(sonars[0])) + ' = ' + str(len(sonars)*len(sonars[0])) + ' sonars')
                    robot.set_transducers(sonars)

                    robot_pose = robot.get_pose()
                    measure_distances(sonars, [obj], robot_pose)
                    measurements = get_measurements_xyz(sonars, robot_pose)

                    cloud_raw = get_cloud_from_measurements(measurements)
                    cloud = pp.centrate(cloud_raw)

                    for point in cloud:
                        cloud_obj.append(point)

                    frame = {
                        "name": obj.name,
                        "cloud": cloud
                    }
                    dataset.append(frame)

        print(obj.name + ' processing done. Saving data...')
        pickle.dump(cloud_obj, open('datasets/synthetic/morphologic/concatenated/' + obj.name+ '_angle_' + str(angle) + '.bin', 'wb'))
        print(obj.name + ' processing finished. Data saved')

    print('Done. Saving dump...')
    pickle.dump(dataset, open('datasets/synthetic/morphologic/separate/morphologic_1_angle_' + str(angle) + '.bin', 'wb'))
    print('finished')


def generate_dataset_united(angle, env, objs, robot, traj_len, num_trajs):
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

            traj_data = []
            iter = 0

            for point in traj:
                print('processing point #' + str(iter) + ' ' + str(point))
                iter += 1

                env.clear()
                env.add_object(obj)
                obj.set_pose(point)

                sonars = create_transducers(-180, 170, -10, 10, angle, 0.1)
                robot.set_transducers(sonars)

                robot_pose = robot.get_pose()
                measure_distances(sonars, [obj], robot_pose)
                measurements = get_measurements_xyz(sonars, robot_pose)

                cloud = get_cloud_from_measurements(measurements)
                traj_data.append(cloud)

            data_obj = {
                "name": obj.name,
                "data": traj_data
            }

            dataset.append(data_obj)

    print('generating completed. Saving...')
    pickle.dump(dataset, open('datasets/synthetic/united/united_validate_1.bin', 'wb'))
    print('saved succesfully')


def extract_features(dataset, sigma, ftr_type='cv'):
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
        clouds = data_obj["data"]

        x_size, y_size, z_size = pp.get_mean_size(pp, clouds)

        iter = -2
        step += 1

        if ftr_type == 'cv7d':
            ftr = CVKalmanFilter_7D(sigma)
            x_cur = np.array([0, 0, 0, 1, 1, 0])  
        if ftr_type == 'ct7d':
            ftr = CTKalmanFilter_7D(sigma)
            x_cur = np.array([0, 0, 0, 1, 1])
        if ftr_type == 'cv':
            ftr = CVKalmanFilter(sigma)
            x_cur = np.array([0, 0, 0, 1, 1, 0])
        if ftr_type == 'ct':
            ftr = CTKalmanFilter(sigma)
            x_cur = np.array([0, 0, 0, 1, 1])
        if ftr_type == 'dub':
            ftr = DubinsKalmanFilter(sigma)
            x_cur = np.array([0, 0, 0, 1, 0, 0])

        x_f = []
        traj = []
        flag = 2
        clouds_len = len(clouds)
        cloud_step = 0

        for cloud in clouds:
            cloud_step += 1
            com = pp.center_of_mass(cloud)
            print('cloud is processing: ' + str(cloud_step) + '/' + str(clouds_len))
            p_xmin, p_xmax, p_ymin, p_ymax, p_zmin, p_zmax = pp.get_margin_points(cloud)

            if ftr_type == 'cv7d':
                y_k = np.block([com, p_xmin, p_xmax, p_ymin, p_ymax, p_zmin, p_zmax])
            if ftr_type == 'ct7d':
                y_k = np.block([com[0:2], p_xmin[0:2], p_xmax[0:2], p_ymin[0:2], p_ymax[0:2], p_zmin[0:2], p_zmax[0:2]])
            if ftr_type == 'cv' or ftr_type == 'dub':
                y_k = com
            if ftr_type == 'ct':
                y_k = com[0:2]

            if flag == 2:
                if ftr_type == 'cv7d' or ftr_type == 'cv' or ftr_type == 'dub':
                    x_cur[0:3] = com
                if ftr_type == 'ct7d' or ftr_type == 'ct':
                    x_cur[0:2] = com[0:2]

                flag -= 1
            elif flag == 1:
                if ftr_type == 'cv7d' or ftr_type == 'cv':
                    x_cur[3:6] = com - x_cur[0:3]
                if ftr_type == 'ct7d' or ftr_type == 'ct':
                    x_cur[2:4] = np.zeros(2)
                if ftr_type == 'dub':
                    x_cur[3:6] = np.zeros(3)

                x_f.append(x_cur)
                traj.append(x_cur[0:3])
                flag -= 1
            else:
                x_new = ftr.EKF(x_f[iter], y_k)
                x_f.append(x_new)
                traj.append(x_new[0:3])

            iter += 1

        v_mean, v_sigma, curvature = pp.get_traj_params(traj)

        write_obj = {
            "name": name,
            "x_size": x_size,
            "y_size": y_size,
            "z_size": z_size,
            "v_mean": v_mean,
            "v_sigma": v_sigma,
            "curvature": curvature
        }

        res_arr.append(write_obj)

    print('feature extracting completed. Saving...')
    pickle.dump(res_arr, open('datasets/synthetic/united/features_validate_1.bin', 'wb'))
    print('saved succesfully')


#Main code in launched here

env = Environment()
robot = Robot()

objs = [
    #Ellipsoid(1, 1, 1, np.array([2, 6, 0, 0, 0, 0])),                        #for sonars testing
    #Ellipsoid(0.5, 3, 0.5, np.array([3, 3, 0, 0, 0, 0])),   #for spinning

    Ellipsoid(1, 1, 1, np.array([3, 3, 0, 0, 0, 0]), name='sphere'),
    Ellipsoid(0.9, 0.4, 0.4, np.array([3, 3, 0, 0, 0, 0]), name='human'),
    Ellipsoid(1.5, 0.5, 0.5, np.array([3, 3, 0, 0, 0, 0]), name='dolphin'),
    Ellipsoid(3, 1, 1.5, np.array([3, 3, 0, 0, 0, 0]), name='drone')
]

for obj in objs:
    print('Object created: type = ' + obj.type + ', pose = ' + str(obj.pose))

start = np.array([20, 20, 0])
angle_start = -2 *np.pi / 3
v_mean = 1
sigma_v = 0
sigma_angle = np.pi / 8
n_points = 20
n_trajs = 100

angle = 1
#generate_dataset_morphologic(angle, env, objs, robot)
generate_dataset_united(angle, env, objs, robot, n_points, n_trajs)

sonars = create_transducers(-180, 170, -4, 4, angle, 0.1)
print('sonars massive created: ' + str(len(sonars)) + 'x' + str(len(sonars[0])) + ' = ' + str(len(sonars)*len(sonars[0])) + ' sonars')
#env.add_object(objs[0])
#robot.set_transducers(sonars)

#single_measure_test(env, sonars, objs, robot, [0, 6.5, 0, 6.5])
sigma = sonars[0][0].get_sigma()
dataset = pickle.load(open('datasets/synthetic/united/united_validate_1.bin', 'rb'))
extract_features(dataset, sigma)
        
traj_rotate = np.loadtxt('trajectories/traj_ellipse_rotate.txt', delimiter=' ')
traj_linear = np.loadtxt('trajectories/traj_linear_right_down_diag.txt', delimiter=' ')
traj_circle = np.loadtxt('trajectories/traj_circle_r5.txt', delimiter=' ')

traj_random = env.generate_random_trajectory(start, angle_start, v_mean, sigma_v, sigma_angle, n_points)

#move_traj_animation(traj_random, sonars, objs[0], robot, clear=False, ftr_type='cv')

#move_traj_and_concat_cloud(traj_linear, sonars, objs[0], robot)
#move_traj_and_save_meas(traj, sonars, obj, robot)

#map = pickle.load(open('datasets/synthetic/sphere_linear_right_down.bin', 'rb'))