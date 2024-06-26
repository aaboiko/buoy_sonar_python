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
from filters import CTKalmanFilter, CVKalmanFilter, CVKalmanFilter_1D
from sklearn import model_selection
from sklearn import linear_model, svm

SONAR_MAX_RANGE = 150
T = 0.1

def angle_pans_from_n_sonars(n, pan_min, pan_max):
    width = pan_max - pan_min
    angle = width / n
    pans = np.linspace(pan_min + angle/2, pan_max - angle/2, n)
    n_rays = int(60 * width / n)

    return angle, n_rays, pans

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


def animate_measures(data):
    for meas in data:
        aa = []
        rr = []

        for i in np.arange(0, SONAR_MAX_RANGE, 0.5):
            rr.append(i)
            aa.append(meas[i])
            
        plt.plot(rr, aa)
        plt.axis([0, 150, 0, 0.1])
        plt.pause(0.1)
        plt.clf()

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
    a = 0
    r_res = SONAR_MAX_RANGE
    
    meas = dict()
    for i in np.arange(0, SONAR_MAX_RANGE + 1, sonar.r_step):
        meas[i] = 0

    progress = 0
    prev = 0
    iter = 0

    center = obj.center
    obj_size = max(obj.a, max(obj.b, obj.c))
    angle_thres = np.arctan2(obj_size, np.linalg.norm(center))

    sonar_pan, sonar_tilt = sonar.get_pan_tilt()
    sonar_axis_ = tf.spherical_to_cart(np.array([1, np.deg2rad(sonar_tilt), np.deg2rad(sonar_pan)]))
    sonar_axis = tf.rotate_vector(sonar_axis_, robot_rpy)

    if tf.angle_between_vectors(sonar_axis, center) > np.deg2rad(sonar.angle / 2):
        print('The object is out of sonar area')
        return r_res, a, meas
    
    center_r, center_phi, center_theta = tf.cart_to_spherical(center)
    center_pan = np.rad2deg(center_theta)
    center_tilt = np.rad2deg(center_phi)
    pan_min = center_pan - np.rad2deg(angle_thres)
    pan_max = center_pan + np.rad2deg(angle_thres)
    tilt_min = center_tilt - np.rad2deg(angle_thres)
    tilt_max = center_tilt + np.rad2deg(angle_thres)

    if pan_min < sonar_pan - sonar.angle / 2:
        pan_min = sonar_pan - sonar.angle / 2
    if pan_max > sonar_pan + sonar.angle / 2:
        pan_max = sonar_pan + sonar.angle / 2
    if tilt_min < sonar_tilt - sonar.angle / 2:
        tilt_min = sonar_tilt - sonar.angle / 2
    if tilt_max > sonar_tilt + sonar.angle / 2:
        tilt_max = sonar_tilt + sonar.angle / 2

    '''print('pan min: ' + str(pan_min))
    print('pan max: ' + str(pan_max))
    print('tilt min: ' + str(tilt_min))
    print('tilt max: ' + str(tilt_max))'''

    i_min = int((tilt_min - sonar_tilt + sonar.angle / 2) // sonar.angle_step)
    i_max = int((tilt_max - sonar_tilt + sonar.angle / 2) // sonar.angle_step)
    j_min = int((pan_min - sonar_pan + sonar.angle / 2) // sonar.angle_step)
    j_max = int((pan_max - sonar_pan + sonar.angle / 2) // sonar.angle_step)

    '''print('i min: ' + str(i_min))
    print('i max: ' + str(i_max))
    print('j min: ' + str(j_min))
    print('j max: ' + str(j_max))'''


    n = (i_max - i_min) * (j_max - j_min)

    for i in range(i_min, i_max):
        for j in range(j_min, j_max):
            progress = int(100 * iter / n)

            if progress > prev:
                print('single sonar measure in progress: ' + str(progress) + '%')
                prev = progress
            iter += 1

            tilt = rays[i][j]["phi"]
            pan = rays[i][j]["theta"]
            k = rays[i][j]["k"]

            phi = np.deg2rad(tilt)
            theta = np.deg2rad(pan)
            a_loc = tf.spherical_to_cart(np.array([1, phi, theta]))
            a_world = tf.rotate_vector(a_loc, robot_rpy)

            angle_between = tf.angle_between_vectors(a_world, center)

            #print('between: ' + str(np.rad2deg(angle_between)) + ', thres: ' + str(np.rad2deg(angle_thres)))

            if angle_between > angle_thres:
                continue

            r = measure_ellipsoid(a_world, center, robot_xyz, obj.A)

            a += k
            r_res = min(r_res, r)
            if r <= SONAR_MAX_RANGE:
                meas[int(r // sonar.r_step) * sonar.r_step] += k

            if visualize and r < SONAR_MAX_RANGE:
                plt.scatter(theta, phi, s=1, color='blue')

    if visualize:
        plt.show()
    print('r = ' + str(r_res) + ', a = ' + str(a))
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


def get_params_from_meas(graph_meas):
    n = len(graph_meas)
    a_dots = []
    r_dots = []

    #1st moments
    l_mean = 0
    r_dot_mean = 0
    a_dot_mean = 0

    r_prev = 0
    a_prev = 0

    first = True

    for meas_arrays in graph_meas:
        a = 0
        r = 0
        l = 0

        for meas in meas_arrays:
            _a, _r, _l = get_arl_from_meas(meas)
            a += _a
            r = max(r, _r)
            l = max(l, _l)

        l_mean += l / n

        if not first:
            r_dot_mean += abs(r - r_prev) / (n - 1)
            a_dot_mean += abs(a - a_prev) / (n - 1)
            a_dots.append(abs(a - a_prev))
            r_dots.append(abs(r - r_prev))

        first = False
        r_prev = r
        a_prev = a

    #2nd moments
    r_dot_sigma = 0
    a_dot_sigma = 0

    for a_dot in a_dots:
        a_dot_sigma += abs(a_dot - a_dot_mean) / (n - 1)
    for r_dot in r_dots:
        r_dot_sigma += abs(r_dot - r_dot_mean) / (n - 1)

    return l_mean, r_dot_mean, a_dot_mean, r_dot_sigma, a_dot_sigma


def move_traj_and_save_meas(traj, robot, obj, sonar, datapath):
    data = []
    ind = 0

    for point in traj:
        print('processing traj point #' + str(ind) + '/' + str(traj.shape[0]))
        ind += 1
        obj.set_pose(point)
        r, a, meas = single_sonar_measure(sonar, obj, robot.get_pose()[0:3], robot.get_pose()[3:6])
        data.append(meas)

    print('saving...')
    pickle.dump(data, open(datapath, 'wb'))
    print('saved')


def move_traj(traj, robot, obj, sonar):
    for point in traj:
        obj.set_pose(point)
        r, a, meas = single_sonar_measure(sonar, obj, robot.get_pose()[0:3], robot.get_pose()[3:6])

        aa = []
        rr = []

        for i in np.arange(0, SONAR_MAX_RANGE, 0.5):
            rr.append(i)
            aa.append(meas[i])
        
        plt.clf()
        plt.plot(rr, aa)
        plt.pause(0.01)

    plt.show()


def draw_arl_from_time(dataset):
    for data_obj in dataset:
        aa1 = []
        rr1 = []
        ll1 = []
        aa2 = []
        rr2 = []
        ll2 = []
        aa3 = []
        rr3 = []
        ll3 = []
        tt = [0.1 * i for i in range(len(data_obj["graph_meas"]))]

        aa = []
        rr = []
        ll = []

        for meas_arrays in data_obj["graph_meas"]:
            meas1 = meas_arrays[0]
            meas2 = meas_arrays[1]
            #meas3 = meas_arrays[2]
            
            a1, r1, l1 = get_arl_from_meas(meas1)
            a2, r2, l2 = get_arl_from_meas(meas2)
            #a3, r3, l3 = get_arl_from_meas(meas3)

            aa1.append(a1)
            rr1.append(r1)
            ll1.append(l1)

            aa2.append(a2)
            rr2.append(r2)
            ll2.append(l2)

            #aa3.append(a3)
            #rr3.append(r3)
            #ll3.append(l3)

            aa.append(a1 + a2)
            rr.append(max(r1, r2))
            ll.append(max(l1, l2))
        
        plt.plot(tt, rr, color='red')
        #plt.plot(tt, ll, color='green')
        #plt.plot(tt, aa, color='blue')

        plt.xlabel('t, secs')
        plt.ylabel('A')
        plt.show()

            
def generate_dataset(sonars, env, objs, robot, traj_len, num_trajs, x_start_bounds, y_start_bounds):
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

            x_start_min, x_start_max = x_start_bounds
            y_start_min, y_start_max = y_start_bounds

            x_start = np.random.uniform(x_start_min, x_start_max)
            y_start = np.random.uniform(y_start_min, y_start_max)
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

                #sonars = create_sonars_by_pan(pans, angle, num_rays)
                #robot.set_transducers(sonars)

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
    pickle.dump(dataset, open('datasets/synthetic/single/single_2.bin', 'wb'))
    print('saved succesfully')


def extract_features(dataset, sigma, index):
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
        l_mean, r_dot_mean, a_dot_mean, r_dot_sigma, a_dot_sigma = get_params_from_meas(graph_meas)
            
        write_obj = {
            "name": name,
            "x_size": x_size,
            "y_size": y_size,
            "z_size": z_size,
            "v_mean": v_mean,
            "v_sigma": v_sigma,
            "curvature": curvature,
            "l_mean": l_mean,
            "r_dot_mean": r_dot_mean,
            "a_dot_mean": a_dot_mean,
            "r_dot_sigma": r_dot_sigma,
            "a_dot_sigma": a_dot_sigma
        }

        res_arr.append(write_obj)

    print('feature extracting completed. Saving...')
    pickle.dump(res_arr, open('datasets/synthetic/single/features_' + str(index) + '.bin', 'wb'))
    print('saved succesfully')


def fit_model(features):
    n = len(features) 

    names = []
    x_sizes = []
    y_sizes = []
    z_sizes = []
    v_means = []
    v_sigmas = []
    curvatures = []
    l_means = []
    r_dot_means = []
    a_dot_means = []
    r_dot_sigmas = []
    a_dot_sigmas = []

    for obj in features:
        name = obj["name"]
        x_size = obj["x_size"]
        y_size = obj["y_size"]
        z_size = obj["z_size"]
        v_mean = obj["v_mean"]
        v_sigma = obj["v_sigma"]
        curvature = obj["curvature"]
        l_mean = obj["l_mean"]
        r_dot_mean = obj["r_dot_mean"]
        a_dot_mean = obj["a_dot_mean"]
        r_dot_sigma = obj["r_dot_sigma"]
        a_dot_sigma = obj["a_dot_sigma"]

        names.append(name)
        x_sizes.append(x_size)
        y_sizes.append(y_size)
        z_sizes.append(z_size)
        v_means.append(v_mean)
        v_sigmas.append(v_sigma)
        curvatures.append(curvature)
        l_means.append(l_mean)
        r_dot_means.append(r_dot_mean)
        a_dot_means.append(a_dot_mean)
        r_dot_sigmas.append(r_dot_sigma)
        a_dot_sigmas.append(a_dot_sigma)

    X = []
    Y = []

    for i in range(n):
        name = names[i]
        x_size = x_sizes[i]
        y_size = y_sizes[i]
        z_size = z_sizes[i]
        v_mean = v_means[i]
        v_sigma = v_sigmas[i]
        curvature = curvatures[i]
        l_mean = l_means[i]
        r_dot_mean = r_dot_means[i]
        a_dot_mean = a_dot_means[i]
        r_dot_sigma = r_dot_sigmas[i]
        a_dot_sigma = a_dot_sigmas[i]

        C = 0
        if name == "human":
            C = 1
        elif name == "dolphin":
            C = 2
        elif name == "drone":
            C = 3

        Y.append(C)
        X.append([l_mean, r_dot_mean, a_dot_mean, r_dot_sigma, a_dot_sigma])

    model_rfb = svm.SVC(kernel='rbf', gamma=0.5, C=10)
    model_rfb.fit(X, Y)

    pickle.dump(model_rfb, open('models/svm/svm_single_1.bin', 'wb'))


def validate(model, num_trajs, traj_len, env, objs, robot, angle, num_rays, pans, x_start_bounds, y_start_bounds, regenerate_sonars):
    #dataset generating
    dataset = []
    n = len(objs) * num_trajs

    sonars = create_sonars_by_pan(pans, angle, num_rays)
    robot.set_transducers(sonars)

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
                print('dataset creation in progress for: ' + obj.name + ' ' + str(progress) + '%')
                prev = progress

            x_start_min, x_start_max = x_start_bounds
            y_start_min, y_start_max = y_start_bounds

            x_start = np.random.uniform(x_start_min, x_start_max)
            y_start = np.random.uniform(y_start_min, y_start_max)
            start = np.array([x_start, y_start, 0])
            angle_start = np.random.uniform(0, 2 * np.pi)
            traj = env.generate_random_trajectory(start, angle_start, v_mean, sigma_v, sigma_angle, traj_len)

            graph_meas = []
            iter = 0

            for point in traj:
                print('processing point: ' + str(iter+1) + '/' + str(traj_len) + '(angle = ' + str(angle) + ')')
                iter += 1

                env.clear()
                env.add_object(obj)
                obj.set_pose(point)

                if regenerate_sonars:
                    sonars = create_sonars_by_pan(pans, angle, num_rays)
                    robot.set_transducers(sonars)

                robot_pose = robot.get_pose()
                cloud, meas_arrays = measure(sonars, obj, robot_pose)
                graph_meas.append(meas_arrays)

            data_obj = {
                "name": obj.name,
                "graph_meas": graph_meas
            }

            dataset.append(data_obj)

    #feature extracting
    n = len(dataset)
    prev = -1
    step = 0

    X = []
    Y = []

    for data_obj in dataset:
        progress = int(100 * step / n)
        if progress > prev:
            print('feature extractor is in progress: ' + str(progress) + '%')
            prev = progress

        name = data_obj["name"]
        graph_meas = data_obj["graph_meas"]
        l_mean, r_dot_mean, a_dot_mean, r_dot_sigma, a_dot_sigma = get_params_from_meas(graph_meas)
        step += 1

        C = 0
        if name == "human":
            C = 1
        elif name == "dolphin":
            C = 2
        elif name == "drone":
            C = 3

        Y.append(C)
        X.append([l_mean, r_dot_mean, a_dot_mean, r_dot_sigma, a_dot_sigma])

    #prediction
    score = model.score(X, Y)
    print('model evaluated with score = ' + str(score))

    return score


#Main code in launched here

env = Environment()
robot = Robot()
obj = Ellipsoid(1, 1, 1, np.array([20, 20, 0, 0, 0, 0]), name='sphere')

objs = [
    #Ellipsoid(1, 1, 1, np.array([2, 6, 0, 0, 0, 0])),                        #for sonars testing
    #Ellipsoid(0.5, 3, 0.5, np.array([3, 3, 0, 0, 0, 0])),   #for spinning

    Ellipsoid(1, 1, 1, np.array([3, 3, 0, 0, 0, 0]), name='sphere'),
    Ellipsoid(0.9, 0.4, 0.4, np.array([3, 3, 0, 0, 0, 0]), name='human'),
    Ellipsoid(1.5, 0.5, 0.5, np.array([3, 3, 0, 0, 0, 0]), name='dolphin'),
    Ellipsoid(3, 1, 1.5, np.array([3, 3, 0, 0, 0, 0]), name='drone')
]

#sonar  = SingleSonar(0, 0, 15, 500)
#r, a, meas = single_sonar_measure(sonar, obj, robot.get_pose()[0:3], robot.get_pose()[3:6])

#visualize_measures(meas)
#draw_scene([obj], 15, 20)
#pans = [0]
#pans = [37.5, 52.5]
#sonars = create_sonars_by_pan(pans, 15, 900)

#clouds, meas_array = measure(sonars, obj, robot.get_pose())
#visualize_measures(meas_array[1])

#traj = np.loadtxt('trajectories/traj_linear_diag.txt', delimiter=' ')
#datapath = 'datasets/synthetic/single/utils/diag.bin'

#move_traj(traj, robot, obj, sonar)

#data = pickle.load(open(datapath, 'rb'))

#generate_dataset(sonars, env, objs, robot, 20, 100, [80, 90], [80, 90])

#dataset1 = pickle.load(open('datasets/synthetic/single/single_1.bin', 'rb'))
#dataset2 = pickle.load(open('datasets/synthetic/single/single_2.bin', 'rb'))
#dataset12 = pickle.load(open('datasets/synthetic/single/single_12.bin', 'rb'))

#score by traj lenght
'''model = pickle.load(open('models/svm/svm_single_12.bin', 'rb'))
scores = []

for traj_len in range(1, 20):
    score = validate(model, 20, traj_len, env, objs, robot, 15, 900, pans, [80, 90], [80, 90], False)
    scores.append(score)

np.savetxt('datasets/synthetic/single/scores_by_traj_len.txt', scores, delimiter=' ')'''

#score by sonar angle with pans array variation
model = pickle.load(open('models/svm/svm_single_12.bin', 'rb'))
scores = []

for n in range(1, 10):
    angle, n_rays, pans = angle_pans_from_n_sonars(n, 30, 60)
    score = validate(model, 20, 20, env, objs, robot, angle, n_rays, pans, [80, 90], [80, 90], False)
    row = [angle, n, score]
    scores.append(row)

np.savetxt('datasets/synthetic/single/scores_by_angle.txt', scores, delimiter=' ')

'''sigma_angle = 15 / (3 * np.sqrt(2))

sigma = np.array([[0.5**2, 0, 0, 0],
                [0, sigma_angle**2, 0, 0],
                [0, 0, sigma_angle**2, 0],
                [0, 0, 0, 0**2]])

extract_features(dataset1, sigma, 1)
extract_features(dataset2, sigma, 2)'''

#features concatenation 1 + 2
'''features1 = pickle.load(open('datasets/synthetic/single/features_1.bin', 'rb'))
features2 = pickle.load(open('datasets/synthetic/single/features_2.bin', 'rb'))
print(len(features1))
print(len(features2))

features12 = features1 + features2
print(len(features12))
pickle.dump(features12, open('datasets/synthetic/single/features_12.bin', 'wb'))'''
##########################################################################

'''for data_obj in dataset:
    aa1 = []
    rr1 = []
    ll1 = []
    aa2 = []
    rr2 = []
    ll2 = []
    aa3 = []
    rr3 = []
    ll3 = []
    tt = [0.1 * i for i in range(len(data_obj["graph_meas"]))]

    aa = []
    rr = []
    ll = []

    #for meas_arrays in data_obj["graph_meas"]:
    l_mean, r_dot_mean, a_dot_mean, r_dot_sigma, a_dot_sigma = get_params_from_meas(data_obj["graph_meas"])

    print('l_mean = ' + str(l_mean))
    print('r_dot_mean = ' + str(r_dot_mean))
    print('a_dot_mean = ' + str(a_dot_mean))
    print('r_dot_sigma = ' + str(r_dot_sigma))
    print('a_dot_sigma = ' + str(a_dot_sigma))'''

