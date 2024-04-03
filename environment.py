import numpy as np
from transform import Transform as tf

class Model:
    def __init__(self, vertices):
        self.vertices_local = vertices
        self.vertices_global = []

        self.state = {
            "pose": {
                "x": 0,
                "y": 0,
                "z": 0,
                "roll" : 0,
                "pitch": 0,
                "yaw": 0
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

        self.trajectory = []

        for vertice in vertices:
            pose = np.array([
                self.state["pose"]["x"],
                self.state["pose"]["y"],
                self.state["pose"]["z"],
                self.state["pose"]["roll"],
                self.state["pose"]["pitch"],
                self.state["pose"]["yaw"],
            ])

            point = tf.local_to_world(vertice, pose)
            self.vertices_global.append(point)


    def get_state(self):
        return self.state
    

    def set_state(self, state):
        self.state = state


    def set_speed_linear(self, vx, vy, vz):
        self.state["speed"] = {
            "x": vx,
            "y": vy,
            "z": vz
        }

    
    def set_pose(self, pose):
        self.state["pose"] = pose


    def get_vertices(self):
        return self.vertices_global
    

class ModelParametric:
    def __init__(self, pose, name=''):
        self.pose = pose
        self.type = "infinite"
        self.name = name
        self.trajectory = []


    def set_pose(self, pose):
        self.pose = pose


    def get_pose(self):
        return self.pose
    
    
    def get_type(self):
        return self.type


    def set_trajectory(self, traj):
        self.trajectory = traj
    

class Sphere(ModelParametric):
    def __init__(self, radius, pose, name=''):
        super().__init__(pose, name=name)
        self.radius = radius
        self.type = "sphere"


class Ellipsoid(ModelParametric):
    def __init__(self, a, b, c, pose, name=''):
        super().__init__(pose, name=name)
        self.a = a
        self.b = b
        self.c = c
        self.type = "ellipsoid"

        self.A_orth = np.eye(3) * np.array([1/(a**2), 1/(b**2), 1/(c**2)])
        self.center = self.pose[0:3]
        self.rpy = self.pose[3:6]
        self.A = tf.rotate_matrix(self.A_orth, self.rpy)

    
    def set_pose(self, pose):
        self.pose = pose
        self.center = self.pose[0:3]
        self.rpy = self.pose[3:6]
        self.A = tf.rotate_matrix(self.A_orth, self.rpy)



class Environment:
    def __init__(self):
        self.objects = []


    def add_object(self, obj):
        self.objects.append(obj)


    def add_objects(self, objects):
        for obj in objects:
            self.objects.append(obj)

    def remove_object(self, name):
        for obj in self.objects:
            if obj.name == name:
                self.objects.remove(obj)


    def clear(self):
        self.objects.clear()


    def get_objects(self):
        return self.objects
    

    def generate_sphere(self, radius, n):
        points = []

        for phi in np.linspace(0, 2 * np.pi, n):
            for theta in np.linspace(-np.pi / 2, np.pi / 2, n):
                xc = radius * np.cos(theta) * np.cos(phi)
                yc = radius * np.cos(theta) * np.sin(phi)
                zc = radius * np.sin(theta)

                point = np.array([xc, yc, zc])
                points.append(point)

        return Model(points)
    

    def generate_circle(self, radius, n):
        points = [np.zeros(3)]

        for theta in np.linspace(0, 2 * np.pi, n):
            for r in np.linspace(radius / n, radius, n):
                x1 = r * np.cos(theta)
                y1 = r * np.sin(theta)
                point = np.array([x1, y1, 0])
                points.append(point)

        return Model(points)
    

    def generate_cube(self, a):
        points = [
            np.array([a/2, a/2, a/2]),
            np.array([-a/2, a/2, a/2]),
            np.array([a/2, -a/2, a/2]),
            np.array([-a/2, -a/2, a/2]),
            np.array([a/2, a/2, -a/2]),
            np.array([-a/2, a/2, -a/2]),
            np.array([a/2, -a/2, -a/2]),
            np.array([-a/2, -a/2, -a/2])
        ]

        return Model(points)
    

    def generate_random_trajectory(self, start, angle_start, v_mean, sigma_v, sigma_angle, n):
        x0, y0, z0 = start
        p_start = np.array([x0, y0, z0, angle_start, 0, 0])
        T = 0.1

        traj = [p_start]

        for t in range(n - 1):
            x_prev, y_prev, z_prev, roll_prev, pitch_prev, yaw_prev = traj[t]
            v = np.random.normal(v_mean, sigma_v)
            angle = roll_prev + np.random.normal(0, sigma_angle)

            x = x_prev + v * T * np.cos(angle)
            y = y_prev + v * T * np.sin(angle)
            z = 0
            roll = angle
            pitch = 0
            yaw = 0

            point_new = np.array([x, y, z, roll, pitch, yaw])
            traj.append(point_new)

        res = np.zeros((n, 6))

        for i in range(n):
            x, y, z, roll, pitch, yaw = traj[i]
            res[i, 0] = x
            res[i, 1] = y
            res[i, 2] = z
            res[i, 3] = roll
            res[i, 4] = pitch
            res[i, 5] = yaw

        return res
    

    def get_trajectory_bounds(self, traj):
        xmin = np.inf
        ymin = np.inf
        xmax = -np.inf
        ymax = -np.inf

        for point in traj:
            x, y, z, roll, pitch, yaw = point
            xmin = min(xmin, x)
            ymin = min(ymin, y)
            xmax = max(xmax, x)
            ymax = max(ymax, y)

        bmax = max(xmax, ymax)
        bmin = min(xmin, ymin)

        return [xmin - 1, xmax + 1, ymin - 1, ymax + 1]

