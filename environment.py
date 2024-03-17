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
    def __init__(self, pose):
        self.pose = pose
        self.type = "infinite"


    def set_pose(self, pose):
        self.pose = pose


    def get_pose(self):
        return self.pose
    
    
    def get_type(self):
        return self.type
    

class Sphere(ModelParametric):
    def __init__(self, radius, pose):
        super().__init__(pose)
        self.radius = radius
        self.type = "sphere"


class Ellipsoid(ModelParametric):
    def __init__(self, a, b, c, pose):
        super().__init__(pose)
        self.a = a
        self.b = b
        self.c = c
        self.type = "ellipsoid"


class Environment:
    def __init__(self):
        self.objects = []


    def add_object(self, obj):
        self.objects.append(obj)


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
