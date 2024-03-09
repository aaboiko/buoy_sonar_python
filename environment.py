import numpy as np
from transform import Transform as tf

class Model:
    def __init__(self, vertices, pose):
        x, y, z, roll, pitch, yaw = pose
        self.vertices = vertices
        self.points = []

        self.state = {
            "pose": {
                "x": x,
                "y": y,
                "z": z,
                "roll" : roll,
                "pitch": pitch,
                "yaw": yaw
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
            point = tf.local_to_world(vertice, pose)
            self.points.append(point)


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


    def get_vertices(self):
        return self.vertices


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
                xc = radius * np.sin(theta) * np.cos(phi)
                yc = radius * np.sin(theta) * np.sin(phi)
                zc = radius * np.cos(theta)

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
