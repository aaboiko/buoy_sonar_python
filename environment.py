import numpy as np

class Model:
    def __init__(self, vertices, x=300, y=0, z=0, roll=0, pitch=0, yaw=0):
        self.vertices = vertices

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
    

    def generate_sphere(self, radius, n, x=300, y=0, z=0):
        points = []
        for phi in np.linspace(0, 2 * np.pi, n):
            for theta in np.linspace(-np.pi / 2, np.pi / 2, n):
                xc = x + radius * np.sin(theta) * np.cos(phi)
                yc = y + radius * np.sin(theta) * np.sin(phi)
                zc = z + radius * np.cos(theta)
                point = np.array([xc, yc, zc])
                points.append(point)

        return Model(points)
    

    def generate_plane(self, normal, n, x=300, y=0, z=0):
        points = []