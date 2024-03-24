import numpy as np

class PointcloudProcessor:
    def __init__(self):
        pass


    def get_bounds(cloud):
        xmin = np.inf
        xmax = -np.inf
        ymin = np.inf
        ymax = -np.inf
        zmin = np.inf
        zmax = -np.inf

        for point in cloud:
            x, y, z = point
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)
            zmin = min(zmin, z)
            zmax = max(zmax, z)

        return xmin, xmax, ymin, ymax, zmin, zmax
    

    def get_size(cloud):
        xmin = np.inf
        xmax = -np.inf
        ymin = np.inf
        ymax = -np.inf
        zmin = np.inf
        zmax = -np.inf

        for point in cloud:
            x, y, z = point
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)
            zmin = min(zmin, z)
            zmax = max(zmax, z)

        size_x = xmax - xmin
        size_y = ymax - ymin
        size_z = zmax - zmin

        return size_x, size_y, size_z
    

    def center_of_mass(cloud):
        sum = np.zeros(3)
        n = len(cloud)

        if n == 0:
            return np.zeros(3)

        for point in cloud:
            sum += point

        return sum / n


    def centrate(cloud):
        sum = np.zeros(3)
        n = len(cloud)

        if n == 0:
            return []

        for point in cloud:
            sum += point

        com = sum / n
        res = []

        for point in cloud:
            point_new = point - com
            res.append(point_new)

        return res
