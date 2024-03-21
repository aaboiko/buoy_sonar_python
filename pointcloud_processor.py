import numpy as np

class PointcloudProcessor:
    def _init__(self):
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
    

    def center_of_mass(cloud):
        sum = np.zeros(3)
        n = len(cloud)

        for point in cloud:
            sum += point

        return sum / n
