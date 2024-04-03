import numpy as np

T = 0.1

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
    

    def get_margin_points(cloud):
        n = len(cloud)

        if n == 0:
            print('Zero pointcloud')
            return np.inf*np.ones(3),  np.inf*np.ones(3),  np.inf*np.ones(3),  np.inf*np.ones(3),  np.inf*np.ones(3),  np.ones(3)
        
        p_xmin = cloud[0]
        p_ymin = cloud[0]
        p_zmin = cloud[0]
        p_xmax = cloud[0]
        p_ymax = cloud[0]
        p_zmax = cloud[0]

        for point in cloud:
            x, y, z = point

            if x < p_xmin[0]:
                p_xmin = point
            if x > p_xmax[0]:
                p_xmax = point
            if y < p_ymin[1]:
                p_ymin = point
            if y > p_ymax[1]:
                p_ymax = point
            if z < p_zmin[2]:
                p_zmin = point
            if z > p_zmax[2]:
                p_zmax = point

        return p_xmin, p_xmax, p_ymin, p_ymax, p_zmin, p_zmax
    

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
    

    def get_mean_size(self, clouds):
        n = len(clouds)
        res = np.zeros(3)

        for cloud in clouds:
            size_x, size_y, size_z = self.get_size(cloud)
            res += np.array([size_x, size_y, size_z]) / n
        
        res_x, res_y, res_z = res
        return res
    

    def get_traj_params(traj):
        traj_len = 0
        point_prev = traj[0]
        n_points = len(traj)

        for point in traj[1:]:
            dist = np.linalg.norm(point - point_prev)
            traj_len += dist
            point_prev = point

        v_mean = traj_len / (T * n_points)

        point_prev = traj[0]
        variance = 0

        for point in traj[1:]:
            v_loc = np.linalg.norm(point - point_prev) / T
            point_prev = point
            variance += ((v_loc - v_mean)**2) / n_points

        v_sigma = np.sqrt(variance)

        traj_1 = [np.zeros(3)]
        traj_2 = [np.zeros(3)]
        point_prev = traj[0]

        for point in traj[1:]:
            v_loc = (point - point_prev) / T
            point_prev = point
            traj_1.append(v_loc)

        point_prev = traj_1[0]
        for point in traj_1[1:]:
            a_loc = (point - point_prev) / T
            point_prev = point
            traj_2.append(a_loc)

        curvature = 0
        for i in range(n_points):
            dot_1 = traj_1[i]
            dot_2 = traj_2[i]
            
            if np.linalg.norm(dot_1) > 0:
                kappa = np.linalg.norm(np.cross(dot_1, dot_2)) / (np.linalg.norm(dot_1)**3)
                curvature += kappa / n_points

        return v_mean, v_sigma, curvature


    def get_embedding(cloud):
        embed = np.array([])
