import numpy as np

class Transform:
    def __init__(self):
        pass

    def euler_from_quaternion(self, q):
        re, v1, v2, v3 = q
        phi = np.arctan2(2 * (v3*v2 + re*v1), re**2 - v1**2 - v2**2 + v3**2)
        theta = np.arcsin(2 * (re*v2 - v1*v3))
        psi = np.arctan2(2 * (v1*v2 + re*v3), re**2 + v1**2 - v2**2 - v3**2)

        return np.array([phi, theta, psi])
    

    def quaternion_from_euler(self, v):
        phi, theta, psi = v
        w = np.cos(phi/2)*np.cos(theta/2)*np.cos(psi/2) + np.sin(phi/2)*np.sin(theta/2)*np.sin(psi/2)
        x = np.sin(phi/2)*np.cos(theta/2)*np.cos(psi/2) - np.cos(phi/2)*np.sin(theta/2)*np.sin(psi/2)
        y = np.cos(phi/2)*np.sin(theta/2)*np.cos(psi/2) + np.sin(phi/2)*np.cos(theta/2)*np.sin(psi/2)
        z = np.cos(phi/2)*np.cos(theta/2)*np.sin(psi/2) - np.sin(phi/2)*np.sin(theta/2)*np.cos(psi/2)

        return np.array([w, x, y, z])
    

    def transform(self, point, angles, trans):
        roll, pitch, yaw = angles

        R_roll = np.array([[np.cos(roll), -np.sin(roll), 0],
                           [np.sin(roll), np.cos(roll), 0],
                           [0,            0,            1]])
        
        R_pitch = np.array([[1,     0,                 0],
                            [0, np.cos(pitch), -np.sin(pitch)],
                            [0, np.sin(pitch), np.cos(pitch)]])
        
        R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                           [np.sin(yaw), np.cos(yaw), 0],
                           [0,            0,            1]])
        
        R = R_roll @ R_pitch @ R_yaw
        return R.T @ (point - trans)
    

    def cart_to_spherical(self, point):
        x, y, z = point
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)
        phi = np.arctan2(z, np.sqrt(x**2 + y**2))

        return np.array([r, phi, theta])