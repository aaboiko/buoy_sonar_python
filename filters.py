import numpy as np
from numpy.linalg import inv
from numpy import sin, cos
from transform import Transform as tf

T = 0.1

class CTKalmanFilter:
    def __init__(self, sigma):
        self.P_prev = 10e6 * np.ones(5)
        self.T = T
        self.sigma_polar = sigma[0:3, 0:3]

        self.C = np.array([[1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0]])

        self.sigma_r = self.sigma_polar[0, 0]
        self.sigma_phi = self.sigma_polar[1, 1]
        self.sigma_theta = self.sigma_polar[2, 2]

        self.sigma_x = 0.2
        self.sigma_y = 0.2
        self.sigma_w = 0.001

        self.Q = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, (T*self.sigma_x)**2, 0, 0],
                            [0, 0, 0, (T*self.sigma_y)**2, 0],
                            [0, 0, 0, 0, (T*self.sigma_w)**2]])
        
        self.R = np.array([[self.sigma_r*T**2, 0],
                           [0, self.sigma_theta*T**2]])


    def jac_3D(self, r, phi, theta):
        return np.array([[cos(phi)*cos(theta), -r*sin(phi)*cos(theta), -r*cos(phi)*sin(theta)],
                         [cos(phi)*sin(theta), -r*sin(phi)*sin(theta), r*cos(phi)*cos(theta)],
                         [sin(phi),             r*cos(phi),             0]])
    

    def jac(self, r, theta):
        return np.array([[np.cos(theta), -r*np.sin(theta)],
                         [np.sin(theta), r*np.cos(theta)]])


    def get_A(self, xv, yv, w):
        return np.array([[1, 0, np.sin(T*w)/w, (-1+np.cos(T*w))/w, (yv + (T*w*xv - yv)*np.cos(T*w) - (xv+T*w*yv) * np.sin(T*w))/(w**2)],
                        [0, 1, (1-np.cos(T*w))/w, np.sin(T*w)/w, (-xv + (xv + T*w*yv)*np.cos(T*w) + (T*w*xv-yv) * np.sin(T*w))/(w**2)],
                        [0, 0,   np.cos(T*w),    -np.sin(T*w),    -T*(yv*np.cos(T*w) + xv*np.sin(T*w))],
                        [0, 0,   np.sin(T*w),     np.cos(T*w),     T*(xv*np.cos(T*w) - yv*np.sin(T*w))],
                        [0, 0, 0, 0, 1]])
    

    def f(self, x_prev):
        x, y, vx, vy, w = x_prev

        Mat = np.array([[1, 0, np.sin(T*w)/w, (np.cos(T*w)-1)/w, 0],
                        [0, 1, (1-np.cos(T*w))/w, np.sin(T*w)/w, 0],
                        [0, 0, np.cos(T*w), -np.sin(T*w), 0],
                        [0, 0, np.sin(T*w), np.cos(T*w), 0],
                        [0, 0, 0, 0, 1]])
    
        return Mat @ x_prev 
    

    def h(self, x_prev):
        return self.C @ x_prev 

    
    def EKF(self, x_prev, y_k):
        x_head = self.f(x_prev)
        x, y, vx, vy, w = x_prev
        A = self.get_A(vx, vy, w)
        P_k = A @ self.P_prev @ A.T + self.Q
        y_head = self.h(x_head)

        x1, y1 = y_head
        r, phi, theta = tf.cart_to_spherical(np.array([x1, y1, 0]))
        J = self.jac(r, theta)
        R = J @ self.R @ J.T

        S = self.C @ P_k @ self.C.T + R
        y_tilda = y_k - y_head
        F = P_k @ self.C.T @ inv(S)
        x_new = x_head + F @ y_tilda
        P_new = (np.eye(5) - F @ self.C) @ P_k

        self.P_prev = P_new

        return x_new
    