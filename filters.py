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

        self.sigma_x = 2.2
        self.sigma_y = 2.2
        self.sigma_w = 0.1

        self.Q = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, (T*self.sigma_x)**2, 0, 0],
                            [0, 0, 0, (T*self.sigma_y)**2, 0],
                            [0, 0, 0, 0, (T*self.sigma_w)**2]])
        
        self.R = np.array([[self.sigma_r*T**2, 0],
                           [0, self.sigma_theta*T**2]])
    

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
    

class CVKalmanFilter:
    def __init__(self, sigma):
        self.P_prev = (10^6) * np.ones(6)
        
        self.T = T
        self.sigma_polar = sigma[0:3, 0:3]

        self.C = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0]])

        self.sigma_r = self.sigma_polar[0, 0]
        self.sigma_phi = self.sigma_polar[1, 1]
        self.sigma_theta = self.sigma_polar[2, 2]

        self.sigma_x = 5.2
        self.sigma_y = 5.2
        self.sigma_z = 5.2

        self.Q = np.array([[(T*self.sigma_x)**2, 0, 0, 0, 0, 0],
                            [0, (T*self.sigma_y)**2, 0, 0, 0, 0],
                            [0, 0, (T*self.sigma_x)**2, 0, 0, 0],
                            [0, 0, 0, (T*self.sigma_x)**2, 0, 0],
                            [0, 0, 0, 0, (T*self.sigma_y)**2, 0],
                            [0, 0, 0, 0, 0, (T*self.sigma_z)**2]])
        
        self.R = np.array([[self.sigma_r*T**2, 0, 0],
                           [0, self.sigma_phi*T**2, 0],
                           [0, 0, self.sigma_theta*T**2]])
        
        self.A = np.array([[1, 0, 0, T, 0, 0],
                           [0, 1, 0, 0, T, 0],
                           [0, 0, 1, 0, 0, T],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])


    def jac(self, r, phi, theta):
        return np.array([[cos(phi)*cos(theta), -r*sin(phi)*cos(theta), -r*cos(phi)*sin(theta)],
                         [cos(phi)*sin(theta), -r*sin(phi)*sin(theta), r*cos(phi)*cos(theta)],
                         [sin(phi),             r*cos(phi),             0]])

    
    def EKF(self, x_prev, y_k):
        x = self.A @ x_prev
        P = self.A @ self.P_prev @ self.A.T + self.Q
        y_head = self.C @ x

        x1, y1, z1 = y_head
        r, phi, theta = tf.cart_to_spherical(np.array([x1, y1, z1]))
        J = self.jac(r, phi, theta)
        R = J @ self.R @ J.T

        y_tilda = y_k - y_head
        S = self.C @ P @ self.C.T + R
        F = P @ self.C.T @ inv(S)
        x_head = x + F @ y_tilda
        P_new = (np.eye(6) - F @ self.C) @ P

        self.P_prev = P_new

        return x_head
    

class CVKalmanFilter_7D:
    def __init__(self, sigma):
        self.P_prev = 10e6 * np.ones(6)
        
        self.T = T
        self.sigma_polar = sigma[0:3, 0:3]

        self.C = np.block([[np.eye(3), np.zeros((3, 3))],
                            [np.eye(3), np.zeros((3, 3))],
                            [np.eye(3), np.zeros((3, 3))],
                            [np.eye(3), np.zeros((3, 3))],
                            [np.eye(3), np.zeros((3, 3))],
                            [np.eye(3), np.zeros((3, 3))],
                            [np.eye(3), np.zeros((3, 3))]])

        self.sigma_r = self.sigma_polar[0, 0]
        self.sigma_phi = self.sigma_polar[1, 1]
        self.sigma_theta = self.sigma_polar[2, 2]

        self.sigma_x = 0.2
        self.sigma_y = 0.2
        self.sigma_z = 0.2

        self.Q = np.array([[(T*self.sigma_x)**2, 0, 0, 0, 0, 0],
                            [0, (T*self.sigma_y)**2, 0, 0, 0, 0],
                            [0, 0, (T*self.sigma_x)**2, 0, 0, 0],
                            [0, 0, 0, (T*self.sigma_x)**2, 0, 0],
                            [0, 0, 0, 0, (T*self.sigma_y)**2, 0],
                            [0, 0, 0, 0, 0, (T*self.sigma_z)**2]])
        
        self.R = np.array([[self.sigma_r*T**2, 0, 0],
                            [0, self.sigma_phi*T**2, 0],
                            [0, 0, self.sigma_theta*T**2]])
        
        self.A = np.array([[1, 0, 0, T, 0, 0],
                            [0, 1, 0, 0, T, 0],
                            [0, 0, 1, 0, 0, T],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]])


    def jac(self, r, phi, theta):
        return np.array([[cos(phi)*cos(theta), -r*sin(phi)*cos(theta), -r*cos(phi)*sin(theta)],
                        [cos(phi)*sin(theta), -r*sin(phi)*sin(theta), r*cos(phi)*cos(theta)],
                        [sin(phi),             r*cos(phi),             0]])

    
    def EKF(self, x_prev, y_k):
        x = self.A @ x_prev
        P = self.A @ self.P_prev @ self.A.T + self.Q
        y_head = self.C @ x

        x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, x5, y5, z5, x6, y6, z6, x7, y7, z7 = y_head
        r1, phi1, theta1 = tf.cart_to_spherical(np.array([x1, y1, z1]))
        r2, phi2, theta2 = tf.cart_to_spherical(np.array([x2, y2, z2]))
        r3, phi3, theta3 = tf.cart_to_spherical(np.array([x3, y3, z3]))
        r4, phi4, theta4 = tf.cart_to_spherical(np.array([x4, y4, z4]))
        r5, phi5, theta5 = tf.cart_to_spherical(np.array([x5, y5, z5]))
        r6, phi6, theta6 = tf.cart_to_spherical(np.array([x6, y6, z6]))
        r7, phi7, theta7 = tf.cart_to_spherical(np.array([x7, y7, z7]))

        J1 = self.jac(r1, phi1, theta1)
        J2 = self.jac(r2, phi2, theta2)
        J3 = self.jac(r3, phi3, theta3)
        J4 = self.jac(r4, phi4, theta4)
        J5 = self.jac(r5, phi5, theta5)
        J6 = self.jac(r6, phi6, theta6)
        J7 = self.jac(r7, phi7, theta7)

        R1 = J1 @ self.R @ J1.T
        R2 = J2 @ self.R @ J2.T
        R3 = J3 @ self.R @ J3.T
        R4 = J4 @ self.R @ J4.T
        R5 = J5 @ self.R @ J5.T
        R6 = J6 @ self.R @ J6.T
        R7 = J7 @ self.R @ J7.T

        O = np.zeros((3, 3))
        R = np.block([[R1, O, O, O, O, O, O],
                        [O, R2, O, O, O, O, O],
                        [O, O, R3, O, O, O, O],
                        [O, O, O, R4, O, O, O],
                        [O, O, O, O, R5, O, O],
                        [O, O, O, O, O, R6, O],
                        [O, O, O, O, O, O, R7]])

        y_tilda = y_k - y_head
        S = self.C @ P @ self.C.T + R
        F = P @ self.C.T @ inv(S)
        x_head = x + F @ y_tilda
        P_new = (np.eye(6) - F @ self.C) @ P

        self.P_prev = P_new

        return x_head
    

class CTKalmanFilter_7D:
    def __init__(self, sigma):
        self.P_prev = 10e6 * np.ones(5)
        self.T = T
        self.sigma_polar = sigma[0:3, 0:3]

        self.C = np.block([[np.eye(2), np.zeros((2, 3))],
                            [np.eye(2), np.zeros((2, 3))],
                            [np.eye(2), np.zeros((2, 3))],
                            [np.eye(2), np.zeros((2, 3))],
                            [np.eye(2), np.zeros((2, 3))],
                            [np.eye(2), np.zeros((2, 3))],
                            [np.eye(2), np.zeros((2, 3))]])

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

        x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7 = y_head
        r1, phi1, theta1 = tf.cart_to_spherical(np.array([x1, y1, 0]))
        r2, phi2, theta2 = tf.cart_to_spherical(np.array([x2, y2, 0]))
        r3, phi3, theta3 = tf.cart_to_spherical(np.array([x3, y3, 0]))
        r4, phi4, theta4 = tf.cart_to_spherical(np.array([x4, y4, 0]))
        r5, phi5, theta5 = tf.cart_to_spherical(np.array([x5, y5, 0]))
        r6, phi6, theta6 = tf.cart_to_spherical(np.array([x6, y6, 0]))
        r7, phi7, theta7 = tf.cart_to_spherical(np.array([x7, y7, 0]))

        J1 = self.jac(r1, theta1)
        J2 = self.jac(r2, theta2)
        J3 = self.jac(r3, theta3)
        J4 = self.jac(r4, theta4)
        J5 = self.jac(r5, theta5)
        J6 = self.jac(r6, theta6)
        J7 = self.jac(r7, theta7)

        R1 = J1 @ self.R @ J1.T
        R2 = J2 @ self.R @ J2.T
        R3 = J3 @ self.R @ J3.T
        R4 = J4 @ self.R @ J4.T
        R5 = J5 @ self.R @ J5.T
        R6 = J6 @ self.R @ J6.T
        R7 = J7 @ self.R @ J7.T

        O = np.zeros((2, 2))
        R = np.block([[R1, O, O, O, O, O, O],
                        [O, R2, O, O, O, O, O],
                        [O, O, R3, O, O, O, O],
                        [O, O, O, R4, O, O, O],
                        [O, O, O, O, R5, O, O],
                        [O, O, O, O, O, R6, O],
                        [O, O, O, O, O, O, R7]])

        S = self.C @ P_k @ self.C.T + R
        y_tilda = y_k - y_head
        F = P_k @ self.C.T @ inv(S)
        x_new = x_head + F @ y_tilda
        P_new = (np.eye(5) - F @ self.C) @ P_k

        self.P_prev = P_new

        return x_new
    

class DubinsKalmanFilter:
    def __init__(self, sigma):
        self.P_prev = 10e6 * np.ones(6)
        self.T = T
        self.sigma_polar = sigma[0:3, 0:3]

        self.C = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0]])

        self.sigma_r = self.sigma_polar[0, 0]
        self.sigma_phi = self.sigma_polar[1, 1]
        self.sigma_theta = self.sigma_polar[2, 2]

        self.sigma_x = 5.2
        self.sigma_y = 5.2
        self.sigma_z = 5.2
        self.sigma_v = 2.2
        self.sigma_ph = 0.1
        self.sigma_th = 0.1

        self.Q = np.array([[(T*self.sigma_x)**2, 0, 0, 0, 0, 0],
                            [0, (T*self.sigma_y)**2, 0, 0, 0, 0],
                            [0, 0, (T*self.sigma_z)**2, 0, 0, 0],
                            [0, 0, 0, (T*self.sigma_v)**2, 0, 0],
                            [0, 0, 0, 0, (T*self.sigma_ph)**2, 0],
                            [0, 0, 0, 0, 0, (T*self.sigma_th)**2]])
        
        self.R = np.array([[self.sigma_r*T**2, 0, 0],
                            [0, self.sigma_phi*T**2, 0],
                            [0, 0, self.sigma_theta*T**2]])
    

    def jac(self, r, phi, theta):
        return np.array([[cos(phi)*cos(theta), -r*sin(phi)*cos(theta), -r*cos(phi)*sin(theta)],
                        [cos(phi)*sin(theta), -r*sin(phi)*sin(theta), r*cos(phi)*cos(theta)],
                        [sin(phi),             r*cos(phi),             0]])


    def get_A(self, v, phi, theta):
        R = np.array([[T*np.cos(phi)*np.cos(theta), -v*T*np.sin(phi)*np.cos(theta), -v*T*np.cos(phi)*np.sin(theta)],
                      [T*np.cos(phi)*np.sin(theta), -v*T*np.sin(phi)*np.sin(theta), v*T*np.cos(phi)*np.cos(theta)],
                      [T*np.sin(phi),                v*T*np.cos(phi),                0]])
        
        I = np.eye(3)
        O = np.zeros((3, 3))
        
        return np.block([[I, R],
                         [O, I]])
    

    def f(self, x_prev):
        x, y, z, v, phi, theta = x_prev

        x_new = x + v * T * np.cos(phi) * np.cos(theta)
        y_new = y + v * T * np.cos(phi) * np.sin(theta)
        z_new = z + v * T * np.sin(phi)
        v_new = v
        phi_new = phi
        theta_new = theta

        return np.array([x_new, y_new, z_new, v_new, phi_new, theta_new])
    

    def h(self, x_prev):
        return self.C @ x_prev 

    
    def EKF(self, x_prev, y_k):
        x_head = self.f(x_prev)
        x, y, z, v, phi, theta = x_prev
        A = self.get_A(v, phi, theta)
        P_k = A @ self.P_prev @ A.T + self.Q
        y_head = self.h(x_head)

        x1, y1, z1 = y_head
        r, phi, theta = tf.cart_to_spherical(np.array([x1, y1, z1]))
        J = self.jac(r, phi, theta)
        R = J @ self.R @ J.T

        S = self.C @ P_k @ self.C.T + R
        y_tilda = y_k - y_head
        F = P_k @ self.C.T @ inv(S)
        x_new = x_head + F @ y_tilda
        P_new = (np.eye(6) - F @ self.C) @ P_k

        self.P_prev = P_new

        return x_new
    

class CVKalmanFilter_1D:
    def __init__(self, sigma):
        self.P_prev = (10^6) * np.ones(2)
        
        self.T = T
        self.C = np.array([1, 0])

        self.sigma_x = sigma

        self.Q = (T*self.sigma_x)**2 * np.eye(2)
        
        self.R = (T*self.sigma_x)**2
        
        self.A = np.array([[1, T],
                           [0, 1]])

    
    def EKF(self, x_prev, y_k):
        x = self.A @ x_prev
        P = self.A @ self.P_prev @ self.A.T + self.Q
        y_head = self.C @ x

        y_tilda = y_k - y_head
        S = self.C @ P @ self.C.T + self.R
        F = P @ self.C.T * (1 / S)
        x_head = x + F * y_tilda
        P_new = (np.eye(2) - F @ self.C) @ P

        self.P_prev = P_new

        return x_head