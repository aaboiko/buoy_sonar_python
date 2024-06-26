import numpy as np
import threading

class Transducer:
    def __init__(self, angle, pan=0, tilt=0, sigma_r=0.5, range_max=300, sigma_a=0):
        self.angle = angle
        self.sigma_r = sigma_r
        self.sigma_a = sigma_a
        self.sigma_angle = self.angle / (3 * np.sqrt(2))

        self.sigma = np.array([[self.sigma_r**2, 0, 0, 0],
                               [0, self.sigma_angle**2, 0, 0],
                               [0, 0, self.sigma_angle**2, 0],
                               [0, 0, 0, self.sigma_a**2]])
        
        self.r = 150
        self.amplitude = 0
        self.pan = pan
        self.tilt = tilt
        self.range_max = range_max


    def diagram(self, angle):
        #np.exp(-(angle**2) / (2 * self.sigma_angle**2))
        
        petal_0 = np.exp(-(angle**2) / (2 * self.sigma_angle**2)),
        petal_1_left = 1/4 * np.exp(-((angle - 2*self.sigma_angle)**2) / (2 * self.sigma_angle**2)),
        petal_1_right = 1/4 * np.exp(-((angle + 2*self.sigma_angle)**2) / (2 * self.sigma_angle**2))
        
        return np.exp(-(angle**2) / (2 * self.sigma_angle**2))
    

    def set_value(self, r, a):
        self.r = r + np.random.normal(0, self.sigma_r)
        self.amplitude = a + np.random.normal(0, self.sigma_a)


    def get_pan_tilt(self):
        return self.pan, self.tilt
    
    
    def get_sigma(self):
        return self.sigma
    

    def get_value(self):
        return np.array([self.r, self.amplitude])
    

    def reset(self):
        self.r = 150
        self.amplitude = 0

    
class SingleSonar:
    def __init__(self, pan, tilt, angle, n, sigma_r=0.5, sigma_a=0):
        self.angle = angle
        self.sigma_r = sigma_r
        self.sigma_a = sigma_a
        self.r_step = 0.5
        self.sigma_angle = self.angle / (3 * np.sqrt(2))
        self.rays = self.generate_rays(pan, tilt, angle, n)

        self.r = 150
        self.a = 0
        self.pan = pan
        self.tilt = tilt
        self.meas = dict()

        self.n_side = n
        self.angle_step = self.angle / (self.n_side - 1)


    def generate_rays(self, pan, tilt, angle, n):
        print('generating rays started. Pan = ' + str(pan) + ', tilt = ' + str(tilt) + ', angle = ' + str(angle))
        rays = []
        progress = 0
        prev = 0
        iter = 0

        for y in np.linspace(-angle/2, angle/2, n):
            rays_row = []

            for x in np.linspace(-angle/2, angle/2, n):
                progress = int(100 * iter / (n**2))
                if progress > prev:
                    print('generating rays in progress: ' + str(progress) + '%')
                    prev = progress
                iter += 1

                
                ray = {
                    "k": self.diagram(np.sqrt(x**2 + y**2)) / (n**2 * np.pi/4),
                    "phi": tilt + y,
                    "theta": pan + x,
                    "r": 150
                }

                rays_row.append(ray)

            rays.append(rays_row)

        return rays


    def diagram(self, angle):
        return np.exp(-(angle**2) / (2 * self.sigma_angle**2))
    

    def get_rays(self):
        return self.rays
    

    def get_pan_tilt(self):
        return self.pan, self.tilt
    

    def set_rays(self, rays):
        self.rays = rays


    def set_value(self, r, a, meas):
        self.r = r + np.random.normal(0, self.sigma_r)
        self.a = a + np.random.normal(0, self.sigma_a)
        self.meas = meas


    def reset(self):
        for rays_row in self.rays:
            for ray in rays_row:
                ray["r"] = 150

        self.r = 150
        self.a = 0
        self.meas = dict()


    def get_r_from_meas(self, meas):
        res = 0
        amax = 0

        for i in np.arange(0, 150, 0.5):
            if meas[i] > amax:
                amax = meas[i]
                res = i

        return res
    

    def get_a_from_meas(self, meas):
        a = 0

        for i in np.arange(0, 150, 0.5):
            a += meas[i]

        return a


    def get_arl_from_meas(self, meas):
        a = 0
        r = 0
        amax = 0
        l = 0

        for i in np.arange(0, 150, 0.5):
            a += meas[i]
            if meas[i] > amax:
                amax = meas[i]
                r = i

        for i in np.arange(r, 150, 0.5):
            if meas[i] > 0:
                l += 0.5
            else:
                break

        return a, r, l