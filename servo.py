import numpy as np

class Servo:
    def __init__(self, emitters=None, sensors=None, pan=0, tilt=0):
        self.pan = pan
        self.tilt = tilt
        self.emitters = emitters
        self.sensors = sensors


    def add_emitter(self, emitter):
        self.emitters.append(emitter)


    def add_sensor(self, sensor):
        self.sensors.append(sensor)


    def set_position(self, pan, tilt):
        self.pan = pan
        self.tilt = tilt