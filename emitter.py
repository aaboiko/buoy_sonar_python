import numpy as np
from sonar_signal import Signal, SignalAM, SignalFM, SignalPulse
import threading

class Emitter:
    def __init__(self, frame, diagram, pan=0, tilt=0, type='pulse'):
        self.frame = frame
        self.diagram = diagram
        self.type = type
        self.pan = pan
        self.tilt = tilt


    def run_LFM(self, amp, duration, interval, f_min, f_max):
        def func(t):
            if t < interval:
                return f_min + (f_max - f_min) / interval
            return 0

        signal = SignalFM(amp, duration, func)


    def run_sine(self, amp, freq, duration):
        def func(t):
            return amp
        
        signal = SignalAM(freq, duration, func)


    def run_pulse(self, amp, duration, pulse_length):
        signal = SignalPulse(amp, duration, pulse_length)