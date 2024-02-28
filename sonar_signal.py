import numpy as np

dt = 10**(-7)

class Signal:
    def __init__(self, duration):
        self.duration = duration
        self.values = np.zeros(duration // dt)
        self.n = self.values.shape[0]


class SignalFM(Signal):
    def __init__(self, amp, duration, freq_func):
        super().__init__(duration)

        for i in range(super().n):
            super().values[i] = amp * np.sin(2 * np.pi * freq_func(i))


class SignalAM(Signal):
    def __init__(self, freq, duration, amp_func):
        super().__init__(duration)

        for i in range(super().n):
            w = 2 * np.pi * freq
            super().values[i] = amp_func(i) * np.sin(w * i)


class SignalPulse(Signal):
    def __init__(self, amp, duration, pulse_length):
        super().__init__(duration)

        for i in range(super().n):
            if i < pulse_length // dt:
                super().values[i] = amp
