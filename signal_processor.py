import tensorflow as tf
import numpy as np
import scipy as sp
import pywt
from scipy.signal import complex_cepstrum

class SignalProcessor:
    def __init__(self) -> None:
        pass

    def get_spectrogram(self, signal, frame_length, frame_step):
        spectrogram = tf.signal.stft(signal, frame_length=frame_length, frame_step=frame_step)
        
        return spectrogram
    
    def get_periodogram(self, signal, fs):
        return sp.signal.periodogram(signal, fs)
    
    def get_wavelet(self, signal):
        return pywt.cwt(signal, 'db1')
    
    def get_cepstrum(self, signal):
        return complex_cepstrum(signal)