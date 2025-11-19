# -*- coding: utf-8 -*-
import math
import numpy as np
from scipy.signal import butter, filtfilt, lfilter, freqz
import matplotlib.pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def lp_filter(filter_input, filter_state, lpf_alpha):
    """Low-pass filter

        Args:
          filter_input ([]): input to be filtered
          filter_state ([]): initial filter state
          lpf_alpha (int): LPF alpha

        Returns:
          filter_state : the filter state
        """
    filter_out = filter_state - (lpf_alpha * (filter_state - filter_input))
    return filter_out


def calculate_lpf_alpha(cutoff_frequency, dt):
    """Low-pass filter

        Args:
          cutoff_frequency (float): cut
          dt (float): timestep dt

        Returns:
          LPF alpha (float)
        """
    return (2 * math.pi * dt * cutoff_frequency) / (2 * math.pi * dt * cutoff_frequency + 1)

