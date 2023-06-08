import numpy as np
import math


def compute_one_sided_spectrum(fft_values):
    """
    Computes the one-sided spectrum of a signal given its FFT.
    """
    N = len(fft_values)
    abs_values = np.abs(fft_values)
    one_sided_spectrum = abs_values[:N // 2]

    # multiply all frequency components by 2, except the DC component
    one_sided_spectrum[1:] *= 2

    return one_sided_spectrum


def get_positive_frequencies(N, T):
    """
    Computes the positive frequencies for FFT of a dataset, given N, the length of the dataset, and T, the time
    spacing between samples.  Returns an array of positive frequencies (Hz) of length N/2.
    """
    frequencies = np.fft.fftfreq(N, T)  # Compute frequencies associated with FFT components
    positive_frequencies = frequencies[:N // 2]  # Get only positive frequencies
    return positive_frequencies


def get_spectrum_fenceposts(opts):
    first, last = opts['freq_range']
    multi = opts['max_lag'] * opts['bin_size']
    first_index = round(first * multi)
    last_index = math.ceil(last * multi)
    return first_index, last_index
