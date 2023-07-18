import numpy as np
import math


def calc_hist(spikes, num_bins, spike_range):
    """Returns a histogram of binned spike times"""
    return np.histogram(spikes, bins=num_bins, range=spike_range)


def calc_rates(spikes, num_bins, spike_range, bin_size):
    """Computes spike rates over bins"""
    hist = calc_hist(spikes, num_bins, spike_range)
    return hist[0] / bin_size


def trim_and_normalize_ac(result, max_lag):
    """
    For symmetrical autocorrelograms, takes every point after the 0th lag. Also normalizes by the value at the 0th lag.
    """
    mid = result.size // 2
    return result[mid + 1:mid + max_lag + 1] / result[mid]


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


def get_spectrum_fenceposts(freq_range, max_lag, bin_size):
    """Returns the indices of a power spectrum between two frequencies."""
    first, last = freq_range
    multi = max_lag * bin_size
    first_index = round(first * multi)
    last_index = math.ceil(last * multi)
    return first_index, last_index


def spectrum(series, freq_range, max_lag, bin_size):
    """Compute the power spectrum of a series."""
    fft = np.fft.fft(series)
    oss = compute_one_sided_spectrum(fft)
    first, last = get_spectrum_fenceposts(freq_range, max_lag, bin_size)
    return oss[first:last]


def sem(children_vals):
    """Take the standard error by columns, over rows of a matrix"""
    all_series = np.vstack(children_vals)
    # Compute standard deviation along the vertical axis (each point in time)
    std_dev = np.std(all_series, axis=0, ddof=1)  # ddof=1 to compute sample standard deviation
    return std_dev / np.sqrt(len(all_series))


