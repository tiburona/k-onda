import numpy as np
import math
from scipy import signal
from scipy.signal import hilbert, butter, filtfilt


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
    if len(children_vals) == 0:
        return np.nan
    all_series = np.vstack(children_vals)
    # Compute standard deviation along the vertical axis (each point in time)
    std_dev = np.nanstd(all_series, axis=0, ddof=1)  # ddof=1 to compute sample standard deviation
    return std_dev / np.sqrt(len(all_series))


def filter_60_hz(signal_with_noise, fs):
    f0 = 60  # Frequency to be removed
    Q = 30  # Quality factor (controls the width of the notch)
    b, a = signal.iirnotch(f0, Q, fs)
    return signal.lfilter(b, a, signal_with_noise)


def divide_by_rms(arr):
    rms = np.sqrt(np.mean(arr ** 2))
    return arr/rms


def bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def compute_phase(data):
    analytic_signal = hilbert(data)
    return np.angle(analytic_signal)


def get_wavelet_scale(frequency, sampling_rate, fc=1.0):
    """Compute the wavelet scale for a given frequency using Morlet wavelet relationship."""
    delta = 1 / sampling_rate  # Sampling period
    return int(fc / (frequency * delta))


def circ_r2_unbiased(alpha, w=None, dim=0):

    r = np.nansum(w * np.exp(1j * alpha), axis=dim)
    n = np.nansum(w, axis=dim)

    coeff2 = -1. / (n - 1)
    coeff1 = 1. / (n ** 2 - n)
    r = (coeff1 * (np.abs(r) ** 2) + coeff2)

    if isinstance(r, float):
        r = np.array([r])

    # Reshape n to ensure it's broadcastable to r
    n = np.broadcast_to(n, r.shape)
    r[n < 2] = np.nan

    return r


def compute_mrl(alpha, w, dim):
    # Convert phase data to Cartesian coordinates
    x = w * np.cos(alpha)
    y = w * np.sin(alpha)

    # Compute the average Cartesian coordinates
    x_bar = np.nansum(x, axis=dim) / np.nansum(w, axis=dim)
    y_bar = np.nansum(y, axis=dim) / np.nansum(w, axis=dim)

    # Compute and return the MRL
    r = np.sqrt(x_bar ** 2 + y_bar ** 2)
    if isinstance(r, float):
        r = np.array([r])
    n = np.nansum(w, axis=dim)
    mask = np.broadcast_to(n < 2, r.shape)
    r[mask] = np.nan

    return r


def cross_correlation(x, y, mode='valid'):
    cross_corr = np.correlate(x, y, mode=mode)
    norm_factor = np.std(x) * np.std(y) * len(x)
    normalized_cross_corr = cross_corr / norm_factor
    return normalized_cross_corr



