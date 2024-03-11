import numpy as np
import math
from scipy import signal
from scipy.signal import hilbert, butter, filtfilt, hann, hamming
from scipy.fft import fft, ifft
from scipy.optimize import curve_fit


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
    return signal.filtfilt(b, a, signal_with_noise)


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


def correlogram(lags, bin_size, spikes1, spikes2, num_pairs):
    to_return = np.zeros(lags * 2 + 1)
    for bn in range(len(to_return)):
        lag = bn - lags
        bin_edge = lag * bin_size
        spike_bins = [(spike + bin_edge - .5 * bin_size, spike + bin_edge + .5 * bin_size) for spike in spikes1]
        to_return[bn] += sum([1 for spike in spikes2 for start, end in spike_bins if start <= spike < end]) / num_pairs
    return to_return


def remove_line_noise_spectrum_estimation(wave, fs=1, opts=''):
    # Adapted from: removeLineNoise_SpectrumEstimation
    # an implementation of the technique presented in Mewett, Nazeran, and Reynolds. "Removing power line noise from
    # recorded EMG," EMBS, IEEE 2001 DOI: 10.1109/IEMBS.2001.1017205.

    # Example usage
    # wave = np.random.randn(10, 1000)  # Replace with your data
    # fs = 1000  # Sampling frequency
    # opts = 'LF=60, NH=5, M=2048'  # Options
    # cleaned_wave = remove_line_noise_spectrum_estimation(wave, fs, opts)

    # Parse options
    if opts is None:
        opts = {}

        # Parse options from the dictionary
    n_harmonics = int(opts.get('nh', 1))
    line_hz = int(opts.get('lf', 50)) * np.arange(1, n_harmonics + 1)
    err_tolerance = float(opts.get('tol', 0.01))
    hw = int(opts.get('hw', 2))
    m = int(opts.get('m', 0))
    window_type = opts.get('win', 'hanning')

    if wave.ndim == 1:
        wave = wave[np.newaxis, :]

    # Determine window size if not specified
    if m == 0:
        z, err, id = 4, np.inf, -1
        while err / line_hz[0] > err_tolerance or id < hw * 2 - 1:
            z += 1
            W = fs * np.linspace(0, 1, 2 ** z)
            err, id = min((abs(w - line_hz[0]), i) for i, w in enumerate(W))
        m = 2 ** z

    # Create window
    if window_type == 'hamming':
        window = hamming(m, sym=False)
    else:
        window = hann(m, sym=False)

    # Prepare for line noise removal
    W = fs * np.linspace(0, 1, m)
    line_id = [np.argmin(abs(W - hz)) for hz in line_hz]

    # Remove line noise
    for cc in range(wave.shape[0]):
        # Padding
        pad_start = _fourier_pad(wave[cc, :m//2], m, fs)
        pad_end = _fourier_pad(wave[cc, -m//2:], m, fs, reverse=True)
        pad_wave = np.concatenate([pad_start, wave[cc, :], pad_end])

        # Filter wave
        filt_wave = np.zeros((2, pad_wave.shape[0]))
        for tt in range(0, pad_wave.shape[0] - m, m // 2):
            snip = pad_wave[tt:tt + m] * window
            spect = fft(snip)
            # Correct the spectrum
            for ii in range(n_harmonics):
                kk = np.arange(-hw, hw + 1) + line_id[ii]
                est = np.linspace(abs(spect[kk[0]]), abs(spect[kk[-1]]), 2 * hw + 1)
                spect[kk] = spect[kk] / np.abs(spect[kk]) * est
                spect[m - kk + 2] = np.conj(spect[kk])
            filt_wave[tt // (m // 2) % 2, tt:tt + m] = ifft(spect)

        filt_wave = np.sum(filt_wave, axis=0) / np.mean(window + np.roll(window, m // 2))
        wave[cc, :] = filt_wave[m//2:-m//2]

    return wave


def _fourier_pad(segment, m, fs, reverse=False):
    # Fit a Fourier series to the segment
    def fourier_series(x, a0, a1, b1, w):
        return a0 + a1 * np.cos(w * x) + b1 * np.sin(w * x)

    x = np.arange(m // 2) + 1 if not reverse else -np.arange(m // 2) - 1
    popt, _ = curve_fit(fourier_series, x, segment, bounds=([-np.inf, -np.inf, -np.inf, 40/fs*2*np.pi], [np.inf, np.inf, np.inf, 70/fs*2*np.pi]))
    fitted = fourier_series(np.arange(m // 2) if not reverse else np.arange(-m, 0), *popt)
    if reverse:
        return fitted - fitted[0] + segment[-1]
    else:
        return fitted - fitted[-1] + segment[0]


def full_width_half_minimum(mean_waveform, sampling_rate, range_of_max_around_midpoint=(0, 35),
                            range_of_min_around_midpoint=(-25, 25)):
    midpoint = len(mean_waveform) // 2
    range_of_max = (ind + midpoint for ind in range_of_max_around_midpoint)
    range_of_min = (ind + midpoint for ind in range_of_min_around_midpoint)
    max_point = np.max(mean_waveform[slice(*range_of_max)])
    half_amplitude = abs(max_point) - np.min(mean_waveform[slice(*range_of_min)]) / 2
    below_half_min = mean_waveform <= (max_point - half_amplitude)
    fwhm_samples = np.sum(below_half_min)
    fwhm_time = fwhm_samples / sampling_rate
    return fwhm_time









