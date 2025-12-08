import numpy as np


def calc_hist(spikes, num_bins, spike_range):
    """Returns a histogram of binned spike times"""
    return np.histogram(spikes, bins=num_bins, range=spike_range)


def calc_rates(spikes, num_bins, spike_range, bin_size):
    """Computes spike rates over bins"""
    hist = calc_hist(spikes, num_bins, spike_range)
    return hist[0] / bin_size