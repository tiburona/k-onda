import numpy as np
from signal_processing import compute_one_sided_spectrum
from utils import cache_method


class FamilyTreeMixin:

    def get_data(self, opts, data_type, neuron_type=None, ac_info=None):
        if data_type in ['autocorr', 'spectrum']:
            return getattr(self, f"get_{data_type}")(opts, neuron_type=neuron_type, ac_info=ac_info)
        else:  # data_type is psth
            return self.get_average(opts, 'get_psth', neuron_type=neuron_type)

    def get_autocorr(self, opts, neuron_type=None, ac_info=None):
        demean = ac_info['mean_correction'] == 'demean'
        result = self.get_all_autocorrelations(opts, method=ac_info['method'], neuron_type=neuron_type,
                                               demean=demean)[ac_info['tag']]
        if ac_info['mean_correction'] == 'grand mean':
            gm_dict = self.parent.get_all_autocorrelations(opts, method=ac_info['method'])
            gm_result = gm_dict[f"{self.parent.name}_by_{ac_info['tag']}"]
            result = result - np.mean(gm_result)
        return result

    def get_spectrum(self, opts, neuron_type=None, ac_info=None):
        demean = ac_info['mean_correction'] == 'demean'
        fft = np.fft.fft(self.get_autocorr(opts, neuron_type=neuron_type, ac_info=ac_info))
        return compute_one_sided_spectrum(fft)

    @cache_method
    def get_average(self, opts, base_method, neuron_type=None):
        child_vals = []
        for child in self.children:
            average = child.get_average(opts, base_method, neuron_type)
            if average.size > 0:
                child_vals.append(average)
        return np.nanmean(child_vals, axis=0) if len(child_vals) else np.array([])
