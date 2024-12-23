import numpy as np

from k_onda.data import Data
from k_onda.calc import EventValidator, SpikePrepMethods, LFPPrepMethods
from k_onda.utils import bandpass_filter, compute_phase, circ_r2_unbiased, compute_mrl


class MRLMethods():

    def get_mrl(self):
        return self.get_average('get_mrl', stop_at='mrl_calculator')


class MRLCalculator(Data, EventValidator):
    _name = 'mrl_calculator'
    
    def __init__(self, unit, period):
        self.unit = unit
        self.period = period
        self.period_type = period.period_type
        self._children = None
        self.neuron_quality = unit.quality
        self.mrl_data = period.unpadded_data
        self.duration = period.duration
        self.identifier = f"{self.period.identifier}_{self.unit.identifier}"
        self.parent = self.unit
        self.spike_period = self.unit.spike_periods[self.period_type][self.period.identifier]
        self.experiment = self.period.experiment
        self._spikes = None
        self._weights = None
        self.brain_region = self.current_brain_region
        self.frequency_band = self.current_frequency_band
        

    @property
    def spikes(self):
        if self._spikes is None:
            spikes = np.array([spike for event in self.spike_period.events for spike in event.spikes])
            start = self.spike_period.start * self.lfp_sampling_rate
            self._spikes = (spikes * self.lfp_sampling_rate - start).astype(int)
        return self._spikes

    @property
    def weights(self):
        if self._weights is None:
            self._weights = self.get_weights()
        return self._weights

           
    @property
    def ancestors(self):
        return [self] + [self.unit] + [self.period] + self.parent.ancestors
    
    @property
    def equivalent_calculator(self):
        other_stage = self.period.reference_period_type
        return [calc for calc in self.parent.mrl_calculators[other_stage] 
                if calc.identifier == self.identifier][0]
    
    def validator(self):
        if self.calc_opts.get('evoked'):
            return np.nansum(self.weights) > 4 and np.nansum(self.equivalent_calculator.weights) > 4
        else:
            return np.nansum(self.weights) > 4
        
    def translate_spikes_to_lfp_events(self, spikes):
        pre_stim = self.pre_event * self.lfp_sampling_rate
        events = np.array(self.period.event_starts) - self.period.event_starts[0] - pre_stim
        indices = {}
        for spike in spikes:
            # Find the index of the event the spike belongs to
            index = np.argmax(events > spike)
            if events[index] > spike:
                indices[spike] = index - 1
            else:
                indices[spike] = len(events) - 1
        return indices
    
    def get_weights(self):
        wt_range = range(self.duration * self.lfp_sampling_rate)
        if not self.calc_opts.get('validate_events'):
            weights = [1 if weight in self.spikes else float('nan') for weight in wt_range]
        else:
            if self.unit.identifier == 'IG156_good_3' and self.period.identifier == 1:
                a = 'foo'
            indices = self.translate_spikes_to_lfp_events(self.spikes) 
            weight_validity = {spike: self.get_event_validity(self.current_brain_region)[event] 
                               for spike, event in indices.items()}
            weights = np.array([1 if weight_validity.get(w) else float('nan') for w in wt_range])
        return np.array(weights)

    def get_phases(self):
        low = self.freq_range[0] + .05
        high = self.freq_range[1]
       
        if isinstance(self.current_frequency_band, type('str')):
            return compute_phase(bandpass_filter(self.mrl_data, low, high, self.sampling_rate))
        else:
            frequency_bands = [(f + .05, f + 1) for f in range(*self.freq_range)]
            return np.array([compute_phase(bandpass_filter(self.period.unpadded_data, low, high, self.sampling_rate))
                                 for low, high in frequency_bands])
    
    def get_angles(self):

        def adjust_angle(angle, weight):
            if np.isnan(weight):
                return np.nan
            return angle % (2 * np.pi)

        phases = self.get_phases().T
        weights = self.weights

        # Apply the function to every element
        vfunc = np.vectorize(adjust_angle)

        if phases.ndim == 1:
            adjusted_phases = vfunc(phases, weights)
        else:
            # Expand the dimensions of weights to make it (60000, 1)
            weights_expanded = weights[:, np.newaxis]
            adjusted_phases = vfunc(phases, weights_expanded)

        # Filter out NaNs
        adjusted_phases = adjusted_phases[~np.isnan(adjusted_phases)]

        return adjusted_phases
    
    def get_mrl(self):
        if not self.validator():
            return np.nan
        w = self.get_weights()
        alpha = self.get_phases()
        dim = int(not isinstance(self.current_frequency_band, type('str')))

        if w.ndim == 1 and alpha.ndim == 2:
            w = w[np.newaxis, :]

        # Handle NaN assignment based on the shape of alpha
        if alpha.ndim == 2:
            w[:, np.isnan(alpha).any(axis=0)] = np.nan
        else:
            w[np.isnan(alpha)] = np.nan

        if self.calc_opts.get('mrl_func') == 'ppc':
            return circ_r2_unbiased(alpha, w, dim=dim)
        else:
            return compute_mrl(alpha, w, dim=dim)
        

class MRLPrepMethods(SpikePrepMethods, LFPPrepMethods):
    
    def mrl_prep(self):
        self.prepare_mrl_calculators()

    def prepare_mrl_calculators(self):
        for unit in self.units['good']:
            unit.mrl_calculators = {
                period_type: [MRLCalculator(unit, period=period) for period in periods] 
                for period_type, periods in self.lfp_periods.items()}
         

    def select_mrl_children(self):
        return self.select_spike_children()
                  
        