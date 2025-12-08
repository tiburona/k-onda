import numpy as np
import xarray as xr

from k_onda.model import Data
from k_onda.modalities import EventValidation
from k_onda.modalities.lfp import LFPProperties
from ...modalities.mixins import BandPassFilterMixin
from k_onda.math import compute_mrl, compute_phase
from k_onda.utils import circ_r2_unbiased


class MRLCalculator(Data, EventValidation, LFPProperties, BandPassFilterMixin):
    _name = 'mrl_calculator'
    
    def __init__(self, unit, period):
        self.unit = unit
        self.period = period
        self.period_type = period.period_type
        self._children = None
        self.neuron_quality = unit.quality
        self.mrl_data = period.padded_data
        self.duration = period.duration
        self.identifier = f"{self.period.identifier}_{self.unit.identifier}"
        self.parent = self.unit
        self.spike_period = self.unit.spike_periods[self.period_type][self.period.identifier]    
        self._weights = None
        self.brain_region = self.selected_brain_region
        self.frequency_band = self.selected_frequency_band
        
    @property
    def spikes(self):
        self.spike_period.spike_times.pint.to('lfp_sample')
        
    @property
    def weights(self):
        if self._weights is None:
            self._weights = self.get_mrl_weights()
        return self._weights
      
    @property
    def ancestors(self):
        return [self] + [self.unit] + [self.period] + self.parent.ancestors
    
    def validator(self):
        if self.calc_opts.get('evoked'):
            return np.nansum(self.weights) > 4 and np.nansum(self.equivalent_calculator.weights) > 4
        else:
            return np.nansum(self.weights) > 4
        
    def translate_spikes_to_lfp_events(self, spikes):

        event_starts = self.period_event_starts.pint.to('lfp_sample')
        onset = self.period.onset.pint.to('lfp_sample')
        pre_event = self.pre_event.pint.to('lfp_sample')
        events = event_starts - onset - pre_event
        indices = {}
        for spike in spikes:
            # Find the index of the event the spike belongs to
            index = np.argmax(events > spike)
            if events[index] > spike:
                indices[spike] = index - 1
            else:
                indices[spike] = len(events) - 1
        return indices
    
    def get_mrl_weights(self):
        wt_range = range(self.to_int(self.duration.pint.to('lfp_sample')))
        if not self.calc_opts.get('validate_events'):
            weights = [1 if weight in self.spikes else float('nan') for weight in wt_range]
        else:
            indices = self.translate_spikes_to_lfp_events(self.spikes) 
            weight_validity = {spike: self.get_event_validity(self.selected_brain_region)[event] 
                               for spike, event in indices.items()}
            weights = np.array([1 if weight_validity.get(w) else float('nan') for w in wt_range])
        return np.array(weights)

    def get_phases(self):
        return compute_phase(self.filter(self.mrl_data), self.lfp_padding)
    
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
        alpha = self.get_phases()          # expect (..., time)
        w = self.get_mrl_weights()             # shape (time,)

        # If alpha is 2D, normalize orientation so time is last
        if alpha.ndim == 2:
            if alpha.shape[-1] == w.shape[-1]:
                pass  # already (bands, time) or (1, time)
            elif alpha.shape[0] == w.shape[-1]:
                alpha = alpha.T            # was (time, bands) -> make time last
            else:
                raise ValueError("alpha time axis doesn't match weights length")

        # Broadcast weights if bands present
        if alpha.ndim == 2:
            w2 = w[np.newaxis, :]          # (1, time)
            # mask any timepoint where phase is NaN in any band
            t_nan = np.isnan(alpha).any(axis=0)
            w2[:, t_nan] = np.nan
            result = compute_mrl(alpha, w2, dim=-1)
        else:
            w1 = w.copy()
            w1[np.isnan(alpha)] = np.nan
            result = compute_mrl(alpha, w1, dim=-1)

        da = xr.DataArray(result, attrs={'mrl_func': 'compute_mrl'})
        return da