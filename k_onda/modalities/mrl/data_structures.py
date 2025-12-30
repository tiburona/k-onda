import numpy as np
import xarray as xr

from k_onda.model import Data
from k_onda.modalities import EventValidation
from k_onda.modalities.lfp import LFPProperties
from ...modalities.mixins import BandPassFilterMixin
from k_onda.math import compute_mrl, compute_phase


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
        spike_times_secs = self.spike_period.spikes - self.period.onset.pint.to("second")
        spike_inds_lfp = self.to_int(spike_times_secs, unit="lfp_sample", convert_to_scalar=False)

        if self.calc_opts.get("validate_events"):
            events = self.period.event_starts_in_period_time - self.pre_event
            ev = events.data            # pint.Quantity (seconds)
            sp = spike_times_secs.data  # pint.Quantity (seconds)

            event_idx = np.searchsorted(
                ev.magnitude, sp.to(ev.units).magnitude, side="right"
            ) - 1

            event_valid = self.get_event_validity(self.selected_brain_region)
            event_valid = np.asarray(event_valid, dtype=bool)
            valid_spike = event_valid[event_idx]

            spike_inds_lfp = spike_inds_lfp[valid_spike]

        return spike_inds_lfp

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
    
    def get_mrl_weights(self):
        n = int(self.to_int(self.duration.pint.to("lfp_sample")))

        spike_idx = self.spikes.astype(float, copy=False)  # ensures NaN-friendly

        valid = np.isfinite(spike_idx)
        idx = spike_idx[valid].astype(np.int64)

        # counts per sample
        counts = np.bincount(idx, minlength=n)

        # convert to NaN-for-zero
        weights = counts.astype(float)
        weights[counts == 0] = np.nan
        return weights

    def get_phases(self):
        return compute_phase(self.filter(self.mrl_data), self.to_int(self.lfp_padding))
    
    def get_angles(self):
        phases = self.get_phases()   # expect (..., time) typically
        w = self.weights             # float array with NaN for 0, counts otherwise

        two_pi = 2 * np.pi

        if phases.ndim == 1:
            valid = np.isfinite(w) & np.isfinite(phases)
            counts = w[valid].astype(np.int64)
            ang = np.mod(phases[valid], two_pi)
            return np.repeat(ang, counts)

        # phases is 2D: make it (time, bands) so it lines up with w
        if phases.shape[-1] == w.shape[0]:
            ph = phases.T                  # (time, bands) from (bands, time)
        elif phases.shape[0] == w.shape[0]:
            ph = phases                    # already (time, bands)
        else:
            raise ValueError("phases time axis doesn't match weights length")

        valid = np.isfinite(w) & np.isfinite(ph).all(axis=1)
        counts = w[valid].astype(np.int64)
        ang = np.mod(ph[valid], two_pi)    # (valid_time, bands)

        # repeat each time row by its spike count; then flatten to 1D
        return np.repeat(ang, counts, axis=0).ravel()

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