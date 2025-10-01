import numpy as np
import xarray as xr
from mne.time_frequency import tfr_array_multitaper
from copy import deepcopy
from scipy.signal.windows import tukey

from k_onda.math import apply_hilbert_to_padded_data
from k_onda.model.period_event import Period, Event
from k_onda.model import Data
from k_onda.model.bins import TimeBin
from k_onda.interfaces import MatlabInterface
from k_onda.math import normalized_xcorr, pearson_xcorr, calc_coherence
from ...modalities.mixins import BandPassFilterMixin
from k_onda.utils import (bandpass_filter, regularize_angles, 
                          is_iterable, contains_nan)
from .methods import LFPMethods


class LFPProperties:

    @property
    def lfp_padding(self):
        padding = self.calc_opts.get('lfp_padding', [0, 0])
        return np.rint(np.array(padding) * self.lfp_sampling_rate).astype(int)


class LFPDataSelector:
    """A class with methods shared by LFPPeriod and LFPEvent that are used to return portions of their data."""

    @property
    def lost_signal(self):
        return self.calc_opts.get('lost_signal', [0, 0])

    def slice_spectrogram(self):                                          
        tol = 0.2                           

        # boolean mask along the *frequency* coord
        fmask = ((self.spectrogram['frequency'] >= self.freq_range[0] - tol) &
                (self.spectrogram['frequency'] <= self.freq_range[1] + tol))         

        return self.spectrogram.sel(frequency=fmask)


    @property
    def sliced_spectrogram(self):
        return self.slice_spectrogram()
    
    
class EventValidator:
    
    def get_event_validity(self, region):
        period = self if self.name == 'period' else self.period
        ev = period.animal.lfp_event_validity[region]
        return {i: valid for i, valid in enumerate(ev[self.period_type][period.identifier])}


class LFPPeriod(Period, LFPMethods, LFPProperties, LFPDataSelector, EventValidator):

    def __init__(self, animal, index, period_type, period_info, onset, events=None, 
                 target_period=None, is_relative=False, experiment=None):
        super().__init__(index, period_type, period_info, onset, experiment=experiment, 
                         target_period=target_period, is_relative=is_relative, events=events)
        self.animal = animal
        self.parent = animal
      
        self.duration_in_lfp_samples = round(self.duration * self.lfp_sampling_rate)
        conversion_factor = self.lfp_sampling_rate/self.sampling_rate 
        self.onset_in_lfp_samples = round(self.onset * conversion_factor)
        self.event_starts = np.array(events)
        self.event_starts_in_seconds = self.event_starts/self.sampling_rate 
        self.event_starts_in_lfp_samples = (np.array(events) * conversion_factor).astype(int) - 1
        self.start_in_lfp_samples = self.onset_in_lfp_samples
        self.stop_in_lfp_samples = self.start_in_lfp_samples + self.duration_in_lfp_samples
        self.pad_start = self.start_in_lfp_samples - self.lfp_padding[0]
        self.pad_stop = self.stop_in_lfp_samples + self.lfp_padding[1]
        # Sanity check: don’t let pads run outside data bounds
        assert self.pad_start >= 0
        for r in self.animal.processed_lfp:
            assert self.pad_stop <= len(self.animal.processed_lfp[r])
        self._spectrogram = None
        self.brain_region = self.selected_brain_region
        self.frequency_band = self.selected_frequency_band
        
    @property
    def padded_data(self):
        return self.get_data_from_animal_dict(self.animal.processed_lfp, 
                                              self.pad_start, self.pad_stop)

    @property
    def event_starts_in_period_time(self):
        if not hasattr(self, "_event_starts_period"):
            self._event_starts_period = [
                t_abs - self.onset_in_seconds     # onset is period’s absolute 0 s
                for t_abs in self.event_starts_in_seconds
            ]
        return self._event_starts_period
        
    @property
    def unpadded_data(self):
        return self.get_data_from_animal_dict(
            self.animal.processed_lfp, self.start_in_lfp_samples, self.stop_in_lfp_samples)
    
    def get_data_from_animal_dict(self, data_source, start, stop):
        if self.selected_brain_region:
            return data_source[self.selected_brain_region][start:stop]
        else:
            return {brain_region: data_source[brain_region][start:stop] 
                    for brain_region in data_source}
    
    @property
    def spectrogram(self):
        if self._spectrogram is None:
            self._spectrogram = self.calc_spectrogram()                   
        last_freq = self.freq_range[1] + .2 # .2 is the tolerance
        spec_trimmed = self._spectrogram.sel(                               
            frequency=slice(None, last_freq))                               
          
        self._spectrogram = spec_trimmed                                   
        return spec_trimmed            

    def get_events(self):
        """
        Build LFPEvent objects, one per stimulus, keyed to spectrogram bins.
        Works entirely in *period_time* space; no absolute-time juggling.
        """
        eps   = 1e-8
        tbins = self.spectrogram.coords['period_time'].values

        events = []
        for i, rel_start in enumerate(self.event_starts_in_period_time):

            win_start = rel_start - self.pre_event
            win_end   = rel_start + self.post_event

            mask = (tbins >= win_start - eps) & (tbins < win_end - eps)
            event_times = tbins[mask]

            events.append(LFPEvent(i, event_times,     # event_times in period space
                                self.event_starts_in_seconds[i],  # keep abs too
                                mask, self))

        self._events = events
        return events
        
    @property
    def extended_data(self):
        data = self.events[0].data
        for event in self.events[1:]:
            data = np.concatenate((data, event.data), axis=1)
        return data
    
    def generate_spectrogram_cache_args(self, power_arg_set, calc_method):
        cache_args = deepcopy(power_arg_set)
        if isinstance(power_arg_set, dict):
            freqs = cache_args['freqs']
            n_cycles = cache_args['n_cycles']
            if is_iterable(freqs):
                cache_args['freqs'] = [freqs[0], freqs[1], freqs[1] - freqs[0]]
            if is_iterable(n_cycles):
                cache_args['n_cycles'] = [n_cycles[0], n_cycles[1], n_cycles[1] - n_cycles[0]]
            flat_args = [str(i) for item in list(cache_args.items()) for i in item]
        else:
            flat_args = [str(i) for i in cache_args]

        arg_set = [self.animal.identifier, self.selected_brain_region, calc_method, 
                   *flat_args, self.period_type, str(self.identifier), 'padding', 
                   *[str(pad) for pad in self.calc_opts.get('lfp_padding', [0, 0])]]
        return arg_set
    
    def calc_spectrogram(self):                                             
        power_arg_set   = self.calc_opts['power_arg_set']
        calc_method     = self.calc_opts.get('calc_method', 'matlab')

        cache_args = self.generate_spectrogram_cache_args(power_arg_set, calc_method)
        
        saved_calc_exists, spectrogram, pickle_path = self.load(
            'lfp_output', 'spectrogram', cache_args)
        
        if saved_calc_exists:
            return spectrogram
        
        else:

            # --- run the calculation -------------
            if calc_method == 'matlab':
                ml      = MatlabInterface(self.env_config['matlab_config'])
                power, freqs, times = ml.mtcsg(self.padded_data, *power_arg_set)
            else:
                data_3d = self.padded_data[np.newaxis, np.newaxis, :]
                power   = tfr_array_multitaper(data_3d, **power_arg_set).squeeze()
                freqs = power_arg_set['freqs']
                times = np.arange(power.shape[-1]) * power_arg_set['decim'] / power_arg_set['sfreq']
    
            # --- WRAP in xarray ----------------------------------------------  # 
            da = xr.DataArray(
                power,
                dims=['frequency', 'time_raw'],
                coords={'frequency': freqs, 'time_raw': times},
                attrs={'calc_method': calc_method}
            )
            
            true_beginning = self.lfp_padding[0] - self.lost_signal[0]

            da = (
                da.assign_coords(time_idx=('time_raw', np.arange(da.sizes['time_raw'])))
                .swap_dims({'time_raw': 'time_idx'})
                .rename({'time_idx': 'time'})
    )
            da = da.assign_coords(
                    spectrogram_time=('time', da['time_raw'].values),
                    period_time=('time', da['time_raw'].values - true_beginning)
                ).drop_vars('time_raw')    

            self.save(da, pickle_path)                                             

        return da      
    
    def index_transformation_function(self, concatenator):
        if concatenator == 'animal':
            return lambda calc: calc.assign_coords(
                time=calc.coords['time'] + self.onset
                ) if isinstance(calc, xr.DataArray) else calc
        else:
            raise NotImplementedError("Period concatenation is currently only supported by animal.")
    

class LFPEvent(Event, LFPMethods, LFPProperties, LFPDataSelector):

    def __init__(self, identifier, event_times, onset, mask, period):
        super().__init__(period, onset, identifier)
        self.event_times = event_times
        self.mask = mask
        if sum(self.mask) == 0:
            raise ValueError("Event mask is empty!")
        self.animal = period.animal
        self.period_type = self.parent.period_type
        self.spectrogram = self.parent.spectrogram

    def __repr__(self):
        return (f"Event {self.animal.identifier} {self.period_type} "
                f"{self.period.identifier} {self.identifier}")

   
    @property          
    def is_valid(self):
        val = self.animal.lfp_event_validity[self.selected_brain_region][self.period_type][
            self.period.identifier][self.identifier]
        if not val:
            print(f"Event {self.animal.identifier} {self.period_type} {self.period.identifier} {self.identifier} is not valid!")
        return val

    def get_power(self):
        
        indices = np.where(self.mask)[0]  # Convert boolean mask to integer indices
        power = self.sliced_spectrogram.isel(time=indices)

        # Extract the first time coordinate (the event start relative to the period start)
        event_start = power.coords['period_time'].values[0] + self.pre_event

        # Create a new coordinate "relative_time" by subtracting the event start time
        power = power.assign_coords(relative_time=power.coords['period_time'] - event_start)

        if self.calc_opts.get('frequency_type') == 'block':
            power = power.mean(dim='frequency')

        if self.calc_opts.get('time_type') == 'block':
            power = power.mean(dim='time')      

        return power
    
    def index_transformation_function(self, concatenator):
        if concatenator == 'period':
            return lambda identifier, calc: calc.assign_coords(
                time=calc.coords['time'] + identifier * self.duration
                )
        elif concatenator == 'animal':
            return lambda identifier, calc: calc.assign_coords(
                time=calc.coords['time'] + identifier * self.duration + self.period.onset
                )
        else:
            raise NotImplementedError("Event concatenation is currently only supported by period and animal.")
    

class RegionRelationshipCalculator(Data, EventValidator, LFPProperties):

    def __init__(self, period, regions):
        self.period = period
        self.parent = self.period.parent
        self.period_id = period.identifier
        self.period_type = period.period_type
        self.regions = regions
        self.identifier = f"{'_'.join(self.regions)}_{self.period.identifier}"
        processed_lfp = self.period.animal.processed_lfp
        self.regions_data = [processed_lfp[r][
            self.period.start_in_lfp_samples:self.period.stop_in_lfp_samples] for r in self.regions]
        self.event_duration = round(self.lfp_sampling_rate * self.period.event_duration)
        self.padded_regions_data = [
            processed_lfp[r][self.period.pad_start:self.period.pad_stop]
            for r in self.regions]
        self._children = []
        self.frequency_bands = [(f + .05, f + 1) for f in range(*self.freq_range)]

    @property
    def children(self):
        if not len(self._children):
            self.get_events()
        return self._children

    def joint_event_validity(self):
        evs = [self.get_event_validity(region) for region in self.regions]
        return {i: all([ev[i] for ev in evs]) for i in evs[0]}

    def divide_data_into_valid_sets(self, region_data, do_pad=False):
        if not self.calc_opts.get('validate_events'):
            return [np.asarray(region_data)]

        N = len(region_data)
        ev_ok = self.joint_event_validity()
        ev_len = int(self.event_duration)

        # Build a boolean vector of event validity across time
        n_events = N // ev_len
        valid_vec = np.zeros(n_events, dtype=bool)
        for k in range(n_events):
            valid_vec[k] = ev_ok.get(k, False)

        # Find contiguous runs of True
        runs = []
        i = 0
        while i < n_events:
            if not valid_vec[i]:
                i += 1
                continue
            j = i
            while j < n_events and valid_vec[j]:
                j += 1
            # events [i, j) are valid
            runs.append((i, j))
            i = j

        valid_sets = []
        # pad lengths in samples
        padL, padR = int(self.lfp_padding[0]), int(self.lfp_padding[1])

        # Minimum length (for filter stability if needed)
        min_len = 0 if self.calc_type == 'coherence' else (self.lfp_sampling_rate + 1) * 3 + 1

        for ev_start, ev_stop in runs:
            start_samp = ev_start * ev_len
            stop_samp  = ev_stop  * ev_len
            if do_pad:
                left  = max(0, start_samp - padL)
                right = min(N, stop_samp + padR)
            else:
                left, right = start_samp, stop_samp

            seg = np.asarray(region_data[left:right])
            if seg.size > min_len:
                valid_sets.append(seg)

        return valid_sets
    
    def get_valid_sets(self, do_pad=False):
        valid_sets = list(zip(*(self.divide_data_into_valid_sets(data, do_pad=do_pad) 
                               for data in self.regions_data)))
        len_sets = [len(a) for a, _ in valid_sets]
        return valid_sets, len_sets
    
    def index_transformation_function(self, concatenator):
        raise NotImplementedError("Not yet implemented for RegionRelationshipCalculator.")
    



class CoherenceCalculator(RegionRelationshipCalculator):

    name = 'coherence_calculator'

    def get_coherence(self):
        if not self.calc_opts.get('validate_events'):
            f, coherence = calc_coherence(*self.regions_data, self.lfp_sampling_rate, 
                                  *self.freq_range)
        else:
            valid_sets, len_sets = self.get_valid_sets()
            if len(valid_sets) == 0:
                return np.nan
            coherence_sets = [calc_coherence(*data, self.lfp_sampling_rate, *self.freq_range) for data in valid_sets]
            f = coherence_sets[0][0]
            coherence = sum(
                [coherence_set[1] * len(valid_sets[i][0])/sum(len_sets) 
                for i, coherence_set in enumerate(coherence_sets)
                ])
        da = xr.DataArray(
            coherence,
            dims=['frequency'],
            coords={'frequency': f})
        if self.calc_opts.get('frequency_type') == 'block':
            da = da.mean(dim='frequency')
        return da
        
    
class AmpXCorrCalculator(RegionRelationshipCalculator, BandPassFilterMixin):

    _name = 'amp_xcorr_calculator'

    def __init__(self, period, regions):
        super().__init__(period, regions)

    @property
    def lags(self):
        if self.calc_type == 'lag_of_max_corr':
            raise ValueError("Data type is max correlation; there are not multiple lags.")
        return [TimeBin(i, data_point, self) for i, data_point in enumerate(self.calc)]
    
    def get_lag_of_max_corr(self):
        # TODO: it is causing all kinds of special pleading elsewhere in the code to have this method
        # I think we should move toward letting the user have a list of operations on the base calc type
        # they can specify, argmax and argmin being two of them.  I mean, arguably that's what `histogram`
        # already is, but it's also an `aggregate`.  I mean just mathematical operations
        amp_xcorr = self.get_amp_xcorr()
        if contains_nan(amp_xcorr):
            return xr.DataArray(np.nan)
        return amp_xcorr.idxmax('lag').item()

    def amplitude(self, signal):
        env = np.abs(apply_hilbert_to_padded_data(self.filter(signal), self.lfp_padding))
        # gentle taper; alpha=0.2 is mild, won’t distort the middle
        return env * tukey(env.size, alpha=0.2)
     
    def get_amp_xcorr(self):
        fs = self.lfp_sampling_rate

        if not self.calc_opts.get('validate_events'):
            n1, n2 = (len(self.padded_regions_data[0]), len(self.padded_regions_data[1]))
            assert n1 == n2, f"LFP regions mismatch: {n1} vs {n2}"

            amp1, amp2 = [self.amplitude(signal) for signal in self.padded_regions_data]
            corr, _ = normalized_xcorr(amp1, amp2, fs=fs)
            result = corr

        else:
            valid_sets, len_sets = self.get_valid_sets(do_pad=True)
            if not valid_sets:
                return xr.DataArray(np.nan)

            len_longest_corr = max(len_sets) * 2 - 1
            corrs = []

            for signal_series in valid_sets:
                amp1, amp2 = [self.amplitude(signal) for signal in signal_series]
                corr, _ = pearson_xcorr(amp1, amp2, fs=fs)
                # todo: this len_longest_corr logic will cause an error if you
                # are trying to average over different calculators with different lengths
                # in practice, often not a problem since you restrict the length of the longest corr
                # but should be fixed
                weight = corr.size / len_longest_corr
                padded = np.full(len_longest_corr, np.nan)
                start = (len_longest_corr - corr.size) // 2
                padded[start:start + corr.size] = corr * weight
                corrs.append(padded)

            result = np.nansum(np.vstack(corrs), axis=0)

        # Build lags deterministically from the final result length
        assert result.size % 2 == 1, "Cross-corr length should be odd (2*N-1)."
        half = (result.size - 1) // 2
        lags = np.arange(-half, half + 1, dtype=float) / fs

        # Optional: support a fixed +/- lag window
        # Back-compat: if user provided 'lags' as samples, respect it; otherwise allow 'max_lag_sec'
        max_lag_sec = self.calc_opts.get('max_lag_sec', None)
        if max_lag_sec is None and 'lags' in self.calc_opts:
            max_lag_sec = self.calc_opts['lags'] / fs
        if max_lag_sec is not None:
            m = (lags >= -max_lag_sec) & (lags <= max_lag_sec)
            result = result[m]
            lags = lags[m]

        return xr.DataArray(result, dims=['lag'], coords={'lag': lags})

            
    
class PhaseRelationshipCalculator(RegionRelationshipCalculator):

    name = 'phase_relationship_calculator'

    @property
    def time_to_use(self):
        if self.calc_opts.get('events'):
            time_to_use = self.calc_opts.get('events')[self.parent.period_type]['post_stim']  # TODO: Fix this to allow for pre stim time
            time_to_use *= self.lfp_sampling_rate
        else:
            time_to_use = self.event_duration
        return time_to_use
    
    def get_event_segment(self):
        time_to_use = self.time_to_use
        ones_segment = np.ones(time_to_use)
        nans_segment = np.full(self.event_duration - time_to_use, np.nan)
        segment = np.concatenate([ones_segment, nans_segment])
        return segment

    def get_event_times(self, data):
        event_duration = int(self.event_duration)
        
        event_times = []
        shape = data.shape[1]  # The length of the second dimension of data

        for _ in range(0, shape, event_duration):
            event_times.append(self.get_event_segment())
        
        event_times = np.concatenate(event_times)
        
        # Pad with NaNs if event_times is shorter than shape
        if len(event_times) < shape:
            padding = np.full(shape - len(event_times), np.nan)
            event_times = np.concatenate([event_times, padding])
        else:
            event_times = event_times[:shape]  # Ensure it matches the required length

        event_times = np.tile(event_times, (data.shape[0], 1))  # Repeat the event_times across rows
        
        return event_times
    
    def get_events(self): # I need to add some pre event stuff ehre
        events = []
        d1, d2 = (self.region_1_data_padded, self.region_2_data_padded)
        events_info = self.calc_opts.get('events')
        if events_info is not None:
            pre_stim = events_info[self.period_type].get('pre_stim', 0)
            post_stim = events_info[self.period_type].get('post_stim', self.event_duration/self.sampling_rate)
        for i, start in enumerate(range(0, len(d1), self.event_duration)):
            slc = slice(*(int(start-pre_stim*self.sampling_rate), int(start+post_stim*self.sampling_rate)))
            events.append(PhaseRelationshipEvent(self, i, d1[slc], d2[slc]))
        self._children = events

    def get_phase_phase_mrl(self):
        valid_sets = self.get_angles()
        results_per_set = [compute_mrl(data, self.get_event_times(data), dim=1) for data in valid_sets]
        weights = np.array([data_set.shape[1] for data_set in valid_sets])
        to_return = np.average(results_per_set, axis=0, weights=weights) # TODO:figure out what's wrong with self.refer here
        return to_return

    def get_angles(self):
        d1, d2 = (self.region_1_data, self.region_2_data)
        phase_diffs = self.get_region_phases(d1) - self.get_region_phases(d2) 
        return regularize_angles(phase_diffs)

    def get_region_phases(self, region_data):
        valid_sets = np.array([
            self.divide_data_into_valid_sets(
                np.array(compute_phase(bandpass_filter(region_data, low, high, self.sampling_rate)))
            ) for low, high in self.frequency_bands])
        return valid_sets.transpose(1, 0, 2)

    

class GrangerFunctions:

    def fetch_granger_stat(self, d1, d2, tags, proc_name, do_weight=True, max_len_sets=None):
        ml = MatlabInterface(self.env_config['matlab_config'], tags=tags)
        data = np.vstack((d1, d2))
        proc = getattr(ml, proc_name)
        result = proc(data)
        if proc_name == 'granger_causality':
            weight = len(d1)/max_len_sets if do_weight else 1
            matrix = np.array(result[0]['f'])
            index_1 = round(self.freq_range[0]*matrix.shape[2]/(self.sampling_rate/2)) 
            index_2 = round(self.freq_range[1]*matrix.shape[2]/(self.sampling_rate/2)) + 1
            forward = []
            backward = []
            if self.calc_opts.get('frequency_type') == 'continuous':
                # check to make sure we have at least 1 Hz res data
                nec_freqs = self.freq_range[1] - self.freq_range[0] + 1
                if index_2 - index_1 < nec_freqs:
                    forward = np.full(nec_freqs, np.nan)
                    backward = np.full(nec_freqs, np.nan)
                else:
                     # TODO: this is more complicated than it need be given that we decided
                     # to make the Granger calculation return whole number frequencies 
                     #  Consider simplifying.
                    for freq in range(self.freq_range[0], self.freq_range[1] + 1):
                        for_bin = []
                        back_bin = []
                        for x in range(index_1, index_2): 
                            if x >= freq + .5:
                                break
                            if x >= freq - .5 and x < freq + .5:
                                for_bin.append(np.real(matrix[0, 1, x]))
                                back_bin.append(np.real(matrix[1, 0, x]))
                        forward.append(np.mean(for_bin)*weight)
                        backward.append(np.mean(back_bin)*weight)
            else:
                forward.append(np.mean(
                    [np.real(matrix[0,1,x]) for x in range(index_1, index_2+1)]) * weight)
                backward.append(np.mean(
                            [np.real(matrix[1,0,x]) for x in range(index_1, index_2+1)]) * weight)
            result = (forward, backward)
            if proc_name == 'ts_data_to_info_crit':
                 result = (len(d1), result)
        return result
    
    
class GrangerCalculator(RegionRelationshipCalculator, GrangerFunctions): # TODO: this is more complicated than it need be given that we went with 

    name = 'granger_calculator'

    def get_events(self): # I need to add some pre event stuff ehre
        events = []
        d1, d2 = (self.region_1_data_padded, self.region_2_data_padded)
        events_info = self.calc_opts.get('events')
        if events_info is not None:
            pre_stim = events_info[self.period_type].get('pre_stim', 0)
            post_stim = events_info[self.period_type].get('post_stim', self.event_duration/self.sampling_rate)
        for i, start in enumerate(range(0, len(d1), self.event_duration)):
            slc = slice(*(int(start-pre_stim*self.sampling_rate), int(start+post_stim*self.sampling_rate)))
            events.append(GrangerEvent(self, i, d1[slc], d2[slc]))
        self._children = events

    def get_segments(self):
        segments = []
        valid_sets, len_sets = self.get_valid_sets() 
        min_set_length = self.calc_opts.get('min_data_length', 8000)
        divisor = self.period.duration*self.sampling_rate
        for i, (set1, set2) in enumerate(valid_sets):
            len_set = len_sets[i]
            divisor = int(len_set/min_set_length) + 1
            segment_length = int(len_set/divisor)
            for s in range(divisor):
                if s != divisor - 1:
                    data1 = set1[s*segment_length:(s+1)*segment_length]
                    data2 = set2[s*segment_length:(s+1)*segment_length]
                else:
                    data1 = set1[s*segment_length:]
                    data2 = set2[s*segment_length:]
            segments.append(GrangerSegment(self, i, data1, data2, len_set))
        return segments

    def get_granger_model_order(self):
        return self.granger_stat('ts_data_to_info_crit')
    
    def get_granger_causality(self):
        ids = [self.period.animal.identifier, self.period_type, str(self.period.identifier), 
               str(self.selected_frequency_band)]
        saved_calc_exists, saved_granger_calc, pickle_path = self.load('lfp_output', 'granger', ids)
        if saved_calc_exists:
            return saved_granger_calc
        fstat = self.get_granger_stat('granger_causality')
        if self.calc_opts.get('frequency_type') == 'continuous':
            forward = np.sum(np.vstack([forward for forward, _ in fstat]), 0)
            backward = np.sum(np.vstack([backward for _, backward in fstat]), 0)
        else:
            forward = sum([forward for forward, _ in fstat])
            backward = sum([backward for _, backward in fstat])
        result = {'forward': forward, 'backward': backward}
        self.save(result, pickle_path)
        return result
    
    def get_granger_stat(self, proc_name):
        valid_sets, len_sets = self.get_valid_sets()
        results = []
        for i, (set1, set2) in enumerate(valid_sets):
            if len(set1) > self.calc_opts.get('min_data_length', 8000):
                tags = ['animal', str(self.period.animal.identifier), self.period_type, 
                        self.period.identifier, 'set', str(i), str(self.selected_frequency_band)]
                results.append(self.fetch_granger_stat(set1, set2, tags, proc_name, do_weight=True, 
                                                       max_len_sets=max(len_sets)))
        return results
    

class GrangerSegment(Data, GrangerFunctions):

    name = 'granger_segment'

    def __init__(self, parent_calculator, i, data1, data2, len_set):
        self.identifier = i
        self.parent= parent_calculator
        self.period = self.parent.period
        self.period_id = self.period.identifier
        self.period_type = self.parent.period_type
        self.region_1_data = data1
        self.region_2_data = data2
        self.length = len_set

    def get_granger_causality(self):
        granger_stat = self.get_granger_stat('granger_causality')
        return {'forward': np.array(granger_stat[0]), 'backward': np.array(granger_stat[1])}

    def get_granger_stat(self, proc_name):
        tags = ['animal', str(self.period.animal.identifier), self.period_type, 
                str(self.period.identifier), 'segment', str(self.identifier), 
                str(self.selected_frequency_band)]
        saved_calc_exists, saved_granger_calc, pickle_path = self.load('lfp_output', 'granger', tags)
        if saved_calc_exists:
            return saved_granger_calc
        else:
            result = self.fetch_granger_stat(self.region_1_data, self.region_2_data, tags, proc_name, 
                                         do_weight=False)
            self.save(result, pickle_path)
        return result
        


        
    
    