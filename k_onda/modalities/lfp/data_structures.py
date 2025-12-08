from collections import OrderedDict
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
from k_onda.math import normalized_xcorr
from ...modalities.mixins import BandPassFilterMixin
from k_onda.utils import (is_iterable, contains_nan)
from .methods import LFPMethods, PSDMethods, CoherenceMethods
from .data_structures_mixins.event_validator import EventValidation
from .data_structures_mixins.descendant_cache import DescendantCache


class LFPProperties:

    @property
    def tolerance(self):
        tolerance = self.calc_opts.get('frequency_tolerance', .2)
        return self.quantity(tolerance, units='Hz', name='tol')

    @property
    def lfp_padding(self):
        padding = self.calc_opts.get('lfp_padding', [0, 0])

        padding = self.quantity(
            padding,
            units='second',
            dims=('side',),
            coords={'side': ['pre', 'post']},
            name='lfp_padding'
        )
        return padding.pint.to('lfp_sample')
        
    @property
    def pad_start(self):
        start = (self.start - self.lfp_padding.sel(side='pre')).pint.to('lfp_sample')
        return start

    @property
    def pad_stop(self):
        return (self.stop + self.lfp_padding.sel(side='post')).pint.to('lfp_sample')
    
    @property
    def lost_signal(self):
        lost_signal = self.calc_opts.get('lost_signal', [0, 0])
        return self.quantity(
            lost_signal,
            units='second',
            dims=('side',),
            coords={'side': ['pre', 'post']},
            name='lost_signal'
        )
    

class LFPPeriod(LFPMethods, Period, LFPProperties, EventValidation, PSDMethods, 
                DescendantCache):

    def __init__(self, animal, index, period_type, period_info, onset, duration, events=None,
                  target_period=None, is_relative=False, experiment=None):
        super().__init__(index, period_type, period_info, onset, duration, experiment=experiment, 
                         target_period=target_period, is_relative=is_relative, events=events,
                         )
        self.animal = animal
        self.parent = animal
        self.event_starts = events    
        self._spectrogram = None
        self._segments = OrderedDict()
        self._events = OrderedDict()
        self.max_event_cache   = self.calc_opts.get('max_event_cache', 4)
        self.max_segment_cache = self.calc_opts.get('max_segment_cache', 4)

    @property
    def padded_data(self):
        return self.get_data_from_animal_dict(pad=True)
    
    @property
    def unpadded_data(self):
        return self.get_data_from_animal_dict(pad=False)

    @property
    def event_starts_in_period_time(self):
        if getattr(self, '_event_starts_in_period_time', None) is None:
            self._event_starts_in_period_time = xr.concat(
                [(t_abs - self.onset).pint.to('second') for t_abs in self.event_starts],
                dim="event"
            )
        return self._event_starts_in_period_time
    
    def get_data_from_animal_dict(self, pad=False):
        data_source = self.animal.processed_lfp
        idx_qs = (self.pad_start, self.pad_stop) if pad else (self.start, self.stop)
        start, stop = [self.to_int(idx_q, 'lfp_sample') for idx_q in idx_qs]

        if self.selected_brain_region:
            return data_source[self.selected_brain_region][start:stop]
        else:
            return {brain_region: data_source[brain_region][start:stop] 
                    for brain_region in data_source}
    
    @property
    def spectrogram(self):
        if self._spectrogram is None:
            self._spectrogram = self.calc_spectrogram()
        return self.frequency_selector(self._spectrogram)
    
    @property
    def children(self):
        if self.calc_type in ['coherence', 'psd'] and self.calc_opts.get('validate_events') == True:
            return self.get_segments()
        else:
            return self.get_events()          

    def get_events(self):

        events = self._get_cache(self._events, self._event_key())

        if events is None:
            events = [LFPEvent(i, event_start, self) 
                      for i, event_start in enumerate(self.event_starts)]
                
            self._set_cache(self._events, self._event_key(), events,
                            self.max_event_cache)

        return events
    
    def get_segments(self):
        segments = self._get_cache(self._segments, self._segment_key())
        if segments is None:
            ev_ok = self.get_event_validity(self.selected_brain_region)
            do_pad = self.calc_type not in ['coherence', 'psd']
            data = self.padded_data if do_pad else self.unpadded_data
            valid_sets = self.divide_data_into_valid_sets(data, ev_ok, do_pad=do_pad) 
            segments = [PeriodSegment(vs) for vs in valid_sets]
            self._set_cache(self._segments, self._segment_key(), segments,
                            self.max_segment_cache)
        return segments
                   
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
        power_arg_set = self.calc_opts["power_arg_set"]
        calc_method = self.calc_opts.get("calc_method", "matlab")

        cache_args = self.generate_spectrogram_cache_args(power_arg_set, calc_method)
        saved_calc_exists, spectrogram, pickle_path = self.load(
            "lfp_output", "spectrogram", cache_args
        )
        if saved_calc_exists:
            return self.rebind_to_ureg(spectrogram)

        # --- run the calculation -------------
        if calc_method == "matlab":
            ml = MatlabInterface(self.env_config["matlab_config"])
            power, freqs, times = ml.mtcsg(self.padded_data, *power_arg_set)
            # `times` is already in seconds (numpy array)
        else:
            data_3d = self.padded_data[np.newaxis, np.newaxis, :]
            power = tfr_array_multitaper(data_3d, **power_arg_set).squeeze()
            freqs = power_arg_set["freqs"]
            dt = power_arg_set["decim"] / power_arg_set["sfreq"]  # seconds per bin
            times = np.arange(power.shape[-1]) * dt  # plain floats in seconds

        n_freq, n_time = power.shape

        # dim indices (unitless bins)
        freq_bin = np.arange(n_freq)
        time_bin = np.arange(n_time)

        # unitful coords
        spectrogram_time = self.standardize_time(
            self.quantity(
                times,
                units="second",
                dims=("time_bin",),
                name="spectrogram_time",
            )
        )

        frequency = self.quantity(
            freqs,
            units="Hz",
            dims=("freq_bin",),
            name="frequency",
        )

        da = xr.DataArray(
            power,
            dims=("freq_bin", "time_bin"),
            coords={
                "freq_bin": freq_bin,              # bin index (no units)
                "time_bin": time_bin,              # bin index (no units)
                "frequency": frequency,            # physical Hz, unitful
                "spectrogram_time": spectrogram_time,  # physical seconds, unitful
            },
            attrs={"calc_method": calc_method},
        )

        # --- time alignment / coords, all unitful ---
        true_beginning = (
            self.lfp_padding.sel(side="pre") - self.lost_signal.sel(side="pre")
        ).pint.to("second").reset_coords(drop=True)

        period_time = self.standardize_time(
            spectrogram_time - true_beginning,
            units="second",
        )

        absolute_time = self.standardize_time(
            spectrogram_time - true_beginning + self.onset,
            units="second",
        )

        da = da.assign_coords(
            period_time=period_time,
            absolute_time=absolute_time,
        )

        # --- infer bin size from spectrogram_time --
        if da.sizes["time_bin"] > 1:
            bin_size = (
                spectrogram_time.isel(time_bin=1)
                - spectrogram_time.isel(time_bin=0)
            ).pint.to("second")
        else:
            bin_size = None 

        da.attrs['bin_size'] = bin_size

        self.save(da, pickle_path)
        return da
    
    def get_power_(self):
        
        mask = (self.spectrogram.period_time >= 0) & (self.spectrogram.period_time < self.duration)
        power = self.spectrogram.sel(time_bin=mask)

        if self.events and self.calc_opts.get('validate_events'):
            ev_ok = self.get_event_validity(self.selected_brain_region)
            r_factor = self.to_int(self.event_duration/self.spectrogram.attrs['bin_size'])
            ev_mask = np.repeat(self.get_valid_vec(ev_ok, self), r_factor)
            power = power.where(ev_mask)
           
        if self.calc_opts.get("frequency_type") == "block":
            power = power.mean(dim="freq_bin", skipna=True)

        if self.calc_opts.get("time_type") == "block":
            power = power.mean(dim="time_bin", skipna=True)

        return power
        
    def index_transformation_function(self, concatenator):
        if concatenator == 'animal':
            return lambda calc: calc.assign_coords(
                time=calc.coords['time'] + self.onset
                ) if isinstance(calc, xr.DataArray) else calc
        else:
            raise NotImplementedError("Period concatenation is currently only supported by animal.")
        
    @property
    def segment_weights(self):
        if len(self.segments) == 0:
            return []
        return [len(seg.unpadded_data) for seg in self.segments]
    
    def get_psd_(self):
        return self.psd_from_contiguous_data(self.unpadded_data)


class PeriodSegment(Data, LFPMethods, PSDMethods):

    _name = 'period_segment'
    
    def __init__(self, data):
        self.unpadded_data = data

    def get_psd_(self):
        return self.psd_from_contiguous_data(self.unpadded_data)


class LFPEvent(Event, LFPMethods, LFPProperties):

    def __init__(self, identifier, onset, period):
        super().__init__(period, onset, identifier)
        self.animal = period.animal
        self.period_type = self.parent.period_type

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
    
    @property
    def spectrogram(self):
        return self.parent.spectrogram
    
    @property
    def spectrogram_mask(self):
        eps   = self.quantity(1e-8, units='second')
        tbins = self.spectrogram.coords['period_time']
        rel_start = self.period.event_starts_in_period_time[self.identifier]
        win_start = rel_start - self.pre_event
        win_end   = rel_start + self.post_event
        mask = (tbins >= win_start - eps) & (tbins < win_end - eps)
        if sum(mask) == 0:
            raise ValueError("Event mask is empty!")
        return mask

    def get_power_(self):

        # Boolean mask → subset times
        power = self.spectrogram.isel(time_bin=self.spectrogram_mask)

        # First time point in this window (period_time is already “time since period start”)
        event_start = power.coords["period_time"].isel(time_bin=0) + self.pre_event

        # Time relative to event onset
        power = power.assign_coords(
            relative_time=power.coords["period_time"] - event_start
        )

        if self.calc_opts.get("frequency_type") == "block":
            power = power.mean(dim="freq_bin")

        if self.calc_opts.get("time_type") == "block":
            power = power.mean(dim="time_bin")

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
    

class RegionRelationshipCalculator(Data, EventValidation, LFPProperties, DescendantCache):

    def __init__(self, period):
        self.period = period
        self.parent = self.period.parent
        self.period_id = period.identifier
        self.period_type = period.period_type
        self.identifier = f"{'_'.join(self.regions)}_{self.period.identifier}"
        self._segments = OrderedDict()
        self._events = OrderedDict()
        self.max_event_cache   = self.calc_opts.get('max_event_cache', 4)
        self.max_segment_cache = self.calc_opts.get('max_segment_cache', 4)

    @property
    def regions_data(self):
        processed_lfp = self.period.animal.processed_lfp
        return [
            processed_lfp[r][self.period.start_in_lfp_samples:self.period.stop_in_lfp_samples]
            for r in self.regions
        ]

    @property
    def padded_regions_data(self):
        processed_lfp = self.period.animal.processed_lfp
        return [
            processed_lfp[r][self.period.pad_start:self.period.pad_stop]
            for r in self.regions
        ]
    
    @property
    def regions(self):
        return tuple(self.selected_brain_regions)

    @property
    def children(self):
        if self.calc_opts.get('validate_events'):
            return self.get_segments()
        else:
            return self.get_events()
     
    def joint_event_validity(self):
        evs = [self.get_event_validity(region) for region in self.regions]
        return {i: all([ev[i] for ev in evs]) for i in evs[0]}
    
    def get_segments(self):

        segments = self._get_cache(self._segments, self._segment_key())

        if segments is None:

            ev_ok = self.joint_event_validity()
            do_pad = self.calc_type not in ['coherence', 'psd']
            regions_data = self.padded_regions_data if do_pad else self.regions_data

            valid_sets = list(zip(*(self.divide_data_into_valid_sets(data, ev_ok, do_pad=do_pad) 
                                for data in regions_data)))
            
            segments = [self.segment_class(self, vs, i) for i, vs in enumerate(valid_sets)]

            self._set_cache(self._segments, self._segment_key(), segments,
                                 self.max_segment_cache)
            
        return segments
        
    def get_events(self):
        events = self._get_cache(self._events, self._event_key())
        if events is None:
            events = [
                self.event_class(self.period, event.onset, self, i)  # todo is this right
                for i, event in enumerate(self.period.events)]
            
    def index_transformation_function(self, concatenator):
        raise NotImplementedError("Not yet implemented for RegionRelationshipCalculator.")
    
    @property
    def segment_weights(self):
        if len(self.segments) == 0:
            return []
        return [len(seg.regions_data[0]) for seg in self.segments]


class RelationshipCalculatorSegment(Data, LFPMethods, LFPProperties):

    def __init__(self, region_relationship_calculator, valid_data, index):
        self.region_relationship_calculator = region_relationship_calculator
        self.parent = region_relationship_calculator
        self.regions_data = valid_data
        self.regions = self.parent.regions
        self.period = self.parent.period
        self.identifier = index


class RelationshipCalculatorEvent(Event, LFPMethods, LFPProperties):

    def __init__(self, period, onset, region_relationship_calculator, index):
        super().__init__(period=period, onset=onset, index=index)
        self.region_relationship_calculator = region_relationship_calculator
        self.parent = region_relationship_calculator
        self.regions = self.parent.regions
        self.period = self.parent.period

    @property
    def regions_data(self):
        processed_lfp = self.period.animal.processed_lfp
        return [
            processed_lfp[r][self.period.start_in_lfp_samples:self.period.stop_in_lfp_samples]
            for r in self.regions
        ]

    @property
    def padded_regions_data(self):
        processed_lfp = self.period.animal.processed_lfp
        return [
            processed_lfp[r][self.period.pad_start:self.period.pad_stop]
            for r in self.regions
        ]
    

class CoherenceCalculatorSegment(RelationshipCalculatorSegment, 
                                 CoherenceMethods, LFPMethods, PSDMethods):

    _name = 'coherence_segment'

    def __init__(self, coherence_calculator, data, index):
        super().__init__(coherence_calculator, data, index)


class CoherenceCalculator(RegionRelationshipCalculator, CoherenceMethods, LFPMethods, PSDMethods):

    _name = 'coherence_calculator'
    segment_class = CoherenceCalculatorSegment
    
   
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
            ev_ok = self.get_event_validity()
            valid_sets, len_sets = self.get_valid_sets(ev_ok, do_pad=True)
            if not valid_sets:
                return xr.DataArray(np.nan)

            len_longest_corr = max(len_sets) * 2 - 1
            corrs = []

            for signal_series in valid_sets:
                amp1, amp2 = [self.amplitude(signal) for signal in signal_series]
                corr, _ = normalized_xcorr(amp1, amp2, fs=fs)
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

        max_lag_sec = self.calc_opts.get('max_lag_sec', None)
        if max_lag_sec is None and 'lags' in self.calc_opts:
            max_lag_sec = self.calc_opts['lags'] / fs
        if max_lag_sec is not None:
            m = (lags >= -max_lag_sec) & (lags <= max_lag_sec)
            result = result[m]
            lags = lags[m]

        return xr.DataArray(result, dims=['lag'], coords={'lag': lags})


class GrangerFunctions:

    def fetch_granger_stat(self, d1, d2, tags, proc_name, do_weight=True, max_len_sets=None):
        ml = MatlabInterface(self.env_config['matlab_config'], tags=tags)
        data = np.vstack(self.padded_regions_data)
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
